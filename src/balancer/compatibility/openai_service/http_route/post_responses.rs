use std::sync::Arc;
use std::sync::Mutex;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use actix_web::Error;
use actix_web::HttpResponse;
use actix_web::post;
use actix_web::web;
use anyhow::anyhow;
use async_trait::async_trait;
use nanoid::nanoid;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;
use tokio_stream::StreamExt as _;

use crate::balancer::chunk_forwarding_session_controller::transforms_outgoing_message::TransformsOutgoingMessage;
use crate::balancer::compatibility::openai_service::app_data::AppData;
use crate::balancer::http_stream_from_agent::http_stream_from_agent;
use crate::balancer::inference_client::Message as OutgoingMessage;
use crate::balancer::inference_client::Response as OutgoingResponse;
use crate::balancer::unbounded_stream_from_agent::unbounded_stream_from_agent;
use crate::conversation_message::ConversationMessage;
use crate::generated_token_result::GeneratedTokenResult;
use crate::jsonrpc::ResponseEnvelope;
use crate::request_params::ContinueFromConversationHistoryParams;

pub fn register(cfg: &mut web::ServiceConfig) {
    cfg.service(respond);
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

#[derive(Deserialize)]
struct OpenAIResponsesRequestParams {
    input: Value,
    max_output_tokens: Option<i32>,
    /// This parameter is ignored here, but is required by the OpenAI API.
    model: String,
    store: Option<bool>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    parallel_tool_calls: Option<bool>,
    truncation: Option<String>,
    tool_choice: Option<Value>,
    tools: Option<Vec<Value>>,
    metadata: Option<Value>,
    user: Option<String>,
    #[serde(default)]
    stream: bool,
}

fn content_value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Array(items) => {
            let mut output = String::new();

            for item in items {
                match item {
                    Value::String(value) => output.push_str(value),
                    Value::Object(object) => {
                        let item_type = object
                            .get("type")
                            .and_then(Value::as_str)
                            .unwrap_or("input_text");
                        if item_type == "input_text" || item_type == "output_text" {
                            if let Some(Value::String(text)) = object.get("text") {
                                output.push_str(text);
                            }
                        }
                    }
                    _ => {}
                }
            }

            if output.is_empty() {
                None
            } else {
                Some(output)
            }
        }
        _ => None,
    }
}

fn input_value_to_messages(input: Value) -> Result<Vec<ConversationMessage>, Error> {
    match input {
        Value::String(input) => Ok(vec![ConversationMessage {
            content: input,
            role: "user".to_string(),
        }]),
        Value::Array(items) => items
            .into_iter()
            .filter_map(|item| match item {
                Value::String(text) => Some(Ok(ConversationMessage {
                    content: text,
                    role: "user".to_string(),
                })),
                Value::Object(object) => {
                    let item_type = object
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("message");
                    let role = object
                        .get("role")
                        .and_then(Value::as_str)
                        .unwrap_or("user")
                        .to_string();
                    let content = match item_type {
                        "input_text" => object
                            .get("text")
                            .and_then(Value::as_str)
                            .map(|value| value.to_string())
                            .ok_or_else(|| anyhow!("Missing input text"))?,
                        "message" => {
                            let content_value = object
                                .get("content")
                                .and_then(content_value_to_string)
                                .or_else(|| {
                                    object
                                        .get("text")
                                        .and_then(Value::as_str)
                                        .map(|value| value.to_string())
                                });
                            match content_value {
                                Some(content) => content,
                                None => {
                                    if object.get("content").is_some_and(Value::is_array) {
                                        return None;
                                    }
                                    return Some(Err(anyhow!("Missing input content")));
                                }
                            }
                        }
                        "input_image" | "input_file" => {
                            return None;
                        }
                        _ => return Some(Err(anyhow!("Unsupported input format"))),
                    };

                    Some(Ok(ConversationMessage { content, role }))
                }
                _ => Some(Err(anyhow!("Unsupported input format"))),
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(actix_web::error::ErrorBadRequest),
        Value::Object(object) => {
            let item_type = object
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("message");
            let role = object
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user")
                .to_string();
            let content = match item_type {
                "input_text" => object
                    .get("text")
                    .and_then(Value::as_str)
                    .map(|value| value.to_string())
                    .ok_or_else(|| actix_web::error::ErrorBadRequest("Missing input text"))?,
                "message" => {
                    let content_value = object
                        .get("content")
                        .and_then(content_value_to_string)
                        .or_else(|| {
                            object
                                .get("text")
                                .and_then(Value::as_str)
                                .map(|value| value.to_string())
                        });
                    match content_value {
                        Some(content) => content,
                        None => {
                            if object.get("content").is_some_and(Value::is_array) {
                                return Ok(vec![]);
                            }
                            return Err(actix_web::error::ErrorBadRequest(
                                "Missing input content",
                            ));
                        }
                    }
                }
                "input_image" | "input_file" => {
                    return Ok(vec![]);
                }
                _ => {
                    return Err(actix_web::error::ErrorBadRequest(
                        "Unsupported input format",
                    ))
                }
            };

            Ok(vec![ConversationMessage { content, role }])
        }
        _ => Err(actix_web::error::ErrorBadRequest(anyhow!(
            "Unsupported input format"
        ))),
    }
}

fn build_response_json(
    response_id: &str,
    model: &str,
    output_text: &str,
    item_id: &str,
    max_output_tokens: Option<i32>,
    store: Option<bool>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    parallel_tool_calls: Option<bool>,
    truncation: Option<&str>,
    tool_choice: Option<&Value>,
    tools: Option<&[Value]>,
    metadata: Option<&Value>,
    user: Option<&str>,
) -> serde_json::Value {
    json!({
        "id": response_id,
        "object": "response",
        "created_at": current_timestamp(),
        "model": model,
        "status": "completed",
        "error": null,
        "incomplete_details": null,
        "instructions": null,
        "max_output_tokens": max_output_tokens,
        "output": [
            {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                        "annotations": []
                    }
                ]
            }
        ],
        "parallel_tool_calls": parallel_tool_calls.unwrap_or(true),
        "previous_response_id": null,
        "reasoning": {
            "effort": null,
            "summary": null
        },
        "store": store.unwrap_or(true),
        "temperature": temperature.unwrap_or(1.0),
        "text": {
            "format": {
                "type": "text"
            }
        },
        "tool_choice": tool_choice.cloned().unwrap_or_else(|| json!("auto")),
        "tools": tools.cloned().unwrap_or_default(),
        "top_p": top_p.unwrap_or(1.0),
        "truncation": truncation.unwrap_or("disabled"),
        "user": user,
        "metadata": metadata.cloned().unwrap_or_else(|| json!({})),
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {
                "cached_tokens": 0
            },
            "output_tokens": 0,
            "output_tokens_details": {
                "reasoning_tokens": 0
            },
            "total_tokens": 0
        }
    })
}

#[derive(Clone)]
struct OpenAIResponsesStreamingResponseTransformer {
    model: String,
    response_id: String,
    item_id: String,
    output_text: Arc<Mutex<String>>,
    max_output_tokens: Option<i32>,
    store: Option<bool>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    parallel_tool_calls: Option<bool>,
    truncation: Option<String>,
    tool_choice: Option<Value>,
    tools: Option<Vec<Value>>,
    metadata: Option<Value>,
    user: Option<String>,
}

#[async_trait]
impl TransformsOutgoingMessage for OpenAIResponsesStreamingResponseTransformer {
    type TransformedMessage = serde_json::Value;

    async fn transform(
        &self,
        message: OutgoingMessage,
    ) -> anyhow::Result<Self::TransformedMessage> {
        match message {
            OutgoingMessage::Response(ResponseEnvelope {
                response: OutgoingResponse::GeneratedToken(GeneratedTokenResult::Done),
                ..
            }) => {
                let output_text = self
                    .output_text
                    .lock()
                    .expect("Failed to lock output text")
                    .clone();

                Ok(json!({
                    "type": "response.completed",
                    "response": build_response_json(
                        &self.response_id,
                        &self.model,
                        &output_text,
                        &self.item_id,
                        self.max_output_tokens,
                        self.store,
                        self.temperature,
                        self.top_p,
                        self.parallel_tool_calls,
                        self.truncation.as_deref(),
                        self.tool_choice.as_ref(),
                        self.tools.as_deref(),
                        self.metadata.as_ref(),
                        self.user.as_deref(),
                    )
                }))
            }
            OutgoingMessage::Response(ResponseEnvelope {
                response: OutgoingResponse::GeneratedToken(GeneratedTokenResult::Token(token)),
                ..
            }) => {
                let mut output_text = self.output_text.lock().expect("Failed to lock output text");
                output_text.push_str(&token);

                Ok(json!({
                    "type": "response.output_text.delta",
                    "response_id": self.response_id,
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": self.item_id,
                    "delta": token
                }))
            }
            _ => Ok(serde_json::to_value(&message)?),
        }
    }
}

#[derive(Clone)]
struct OpenAIResponsesCombinedResponseTransformer {}

#[async_trait]
impl TransformsOutgoingMessage for OpenAIResponsesCombinedResponseTransformer {
    type TransformedMessage = String;

    fn stringify(&self, message: &Self::TransformedMessage) -> anyhow::Result<String> {
        Ok(message.clone())
    }

    async fn transform(
        &self,
        message: OutgoingMessage,
    ) -> anyhow::Result<Self::TransformedMessage> {
        match message {
            OutgoingMessage::Response(ResponseEnvelope {
                response: OutgoingResponse::GeneratedToken(GeneratedTokenResult::Done),
                ..
            }) => Ok("".to_string()),
            OutgoingMessage::Response(ResponseEnvelope {
                response: OutgoingResponse::GeneratedToken(GeneratedTokenResult::Token(token)),
                ..
            }) => Ok(token),
            _ => Err(anyhow!("Unexpected message type: {:?}", message)),
        }
    }
}

#[post("/v1/responses")]
async fn respond(
    app_data: web::Data<AppData>,
    openai_params: web::Json<OpenAIResponsesRequestParams>,
) -> Result<HttpResponse, Error> {
    let conversation_history = input_value_to_messages(openai_params.input.clone())?;

    let paddler_params = ContinueFromConversationHistoryParams {
        add_generation_prompt: true,
        conversation_history,
        enable_thinking: true,
        max_tokens: openai_params.max_output_tokens.unwrap_or(2000),
        tools: vec![],
    };

    if openai_params.stream {
        http_stream_from_agent(
            app_data.buffered_request_manager.clone(),
            app_data.inference_service_configuration.clone(),
            paddler_params,
            OpenAIResponsesStreamingResponseTransformer {
                model: openai_params.model.clone(),
                response_id: nanoid!(),
                item_id: nanoid!(),
                output_text: Arc::new(Mutex::new(String::new())),
                max_output_tokens: openai_params.max_output_tokens,
                store: openai_params.store,
                temperature: openai_params.temperature,
                top_p: openai_params.top_p,
                parallel_tool_calls: openai_params.parallel_tool_calls,
                truncation: openai_params.truncation.clone(),
                tool_choice: openai_params.tool_choice.clone(),
                tools: openai_params.tools.clone(),
                metadata: openai_params.metadata.clone(),
                user: openai_params.user.clone(),
            },
        )
    } else {
        let combined_response = unbounded_stream_from_agent(
            app_data.buffered_request_manager.clone(),
            app_data.inference_service_configuration.clone(),
            paddler_params,
            OpenAIResponsesCombinedResponseTransformer {},
        )?
        .collect::<Vec<String>>()
        .await
        .join("");

        Ok(HttpResponse::Ok().json(build_response_json(
            &nanoid!(),
            &openai_params.model,
            &combined_response,
            &nanoid!(),
            openai_params.max_output_tokens,
            openai_params.store,
            openai_params.temperature,
            openai_params.top_p,
            openai_params.parallel_tool_calls,
            openai_params.truncation.as_deref(),
            openai_params.tool_choice.as_ref(),
            openai_params.tools.as_deref(),
            openai_params.metadata.as_ref(),
            openai_params.user.as_deref(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_content_value_to_string() {
        // String input
        assert_eq!(
            content_value_to_string(&json!("hello")),
            Some("hello".to_string())
        );

        // Array input
        assert_eq!(
            content_value_to_string(&json!(["hello", " ", "world"])),
            Some("hello world".to_string())
        );

        // Array with objects
        assert_eq!(
            content_value_to_string(&json!([
                {"text": "hello"},
                " ",
                {"text": "world"}
            ])),
            Some("hello world".to_string())
        );

        // Invalid inputs
        assert_eq!(content_value_to_string(&json!(123)), None);
        assert_eq!(content_value_to_string(&json!([])), None);
        assert_eq!(content_value_to_string(&json!([123])), None);
    }

    #[test]
    fn test_input_value_to_messages_string() {
        let input = json!("hello world");
        let messages = input_value_to_messages(input).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "hello world");
    }

    #[test]
    fn test_input_value_to_messages_array_strings() {
        let input = json!(["message 1", "message 2"]);
        let messages = input_value_to_messages(input).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "message 1");
        assert_eq!(messages[1].content, "message 2");
    }

    #[test]
    fn test_input_value_to_messages_array_objects() {
        let input = json!([
            {"role": "system", "content": "you are a bot"},
            {"role": "user", "text": "hello"}
        ]);
        let messages = input_value_to_messages(input).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content, "you are a bot");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content, "hello");
    }

    #[test]
    fn test_input_value_to_messages_input_text_item() {
        let input = json!([
            {"type": "input_text", "text": "hello"}
        ]);
        let messages = input_value_to_messages(input).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "hello");
    }

    #[test]
    fn test_input_value_to_messages_message_with_input_text_content() {
        let input = json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}]
        });
        let messages = input_value_to_messages(input).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "hello");
    }

    #[test]
    fn test_input_value_to_messages_single_object() {
        let input = json!({"role": "system", "content": "system message"});
        let messages = input_value_to_messages(input).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content, "system message");
    }

    #[test]
    fn test_input_value_to_messages_invalid() {
        assert!(input_value_to_messages(json!(123)).is_err());
        assert!(input_value_to_messages(json!([123])).is_err());
        assert!(input_value_to_messages(json!({"foo": "bar"})).is_err());
        assert!(input_value_to_messages(json!({"type": "input_text"})).is_err());
        assert!(input_value_to_messages(json!({"type": "message"})).is_err());
        assert!(input_value_to_messages(json!({"type": "unknown", "text": "hello"})).is_err());
        assert!(input_value_to_messages(json!([{"type": "input_text"}])).is_err());
        assert!(input_value_to_messages(json!([{"type": "unknown", "text": "hello"}])).is_err());
    }

    #[test]
    fn test_input_value_to_messages_ignores_non_text_items() {
        let input = json!([
            {"type": "input_image", "image_url": "https://example.com/image.png"},
            {"type": "input_text", "text": "hello"}
        ]);
        let messages = input_value_to_messages(input).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "hello");
    }

    #[test]
    fn test_input_value_to_messages_only_non_text_items() {
        let input = json!([
            {"type": "input_image", "image_url": "https://example.com/image.png"},
            {"type": "input_file", "file_id": "file_123"}
        ]);
        let messages = input_value_to_messages(input).unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_input_value_to_messages_message_only_non_text_content() {
        let input = json!([{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": "https://example.com/image.png"}]
        }]);
        let messages = input_value_to_messages(input).unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_input_value_to_messages_single_message_only_non_text_content() {
        let input = json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_file", "file_id": "file_123"}]
        });
        let messages = input_value_to_messages(input).unwrap();
        assert!(messages.is_empty());
    }
}

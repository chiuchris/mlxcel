// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::{build_chat_messages, build_raw_json_messages, prepare_chat_request};
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::types::request::{
    FunctionDefinition, Tool, ToolCallFunction, ToolCallInMessage, ToolChoice,
};
use crate::server::types::{
    ChatCompletionRequest, ContentPart, ImageUrl, Message, MessageContent, Role, SamplingParams,
};

fn request_with_messages(messages: Vec<Message>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "test-model".to_string(),
        messages,
        stream: false,
        stream_options: None,
        logprobs: None,
        top_logprobs: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        params: SamplingParams::default(),
    }
}

#[test]
fn build_chat_messages_flattens_text_parts() {
    let request = request_with_messages(vec![Message {
        role: Role::User,
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Hello".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,aGVsbG8=".to_string(),
                },
            },
            ContentPart::Text {
                text: " world".to_string(),
            },
        ]),
        name: None,
        tool_call_id: None,
        tool_calls: None,
    }]);

    let messages = build_chat_messages(&request);
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content, "Hello world");
}

#[tokio::test]
async fn prepare_chat_request_uses_template_output_and_extracts_images() {
    let request = request_with_messages(vec![Message {
        role: Role::User,
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Look".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,aGVsbG8=".to_string(),
                },
            },
        ]),
        name: None,
        tool_call_id: None,
        tool_calls: None,
    }]);
    let processor =
        ChatTemplateProcessor::with_template("Prompt: {{ messages[0].content }}".to_string());

    let prepared = prepare_chat_request(&processor, &request).await;
    assert_eq!(prepared.prompt, "Prompt: Look");
    assert_eq!(prepared.image_data, vec![b"hello".to_vec()]);
}

#[tokio::test]
async fn prepare_chat_request_falls_back_to_simple_prompt_on_template_error() {
    let request = request_with_messages(vec![Message {
        role: Role::User,
        content: MessageContent::Text("Hello".to_string()),
        name: None,
        tool_call_id: None,
        tool_calls: None,
    }]);
    let processor = ChatTemplateProcessor::with_template("{% if %}".to_string());

    let prepared = prepare_chat_request(&processor, &request).await;
    assert_eq!(prepared.prompt, request.to_prompt());
}

// -- Tool calling request deserialization tests --

#[test]
fn deserialize_request_with_tools() {
    let json = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }],
        "tool_choice": "auto"
    }"#;

    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert!(req.tools.is_some());
    let tools = req.tools.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
    assert!(tools[0].function.parameters.is_some());
}

#[test]
fn deserialize_request_tool_choice_string() {
    let json = r#"{
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "none"
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert!(req.tool_choice.is_some());
    let tc = req.tool_choice.unwrap();
    assert!(tc.is_none());
}

#[test]
fn deserialize_request_tool_choice_specific_function() {
    let json = r#"{
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "function", "function": {"name": "my_fn"}}
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    let tc = req.tool_choice.unwrap();
    assert_eq!(tc.specific_function(), Some("my_fn"));
    assert_eq!(tc.mode(), "specific");
}

#[test]
fn deserialize_message_with_tool_calls() {
    let json = r#"{
        "model": "test",
        "messages": [
            {"role": "user", "content": "Call weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}
                }]
            },
            {
                "role": "tool",
                "content": "Sunny, 25C",
                "tool_call_id": "call_abc"
            }
        ]
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.messages.len(), 3);
    // assistant message has tool_calls
    assert!(req.messages[1].tool_calls.is_some());
    let tc = req.messages[1].tool_calls.as_ref().unwrap();
    assert_eq!(tc[0].function.name, "get_weather");
    // tool message has tool_call_id
    assert_eq!(req.messages[2].role, Role::Tool);
    assert_eq!(req.messages[2].tool_call_id.as_deref(), Some("call_abc"));
}

#[test]
fn deserialize_request_without_tools() {
    let json = r#"{
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}]
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert!(req.tools.is_none());
    assert!(req.tool_choice.is_none());
    assert!(req.parallel_tool_calls.is_none());
}

#[test]
fn build_raw_json_messages_includes_tool_fields() {
    let request = ChatCompletionRequest {
        model: "test".to_string(),
        messages: vec![
            Message {
                role: Role::Assistant,
                content: MessageContent::Text(String::new()),
                name: None,
                tool_call_id: None,
                tool_calls: Some(vec![ToolCallInMessage {
                    id: "call_123".to_string(),
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "get_weather".to_string(),
                        arguments: r#"{"location":"Paris"}"#.to_string(),
                    },
                }]),
            },
            Message {
                role: Role::Tool,
                content: MessageContent::Text("Sunny".to_string()),
                name: None,
                tool_call_id: Some("call_123".to_string()),
                tool_calls: None,
            },
        ],
        stream: false,
        stream_options: None,
        logprobs: None,
        top_logprobs: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        params: SamplingParams::default(),
    };

    let raw = build_raw_json_messages(&request);
    let arr = raw.as_array().unwrap();
    assert_eq!(arr.len(), 2);

    // assistant message should have tool_calls
    assert!(arr[0].get("tool_calls").is_some());
    let tc = arr[0]["tool_calls"].as_array().unwrap();
    assert_eq!(tc[0]["function"]["name"].as_str().unwrap(), "get_weather");

    // tool message should have tool_call_id
    assert_eq!(arr[1]["tool_call_id"].as_str().unwrap(), "call_123");
    assert_eq!(arr[1]["role"].as_str().unwrap(), "tool");
}

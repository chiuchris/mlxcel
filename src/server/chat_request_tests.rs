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
use crate::server::types::request::{ToolCallFunction, ToolCallInMessage};
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
        chat_template_kwargs: None,
        extra_body: None,
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

    let prepared = prepare_chat_request(&processor, &request, None).await;
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

    let prepared = prepare_chat_request(&processor, &request, None).await;
    assert_eq!(prepared.prompt, request.to_prompt());
}

// ---------------------------------------------------------------------------
// Security H-1: Jinja fallback path must not bypass rolling-checkpoint strip
// ---------------------------------------------------------------------------

/// A deliberately-broken Jinja template that forces the fallback path by
/// producing a parse error at render time.
fn broken_template() -> String {
    // `{% if %}` is a Jinja parse error (missing condition expression).
    "{% if %}".to_string()
}

#[tokio::test]
async fn h1_fallback_strips_think_blocks_when_preserve_thinking_false() {
    // 3-turn conversation: u1, a1 (with <think>), u2, a2 (with <think>), u3.
    // With preserve_thinking=false (default) and a broken Jinja template that
    // forces the fallback path, the resulting prompt MUST NOT contain any
    // <think> markers from a1 or a2 (both are strictly before u3).
    let request = three_turn_request_with_think_blocks();
    let processor = ChatTemplateProcessor::with_template(broken_template());

    let prepared = prepare_chat_request(&processor, &request, None).await;

    assert!(
        !prepared.prompt.contains("<think>"),
        "fallback must strip <think> markers when preserve_thinking=false; got: {:?}",
        prepared.prompt
    );
    assert!(
        !prepared.prompt.contains("calc 2+2"),
        "fallback must strip thinking content when preserve_thinking=false; got: {:?}",
        prepared.prompt
    );
    assert!(
        !prepared.prompt.contains("calc 3+3"),
        "fallback must strip thinking content when preserve_thinking=false; got: {:?}",
        prepared.prompt
    );
    // Answer text (outside <think> blocks) must survive.
    assert!(
        prepared.prompt.contains("The answer is 4."),
        "fallback must preserve non-thinking content; got: {:?}",
        prepared.prompt
    );
    assert!(
        prepared.prompt.contains("Six."),
        "fallback must preserve non-thinking content; got: {:?}",
        prepared.prompt
    );
}

#[tokio::test]
async fn h1_fallback_preserves_think_blocks_when_preserve_thinking_true() {
    // Same broken-template setup, but with preserve_thinking=true:
    // all <think> blocks must survive in the fallback output.
    let mut request = three_turn_request_with_think_blocks();
    let mut map = serde_json::Map::new();
    map.insert(
        "preserve_thinking".to_string(),
        serde_json::Value::Bool(true),
    );
    request.chat_template_kwargs = Some(map);

    let processor = ChatTemplateProcessor::with_template(broken_template());

    let prepared = prepare_chat_request(&processor, &request, None).await;

    assert!(
        prepared.prompt.contains("<think>"),
        "fallback must retain <think> markers when preserve_thinking=true; got: {:?}",
        prepared.prompt
    );
    assert!(
        prepared.prompt.contains("calc 2+2"),
        "fallback must retain thinking content when preserve_thinking=true; got: {:?}",
        prepared.prompt
    );
    assert!(
        prepared.prompt.contains("calc 3+3"),
        "fallback must retain thinking content when preserve_thinking=true; got: {:?}",
        prepared.prompt
    );
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
        chat_template_kwargs: None,
        extra_body: None,
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

// ---------------------------------------------------------------------------
// Issue #410 — preserve_thinking plumbing through prepare_chat_request
// ---------------------------------------------------------------------------

use crate::server::chat_template_kwargs::ChatTemplateKwargs;

fn three_turn_request_with_think_blocks() -> ChatCompletionRequest {
    // Canonical 3-turn conversation: u1, a1 (with think), u2, a2 (with think), u3.
    // Per rolling checkpoint, a1 is before the latest user turn (u3), so its
    // think block should be stripped when preserve_thinking=false. a2 is also
    // strictly before u3 — both get stripped.
    request_with_messages(vec![
        Message {
            role: Role::User,
            content: MessageContent::Text("What is 2+2?".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            role: Role::Assistant,
            content: MessageContent::Text(
                "<think>\ncalc 2+2\n</think>\n\nThe answer is 4.".to_string(),
            ),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            role: Role::User,
            content: MessageContent::Text("And 3+3?".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            role: Role::Assistant,
            content: MessageContent::Text("<think>\ncalc 3+3\n</think>\n\nSix.".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            role: Role::User,
            content: MessageContent::Text("And 4+4?".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
    ])
}

fn dump_template() -> String {
    // Template that echoes every message's raw content — lets us assert on
    // exactly what reached the renderer.
    "{% for m in messages %}[{{ m.role }}: {{ m.content }}]{% endfor %}".to_string()
}

#[tokio::test]
async fn preserve_thinking_false_strips_all_prior_think_blocks() {
    // Default: preserve_thinking=false → both a1 and a2's <think> blocks
    // are stripped (both are strictly before u3).
    let request = three_turn_request_with_think_blocks();
    let processor = ChatTemplateProcessor::with_template(dump_template());
    let prepared = prepare_chat_request(&processor, &request, None).await;

    // Neither think content nor markers should remain.
    assert!(
        !prepared.prompt.contains("<think>"),
        "no <think> open should survive strip, got: {:?}",
        prepared.prompt
    );
    assert!(!prepared.prompt.contains("calc 2+2"));
    assert!(!prepared.prompt.contains("calc 3+3"));
    // Answer content is preserved.
    assert!(prepared.prompt.contains("The answer is 4."));
    assert!(prepared.prompt.contains("Six."));
}

#[tokio::test]
async fn preserve_thinking_true_retains_all_think_blocks() {
    let mut request = three_turn_request_with_think_blocks();
    // Per-request top-level preserve_thinking=true.
    let mut map = serde_json::Map::new();
    map.insert(
        "preserve_thinking".to_string(),
        serde_json::Value::Bool(true),
    );
    request.chat_template_kwargs = Some(map);

    let processor = ChatTemplateProcessor::with_template(dump_template());
    let prepared = prepare_chat_request(&processor, &request, None).await;

    // All think blocks retained.
    assert!(prepared.prompt.contains("<think>"));
    assert!(prepared.prompt.contains("calc 2+2"));
    assert!(prepared.prompt.contains("calc 3+3"));
    assert!(prepared.prompt.contains("The answer is 4."));
    assert!(prepared.prompt.contains("Six."));
}

#[tokio::test]
async fn preserve_thinking_via_extra_body_nested_works() {
    // vLLM/OpenAI client shape: nested under extra_body.
    let mut request = three_turn_request_with_think_blocks();
    let mut nested = serde_json::Map::new();
    nested.insert(
        "preserve_thinking".to_string(),
        serde_json::Value::Bool(true),
    );
    let mut extra = serde_json::Map::new();
    extra.insert(
        "chat_template_kwargs".to_string(),
        serde_json::Value::Object(nested),
    );
    request.extra_body = Some(extra);

    let processor = ChatTemplateProcessor::with_template(dump_template());
    let prepared = prepare_chat_request(&processor, &request, None).await;
    assert!(prepared.prompt.contains("<think>"));
}

#[tokio::test]
async fn preserve_thinking_via_extra_body_flat_dashscope_works() {
    // DashScope shape: flat extra_body.preserve_thinking.
    let mut request = three_turn_request_with_think_blocks();
    let mut extra = serde_json::Map::new();
    extra.insert(
        "preserve_thinking".to_string(),
        serde_json::Value::Bool(true),
    );
    request.extra_body = Some(extra);

    let processor = ChatTemplateProcessor::with_template(dump_template());
    let prepared = prepare_chat_request(&processor, &request, None).await;
    assert!(prepared.prompt.contains("<think>"));
}

#[tokio::test]
async fn server_default_preserve_thinking_applies_when_request_empty() {
    // Server sets preserve_thinking=true; request does not override.
    let request = three_turn_request_with_think_blocks();
    let server_default =
        ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": true}"#).unwrap();
    let processor = ChatTemplateProcessor::with_template(dump_template());
    let prepared = prepare_chat_request(&processor, &request, Some(&server_default)).await;
    assert!(prepared.prompt.contains("<think>"));
}

#[tokio::test]
async fn per_request_overrides_server_default_for_single_key() {
    // Server: preserve_thinking=true. Request: preserve_thinking=false.
    // Merge → per-request wins → think blocks stripped.
    let mut request = three_turn_request_with_think_blocks();
    let mut map = serde_json::Map::new();
    map.insert(
        "preserve_thinking".to_string(),
        serde_json::Value::Bool(false),
    );
    request.chat_template_kwargs = Some(map);

    let server_default =
        ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": true}"#).unwrap();
    let processor = ChatTemplateProcessor::with_template(dump_template());
    let prepared = prepare_chat_request(&processor, &request, Some(&server_default)).await;
    assert!(
        !prepared.prompt.contains("<think>"),
        "per-request false must override server default true"
    );
}

#[tokio::test]
async fn prefix_stability_across_turns_when_preserve_thinking_true() {
    // KV-cache prefix stability regression guard: when preserve_thinking=true,
    // rendering a conversation with N turns must produce a prompt that BEGINS
    // WITH the rendering of the same conversation with the final user turn
    // removed. Without this guarantee, older turns would be re-serialized
    // differently between turns and the KV cache could not be reused.
    //
    // The per-request kwarg is set to true; the template is the identity dump
    // template used above.
    let processor = ChatTemplateProcessor::with_template(dump_template());

    // Turn 1+2: u1, a1 (with think), u2 — a ongoing conversation.
    let messages_t2 = vec![
        Message {
            role: Role::User,
            content: MessageContent::Text("q1".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            role: Role::Assistant,
            content: MessageContent::Text("<think>think1</think>a1".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            role: Role::User,
            content: MessageContent::Text("q2".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
    ];

    // Turn 1+2+3: same plus a2 (with think), u3.
    let mut messages_t3 = messages_t2.clone();
    messages_t3.push(Message {
        role: Role::Assistant,
        content: MessageContent::Text("<think>think2</think>a2".to_string()),
        name: None,
        tool_call_id: None,
        tool_calls: None,
    });
    messages_t3.push(Message {
        role: Role::User,
        content: MessageContent::Text("q3".to_string()),
        name: None,
        tool_call_id: None,
        tool_calls: None,
    });

    let kwargs = {
        let mut m = serde_json::Map::new();
        m.insert(
            "preserve_thinking".to_string(),
            serde_json::Value::Bool(true),
        );
        Some(m)
    };

    let request_t2 = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: messages_t2,
        stream: false,
        stream_options: None,
        logprobs: None,
        top_logprobs: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        chat_template_kwargs: kwargs.clone(),
        extra_body: None,
        params: SamplingParams::default(),
    };
    let request_t3 = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: messages_t3,
        stream: false,
        stream_options: None,
        logprobs: None,
        top_logprobs: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        chat_template_kwargs: kwargs,
        extra_body: None,
        params: SamplingParams::default(),
    };

    let prep_t2 = prepare_chat_request(&processor, &request_t2, None).await;
    let prep_t3 = prepare_chat_request(&processor, &request_t3, None).await;

    // Prefix stability: prep_t3 starts with prep_t2's rendering. This
    // guarantees the KV cache built for turn 2 can be reused as a prefix
    // for turn 3.
    assert!(
        prep_t3.prompt.starts_with(&prep_t2.prompt),
        "preserve_thinking=true must yield stable prefix across turns;\nt2: {:?}\nt3: {:?}",
        prep_t2.prompt,
        prep_t3.prompt
    );
    // And of course both must retain their think blocks.
    assert!(prep_t2.prompt.contains("think1"));
    assert!(prep_t3.prompt.contains("think1"));
    assert!(prep_t3.prompt.contains("think2"));
}

#[test]
fn deserialize_request_with_top_level_chat_template_kwargs() {
    // Primary llama.cpp shape.
    let json = r#"{
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"preserve_thinking": true}
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert!(req.chat_template_kwargs.is_some());
    let k = req.chat_template_kwargs.unwrap();
    assert_eq!(
        k.get("preserve_thinking"),
        Some(&serde_json::Value::Bool(true))
    );
}

#[test]
fn deserialize_request_with_extra_body() {
    let json = r#"{
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"preserve_thinking": true}
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert!(req.extra_body.is_some());
    let e = req.extra_body.unwrap();
    assert_eq!(
        e.get("preserve_thinking"),
        Some(&serde_json::Value::Bool(true))
    );
}

#[tokio::test]
async fn top_level_wins_over_extra_body_when_both_present() {
    // AC: primary llama.cpp shape beats DashScope flat shape.
    let mut request = three_turn_request_with_think_blocks();
    let mut top = serde_json::Map::new();
    top.insert(
        "preserve_thinking".to_string(),
        serde_json::Value::Bool(true),
    );
    request.chat_template_kwargs = Some(top);
    let mut extra = serde_json::Map::new();
    extra.insert(
        "preserve_thinking".to_string(),
        serde_json::Value::Bool(false),
    );
    request.extra_body = Some(extra);

    let processor = ChatTemplateProcessor::with_template(dump_template());
    let prepared = prepare_chat_request(&processor, &request, None).await;
    // Top-level says true → think blocks retained despite extra_body saying false.
    assert!(prepared.prompt.contains("<think>"));
}

#[tokio::test]
async fn rolling_checkpoint_tolerates_tool_turn_in_middle() {
    // Conversation with a tool turn between user turns: must not confuse the
    // latest-user-turn anchor.
    let messages = vec![
        Message {
            role: Role::User,
            content: MessageContent::Text("q1".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            role: Role::Assistant,
            content: MessageContent::Text("<think>plan</think>call tool".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            role: Role::User,
            content: MessageContent::Text("q2".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            role: Role::Tool,
            content: MessageContent::Text("tool-result".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        },
    ];
    let request = request_with_messages(messages);
    let processor = ChatTemplateProcessor::with_template(dump_template());
    // preserve_thinking=false (default): threshold = index 2 (last "user"),
    // strip only the assistant message at index 1.
    let prepared = prepare_chat_request(&processor, &request, None).await;
    assert!(
        !prepared.prompt.contains("<think>"),
        "think in first assistant reply must be stripped"
    );
    assert!(
        prepared.prompt.contains("call tool"),
        "assistant reply content must be preserved after strip"
    );
    assert!(prepared.prompt.contains("tool-result"));
}

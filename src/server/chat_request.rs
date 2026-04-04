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

//! Shared chat-request preparation helpers.
//!
//! Both streaming and non-streaming chat routes should apply the same message
//! flattening, template rendering, and image extraction rules.
//!
//! Used by: routes/chat

use super::chat_template::{ChatMessage, ChatTemplateProcessor};
use super::media::{extract_chat_audio_data, extract_chat_image_data};
use super::types::ChatCompletionRequest;
use super::types::request::Tool;

pub(crate) struct PreparedChatRequest {
    pub(crate) prompt: String,
    pub(crate) image_data: Vec<Vec<u8>>,
    pub(crate) audio_data: Vec<Vec<u8>>,
}

pub(crate) async fn prepare_chat_request(
    processor: &ChatTemplateProcessor,
    request: &ChatCompletionRequest,
) -> PreparedChatRequest {
    // Determine effective tools based on tool_choice
    let effective_tools = effective_tools(request);

    let prompt = if has_tool_fields(request) {
        // When messages contain tool_calls or tool_call_id, use raw JSON
        // rendering so the Jinja2 template can access all fields.
        let raw_messages = build_raw_json_messages(request);
        processor
            .apply_raw(&raw_messages, effective_tools)
            .unwrap_or_else(|err| {
                tracing::warn!(
                    "Chat template render (raw) failed, using fallback: {:#}",
                    err
                );
                request.to_prompt()
            })
    } else {
        let messages = build_chat_messages(request);
        processor
            .apply(&messages, effective_tools)
            .unwrap_or_else(|err| {
                tracing::warn!("Chat template render failed, using fallback: {:#}", err);
                request.to_prompt()
            })
    };

    let (image_data, audio_data) = tokio::join!(
        extract_chat_image_data(request),
        extract_chat_audio_data(request),
    );

    PreparedChatRequest {
        prompt,
        image_data,
        audio_data,
    }
}

/// Determine the effective tools slice to pass to the template.
///
/// Returns `None` when tool_choice is "none" or no tools are provided.
fn effective_tools(request: &ChatCompletionRequest) -> Option<&[Tool]> {
    // If tool_choice is "none", do not pass tools to template
    if let Some(ref tc) = request.tool_choice
        && tc.is_none()
    {
        return None;
    }
    request.tools.as_deref()
}

/// Check if any message in the request has tool-related fields that
/// require raw JSON rendering (tool_calls, tool_call_id).
fn has_tool_fields(request: &ChatCompletionRequest) -> bool {
    request
        .messages
        .iter()
        .any(|m| m.tool_call_id.is_some() || m.tool_calls.is_some())
}

/// Build raw JSON messages for template rendering.
///
/// This preserves all fields (including tool_calls, tool_call_id) so that
/// Jinja2 templates can access them for multi-turn tool use conversations.
fn build_raw_json_messages(request: &ChatCompletionRequest) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = request
        .messages
        .iter()
        .map(|m| {
            let mut msg = serde_json::json!({
                "role": m.role.as_str(),
                "content": m.content.text(),
            });

            if let Some(ref name) = m.name {
                msg["name"] = serde_json::Value::String(name.clone());
            }
            if let Some(ref tool_call_id) = m.tool_call_id {
                msg["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
            }
            if let Some(ref tool_calls) = m.tool_calls {
                msg["tool_calls"] =
                    serde_json::to_value(tool_calls).unwrap_or(serde_json::Value::Null);
            }

            msg
        })
        .collect();

    serde_json::Value::Array(messages)
}

pub(crate) fn build_chat_messages(request: &ChatCompletionRequest) -> Vec<ChatMessage> {
    request
        .messages
        .iter()
        .map(|message| ChatMessage {
            role: message.role.as_str().to_string(),
            content: message.content.text(),
        })
        .collect()
}

#[cfg(test)]
#[path = "chat_request_tests.rs"]
mod tests;

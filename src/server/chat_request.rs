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

use super::chat_template::{ChatMessage, ChatTemplateProcessor};
use super::media::extract_chat_image_data;
use super::types::ChatCompletionRequest;

pub(crate) struct PreparedChatRequest {
    pub(crate) prompt: String,
    pub(crate) image_data: Vec<Vec<u8>>,
}

pub(crate) fn prepare_chat_request(
    processor: &ChatTemplateProcessor,
    request: &ChatCompletionRequest,
) -> PreparedChatRequest {
    let messages = build_chat_messages(request);
    let prompt = processor.apply(&messages).unwrap_or_else(|err| {
        tracing::warn!("Chat template render failed, using fallback: {:#}", err);
        request.to_prompt()
    });

    PreparedChatRequest {
        prompt,
        image_data: extract_chat_image_data(request),
    }
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

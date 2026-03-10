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

use super::{build_chat_messages, prepare_chat_request};
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::types::{
    ChatCompletionRequest, ContentPart, ImageUrl, Message, MessageContent, Role, SamplingParams,
};

fn request_with_messages(messages: Vec<Message>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "test-model".to_string(),
        messages,
        stream: false,
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
    }]);
    let processor = ChatTemplateProcessor::with_template("{% if %}".to_string());

    let prepared = prepare_chat_request(&processor, &request).await;
    assert_eq!(prepared.prompt, request.to_prompt());
}

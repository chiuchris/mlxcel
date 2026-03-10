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

use std::fs;

use super::{collect_image_data, extract_chat_image_data, read_image_url};
use crate::server::types::{
    ChatCompletionRequest, ContentPart, ImageUrl, Message, MessageContent, Role, SamplingParams,
};

fn build_chat_request(parts: Vec<ContentPart>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Parts(parts),
            name: None,
        }],
        stream: false,
        params: SamplingParams::default(),
    }
}

#[test]
fn read_image_url_decodes_base64_data_uri() {
    let bytes = read_image_url("data:image/png;base64,aGVsbG8=").unwrap();
    assert_eq!(bytes, b"hello");
}

#[test]
fn read_image_url_rejects_non_base64_data_uri() {
    assert!(read_image_url("data:image/png,hello").is_none());
}

#[test]
fn collect_image_data_skips_invalid_entries() {
    let images = collect_image_data([
        "data:image/png;base64,aGVsbG8=",
        "https://example.com/image.png",
        "data:image/png;base64,%%%bad%%%",
    ]);
    assert_eq!(images, vec![b"hello".to_vec()]);
}

#[test]
fn extract_chat_image_data_reads_file_urls() {
    let path = std::env::temp_dir().join(format!("mlxcel-media-{}.bin", uuid::Uuid::new_v4()));
    fs::write(&path, b"image-bytes").unwrap();

    let request = build_chat_request(vec![ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: format!("file://{}", path.display()),
        },
    }]);

    let images = extract_chat_image_data(&request);
    assert_eq!(images, vec![b"image-bytes".to_vec()]);

    fs::remove_file(path).unwrap();
}

#[test]
fn extract_chat_image_data_collects_images_across_messages() {
    let request = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("ignore".to_string()),
                name: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "look".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "data:image/png;base64,aGVsbG8=".to_string(),
                        },
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "data:image/png;base64,d29ybGQ=".to_string(),
                        },
                    },
                ]),
                name: None,
            },
        ],
        stream: false,
        params: SamplingParams::default(),
    };

    let images = extract_chat_image_data(&request);
    assert_eq!(images, vec![b"hello".to_vec(), b"world".to_vec()]);
}

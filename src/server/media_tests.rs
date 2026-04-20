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
use base64::Engine;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn build_chat_request(parts: Vec<ContentPart>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Parts(parts),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }],
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

fn tiny_png_bytes() -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2N1ekAAAAASUVORK5CYII=")
        .unwrap()
}

#[tokio::test]
async fn read_image_url_decodes_base64_data_uri() {
    let bytes = read_image_url("data:image/png;base64,aGVsbG8=")
        .await
        .unwrap();
    assert_eq!(bytes, b"hello");
}

#[tokio::test]
async fn read_image_url_rejects_non_base64_data_uri() {
    assert!(read_image_url("data:image/png,hello").await.is_none());
}

#[tokio::test]
async fn read_image_url_rejects_malformed_data_uri() {
    assert!(read_image_url("data:image/png;base64").await.is_none());
}

#[tokio::test]
async fn read_image_url_reads_bare_local_paths() {
    let path = std::env::temp_dir().join(format!("mlxcel-media-{}.png", uuid::Uuid::new_v4()));
    let payload = tiny_png_bytes();
    fs::write(&path, &payload).unwrap();

    let bytes = read_image_url(path.to_str().unwrap()).await.unwrap();
    assert_eq!(bytes, payload);

    fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn read_image_url_fetches_http_urls() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let payload = tiny_png_bytes();
    let payload_clone = payload.clone();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0u8; 1024];
        let _ = socket.read(&mut request).await.unwrap();

        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: image/png\r\nConnection: close\r\n\r\n",
            payload_clone.len()
        );
        socket.write_all(header.as_bytes()).await.unwrap();
        socket.write_all(&payload_clone).await.unwrap();
    });

    let url = format!("http://{}/image.png", addr);
    let bytes = read_image_url(&url).await.unwrap();
    assert_eq!(bytes, payload);

    server.await.unwrap();
}

#[tokio::test]
async fn collect_image_data_skips_invalid_entries() {
    let images = collect_image_data([
        "data:image/png;base64,aGVsbG8=",
        "ftp://example.com/image.png",
        "data:image/png;base64,%%%bad%%%",
    ])
    .await;
    assert_eq!(images, vec![b"hello".to_vec()]);
}

#[tokio::test]
async fn extract_chat_image_data_reads_file_urls() {
    let path = std::env::temp_dir().join(format!("mlxcel-media-{}.bin", uuid::Uuid::new_v4()));
    fs::write(&path, b"image-bytes").unwrap();

    let request = build_chat_request(vec![ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: format!("file://{}", path.display()),
        },
    }]);

    let images = extract_chat_image_data(&request).await;
    assert_eq!(images, vec![b"image-bytes".to_vec()]);

    fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn extract_chat_image_data_collects_images_across_messages() {
    let request = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("ignore".to_string()),
                name: None,
                tool_call_id: None,
                tool_calls: None,
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
                tool_call_id: None,
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

    let images = extract_chat_image_data(&request).await;
    assert_eq!(images, vec![b"hello".to_vec(), b"world".to_vec()]);
}

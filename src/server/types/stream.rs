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

//! SSE streaming response types

use serde::Serialize;
use serde_json::Value;

/// Delta content for streaming
#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Streaming choice
///
/// `finish_reason` is always serialized (as `null` in content chunks, as a
/// string value in the final chunk). This is required by the OpenAI spec and
/// expected by clients such as opencode, Continue, and Cursor.
#[derive(Debug, Clone, Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: Delta,
    /// Always serialized: `null` in content chunks, "stop"/"length" in final
    pub finish_reason: Option<String>,
    /// Always null for now; present to satisfy strict client parsers
    pub logprobs: Option<Value>,
}

/// Chat completion chunk for streaming
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    /// Always null for now; present to satisfy strict client parsers
    pub system_fingerprint: Option<String>,
    pub choices: Vec<StreamChoice>,
    /// Only present in the final usage chunk (when stream_options.include_usage
    /// is true). Omitted from all other chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<super::response::Usage>,
}

impl ChatCompletionChunk {
    /// Create initial chunk with role
    pub fn initial(id: String, model: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Create content chunk
    pub fn content(id: String, model: String, content: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(content),
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Create final chunk with finish reason
    pub fn finish(id: String, model: String, finish_reason: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Create usage chunk sent when `stream_options.include_usage` is true.
    ///
    /// The OpenAI spec requires this to be a separate chunk with an empty
    /// `choices` array and a populated `usage` object.
    pub fn usage(
        id: String,
        model: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![],
            usage: Some(super::response::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
        }
    }
}

/// Text completion chunk for streaming
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    /// Always null for now; present to satisfy strict client parsers
    pub system_fingerprint: Option<String>,
    pub choices: Vec<CompletionStreamChoice>,
    /// Only present in the final usage chunk (when stream_options.include_usage
    /// is true). Omitted from all other chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<super::response::Usage>,
}

/// Streaming choice for text completions
///
/// `finish_reason` is always serialized (as `null` in content chunks) per the
/// OpenAI spec.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionStreamChoice {
    pub index: usize,
    pub text: String,
    /// Always serialized: `null` in content chunks, "stop"/"length" in final
    pub finish_reason: Option<String>,
    /// Always null for now; present to satisfy strict client parsers
    pub logprobs: Option<Value>,
}

impl CompletionChunk {
    /// Create content chunk
    pub fn content(id: String, model: String, text: String) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![CompletionStreamChoice {
                index: 0,
                text,
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Create final chunk with finish reason
    pub fn finish(id: String, model: String, finish_reason: String) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![CompletionStreamChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Create usage chunk sent when `stream_options.include_usage` is true.
    pub fn usage(
        id: String,
        model: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![],
            usage: Some(super::response::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
        }
    }
}

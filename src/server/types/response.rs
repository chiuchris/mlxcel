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

//! OpenAI and llama-server compatible response types

use serde::Serialize;

// ---------------------------------------------------------------------------
// Logprobs structures
// ---------------------------------------------------------------------------

/// Top-logprob alternative for a single token position (chat completions)
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    /// Text representation of the token
    pub token: String,
    /// Log probability of this token
    pub logprob: f32,
    /// UTF-8 byte encoding of the token text (`null` for non-UTF-8 tokens)
    pub bytes: Option<Vec<u8>>,
}

/// Log probability data for a single generated token (chat completions)
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    /// Text representation of the token
    pub token: String,
    /// Log probability of this token
    pub logprob: f32,
    /// UTF-8 byte encoding of the token text (`null` for non-UTF-8 tokens)
    pub bytes: Option<Vec<u8>>,
    /// Top-k alternative tokens at this position (empty when `top_logprobs` is 0)
    pub top_logprobs: Vec<TopLogprob>,
}

/// Logprobs container for chat completion choices
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogprobs {
    /// Per-token log probability data for the generated content
    pub content: Option<Vec<TokenLogprob>>,
}

/// Logprobs container for legacy text completion choices
#[derive(Debug, Clone, Serialize)]
pub struct CompletionLogprobs {
    /// Decoded token strings
    pub tokens: Vec<String>,
    /// Log probability of each token
    pub token_logprobs: Vec<f32>,
    /// Character offset of each token within the completion text
    pub text_offset: Vec<usize>,
    /// Top-k alternatives at each position (each entry is `null` or a map of
    /// token→logprob)
    pub top_logprobs: Vec<Option<std::collections::HashMap<String, f32>>>,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Chat completion choice
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
    /// Log probabilities for output tokens; `null` when not requested
    pub logprobs: Option<ChatLogprobs>,
}

/// Chat message in response
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat completion response
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    pub fn new(
        id: String,
        model: String,
        content: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: Option<String>,
    ) -> Self {
        Self::new_with_logprobs(
            id,
            model,
            content,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            None,
        )
    }

    pub fn new_with_logprobs(
        id: String,
        model: String,
        content: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: Option<String>,
        logprobs: Option<ChatLogprobs>,
    ) -> Self {
        Self {
            id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason,
                logprobs,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        }
    }
}

/// Text completion choice
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
    /// Log probabilities for output tokens; `null` when not requested
    pub logprobs: Option<CompletionLogprobs>,
}

/// Text completion response
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

impl CompletionResponse {
    pub fn new(
        id: String,
        model: String,
        text: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: Option<String>,
    ) -> Self {
        Self::new_with_logprobs(
            id,
            model,
            text,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            None,
        )
    }

    pub fn new_with_logprobs(
        id: String,
        model: String,
        text: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: Option<String>,
        logprobs: Option<CompletionLogprobs>,
    ) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![CompletionChoice {
                index: 0,
                text,
                finish_reason,
                logprobs,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        }
    }
}

/// Model information
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// Models list response
#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Health check response
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch: Option<BatchStatusInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observability: Option<crate::server::batch::ObservabilitySnapshot>,
}

/// Batch status included in the health check response.
#[derive(Debug, Clone, Serialize)]
pub struct BatchStatusInfo {
    pub active_sequences: usize,
    pub queue_depth: usize,
    pub max_batch_size: usize,
}

/// Error response
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
    /// HTTP status code (not serialized — used by IntoResponse)
    #[serde(skip)]
    pub status: axum::http::StatusCode,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

impl ErrorResponse {
    pub fn new(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                code: None,
            },
            status: axum::http::StatusCode::BAD_REQUEST,
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: "server_busy".into(),
                code: None,
            },
            status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

impl axum::response::IntoResponse for ErrorResponse {
    fn into_response(self) -> axum::response::Response {
        (self.status, axum::Json(self)).into_response()
    }
}

/// Native llama-server completion response (POST /completion)
#[derive(Debug, Clone, Serialize)]
pub struct NativeCompletionResponse {
    pub content: String,
    pub stop: bool,
    pub generation_settings: serde_json::Value,
    pub model: String,
    pub tokens_predicted: usize,
    pub tokens_evaluated: usize,
    pub timings: TimingInfo,
}

/// Timing information for generation (llama-server compatible)
#[derive(Debug, Clone, Serialize)]
pub struct TimingInfo {
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub prompt_per_token_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub predicted_per_token_ms: f64,
    pub predicted_per_second: f64,
}

/// Tokenize response (POST /tokenize)
#[derive(Debug, Clone, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<i32>,
}

/// Detokenize response (POST /detokenize)
#[derive(Debug, Clone, Serialize)]
pub struct DetokenizeResponse {
    pub content: String,
}

/// Server properties response (GET /props)
#[derive(Debug, Clone, Serialize)]
pub struct PropsResponse {
    pub default_generation_settings: serde_json::Value,
    pub total_slots: usize,
}

/// Slot information (GET /slots)
#[derive(Debug, Clone, Serialize)]
pub struct SlotInfo {
    pub id: usize,
    pub state: String,
    pub model: String,
    pub is_processing: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
}

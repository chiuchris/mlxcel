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

//! OpenAI and llama-server compatible request types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chat message role
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// Convert role to lowercase string for chat templates
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// Content part for multimodal messages (OpenAI format)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content (supports base64 data URIs and file:// URLs)
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// Image URL reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL: base64 data URI (data:image/...;base64,...) or file:// path
    pub url: String,
}

/// Message content: either a plain string or multimodal array
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content
    Text(String),
    /// Multimodal content parts (text + images)
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract the text content from the message
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Extract image data URIs/paths from multimodal content
    pub fn image_urls(&self) -> Vec<String> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url.url.clone()),
                    _ => None,
                })
                .collect(),
        }
    }
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Sampling parameters shared across endpoints
///
/// All parameters are optional. When not specified in the request,
/// server defaults (from CLI arguments) will be used.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 = greedy, higher = more random)
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling threshold
    pub top_p: Option<f32>,
    /// Top-k sampling (0 = disabled)
    pub top_k: Option<usize>,
    /// Min-p sampling threshold (0.0 = disabled)
    pub min_p: Option<f32>,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: Option<f32>,
    /// Context size for repetition penalty
    pub repetition_context_size: Option<usize>,
    /// Logit bias for specific tokens
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,

    // DRY (Don't Repeat Yourself) sampling parameters
    /// DRY penalty multiplier (0.0 = disabled, typical: 0.8-1.3)
    pub dry_multiplier: Option<f32>,
    /// DRY penalty base for exponential scaling (typical: 1.75)
    pub dry_base: Option<f32>,
    /// Minimum sequence length before DRY penalties apply (typical: 2)
    pub dry_allowed_length: Option<usize>,
    /// Number of recent tokens to scan for DRY (0 = entire context)
    pub dry_penalty_last_n: Option<usize>,
    /// Sequence breaker tokens for DRY (resets matching)
    pub dry_sequence_breakers: Option<Vec<i32>>,

    // XTC (Exclude Top Choices) sampling parameters
    /// XTC probability (0.0 = disabled)
    pub xtc_probability: Option<f32>,
    /// XTC probability threshold
    pub xtc_threshold: Option<f32>,

    // OpenAI-compatible frequency/presence penalties
    /// Frequency penalty (0.0 = disabled) - penalizes based on frequency
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (0.0 = disabled) - penalizes based on presence
    pub presence_penalty: Option<f32>,
}

/// Chat completion request (POST /v1/chat/completions)
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier
    pub model: String,
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
    /// Sampling parameters (flattened)
    #[serde(flatten)]
    pub params: SamplingParams,
}

/// Text completion request (POST /v1/completions)
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier
    pub model: String,
    /// Input prompt
    pub prompt: String,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
    /// Sampling parameters (flattened)
    #[serde(flatten)]
    pub params: SamplingParams,
}

impl ChatCompletionRequest {
    /// Convert messages to a prompt string using a simple format
    pub fn to_prompt(&self) -> String {
        let mut prompt = String::new();
        for msg in &self.messages {
            let text = msg.content.text();
            match msg.role {
                Role::System => {
                    prompt.push_str(&format!("System: {}\n\n", text));
                }
                Role::User => {
                    prompt.push_str(&format!("User: {}\n\n", text));
                }
                Role::Assistant => {
                    prompt.push_str(&format!("Assistant: {}\n\n", text));
                }
                Role::Tool => {
                    prompt.push_str(&format!("Tool: {}\n\n", text));
                }
            }
        }
        prompt.push_str("Assistant: ");
        prompt
    }

    /// Extract all image URLs from messages
    pub fn image_urls(&self) -> Vec<String> {
        self.messages
            .iter()
            .flat_map(|m| m.content.image_urls())
            .collect()
    }
}

/// Native llama-server completion request (POST /completion)
#[derive(Debug, Clone, Deserialize)]
pub struct NativeCompletionRequest {
    /// Input prompt
    pub prompt: String,
    /// Maximum number of tokens to predict
    pub n_predict: Option<usize>,
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Sampling temperature
    pub temperature: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<i32>,
    /// Top-p sampling
    pub top_p: Option<f32>,
    /// Min-p sampling
    pub min_p: Option<f32>,
    /// Repetition penalty
    pub repeat_penalty: Option<f32>,
    /// Repetition penalty last N tokens
    pub repeat_last_n: Option<usize>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Random seed
    pub seed: Option<u64>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// DRY penalty multiplier (0.0 = disabled)
    pub dry_multiplier: Option<f32>,
    /// DRY exponential base
    pub dry_base: Option<f32>,
    /// DRY minimum match length before penalty
    pub dry_allowed_length: Option<usize>,
    /// DRY lookback window (-1 = full context)
    pub dry_penalty_last_n: Option<usize>,
    /// DRY sequence breaker token IDs
    pub dry_sequence_breakers: Option<Vec<i32>>,
}

/// Tokenize request (POST /tokenize)
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeRequest {
    /// Text content to tokenize
    pub content: String,
    /// Whether to add special tokens (BOS/EOS)
    pub add_special: Option<bool>,
}

/// Detokenize request (POST /detokenize)
#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeRequest {
    /// Token IDs to decode
    pub tokens: Vec<i32>,
}

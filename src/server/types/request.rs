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

// ---------------------------------------------------------------------------
// Tool calling types (OpenAI-compatible)
// Used by: ChatCompletionRequest, chat_template, routes/chat
// ---------------------------------------------------------------------------

/// A tool definition (OpenAI format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool type (always "function" for now)
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition
    pub function: FunctionDefinition,
}

/// Function definition within a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name
    pub name: String,
    /// Human-readable description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Tool choice specification
///
/// Can be a string ("auto", "none", "required") or an object specifying a
/// particular function.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String mode: "auto", "none", or "required"
    Mode(String),
    /// Specific function: {"type": "function", "function": {"name": "X"}}
    Specific(ToolChoiceFunction),
}

impl ToolChoice {
    /// Returns the mode string for simple string choices, or "specific" for
    /// named function selections.
    pub fn mode(&self) -> &str {
        match self {
            ToolChoice::Mode(s) => s.as_str(),
            ToolChoice::Specific(_) => "specific",
        }
    }

    /// Returns true if tool calling is effectively disabled.
    pub fn is_none(&self) -> bool {
        matches!(self, ToolChoice::Mode(s) if s == "none")
    }

    /// Returns the specific function name if this is a named choice.
    pub fn specific_function(&self) -> Option<&str> {
        match self {
            ToolChoice::Specific(f) => Some(&f.function.name),
            _ => None,
        }
    }
}

/// Named function tool choice
#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub function: ToolChoiceFunctionName,
}

/// Function name within a tool choice
#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunctionName {
    pub name: String,
}

/// A tool call within an assistant message (multi-turn history)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInMessage {
    /// Unique tool call ID
    pub id: String,
    /// Tool type (always "function")
    #[serde(rename = "type")]
    pub call_type: String,
    /// Function call details
    pub function: ToolCallFunction,
}

/// Function name + arguments within a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// Function name
    pub name: String,
    /// Stringified JSON arguments
    pub arguments: String,
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
    /// Image URL content (supports base64 data URIs, `file://`, local paths,
    /// and `http(s)` URLs)
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    /// Audio input content (base64-encoded audio data)
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudio },
}

/// Image URL reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL: `data:image/...;base64,...`, `file://...`, bare local path, or
    /// `http(s)://...`
    pub url: String,
}

/// Audio input reference (OpenAI-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudio {
    /// Base64-encoded audio data, or a URL/file path
    pub data: String,
    /// Audio format: "wav", "mp3", etc.
    #[serde(default = "default_audio_format")]
    pub format: String,
}

fn default_audio_format() -> String {
    "wav".to_string()
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

    /// Extract audio input data from multimodal content
    pub fn audio_inputs(&self) -> Vec<InputAudio> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::InputAudio { input_audio } => Some(input_audio.clone()),
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
    /// Tool call ID for `role: "tool"` messages (references a previous tool call)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls made by the assistant (multi-turn history)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallInMessage>>,
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

    // Issue #409: thinking-token budget (Qwen3-family reasoning cap).
    //
    // Three aliases accepted, first non-None wins (see
    // `thinking_budget::pick_budget_alias`). llama.cpp-compatible primary
    // name, vLLM alias, and Qwen alias. Value semantics: -1 unrestricted,
    // 0 immediate close, N > 0 cap at N tokens inside the `<think>` block.
    /// Primary / llama.cpp-compatible name for the reasoning-token cap.
    pub thinking_budget_tokens: Option<i32>,
    /// vLLM-compatible alias for `thinking_budget_tokens`.
    pub thinking_token_budget: Option<i32>,
    /// Qwen-official alias for `thinking_budget_tokens`.
    pub thinking_budget: Option<i32>,
}

/// Stream options for controlling streaming behavior
#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    /// Include token usage statistics in the final streaming chunk
    #[serde(default)]
    pub include_usage: bool,
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
    /// Options controlling streaming behavior (only used when stream=true)
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Whether to return log probabilities of output tokens
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top log-probability alternatives to return per token (0–20)
    #[serde(default)]
    pub top_logprobs: Option<u8>,

    // Tool calling fields (OpenAI-compatible)
    /// Tool definitions available for the model to call
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// Controls how the model selects tools: "auto", "none", "required", or a
    /// specific function object
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may issue multiple tool calls in parallel
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    /// Issue #410: top-level `chat_template_kwargs` (llama.cpp shape).
    ///
    /// A JSON object whose keys are forwarded as Jinja template kwargs when
    /// rendering the conversation. Primary shape; wins over nested
    /// `extra_body.chat_template_kwargs`, flattened OpenAI-SDK `extra_body`
    /// aliases, and DashScope flat `extra_body.preserve_thinking`. See
    /// [`crate::server::chat_template_kwargs::extract_request_kwargs`] for
    /// the full precedence chain.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,

    /// Issue #410: nested `extra_body` compatibility (vLLM / manual callers).
    ///
    /// Some callers send an actual top-level `extra_body` object. Only the
    /// keys we currently recognize are read back out; unknown keys are
    /// silently ignored to match llama.cpp's lenient behavior.
    #[serde(default)]
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,

    /// Issue #410: OpenAI SDK `extra_body={...}` flattened into the request root.
    ///
    /// The official OpenAI Python client merges `extra_body` into the top-level
    /// JSON object instead of emitting a nested `"extra_body": {...}` wrapper.
    /// Capture those unknown root keys here so request-kwarg extraction can
    /// treat them the same as nested `extra_body` aliases.
    #[serde(default, flatten)]
    pub extra_body_fields: serde_json::Map<String, serde_json::Value>,

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
    /// Options controlling streaming behavior (only used when stream=true)
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Number of top log-probability alternatives to return (legacy format: 0–5)
    #[serde(default)]
    pub logprobs: Option<u8>,
    /// Sampling parameters (flattened)
    #[serde(flatten)]
    pub params: SamplingParams,
}

impl ChatCompletionRequest {
    /// Merge nested `extra_body` with flattened OpenAI-SDK root fields.
    ///
    /// Flattened root keys win over the nested object on collision because the
    /// request body already exposed them at the higher-precedence top level.
    pub fn merged_extra_body(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        match (&self.extra_body, self.extra_body_fields.is_empty()) {
            (None, true) => None,
            (Some(extra), true) => Some(extra.clone()),
            (None, false) => Some(self.extra_body_fields.clone()),
            (Some(extra), false) => {
                let mut merged = extra.clone();
                for (key, value) in &self.extra_body_fields {
                    merged.insert(key.clone(), value.clone());
                }
                Some(merged)
            }
        }
    }

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

    /// Extract all audio inputs from messages
    pub fn audio_inputs(&self) -> Vec<InputAudio> {
        self.messages
            .iter()
            .flat_map(|m| m.content.audio_inputs())
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

    // Issue #409: thinking-token budget (Qwen3-family reasoning cap).
    /// Primary / llama.cpp-compatible name for the reasoning-token cap.
    pub thinking_budget_tokens: Option<i32>,
    /// vLLM-compatible alias for `thinking_budget_tokens`.
    pub thinking_token_budget: Option<i32>,
    /// Qwen-official alias for `thinking_budget_tokens`.
    pub thinking_budget: Option<i32>,
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

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

//! Common configuration utilities
//!
//! Shared default values and serde deserializers used across model configurations.

use serde::Deserialize;

// Common Default Functions.
/// Default RoPE theta (base frequency)
pub fn default_rope_theta() -> f32 {
    10000.0
}

/// Default RoPE theta for high-context models (500K)
pub fn default_rope_theta_500k() -> f32 {
    500000.0
}

/// Default RoPE theta for 1M context models
pub fn default_rope_theta_1m() -> f32 {
    1000000.0
}

/// Default RoPE traditional mode (false for most modern models)
pub fn default_rope_traditional() -> bool {
    false
}

/// Default max position embeddings
pub fn default_max_position_embeddings() -> usize {
    2048
}

/// Default max position embeddings (8K context)
pub fn default_max_position_embeddings_8k() -> usize {
    8192
}

/// Default max position embeddings (128K context)
pub fn default_max_position_embeddings_128k() -> usize {
    131072
}

/// Default attention bias (false for most models)
pub fn default_attention_bias() -> bool {
    false
}

/// Default MLP bias (false for most models)
pub fn default_mlp_bias() -> bool {
    false
}

/// Default tie word embeddings
pub fn default_tie_word_embeddings() -> bool {
    true
}

/// Default tie word embeddings (false variant)
pub fn default_tie_word_embeddings_false() -> bool {
    false
}

/// Default RMS norm epsilon
pub fn default_rms_norm_eps() -> f32 {
    1e-5
}

/// Default RMS norm epsilon (1e-6 variant)
pub fn default_rms_norm_eps_small() -> f32 {
    1e-6
}

/// Default layer norm epsilon
pub fn default_layer_norm_eps() -> f32 {
    1e-5
}

/// Default boolean true
pub fn default_true() -> bool {
    true
}

/// Default boolean false
pub fn default_false() -> bool {
    false
}

// MoE Defaults.
/// Default number of groups for grouped MoE
pub fn default_n_group() -> usize {
    1
}

/// Default top-k per group
pub fn default_topk_group() -> usize {
    1
}

/// Default routed scaling factor
pub fn default_routed_scaling_factor() -> f32 {
    1.0
}

/// Default norm top-k probability
pub fn default_norm_topk_prob() -> bool {
    false
}

/// Default scoring function (softmax)
pub fn default_scoring_func() -> String {
    "softmax".to_string()
}

/// Default scoring function (sigmoid for DeepSeek v3)
pub fn default_scoring_func_sigmoid() -> String {
    "sigmoid".to_string()
}

/// Default top-k method (greedy)
pub fn default_topk_method() -> String {
    "greedy".to_string()
}

/// Default top-k method (group_limited_greedy for DeepSeek)
pub fn default_topk_method_group() -> String {
    "group_limited_greedy".to_string()
}

// SSM Defaults.
/// Default Mamba d_conv (convolution size)
pub fn default_mamba_d_conv() -> usize {
    4
}

/// Default Mamba d_state (state dimension)
pub fn default_mamba_d_state() -> usize {
    16
}

/// Default Mamba expand factor
pub fn default_mamba_expand() -> usize {
    2
}

/// Default time step limit for Mamba2
pub fn default_time_step_limit() -> (f32, f32) {
    (0.0, f32::INFINITY)
}

// RoPE Scaling Defaults.
/// Default original max position embeddings for SuScaled RoPE
pub fn default_original_max_position_embeddings() -> usize {
    4096
}

/// Default Llama 4 scaling beta
pub fn default_llama_4_scaling_beta() -> f32 {
    0.5
}

// Custom Deserializers.
/// Deserialize time_step_rank which can be "auto" or a number
pub fn deserialize_time_step_rank<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TimeStepRank {
        Auto(String),
        Explicit(usize),
    }

    match TimeStepRank::deserialize(deserializer)? {
        TimeStepRank::Auto(s) if s == "auto" => Ok(0), // Will be computed later
        TimeStepRank::Auto(_) => Ok(0),
        TimeStepRank::Explicit(v) => Ok(v),
    }
}

/// Deserialize time_step_limit which can be a tuple or array
pub fn deserialize_time_step_limit<'de, D>(deserializer: D) -> Result<(f32, f32), D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TimeStepLimit {
        Tuple((f32, f32)),
        Array(Vec<f32>),
    }

    match TimeStepLimit::deserialize(deserializer)? {
        TimeStepLimit::Tuple(t) => Ok(t),
        TimeStepLimit::Array(arr) if arr.len() >= 2 => Ok((arr[0], arr[1])),
        _ => Ok(default_time_step_limit()),
    }
}

/// Deserialize hybrid pattern from comma-separated string or list
pub fn deserialize_hybrid_pattern<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum HybridPattern {
        String(String),
        List(Vec<String>),
    }

    match HybridPattern::deserialize(deserializer)? {
        HybridPattern::String(s) => Ok(s.split(',').map(|s| s.trim().to_string()).collect()),
        HybridPattern::List(v) => Ok(v),
    }
}

/// Deserialize mamba_dt_rank which can be "auto" or explicit
pub fn deserialize_mamba_dt_rank<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_time_step_rank(deserializer)
}

// Quantization Configuration.
/// Common quantization arguments
#[derive(Debug, Clone, Default, Deserialize)]
pub struct QuantizationArgs {
    #[serde(default)]
    pub group_size: Option<i32>,
    #[serde(default)]
    pub bits: Option<i32>,
}

impl QuantizationArgs {
    pub fn is_quantized(&self) -> bool {
        self.group_size.is_some() || self.bits.is_some()
    }

    pub fn get_group_size(&self) -> i32 {
        self.group_size.unwrap_or(64)
    }

    pub fn get_bits(&self) -> i32 {
        self.bits.unwrap_or(4)
    }
}

// RoPE Scaling Configuration.
/// RoPE scaling configuration
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeScaling {
    #[serde(rename = "type", alias = "rope_type")]
    pub scaling_type: Option<String>,
    pub factor: Option<f32>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    // Llama 4 specific
    pub beta: Option<f32>,
    // YaRN specific
    pub attention_factor: Option<f32>,
    pub beta_fast: Option<f32>,
    pub beta_slow: Option<f32>,
    pub mscale: Option<f32>,
    pub mscale_all_dim: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        assert_eq!(default_rope_theta(), 10000.0);
        assert_eq!(default_max_position_embeddings(), 2048);
        assert!(!default_attention_bias());
        assert!(default_tie_word_embeddings());
    }

    #[test]
    fn test_quantization_args() {
        let args = QuantizationArgs::default();
        assert!(!args.is_quantized());
        assert_eq!(args.get_group_size(), 64);
        assert_eq!(args.get_bits(), 4);
    }
}

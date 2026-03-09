//! Shared server configuration types.
//!
//! These structs are reused by route handlers, startup normalization, and the
//! model worker, so keeping them separate from the startup side effects makes
//! server policy easier to extend and test.

use std::path::PathBuf;

use crate::SamplingConfig;

/// Bridge between server request params and `mlxcel-core` `SamplingConfig`.
#[derive(Debug, Clone)]
pub struct ServerGenerateOptions {
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_sequences: Option<Vec<String>>,
}

/// Server configuration derived from CLI-compatible startup arguments.
///
/// Default values intentionally track `llama-server` behavior where practical
/// so route handlers can apply one consistent set of defaults.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
    pub model_alias: Option<String>,
    pub context_size: usize,
    pub n_parallel: usize,
    pub enable_slots_endpoint: bool,
    pub enable_props_endpoint: bool,
    pub enable_metrics_endpoint: bool,
    pub default_temperature: f32,
    pub default_top_p: f32,
    pub default_top_k: i32,
    pub default_min_p: f32,
    pub default_repetition_penalty: f32,
    pub default_repetition_context_size: usize,
    pub default_max_tokens: usize,
    pub default_seed: Option<u64>,
    pub default_frequency_penalty: f32,
    pub default_presence_penalty: f32,
    pub default_dry_multiplier: f32,
    pub default_dry_base: f32,
    pub default_dry_allowed_length: usize,
    pub default_dry_penalty_last_n: usize,
    pub draft_model_path: Option<PathBuf>,
    pub num_draft_tokens: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            timeout_seconds: 600,
            model_alias: None,
            context_size: 0,
            n_parallel: 1,
            enable_slots_endpoint: true,
            enable_props_endpoint: false,
            enable_metrics_endpoint: false,
            default_temperature: 0.8,
            default_top_p: 0.9,
            default_top_k: 40,
            default_min_p: 0.1,
            default_repetition_penalty: 1.0,
            default_repetition_context_size: 64,
            default_max_tokens: 512,
            default_seed: None,
            default_frequency_penalty: 0.0,
            default_presence_penalty: 0.0,
            default_dry_multiplier: 0.0,
            default_dry_base: 1.75,
            default_dry_allowed_length: 2,
            default_dry_penalty_last_n: 0,
            draft_model_path: None,
            num_draft_tokens: 3,
        }
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;

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

//! Shared server configuration types.
//!
//! These structs are reused by route handlers, startup normalization, and the
//! model worker, so keeping them separate from the startup side effects makes
//! server policy easier to extend and test.

use std::path::PathBuf;

use crate::SamplingConfig;
use crate::server::batch::RequestPriority;

/// Bridge between server request params and `mlxcel-core` `SamplingConfig`.
#[derive(Debug, Clone)]
pub struct ServerGenerateOptions {
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_sequences: Option<Vec<String>>,
    /// Request priority for prefill queue ordering.
    pub priority: RequestPriority,
}

/// Policy for selecting which sequence to evict when preemption is enabled
/// and the batch is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreemptionPolicy {
    /// Evict the sequence that has generated the most tokens.
    #[default]
    LongestFirst,
    /// Evict the lowest-priority sequence; break ties by longest running.
    LowestPriority,
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
    /// Maximum number of sequences in the active decode batch.
    /// Defaults to `n_parallel` (typically 1) for backwards compatibility.
    pub max_batch_size: usize,
    /// Maximum number of requests waiting in the prefill queue.
    pub max_queue_depth: usize,
    /// Number of tokens per prefill chunk. When 0, chunking is disabled and
    /// the full prompt is prefilled in a single pass.
    pub prefill_chunk_size: usize,
    /// Whether preemptive eviction is enabled. When true and the batch is
    /// full, a high-priority incoming request may evict a lower-priority
    /// or longer-running active sequence.
    pub enable_preemption: bool,
    /// Policy used to select the eviction victim.
    pub preemption_policy: PreemptionPolicy,
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
            max_batch_size: 1,
            max_queue_depth: 1024,
            prefill_chunk_size: 512,
            enable_preemption: false,
            preemption_policy: PreemptionPolicy::default(),
        }
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;

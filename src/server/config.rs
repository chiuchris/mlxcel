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
use crate::distributed::ShardConfig;
use crate::distributed::TransportBackend;
use crate::distributed::pipeline::RemotePipelineRuntimeConfig;
use crate::server::batch::RequestPriority;
use mlxcel_core::sampling::LogprobsConfig;

/// Storage backend used by the server batch scheduler for decode-time state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeStorageBackend {
    /// Select paged decode automatically for workers that support it.
    #[default]
    Auto,
    /// Existing dense per-sequence KV caches.
    Dense,
    /// Paged block-table state mirrored alongside dense compatibility caches.
    Paged,
}

impl std::str::FromStr for DecodeStorageBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "dense" => Ok(Self::Dense),
            "paged" => Ok(Self::Paged),
            other => Err(format!(
                "unknown decode storage backend \"{other}\"; expected \"auto\", \"dense\", or \"paged\""
            )),
        }
    }
}

/// Bridge between server request params and `mlxcel-core` `SamplingConfig`.
#[derive(Debug, Clone)]
pub struct ServerGenerateOptions {
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_sequences: Option<Vec<String>>,
    /// Request priority for prefill queue ordering.
    pub priority: RequestPriority,
    /// Log probability configuration; disabled by default (zero overhead).
    pub logprobs: LogprobsConfig,
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

/// Normalized pipeline-parallel runtime mode for the server worker.
#[derive(Debug, Clone)]
pub enum PipelineParallelRuntimeConfig {
    /// Existing single-process stage-partitioned runtime.
    InProcess {
        layers: String,
        micro_batch_size: usize,
    },
    /// Coordinator runtime that dispatches requests to remote stages.
    RemoteCoordinator(RemotePipelineRuntimeConfig),
}

impl PipelineParallelRuntimeConfig {
    pub fn describe(&self) -> String {
        match self {
            Self::InProcess {
                layers,
                micro_batch_size,
            } => {
                format!("in_process(pp_layers={layers}, pp_micro_batch_size={micro_batch_size})")
            }
            Self::RemoteCoordinator(config) => format!(
                "remote_coordinator(stages={}, transport={}, bind_address={})",
                config.stage_peers.len(),
                config.transport_backend,
                config.bind_address
            ),
        }
    }
}

/// Startup-only config for launching this process as a remote pipeline stage.
#[derive(Debug, Clone)]
pub struct RemotePipelineStageConfig {
    pub bind_address: String,
    pub stage_index: u32,
    pub num_stages: u32,
    pub upstream_peer: Option<String>,
    pub downstream_peer: Option<String>,
    pub transport_backend: TransportBackend,
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
    /// When true, disable the batch scheduler and use the legacy sequential
    /// worker. Equivalent to `max_batch_size <= 1` for scheduling purposes
    /// but makes the intent explicit and guarantees zero scheduler overhead.
    pub no_batch: bool,
    /// Maximum number of requests to batch together for prefill.
    ///
    /// When `> 1`, the scheduler collects up to this many pending requests and
    /// runs a single batched forward pass `[batch_size, max_seq_len]` so that
    /// larger matmul operations better saturate Neural Accelerator cores.
    /// Falls back to sequential (per-request) prefill when only one request
    /// is pending or on any error.
    ///
    /// Default: 1 (no batching, backward compatible).
    /// Recommended: 4–8 on M5 Pro/Max hardware.
    pub max_batch_prefill: usize,
    /// Decode-time storage backend used by the batch scheduler.
    pub decode_storage_backend: DecodeStorageBackend,
    /// Normalized pipeline-parallel runtime mode for the server worker.
    pub pipeline_parallel_runtime: Option<PipelineParallelRuntimeConfig>,
    /// When present, launch this process as a remote pipeline stage instead of
    /// the HTTP API server.
    pub remote_pipeline_stage: Option<RemotePipelineStageConfig>,
    /// Tensor-parallel loading/runtime options resolved at startup.
    pub tensor_parallel: ShardConfig,
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
            no_batch: false,
            max_batch_prefill: 1,
            decode_storage_backend: DecodeStorageBackend::Auto,
            pipeline_parallel_runtime: None,
            remote_pipeline_stage: None,
            tensor_parallel: ShardConfig::default(),
        }
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;

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

//! Edge-only server startup inputs from CLI compatibility surfaces.
//!
//! `ServerStartupConfig` is the normalized runtime-facing policy object.
//! This module keeps raw CLI concerns such as `--no-slots` overrides and
//! negative seed sentinels at the edge so server startup and request handling do
//! not need to remember llama-server compatibility rules.

use std::path::PathBuf;

use super::ServerStartupConfig;

/// Raw server startup input captured from CLI/front-end binaries.
///
/// These fields intentionally preserve edge-specific conventions such as
/// negative seed sentinels and `--no-*` compatibility flags. Normalize them
/// exactly once with [`ServerStartupInput::into_startup_config`].
#[derive(Debug)]
pub struct ServerStartupInput {
    pub model_path: PathBuf,
    pub adapter_path: Option<PathBuf>,
    pub model_alias: Option<String>,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub api_key_file: Option<PathBuf>,
    pub n_parallel: usize,
    pub ctx_size: usize,
    pub n_predict: i32,
    pub timeout: u64,
    pub draft_model_path: Option<PathBuf>,
    pub draft_max: usize,
    pub max_batch_size: Option<usize>,
    pub max_queue_depth: usize,
    pub prefill_chunk_size: usize,
    pub enable_preemption: bool,
    pub preemption_policy: String,
    pub chat_template: Option<String>,
    pub chat_template_file: Option<PathBuf>,
    pub slots: bool,
    pub no_slots: bool,
    pub props: bool,
    pub metrics: bool,
    pub warmup: bool,
    pub no_warmup: bool,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: i64,
    pub repeat_last_n: usize,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_penalty_last_n: i32,
    pub dry_sequence_breakers: Vec<String>,
    pub verbose: bool,
    pub log_disable: bool,
    pub log_file: Option<PathBuf>,
}

impl ServerStartupInput {
    /// Normalize edge-only CLI conventions into runtime startup policy.
    pub fn into_startup_config(self) -> ServerStartupConfig {
        ServerStartupConfig {
            model_path: self.model_path,
            adapter_path: self.adapter_path,
            model_alias: self.model_alias,
            host: self.host,
            port: self.port,
            api_key: self.api_key,
            api_key_file: self.api_key_file,
            n_parallel: self.n_parallel,
            ctx_size: self.ctx_size,
            n_predict: self.n_predict,
            timeout: self.timeout,
            draft_model_path: self.draft_model_path,
            draft_max: self.draft_max,
            max_batch_size: self.max_batch_size,
            max_queue_depth: self.max_queue_depth,
            prefill_chunk_size: self.prefill_chunk_size,
            enable_preemption: self.enable_preemption,
            preemption_policy: self.preemption_policy,
            chat_template: self.chat_template,
            chat_template_file: self.chat_template_file,
            enable_slots: resolve_compat_toggle(self.slots, self.no_slots),
            enable_props: self.props,
            enable_metrics: self.metrics,
            warmup: resolve_compat_toggle(self.warmup, self.no_warmup),
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            seed: resolve_seed(self.seed),
            repeat_last_n: self.repeat_last_n,
            repeat_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_penalty_last_n: self.dry_penalty_last_n,
            dry_sequence_breakers: self.dry_sequence_breakers,
            verbose: self.verbose,
            log_disable: self.log_disable,
            log_file: self.log_file,
        }
    }
}

/// Resolve a llama-server style `--flag` / `--no-flag` pair into a policy bool.
pub fn resolve_compat_toggle(enabled: bool, disabled: bool) -> bool {
    enabled && !disabled
}

/// Convert the CLI seed sentinel into the runtime representation.
pub fn resolve_seed(seed: i64) -> Option<u64> {
    if seed < 0 { None } else { Some(seed as u64) }
}

#[cfg(test)]
#[path = "cli_input_tests.rs"]
mod tests;

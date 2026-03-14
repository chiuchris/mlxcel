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

//! CLI `serve` command wiring.
//!
//! This module owns the binary-only translation from clap-facing arguments to
//! the normalized server startup input. The actual llama-server compatibility
//! rules live in `mlxcel::server::ServerStartupInput` so `main.rs` stays focused
//! on schema and routing.

use mlxcel::server::{ServerStartupInput, start_server};

/// Run the `mlxcel serve` subcommand.
#[tokio::main]
pub(crate) async fn run_serve(args: crate::ServeArgs) -> anyhow::Result<()> {
    start_server(build_startup_input(args).into_startup_config()).await
}

fn build_startup_input(args: crate::ServeArgs) -> ServerStartupInput {
    ServerStartupInput {
        model_path: args.model,
        adapter_path: args.adapter,
        model_alias: args.alias,
        host: args.host,
        port: args.port,
        api_key: args.api_key,
        api_key_file: args.api_key_file,
        n_parallel: args.n_parallel,
        ctx_size: args.ctx_size,
        n_predict: args.n_predict,
        timeout: args.timeout,
        draft_model_path: args.draft_model,
        draft_max: args.draft_max,
        max_batch_size: args.max_batch_size,
        no_batch: args.no_batch,
        max_queue_depth: args.max_queue_depth,
        prefill_chunk_size: args.prefill_chunk_size,
        enable_preemption: args.enable_preemption,
        preemption_policy: args.preemption_policy,
        chat_template: args.chat_template,
        chat_template_file: args.chat_template_file,
        slots: args.slots,
        no_slots: args._no_slots,
        props: args.props,
        metrics: args.metrics,
        warmup: args.warmup,
        no_warmup: args._no_warmup,
        temperature: args.temp,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
        repeat_last_n: args.repeat_last_n,
        repeat_penalty: args.repeat_penalty,
        presence_penalty: args.presence_penalty,
        frequency_penalty: args.frequency_penalty,
        dry_multiplier: args.dry_multiplier,
        dry_base: args.dry_base,
        dry_allowed_length: args.dry_allowed_length,
        dry_penalty_last_n: args.dry_penalty_last_n,
        dry_sequence_breakers: args.dry_sequence_breakers,
        verbose: args.verbose,
        log_disable: args.log_disable,
        log_file: args.log_file,
    }
}

#[cfg(test)]
#[path = "serve_tests.rs"]
mod tests;

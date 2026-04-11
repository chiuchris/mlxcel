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

//! Prometheus-compatible metrics endpoint.
//!
//! This route is read-only and should remain separate from generation policy.
//! Includes both server-level request counters and batch observability gauges.

use std::sync::atomic::Ordering;

use axum::{
    extract::State,
    http::header,
    response::{IntoResponse, Response},
};

use crate::server::AppState;

/// GET /metrics -- Prometheus text format
pub async fn metrics(State(state): State<AppState>) -> Response {
    let m = &state.metrics;

    let requests = m.requests_total.load(Ordering::Relaxed);
    let prompt_tokens = m.prompt_tokens_total.load(Ordering::Relaxed);
    let completion_tokens = m.completion_tokens_total.load(Ordering::Relaxed);
    let gen_time_ms = m.generation_time_ms_total.load(Ordering::Relaxed);

    let slots_total = state.config.max_batch_size;
    let active = state.batch_metrics.active_count();
    let slots_available = slots_total.saturating_sub(active);
    let queue_depth = state.batch_metrics.queue_depth();

    // Batch observability counters
    let obs = state.batch_observability.snapshot();

    let body = format!(
        "# HELP mlxcel_requests_total Total number of generation requests\n\
         # TYPE mlxcel_requests_total counter\n\
         mlxcel_requests_total {requests}\n\
         # HELP mlxcel_prompt_tokens_total Total prompt tokens processed\n\
         # TYPE mlxcel_prompt_tokens_total counter\n\
         mlxcel_prompt_tokens_total {prompt_tokens}\n\
         # HELP mlxcel_completion_tokens_total Total completion tokens generated\n\
         # TYPE mlxcel_completion_tokens_total counter\n\
         mlxcel_completion_tokens_total {completion_tokens}\n\
         # HELP mlxcel_generation_time_seconds_total Total generation time in seconds\n\
         # TYPE mlxcel_generation_time_seconds_total counter\n\
         mlxcel_generation_time_seconds_total {gen_time_sec:.3}\n\
         # HELP mlxcel_slots_total Total number of parallel slots\n\
         # TYPE mlxcel_slots_total gauge\n\
         mlxcel_slots_total {slots_total}\n\
         # HELP mlxcel_slots_available Available parallel slots\n\
         # TYPE mlxcel_slots_available gauge\n\
         mlxcel_slots_available {slots_available}\n\
         # HELP mlxcel_queue_depth Current prefill queue depth\n\
         # TYPE mlxcel_queue_depth gauge\n\
         mlxcel_queue_depth {queue_depth}\n\
         # HELP mlxcel_batch_sequences_started Total sequences that entered prefill\n\
         # TYPE mlxcel_batch_sequences_started counter\n\
         mlxcel_batch_sequences_started {seq_started}\n\
         # HELP mlxcel_batch_sequences_completed Total sequences that completed generation\n\
         # TYPE mlxcel_batch_sequences_completed counter\n\
         mlxcel_batch_sequences_completed {seq_completed}\n\
         # HELP mlxcel_batch_prefill_tokens_total Cumulative prefill tokens processed\n\
         # TYPE mlxcel_batch_prefill_tokens_total counter\n\
         mlxcel_batch_prefill_tokens_total {prefill_tokens}\n\
         # HELP mlxcel_batch_decode_tokens_total Cumulative decode tokens generated\n\
         # TYPE mlxcel_batch_decode_tokens_total counter\n\
         mlxcel_batch_decode_tokens_total {decode_tokens}\n\
         # HELP mlxcel_batch_decode_steps_total Total decode steps executed\n\
         # TYPE mlxcel_batch_decode_steps_total counter\n\
         mlxcel_batch_decode_steps_total {decode_steps}\n\
         # HELP mlxcel_batch_prefill_chunks_total Total prefill chunks processed\n\
         # TYPE mlxcel_batch_prefill_chunks_total counter\n\
         mlxcel_batch_prefill_chunks_total {prefill_chunks}\n\
         # HELP mlxcel_batch_current_size Current active batch size\n\
         # TYPE mlxcel_batch_current_size gauge\n\
         mlxcel_batch_current_size {batch_size}\n\
         # HELP mlxcel_cache_pool_active Active cache entries\n\
         # TYPE mlxcel_cache_pool_active gauge\n\
         mlxcel_cache_pool_active {cache_active}\n\
         # HELP mlxcel_cache_pool_paged_block_size Paged decode block size in tokens\n\
         # TYPE mlxcel_cache_pool_paged_block_size gauge\n\
         mlxcel_cache_pool_paged_block_size {paged_block_size}\n\
         # HELP mlxcel_cache_pool_paged_blocks_allocated Total paged KV blocks tracked by the allocator\n\
         # TYPE mlxcel_cache_pool_paged_blocks_allocated gauge\n\
         mlxcel_cache_pool_paged_blocks_allocated {paged_blocks_allocated}\n\
         # HELP mlxcel_cache_pool_paged_blocks_live Paged KV blocks currently in live use\n\
         # TYPE mlxcel_cache_pool_paged_blocks_live gauge\n\
         mlxcel_cache_pool_paged_blocks_live {paged_blocks_live}\n\
         # HELP mlxcel_cache_pool_paged_blocks_free Paged KV blocks currently free for reuse\n\
         # TYPE mlxcel_cache_pool_paged_blocks_free gauge\n\
         mlxcel_cache_pool_paged_blocks_free {paged_blocks_free}\n\
         # HELP mlxcel_cache_pool_paged_bytes_reserved Reserved bytes across paged sequences\n\
         # TYPE mlxcel_cache_pool_paged_bytes_reserved gauge\n\
         mlxcel_cache_pool_paged_bytes_reserved {paged_bytes_reserved}\n\
         # HELP mlxcel_cache_pool_paged_bytes_in_use Visible bytes in use across paged sequences\n\
         # TYPE mlxcel_cache_pool_paged_bytes_in_use gauge\n\
         mlxcel_cache_pool_paged_bytes_in_use {paged_bytes_in_use}\n\
         # HELP mlxcel_decode_storage_fallbacks_total Number of paged decode fallback events\n\
         # TYPE mlxcel_decode_storage_fallbacks_total counter\n\
         mlxcel_decode_storage_fallbacks_total {decode_storage_fallbacks}\n",
        gen_time_sec = gen_time_ms as f64 / 1000.0,
        seq_started = obs.sequences_started,
        seq_completed = obs.sequences_completed,
        prefill_tokens = obs.total_prefill_tokens,
        decode_tokens = obs.total_decode_tokens,
        decode_steps = obs.decode_steps_processed,
        prefill_chunks = obs.prefill_chunks_processed,
        batch_size = obs.current_batch_size,
        cache_active = obs.cache_pool_active,
        paged_block_size = obs.cache_pool_paged_block_size,
        paged_blocks_allocated = obs.cache_pool_paged_blocks_allocated,
        paged_blocks_live = obs.cache_pool_paged_blocks_live,
        paged_blocks_free = obs.cache_pool_paged_blocks_free,
        paged_bytes_reserved = obs.cache_pool_paged_bytes_reserved,
        paged_bytes_in_use = obs.cache_pool_paged_bytes_in_use,
        decode_storage_fallbacks = obs.decode_storage_fallbacks,
    );

    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

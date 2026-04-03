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

//! Core batch scheduler with iteration-level scheduling and chunked prefill.
//!
//! [`BatchScheduler`] replaces the sequential `loop { request_rx.recv() }`
//! pattern in the model worker. At each tick it decides whether to:
//!
//! - **Prefill** (or continue a chunked prefill of) a queued request,
//! - **Decode** one token for each active sequence, or
//! - **Idle** (block until the next request arrives).
//!
//! When `prefill_chunk_size > 0`, long prompts are broken into chunks and
//! decode steps are interleaved between chunks so active sequences are not
//! starved during prefill of large prompts.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::{CachePool, SequenceId};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::generation_policy::{
    initial_token_history, merged_eos_token_ids, seed_rng_if_needed,
};
use mlxcel_core::hardware;
use mlxcel_core::sampling::{compute_logprobs, sample_token_optimized};
use mlxcel_core::streams::{install_default_stream, new_generation_stream};
use mlxcel_core::utils::{align_to_na_tile, create_padded_prefill_mask};
use mlxcel_core::{MlxStream, UniquePtr};

use crate::LoadedModel;
use crate::server::ServerGenerateOptions;
use crate::server::batch::observability::BatchObservability;
use crate::server::config::PreemptionPolicy;
use crate::server::model_provider::model_worker::{
    StreamingDecodeState, build_generation_result, merge_config_stop_tokens,
    prepare_request_vlm_embeddings,
};
use crate::server::model_provider::{GenerateEvent, ModelRequest};
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;
use crate::vlm_runtime::prepared_embedding_refs;

use super::active::ActiveBatch;
use super::queue::PrefillQueue;
use super::sequence::{
    BatchSchedulerAction, FinishReason, RequestPriority, SequenceInfo, SequenceState,
};

/// Returns true when the current hardware is M5+ with Neural Accelerator
/// support and tile-aligned prefill should be applied.
#[inline]
fn should_align_prefill() -> bool {
    let hw = hardware::get_hardware();
    hw.has_neural_accelerator && hw.macos_supports_na
}

/// Core batch scheduler that drives the model worker loop.
///
/// Replaces the old sequential `recv()` loop with an iteration-level scheduler
/// that interleaves prefill and decode operations. When `max_batch_size == 1`
/// (the default), behavior is identical to the pre-scheduler worker loop.
///
/// When `prefill_chunk_size > 0`, long prompts are processed in chunks with
/// decode interleaving to prevent latency spikes for active sequences.
pub struct BatchScheduler {
    // -- Pool & scheduling structures --
    cache_pool: CachePool,
    prefill_queue: PrefillQueue,
    active_batch: ActiveBatch,

    // -- Model & tokenizer --
    model: LoadedModel,
    tokenizer: MlxcelTokenizer,

    // -- Generation infrastructure --
    generation_stream: Option<UniquePtr<MlxStream>>,

    // -- Request channel --
    request_rx: mpsc::Receiver<ModelRequest>,

    // -- Metrics --
    /// Shared metrics updated atomically for HTTP handlers to read.
    batch_metrics: Arc<BatchMetrics>,
    /// Detailed observability counters (prefill, decode, cache).
    batch_observability: Arc<BatchObservability>,

    // -- Configuration --
    config_eos: Vec<i32>,
    /// Number of prompt tokens per prefill chunk. 0 = chunking disabled.
    prefill_chunk_size: usize,
    /// Whether preemptive eviction is enabled.
    enable_preemption: bool,
    /// Policy for selecting the eviction victim.
    preemption_policy: PreemptionPolicy,

    // -- Chunked prefill in-progress state --
    /// Sequence currently undergoing chunked prefill. `None` when no chunked
    /// prefill is in progress.
    chunked_prefill_seq: Option<SequenceInfo>,

    // -- Shutdown flag --
    shutdown_requested: bool,

    // -- Batched prefill config --
    /// Maximum number of pending requests to batch together for prefill.
    /// When `> 1`, the scheduler may collect multiple queued requests and
    /// run a single batched forward pass. Falls back to sequential prefill
    /// when only one request is pending or on any error.
    max_batch_prefill: usize,
}

impl BatchScheduler {
    /// Create a new batch scheduler, taking ownership of the model and channel.
    pub fn new(
        model: LoadedModel,
        tokenizer: MlxcelTokenizer,
        config_eos: Vec<i32>,
        request_rx: mpsc::Receiver<ModelRequest>,
        max_batch_size: usize,
        max_queue_depth: usize,
        batch_metrics: Arc<BatchMetrics>,
    ) -> Self {
        Self::with_config(
            model,
            tokenizer,
            config_eos,
            request_rx,
            max_batch_size,
            max_queue_depth,
            batch_metrics,
            Arc::new(BatchObservability::new()),
            0,
            false,
            PreemptionPolicy::default(),
            1,
        )
    }

    /// Create a new batch scheduler with chunked-prefill and preemption config.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        model: LoadedModel,
        tokenizer: MlxcelTokenizer,
        config_eos: Vec<i32>,
        request_rx: mpsc::Receiver<ModelRequest>,
        max_batch_size: usize,
        max_queue_depth: usize,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
        prefill_chunk_size: usize,
        enable_preemption: bool,
        preemption_policy: PreemptionPolicy,
        max_batch_prefill: usize,
    ) -> Self {
        let generation_stream = new_generation_stream();
        let max_batch_size = max_batch_size.max(1);
        // Non-batching models use lightweight placeholder entries in the pool
        // (no real KV caches), so we size the pool to cover both the active
        // batch and the prefill queue so requests can be queued while another
        // sequence is generating.
        let pool_capacity = max_batch_size + max_queue_depth;
        Self {
            cache_pool: CachePool::new(pool_capacity),
            prefill_queue: PrefillQueue::with_capacity(max_queue_depth),
            active_batch: ActiveBatch::new(max_batch_size),
            model,
            tokenizer,
            generation_stream,
            request_rx,
            batch_metrics,
            batch_observability,
            config_eos,
            prefill_chunk_size,
            enable_preemption,
            preemption_policy,
            chunked_prefill_seq: None,
            shutdown_requested: false,
            max_batch_prefill: max_batch_prefill.max(1),
        }
    }

    /// Run the scheduler loop until shutdown or channel close.
    pub fn run(&mut self) {
        install_default_stream(self.generation_stream.as_ref());

        loop {
            // 1. Non-blocking drain of all pending requests
            self.drain_incoming_requests();

            if self.shutdown_requested {
                break;
            }

            self.publish_metrics();

            // 2. Decide what to do this tick
            let action = self.decide_action();

            // 3. Execute
            match action {
                BatchSchedulerAction::Prefill(seq_id) => {
                    // Use batched prefill when max_batch_prefill > 1 and at
                    // least 2 requests are waiting, otherwise take the regular
                    // single-request path so there is zero overhead for the
                    // common case.
                    if self.max_batch_prefill > 1
                        && self.prefill_queue.len() >= 2
                        && self.chunked_prefill_seq.is_none()
                    {
                        self.execute_batched_prefill();
                    } else {
                        self.execute_prefill(seq_id);
                    }
                    self.publish_metrics();
                }
                BatchSchedulerAction::Decode(ids) => {
                    self.execute_decode_step(&ids);
                }
                BatchSchedulerAction::Idle => match self.request_rx.recv() {
                    Ok(req) => {
                        if self.handle_incoming(req) {
                            break;
                        }
                        self.publish_metrics();
                    }
                    Err(_) => {
                        tracing::info!("Request channel closed, scheduler exiting");
                        break;
                    }
                },
            }

            // 4. Clean up completed sequences
            self.finalize_completed();
        }
    }

    fn publish_metrics(&self) {
        let active = self.active_batch.len();
        let queued = self.prefill_queue.len();
        self.batch_metrics.set_active_count(active);
        self.batch_metrics.set_queue_depth(queued);
        self.batch_observability.update_gauges(
            active,
            queued,
            self.cache_pool.active_count(),
            0, // cache bytes estimation not yet implemented
        );
    }

    // ------------------------------------------------------------------
    // Request ingestion
    // ------------------------------------------------------------------

    fn drain_incoming_requests(&mut self) {
        loop {
            match self.request_rx.try_recv() {
                Ok(req) => {
                    if self.handle_incoming(req) {
                        self.shutdown_requested = true;
                        return;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.shutdown_requested = true;
                    return;
                }
            }
        }
    }

    fn handle_incoming(&mut self, req: ModelRequest) -> bool {
        match req {
            ModelRequest::Generate {
                prompt,
                options,
                images,
                response_tx,
            } => {
                self.enqueue_request(prompt, options, images, response_tx);
                false
            }
            ModelRequest::Shutdown => {
                tracing::info!("BatchScheduler received shutdown signal");
                true
            }
        }
    }

    fn enqueue_request(
        &mut self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        response_tx: mpsc::Sender<GenerateEvent>,
    ) {
        let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
        let token_ids = match self.tokenizer.encode(&prompt, add_special) {
            Ok(ids) => ids,
            Err(err) => {
                let _ =
                    response_tx.send(GenerateEvent::Error(format!("Tokenization error: {err}")));
                return;
            }
        };
        let mut prompt_tokens: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let sampling = merge_config_stop_tokens(options.sampling.clone(), &self.config_eos);

        let seq_id = match self.cache_pool.allocate(&self.model) {
            Ok(id) => id,
            Err(err) => {
                tracing::warn!("Cache pool allocation failed: {err}");
                let _ = response_tx.send(GenerateEvent::Error(format!("Server busy: {err}")));
                return;
            }
        };

        let vlm_embeddings = match prepare_request_vlm_embeddings(
            &self.model,
            &self.tokenizer,
            &prompt,
            &mut prompt_tokens,
            &images,
        ) {
            Ok(emb) => emb,
            Err(err) => {
                self.cache_pool.release(seq_id);
                let _ = response_tx.send(GenerateEvent::Error(err.to_string()));
                return;
            }
        };

        let decode_state = StreamingDecodeState::new(&self.tokenizer, &prompt_tokens);

        let seq = SequenceInfo {
            seq_id,
            state: SequenceState::Queued,
            prompt_tokens,
            sampling,
            max_tokens: options.max_tokens,
            eos_token_ids: self.config_eos.clone(),
            priority: options.priority,
            logprobs_config: options.logprobs,
            vlm_embeddings,
            images,
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            decode_state,
            prefill_offset: 0,
            response_tx,
            created_at: Instant::now(),
            prefill_start: None,
            first_token_time: None,
            token_history: Vec::new(),
            merged_eos: Vec::new(),
        };

        if let Err(rejected) = self.prefill_queue.enqueue(seq) {
            self.cache_pool.release(rejected.seq_id);
            let _ = rejected.response_tx.send(GenerateEvent::Error(
                "Server busy: prefill queue full".to_string(),
            ));
        }
    }

    // ------------------------------------------------------------------
    // Scheduling decision
    // ------------------------------------------------------------------

    /// Determine the next action. Runs in O(1) time.
    ///
    /// Policy:
    /// 1. If a chunked prefill is in progress and active sequences exist,
    ///    decode first (interleave).
    /// 2. If a chunked prefill is in progress and no active sequences,
    ///    continue the prefill.
    /// 3. If active sequences exist, decode first.
    /// 4. If the batch is not full and the queue has work, prefill.
    /// 5. Otherwise idle.
    fn decide_action(&self) -> BatchSchedulerAction {
        tracing::debug!(
            active = self.active_batch.len(),
            queued = self.prefill_queue.len(),
            chunked_in_progress = self.chunked_prefill_seq.is_some(),
            "scheduler tick"
        );
        // Chunked prefill in progress: interleave decode with prefill
        if self.chunked_prefill_seq.is_some() {
            if !self.active_batch.is_empty() {
                // Interleave: decode active sequences first, then continue
                // prefill on the next tick.
                return BatchSchedulerAction::Decode(self.active_batch.sequence_ids());
            }
            // No active sequences, continue the prefill
            return BatchSchedulerAction::Prefill(SequenceId::from_raw(0));
        }

        if self.active_batch.is_empty() && self.prefill_queue.is_empty() {
            return BatchSchedulerAction::Idle;
        }

        // When active sequences exist:
        // 1. If batch is NOT full and queue has work → admit one new sequence
        //    (this grows the batch to improve decode throughput via batching)
        // 2. If batch is full or queue is empty → decode existing sequences
        // 3. Preemption overrides when enabled and a higher-priority request waits
        if !self.active_batch.is_empty() {
            if self.should_preempt() {
                return BatchSchedulerAction::Prefill(SequenceId::from_raw(0));
            }
            if !self.active_batch.is_full() && !self.prefill_queue.is_empty() {
                // Admit one queued request to grow the batch before decoding.
                // This is critical for throughput: larger batches amortize
                // weight-loading bandwidth across more sequences.
                return BatchSchedulerAction::Prefill(SequenceId::from_raw(0));
            }
            return BatchSchedulerAction::Decode(self.active_batch.sequence_ids());
        }

        // Batch is empty but queue has work
        BatchSchedulerAction::Prefill(SequenceId::from_raw(0))
    }

    /// Check if preemption should occur: batch is full, preemption is
    /// enabled, and a higher-priority request is waiting.
    fn should_preempt(&self) -> bool {
        if !self.enable_preemption || !self.active_batch.is_full() {
            return false;
        }
        // Only preempt if waiting request has higher priority than some
        // active sequence.
        let waiting_priority = match self.prefill_queue.peek_priority() {
            Some(p) => p,
            None => return false,
        };
        // Find the lowest-priority active sequence
        let min_active_priority = self
            .active_batch
            .iter_min_priority()
            .unwrap_or(RequestPriority::High);

        waiting_priority > min_active_priority
    }

    // ------------------------------------------------------------------
    // Prefill execution (chunked or full)
    // ------------------------------------------------------------------

    /// Prefill a sequence. If `prefill_chunk_size > 0` and the prompt
    /// exceeds one chunk, the prefill is split across multiple ticks with
    /// decode interleaving.
    fn execute_prefill(&mut self, _action_id: SequenceId) {
        // Resume a chunked prefill already in progress?
        if self.chunked_prefill_seq.is_some() {
            self.continue_chunked_prefill();
            return;
        }

        // Preemption: if batch is full and preemption is enabled, evict
        // a lower-priority sequence to make room.
        if self.active_batch.is_full() && self.enable_preemption && !self.try_evict_for_preemption()
        {
            // Cannot evict -- skip prefill this tick
            return;
        }

        let mut seq = match self.prefill_queue.dequeue() {
            Some(s) => s,
            None => return,
        };

        if let Err(err) = seq.state.transition_to(SequenceState::Prefilling) {
            tracing::error!("State transition error: {err}");
            self.abort_sequence(seq, &err);
            return;
        }
        seq.prefill_start = Some(Instant::now());
        seed_rng_if_needed(&seq.sampling);

        let prompt_len = seq.prompt_tokens.len();

        // Decide: chunked vs full prefill
        if self.prefill_chunk_size > 0 && prompt_len > self.prefill_chunk_size {
            // Start chunked prefill: process first chunk
            self.start_chunked_prefill(seq);
        } else {
            // Full-prompt prefill (original path)
            self.execute_full_prefill(seq);
        }
    }

    /// Batched prefill: drain up to `max_batch_prefill` requests from the
    /// prefill queue and process them in a single forward pass.
    ///
    /// Sequences are padded to the longest prompt in the batch (aligned to a
    /// 32-token tile on M5+). Each sequence gets a per-sequence causal +
    /// padding attention mask so padding tokens are excluded from attention.
    ///
    /// On any error the method falls back to sequential single-request prefill
    /// for the remaining requests so no requests are lost.
    fn execute_batched_prefill(&mut self) {
        let batch_size = self.max_batch_prefill.min(self.prefill_queue.len());

        // Collect up to `batch_size` requests from the queue.
        let mut seqs: Vec<SequenceInfo> = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            match self.prefill_queue.dequeue() {
                Some(s) => seqs.push(s),
                None => break,
            }
        }

        if seqs.is_empty() {
            return;
        }

        // Single-request fast path: fall through to the regular prefill so
        // there is no overhead for constructing a padded batch.
        if seqs.len() == 1 {
            let seq = seqs.remove(0);
            self.execute_full_prefill(seq);
            return;
        }

        // Any VLM request cannot currently be batched (embeddings are
        // per-sequence and would need separate handling). Fall back for the
        // whole batch when any request carries VLM embeddings.
        if seqs.iter().any(|s| s.vlm_embeddings.is_some()) {
            tracing::debug!("batched prefill: falling back to sequential (VLM request in batch)");
            for seq in seqs {
                self.execute_full_prefill(seq);
            }
            return;
        }

        let b = seqs.len();
        let max_len = seqs.iter().map(|s| s.prompt_tokens.len()).max().unwrap();
        let padded_len = if should_align_prefill() {
            align_to_na_tile(max_len)
        } else {
            max_len
        };

        tracing::debug!("batched prefill: {} requests, padded to {}", b, padded_len);

        // Transition all sequences to Prefilling.
        for seq in &mut seqs {
            if let Err(err) = seq.state.transition_to(SequenceState::Prefilling) {
                tracing::error!("Batched prefill state transition error: {err}");
            }
            seq.prefill_start = Some(Instant::now());
            seed_rng_if_needed(&seq.sampling);
        }

        // Build padded input: [B, padded_len]
        let mut flat_tokens: Vec<i32> = Vec::with_capacity(b * padded_len);
        for seq in &seqs {
            let tokens = &seq.prompt_tokens;
            flat_tokens.extend_from_slice(tokens);
            // Pad with 0 to padded_len
            flat_tokens.extend(std::iter::repeat_n(0, padded_len - tokens.len()));
        }
        let input = mlxcel_core::from_slice_i32(&flat_tokens, &[b as i32, padded_len as i32]);

        // Build per-sequence attention masks and collect cache pointers.
        // Each mask has shape [padded_len, padded_len]; we add a batch dim
        // by expanding to [1, padded_len, padded_len] and stacking into
        // [B, padded_len, padded_len] so models can use it directly.
        let mut batch_masks: Vec<UniquePtr<mlxcel_core::MlxArray>> = Vec::with_capacity(b);
        for seq in &seqs {
            let actual = seq.prompt_tokens.len() as i32;
            let padded = padded_len as i32;
            let mask = create_padded_prefill_mask(actual, padded, 0);
            // Expand to [1, padded_len, padded_len] for stacking.
            let expanded = mlxcel_core::expand_dims(&mask, 0);
            batch_masks.push(expanded);
        }

        // Stack masks into [B, padded_len, padded_len].
        let stacked_mask = mlxcel_core::stack_owned(&batch_masks, 0);

        // Collect cache pointers (one cache slice per sequence).
        let mut cache_ptrs: Vec<(*mut mlxcel_core::layers::KVCache, usize)> = Vec::with_capacity(b);
        let mut valid = true;
        for seq in &seqs {
            match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(caches) => {
                    cache_ptrs.push((caches.as_mut_ptr(), caches.len()));
                }
                None => {
                    tracing::warn!(
                        "batched prefill: cache not found for seq {}, falling back",
                        seq.seq_id
                    );
                    valid = false;
                    break;
                }
            }
        }

        if !valid {
            // Re-queue all sequences for sequential processing.
            for seq in seqs {
                self.execute_full_prefill(seq);
            }
            return;
        }

        // SAFETY: Each seq_id maps to a distinct SequenceCacheSet entry in the
        // CachePool HashMap (allocation guarantees uniqueness). No two slices
        // alias the same memory. Pointers remain valid because cache_pool is not
        // mutated between extraction and the forward_batched call.
        let mut batch_caches: Vec<&mut [mlxcel_core::layers::KVCache]> = cache_ptrs
            .iter()
            .map(|&(ptr, len)| unsafe { std::slice::from_raw_parts_mut(ptr, len) })
            .collect();

        // Single batched forward pass: [B, padded_len] → [B, padded_len, vocab]
        let raw_logits = self
            .model
            .forward_batched(&input, &mut batch_caches, Some(&stacked_mask));

        mlxcel_core::eval(&raw_logits);
        mlxcel_core::clear_memory_cache();

        // Process per-sequence results.
        for (i, mut seq) in seqs.into_iter().enumerate() {
            let actual_len = seq.prompt_tokens.len();
            let padded = padded_len;

            // Extract logits at the last real token position: index [i, actual_len-1, :]
            let last_pos = actual_len as i32 - 1;
            let vocab = {
                let shape = mlxcel_core::array_shape(&raw_logits);
                shape[2]
            };
            let seq_logits = mlxcel_core::slice(
                &raw_logits,
                &[i as i32, last_pos, 0],
                &[i as i32 + 1, last_pos + 1, vocab],
            );

            // Trim padding positions from this sequence's KV cache so that the
            // decode phase starts with the correct cache offset.
            let excess = (padded - actual_len) as i32;
            if excess > 0
                && let Some(caches) = self.cache_pool.get_caches_mut(seq.seq_id)
            {
                for c in caches.iter_mut() {
                    c.trim(excess);
                }
            }

            seq.prefill_offset = actual_len;
            self.batch_observability.record_prefill_start(actual_len);

            let eos_tokens =
                merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
            let needs_history = seq.sampling.needs_token_history();
            let token_history = initial_token_history(&seq.prompt_tokens, needs_history);

            self.finish_prefill(seq, seq_logits, eos_tokens, token_history, needs_history);
        }
    }

    /// Full-prompt prefill: process the entire prompt in one pass.
    fn execute_full_prefill(&mut self, mut seq: SequenceInfo) {
        let _span = tracing::info_span!(
            "prefill",
            seq_id = %seq.seq_id,
            prompt_len = seq.prompt_tokens.len(),
        )
        .entered();
        self.batch_observability
            .record_prefill_start(seq.prompt_tokens.len());

        // Non-batching models use internal RefCell caches that are shared
        // across all sequences.  Reset them now (at prefill time) rather
        // than at enqueue time so that queued requests don't corrupt an
        // in-flight generation.
        if !self.model.supports_batching() {
            let _ = self.model.make_caches();
        }

        let eos_tokens =
            merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
        let needs_history = seq.sampling.needs_token_history();
        let token_history = initial_token_history(&seq.prompt_tokens, needs_history);

        let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
            Some(c) => c,
            None => {
                self.abort_sequence(seq, "Cache not found for sequence during prefill");
                return;
            }
        };

        // Run prefill (with or without VLM embeddings).
        // On M5+ hardware pad the prompt to a 32-token tile boundary for
        // optimal Neural Accelerator throughput.
        let actual_len = seq.prompt_tokens.len();
        let (effective_tokens, pad_mask_opt) = if should_align_prefill() {
            let padded_len = align_to_na_tile(actual_len);
            if padded_len > actual_len {
                let mut padded = seq.prompt_tokens.clone();
                padded.resize(padded_len, 0);
                let mask = create_padded_prefill_mask(actual_len as i32, padded_len as i32, 0);
                (padded, Some(mask))
            } else {
                (seq.prompt_tokens.clone(), None)
            }
        } else {
            (seq.prompt_tokens.clone(), None)
        };

        let eff_len = effective_tokens.len() as i32;
        let input = mlxcel_core::from_slice_i32(&effective_tokens, &[1, eff_len]);

        let raw_logits = if let Some(ref embeddings) = seq.vlm_embeddings {
            // VLM path: apply provided mask or the tile-alignment mask.
            match prepared_embedding_refs(embeddings) {
                Ok((input_embeds, caller_mask)) => {
                    // Caller-supplied mask takes precedence; tile-alignment mask
                    // is used only when the caller does not provide one.
                    let effective_mask =
                        caller_mask.or(pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()));
                    let logits = self.model.forward_with_embeddings(
                        &input,
                        Some(input_embeds),
                        caches,
                        effective_mask,
                    );
                    mlxcel_core::eval(&logits);
                    self.model.after_prefill();
                    logits
                }
                Err(err) => {
                    self.abort_sequence(seq, &err.to_string());
                    return;
                }
            }
        } else {
            self.model.forward(
                &input,
                caches,
                pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
            )
        };

        // Extract logits at the last real token position and trim padding from
        // KV caches so the decode phase begins with the correct cache offset.
        let logits = if pad_mask_opt.is_some() && effective_tokens.len() > actual_len {
            let padded_len = effective_tokens.len();
            let shape = mlxcel_core::array_shape(&raw_logits);
            let vocab = shape[2];
            let sliced = mlxcel_core::slice(
                &raw_logits,
                &[0, actual_len as i32 - 1, 0],
                &[shape[0], actual_len as i32, vocab],
            );
            // Trim padding positions from all KV caches.
            let excess = (padded_len - actual_len) as i32;
            for c in caches.iter_mut() {
                c.trim(excess);
            }
            sliced
        } else {
            raw_logits
        };

        mlxcel_core::clear_memory_cache();
        seq.prefill_offset = actual_len;

        self.finish_prefill(seq, logits, eos_tokens, token_history, needs_history);
    }

    /// Begin a chunked prefill: process the first chunk and store the
    /// sequence for continuation on subsequent ticks.
    fn start_chunked_prefill(&mut self, mut seq: SequenceInfo) {
        let _span = tracing::info_span!(
            "chunked_prefill_start",
            seq_id = %seq.seq_id,
            prompt_len = seq.prompt_tokens.len(),
            chunk_size = self.prefill_chunk_size,
        )
        .entered();
        self.batch_observability
            .record_prefill_start(seq.prompt_tokens.len());

        // Reset internal caches for non-batching models (same as execute_full_prefill).
        if !self.model.supports_batching() {
            let _ = self.model.make_caches();
        }

        let chunk_size = self.prefill_chunk_size;
        let end = chunk_size.min(seq.prompt_tokens.len());
        let chunk = &seq.prompt_tokens[..end];

        let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
            Some(c) => c,
            None => {
                self.abort_sequence(seq, "Cache not found for sequence during chunked prefill");
                return;
            }
        };

        // Align the first chunk to a 32-token tile boundary on M5+ hardware.
        let actual_chunk_len = chunk.len();
        let (eff_chunk, pad_mask_opt) = if should_align_prefill() {
            let padded_len = align_to_na_tile(actual_chunk_len);
            if padded_len > actual_chunk_len {
                let mut padded = chunk.to_vec();
                padded.resize(padded_len, 0);
                let mask =
                    create_padded_prefill_mask(actual_chunk_len as i32, padded_len as i32, 0);
                (padded, Some(mask))
            } else {
                (chunk.to_vec(), None)
            }
        } else {
            (chunk.to_vec(), None)
        };

        let eff_len = eff_chunk.len() as i32;
        let input = mlxcel_core::from_slice_i32(&eff_chunk, &[1, eff_len]);

        // VLM embeddings are applied only on the first chunk.
        if let Some(ref embeddings) = seq.vlm_embeddings {
            match prepared_embedding_refs(embeddings) {
                Ok((input_embeds, caller_mask)) => {
                    let effective_mask =
                        caller_mask.or(pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()));
                    let logits = self.model.forward_with_embeddings(
                        &input,
                        Some(input_embeds),
                        caches,
                        effective_mask,
                    );
                    mlxcel_core::eval(&logits);
                    self.model.after_prefill();
                }
                Err(err) => {
                    self.abort_sequence(seq, &err.to_string());
                    return;
                }
            }
        } else {
            let logits = self.model.forward(
                &input,
                caches,
                pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
            );
            mlxcel_core::eval(&logits);
        }

        // Trim padding positions from KV caches when the chunk was padded.
        if pad_mask_opt.is_some() && eff_chunk.len() > actual_chunk_len {
            let excess = (eff_chunk.len() - actual_chunk_len) as i32;
            for c in caches.iter_mut() {
                c.trim(excess);
            }
        }

        mlxcel_core::clear_memory_cache();
        seq.prefill_offset = end;

        tracing::debug!(
            "Chunked prefill: seq {} chunk 0..{end}/{} tokens",
            seq.seq_id,
            seq.prompt_tokens.len()
        );

        // Store the sequence for continuation
        self.chunked_prefill_seq = Some(seq);
    }

    /// Continue a chunked prefill that is already in progress.
    fn continue_chunked_prefill(&mut self) {
        let mut seq = match self.chunked_prefill_seq.take() {
            Some(s) => s,
            None => return,
        };

        let _span = tracing::info_span!(
            "chunked_prefill_continue",
            seq_id = %seq.seq_id,
            offset = seq.prefill_offset,
            total = seq.prompt_tokens.len(),
        )
        .entered();
        self.batch_observability.record_prefill_chunk();

        let chunk_size = self.prefill_chunk_size;
        let offset = seq.prefill_offset;
        let total = seq.prompt_tokens.len();
        let end = (offset + chunk_size).min(total);
        let chunk = &seq.prompt_tokens[offset..end];

        let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
            Some(c) => c,
            None => {
                self.abort_sequence(seq, "Cache not found during chunked prefill continuation");
                return;
            }
        };

        // Align each continuation chunk to a 32-token tile boundary on M5+.
        let actual_chunk_len = chunk.len();
        let kv_offset = caches.first().map_or(0, |c| c.offset);
        let (eff_chunk, pad_mask_opt) = if should_align_prefill() {
            let padded_len = align_to_na_tile(actual_chunk_len);
            if padded_len > actual_chunk_len {
                let mut padded = chunk.to_vec();
                padded.resize(padded_len, 0);
                let mask = create_padded_prefill_mask(
                    actual_chunk_len as i32,
                    padded_len as i32,
                    kv_offset,
                );
                (padded, Some(mask))
            } else {
                (chunk.to_vec(), None)
            }
        } else {
            (chunk.to_vec(), None)
        };

        let eff_len = eff_chunk.len() as i32;
        let input = mlxcel_core::from_slice_i32(&eff_chunk, &[1, eff_len]);
        let logits = self.model.forward(
            &input,
            caches,
            pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
        );

        // Trim padding positions from KV caches when the chunk was padded.
        if pad_mask_opt.is_some() && eff_chunk.len() > actual_chunk_len {
            let excess = (eff_chunk.len() - actual_chunk_len) as i32;
            for c in caches.iter_mut() {
                c.trim(excess);
            }
        }

        seq.prefill_offset = end;

        tracing::debug!(
            "Chunked prefill: seq {} chunk {offset}..{end}/{total} tokens",
            seq.seq_id,
        );

        if end < total {
            // More chunks remain -- store and yield back to the scheduler
            mlxcel_core::eval(&logits);
            mlxcel_core::clear_memory_cache();
            self.chunked_prefill_seq = Some(seq);
            return;
        }

        // Final chunk -- complete the prefill and sample the first token
        mlxcel_core::clear_memory_cache();

        let eos_tokens =
            merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
        let needs_history = seq.sampling.needs_token_history();
        let token_history = initial_token_history(&seq.prompt_tokens, needs_history);

        self.finish_prefill(seq, logits, eos_tokens, token_history, needs_history);
    }

    /// Complete a prefill (full or chunked): sample the first token,
    /// handle EOS, and either finish immediately or move to the active
    /// decode batch.
    fn finish_prefill(
        &mut self,
        mut seq: SequenceInfo,
        logits: UniquePtr<mlxcel_core::MlxArray>,
        eos_tokens: Vec<i32>,
        mut token_history: Vec<i32>,
        needs_history: bool,
    ) {
        let (first_token_arr, adjusted_logits) =
            sample_token_optimized(&logits, &seq.sampling, &token_history);
        mlxcel_core::eval(&first_token_arr);
        let first_token = mlxcel_core::item_i32(&first_token_arr);

        seq.first_token_time = Some(Instant::now());

        // Check for immediate EOS
        if eos_tokens.contains(&first_token) {
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Stop))
            {
                tracing::error!("State transition error: {err}");
            }
            let result = build_generation_result(
                String::new(),
                seq.prompt_tokens.len(),
                0,
                seq.created_at.elapsed().as_millis() as u64,
                seq.prefill_start
                    .map(|t| (Instant::now() - t).as_millis() as u64)
                    .unwrap_or(0),
                seq.max_tokens,
            );
            let _ = seq.response_tx.send(GenerateEvent::Done(result));
            self.cache_pool.release(seq.seq_id);
            return;
        }

        // Optionally compute logprobs for the first token.
        let token_lp = compute_logprobs(&adjusted_logits, first_token, &seq.logprobs_config);

        seq.generated_tokens.push(first_token);
        if needs_history {
            token_history.push(first_token);
        }

        // Store merged EOS and token history on the sequence so decode_single_step
        // can reuse them without per-step reconstruction.
        seq.merged_eos = eos_tokens;
        seq.token_history = token_history;

        if let Some(new_text) = seq.decode_state.on_token(first_token, &self.tokenizer) {
            let event = match token_lp {
                Some(lp) => GenerateEvent::TokenWithLogprobs(new_text, lp),
                None => GenerateEvent::Token(new_text),
            };
            let _ = seq.response_tx.send(event);
        }

        if seq.generated_tokens.len() >= seq.max_tokens {
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Length))
            {
                tracing::error!("State transition error: {err}");
            }
            seq.decode_state.flush(&self.tokenizer);
            let result =
                seq.decode_state
                    .finish(seq.created_at, seq.prompt_tokens.len(), seq.max_tokens);
            let _ = seq.response_tx.send(GenerateEvent::Done(result));
            self.cache_pool.release(seq.seq_id);
            return;
        }

        if let Err(err) = seq.state.transition_to(SequenceState::Decoding) {
            tracing::error!("State transition error: {err}");
            self.abort_sequence(seq, &err);
            return;
        }

        let prompt_len = seq.prompt_tokens.len() as i32;
        if let Some(cache_set) = self.cache_pool.get_mut(seq.seq_id) {
            cache_set.prompt_len = seq.prompt_tokens.len();
            cache_set.current_offset = prompt_len + 1;
        }

        if let Err(err) = self.active_batch.add(seq) {
            tracing::error!("Failed to add sequence to active batch: {err}");
        }
    }

    // ------------------------------------------------------------------
    // Preemptive eviction
    // ------------------------------------------------------------------

    /// Attempt to evict one sequence from the active batch to make room
    /// for a higher-priority queued request.
    ///
    /// Returns `true` if eviction succeeded (a slot is now free).
    ///
    /// **Streaming caveat:** Tokens already streamed to the client via
    /// `GenerateEvent::Token` are not recalled. When the evicted sequence
    /// is re-prefilled, duplicate tokens may be streamed. This is
    /// acceptable for preemptive scheduling (the client sees a retry)
    /// and is consistent with vLLM's eviction semantics.
    fn try_evict_for_preemption(&mut self) -> bool {
        let victim_id = match self.select_eviction_victim() {
            Some(id) => id,
            None => return false,
        };

        if let Some(mut victim) = self.active_batch.remove(victim_id) {
            tracing::info!(
                "Preempting sequence {} (priority={:?}, {} tokens generated)",
                victim.seq_id,
                victim.priority,
                victim.generated_tokens.len()
            );

            // Release its KV cache
            self.cache_pool.release(victim.seq_id);

            // Reset the sequence for re-prefill: clear generated tokens,
            // reset decode state, and re-allocate a cache slot.
            victim.generated_tokens.clear();
            victim.generated_text.clear();
            victim.prefill_offset = 0;
            victim.decode_state = StreamingDecodeState::new(&self.tokenizer, &victim.prompt_tokens);
            victim.token_history.clear();
            victim.merged_eos.clear();

            // Allocate a fresh cache slot
            match self.cache_pool.allocate(&self.model) {
                Ok(new_id) => {
                    victim.seq_id = new_id;
                    if let Err(err) = victim.state.transition_to(SequenceState::Queued) {
                        tracing::error!("Eviction state transition error: {err}");
                        self.cache_pool.release(new_id);
                        let _ = victim
                            .response_tx
                            .send(GenerateEvent::Error(format!("Eviction state error: {err}")));
                        return true; // Slot is still freed
                    }
                }
                Err(err) => {
                    tracing::warn!("Re-allocation failed for evicted sequence: {err}");
                    let _ = victim.response_tx.send(GenerateEvent::Error(format!(
                        "Preemption re-queue failed: {err}"
                    )));
                    return true; // Slot is still freed
                }
            }

            // Re-queue the evicted sequence (it will re-prefill when admitted)
            if let Err(rejected) = self.prefill_queue.enqueue(victim) {
                self.cache_pool.release(rejected.seq_id);
                let _ = rejected.response_tx.send(GenerateEvent::Error(
                    "Preemption re-queue failed: prefill queue full".to_string(),
                ));
            }

            self.batch_metrics.record_preemption();
            true
        } else {
            false
        }
    }

    /// Select the eviction victim based on the configured policy.
    fn select_eviction_victim(&self) -> Option<SequenceId> {
        match self.preemption_policy {
            PreemptionPolicy::LongestFirst => {
                // Evict the sequence with the most generated tokens
                self.active_batch
                    .iter_sequences()
                    .max_by_key(|seq| seq.generated_tokens.len())
                    .map(|seq| seq.seq_id)
            }
            PreemptionPolicy::LowestPriority => {
                // Evict the lowest-priority sequence; break ties by longest
                self.active_batch
                    .iter_sequences()
                    .min_by(|a, b| {
                        a.priority
                            .cmp(&b.priority)
                            .then_with(|| b.generated_tokens.len().cmp(&a.generated_tokens.len()))
                    })
                    .map(|seq| seq.seq_id)
            }
        }
    }

    // ------------------------------------------------------------------
    // Decode execution (batched when B > 1, sequential fallback otherwise)
    // ------------------------------------------------------------------

    /// Run one decode step for the active sequences.
    fn execute_decode_step(&mut self, seq_ids: &[SequenceId]) {
        let _span = tracing::info_span!("decode_step", batch_size = seq_ids.len(),).entered();
        self.batch_observability.record_decode_step(seq_ids.len());

        if seq_ids.len() <= 1 || !self.model.supports_batching() {
            for &seq_id in seq_ids {
                self.decode_single_step(seq_id);
            }
            return;
        }

        self.execute_batched_decode(seq_ids);
    }

    /// Batched decode: one forward_batched() call for all active sequences.
    fn execute_batched_decode(&mut self, seq_ids: &[SequenceId]) {
        let b = seq_ids.len();

        let mut last_tokens: Vec<i32> = Vec::with_capacity(b);

        for &seq_id in seq_ids {
            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => {
                    self.execute_decode_step_sequential_remaining(seq_ids, last_tokens.len());
                    return;
                }
            };
            last_tokens.push(*seq.generated_tokens.last().unwrap_or(&0));
        }

        let input = mlxcel_core::from_slice_i32(&last_tokens, &[b as i32, 1]);

        debug_assert!(
            {
                let unique: HashSet<_> = seq_ids.iter().collect();
                unique.len() == seq_ids.len()
            },
            "execute_batched_decode: duplicate SequenceId in seq_ids"
        );

        let mut cache_ptrs: Vec<(*mut mlxcel_core::layers::KVCache, usize)> = Vec::with_capacity(b);
        for &seq_id in seq_ids {
            match self.cache_pool.get_caches_mut(seq_id) {
                Some(caches) => {
                    cache_ptrs.push((caches.as_mut_ptr(), caches.len()));
                }
                None => {
                    tracing::error!("Cache not found for {seq_id} during batched decode");
                    return;
                }
            }
        }

        // SAFETY: CachePool stores each SequenceId in a separate HashMap entry
        // (SequenceCacheSet), so the Vec<KVCache> backing each slice is a
        // distinct heap allocation. The debug_assert above verifies no
        // duplicate SequenceIds are present, guaranteeing no two slices alias
        // the same memory. The raw pointers remain valid because cache_pool is
        // not mutated between pointer extraction and the forward_batched call.
        let mut batch_caches: Vec<&mut [mlxcel_core::layers::KVCache]> = cache_ptrs
            .iter()
            .map(|&(ptr, len)| unsafe { std::slice::from_raw_parts_mut(ptr, len) })
            .collect();

        let logits = self.model.forward_batched(&input, &mut batch_caches, None);

        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let seq_logits =
                mlxcel_core::slice(&logits, &[i as i32, 0, 0], &[i as i32 + 1, 1, i32::MAX]);

            // Use cached token_history (incrementally maintained) instead of
            // rebuilding per step. Use cached merged_eos computed once at prefill.
            let (token_val, token_lp) = {
                let seq = match self.active_batch.get_mut(seq_id) {
                    Some(s) => s,
                    None => continue,
                };
                let (token_arr, adjusted_logits) =
                    sample_token_optimized(&seq_logits, &seq.sampling, &seq.token_history);
                mlxcel_core::eval(&token_arr);
                let tok = mlxcel_core::item_i32(&token_arr);
                let lp = compute_logprobs(&adjusted_logits, tok, &seq.logprobs_config);
                (tok, lp)
            };

            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => continue,
            };

            if seq.merged_eos.contains(&token_val) {
                if let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Stop))
                {
                    tracing::error!("State transition error: {err}");
                }
                continue;
            }

            seq.generated_tokens.push(token_val);

            // Incrementally update token_history
            if seq.sampling.needs_token_history() {
                seq.token_history.push(token_val);
            }

            if let Some(new_text) = seq.decode_state.on_token(token_val, &self.tokenizer) {
                let event = match token_lp {
                    Some(lp) => GenerateEvent::TokenWithLogprobs(new_text, lp),
                    None => GenerateEvent::Token(new_text),
                };
                let _ = seq.response_tx.send(event);
            }

            if seq.generated_tokens.len() >= seq.max_tokens
                && let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Length))
            {
                tracing::error!("State transition error: {err}");
            }

            // Periodic cache clearing (matches Python mlx-lm which clears every 256)
            if seq.generated_tokens.len() % 256 == 0 {
                mlxcel_core::clear_memory_cache();
            }

            if let Some(cache_set) = self.cache_pool.get_mut(seq_id) {
                cache_set.current_offset += 1;
            }
        }
    }

    fn execute_decode_step_sequential_remaining(
        &mut self,
        seq_ids: &[SequenceId],
        start_from: usize,
    ) {
        for &seq_id in &seq_ids[start_from..] {
            self.decode_single_step(seq_id);
        }
    }

    fn decode_single_step(&mut self, seq_id: SequenceId) {
        let last_token = {
            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => return,
            };
            *seq.generated_tokens.last().unwrap_or(&0)
        };

        let caches = match self.cache_pool.get_caches_mut(seq_id) {
            Some(c) => c,
            None => {
                tracing::error!("Cache not found for {seq_id} during decode");
                return;
            }
        };

        let input = mlxcel_core::from_slice_i32(&[last_token], &[1, 1]);
        let logits = self.model.forward(&input, caches, None);

        // Use cached token_history from SequenceInfo (incrementally maintained)
        // and cached merged_eos (computed once during prefill) to avoid
        // per-step allocation and reconstruction overhead.
        let (token_val, token_lp) = {
            let seq = self.active_batch.get_mut(seq_id).unwrap();
            let (token_arr, adjusted_logits) =
                sample_token_optimized(&logits, &seq.sampling, &seq.token_history);
            mlxcel_core::eval(&token_arr);
            let tok = mlxcel_core::item_i32(&token_arr);
            let lp = compute_logprobs(&adjusted_logits, tok, &seq.logprobs_config);
            (tok, lp)
        };

        let seq = match self.active_batch.get_mut(seq_id) {
            Some(s) => s,
            None => return,
        };

        if seq.merged_eos.contains(&token_val) {
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Stop))
            {
                tracing::error!("State transition error: {err}");
            }
            return;
        }

        seq.generated_tokens.push(token_val);

        // Incrementally update token_history instead of rebuilding from scratch
        if seq.sampling.needs_token_history() {
            seq.token_history.push(token_val);
        }

        if let Some(new_text) = seq.decode_state.on_token(token_val, &self.tokenizer) {
            let event = match token_lp {
                Some(lp) => GenerateEvent::TokenWithLogprobs(new_text, lp),
                None => GenerateEvent::Token(new_text),
            };
            let _ = seq.response_tx.send(event);
        }

        if seq.generated_tokens.len() >= seq.max_tokens
            && let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Length))
        {
            tracing::error!("State transition error: {err}");
        }

        // Periodic cache clearing (matches Python mlx-lm which clears every 256)
        if seq.generated_tokens.len() % 256 == 0 {
            mlxcel_core::clear_memory_cache();
        }

        if let Some(cache_set) = self.cache_pool.get_mut(seq_id) {
            cache_set.current_offset += 1;
        }
    }

    // ------------------------------------------------------------------
    // Completion and cleanup
    // ------------------------------------------------------------------

    fn finalize_completed(&mut self) {
        // Collect finished IDs by scanning active sequences. Uses iter_sequences()
        // to avoid allocating a full key snapshot when no sequences are finished.
        let finished_ids: Vec<SequenceId> = self
            .active_batch
            .iter_sequences()
            .filter(|s| s.state.is_finished())
            .map(|s| s.seq_id)
            .collect();

        let has_completed = !finished_ids.is_empty();
        for id in finished_ids {
            if let Some(mut seq) = self.active_batch.remove(id) {
                let tokens_generated = seq.generated_tokens.len();

                seq.decode_state.flush(&self.tokenizer);
                let result = seq.decode_state.finish(
                    seq.created_at,
                    seq.prompt_tokens.len(),
                    seq.max_tokens,
                );
                let _ = seq.response_tx.send(GenerateEvent::Done(result));

                self.cache_pool.release(id);
                self.batch_metrics
                    .record_sequence_completed(tokens_generated);
                self.batch_observability.record_sequence_completed();

                tracing::debug!("Sequence {id} completed ({tokens_generated} tokens)");
            }
        }

        if has_completed {
            self.publish_metrics();
        }
    }

    fn abort_sequence(&mut self, seq: SequenceInfo, error: &str) {
        let _ = seq
            .response_tx
            .send(GenerateEvent::Error(error.to_string()));
        self.cache_pool.release(seq.seq_id);
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;

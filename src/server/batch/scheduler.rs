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

//! Core batch scheduler with iteration-level scheduling.
//!
//! [`BatchScheduler`] replaces the sequential `loop { request_rx.recv() }`
//! pattern in the model worker. At each tick it decides whether to:
//!
//! - **Prefill** a new queued request (prompt processing + first token),
//! - **Decode** one token for each active sequence, or
//! - **Idle** (block until the next request arrives).
//!
//! Phase 1 uses a sequential loop for decode: one `forward()` call per
//! active sequence. Sub-issue 4 (#16) upgrades this to batched decode.

use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::{CachePool, SequenceId};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::generation_policy::{
    initial_token_history, merged_eos_token_ids, seed_rng_if_needed,
};
use mlxcel_core::sampling::sample_token_optimized;
use mlxcel_core::streams::{install_default_stream, new_generation_stream};
use mlxcel_core::{MlxStream, UniquePtr};

use crate::LoadedModel;
use crate::server::ServerGenerateOptions;
use crate::server::model_provider::model_worker::{
    StreamingDecodeState, build_generation_result, merge_config_stop_tokens,
    prepare_request_vlm_embeddings,
};
use crate::server::model_provider::{GenerateEvent, ModelRequest};
use crate::tokenizer::MlxcelTokenizer;
use crate::vlm_runtime::prepared_embedding_refs;

use super::active::ActiveBatch;
use super::queue::PrefillQueue;
use super::sequence::{BatchSchedulerAction, FinishReason, SequenceInfo, SequenceState};

/// Core batch scheduler that drives the model worker loop.
///
/// Replaces the old sequential `recv()` loop with an iteration-level scheduler
/// that interleaves prefill and decode operations. When `max_batch_size == 1`
/// (the default), behavior is identical to the pre-scheduler worker loop.
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

    // -- Configuration --
    config_eos: Vec<i32>,

    // -- Shutdown flag --
    /// Set to `true` when a `Shutdown` message is received during the
    /// non-blocking drain phase, so the main loop can observe it and exit.
    shutdown_requested: bool,
}

impl BatchScheduler {
    /// Create a new batch scheduler, taking ownership of the model and channel.
    ///
    /// `max_batch_size` controls the maximum number of concurrent decode
    /// sequences. `max_queue_depth` controls the maximum number of pending
    /// prefill requests.
    pub fn new(
        model: LoadedModel,
        tokenizer: MlxcelTokenizer,
        config_eos: Vec<i32>,
        request_rx: mpsc::Receiver<ModelRequest>,
        max_batch_size: usize,
        max_queue_depth: usize,
    ) -> Self {
        let generation_stream = new_generation_stream();
        let max_batch_size = max_batch_size.max(1);
        Self {
            cache_pool: CachePool::new(max_batch_size),
            prefill_queue: PrefillQueue::with_capacity(max_queue_depth),
            active_batch: ActiveBatch::new(max_batch_size),
            model,
            tokenizer,
            generation_stream,
            request_rx,
            config_eos,
            shutdown_requested: false,
        }
    }

    /// Run the scheduler loop until shutdown or channel close.
    ///
    /// This is the main entry point, called from the worker thread. It never
    /// returns under normal operation; it exits when:
    /// - A `ModelRequest::Shutdown` is received, or
    /// - The request channel is closed (all senders dropped).
    pub fn run(&mut self) {
        install_default_stream(self.generation_stream.as_ref());

        loop {
            // 1. Non-blocking drain of all pending requests
            self.drain_incoming_requests();

            // Exit if shutdown was received during drain
            if self.shutdown_requested {
                break;
            }

            // 2. Decide what to do this tick
            let action = self.decide_action();

            // 3. Execute
            match action {
                BatchSchedulerAction::Prefill(seq_id) => {
                    self.execute_prefill(seq_id);
                }
                BatchSchedulerAction::Decode(ids) => {
                    self.execute_decode_step(&ids);
                }
                BatchSchedulerAction::Idle => {
                    // Block until next request arrives
                    match self.request_rx.recv() {
                        Ok(req) => {
                            if self.handle_incoming(req) {
                                // Shutdown requested
                                break;
                            }
                        }
                        Err(_) => {
                            tracing::info!("Request channel closed, scheduler exiting");
                            break;
                        }
                    }
                }
            }

            // 4. Clean up completed sequences
            self.finalize_completed();
        }
    }

    // ------------------------------------------------------------------
    // Request ingestion
    // ------------------------------------------------------------------

    /// Non-blocking drain: pull all pending requests from the channel.
    ///
    /// Sets `self.shutdown_requested = true` if a `Shutdown` message is
    /// received, so the caller can break out of the main loop.
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

    /// Process a single incoming request. Returns `true` if shutdown.
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

    /// Convert a `ModelRequest::Generate` into a `SequenceInfo` and enqueue.
    fn enqueue_request(
        &mut self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        response_tx: mpsc::Sender<GenerateEvent>,
    ) {
        // Tokenize
        let token_ids = match self.tokenizer.encode(&prompt, true) {
            Ok(ids) => ids,
            Err(err) => {
                let _ =
                    response_tx.send(GenerateEvent::Error(format!("Tokenization error: {err}")));
                return;
            }
        };
        let mut prompt_tokens: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();

        // Merge config-level EOS tokens into sampling
        let sampling = merge_config_stop_tokens(options.sampling.clone(), &self.config_eos);

        // Allocate a sequence ID from the cache pool
        let seq_id = match self.cache_pool.allocate(&self.model) {
            Ok(id) => id,
            Err(err) => {
                tracing::warn!("Cache pool allocation failed: {err}");
                let _ = response_tx.send(GenerateEvent::Error(format!("Server busy: {err}")));
                return;
            }
        };

        // Prepare VLM embeddings (if applicable)
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
            vlm_embeddings,
            images,
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            decode_state,
            response_tx,
            created_at: Instant::now(),
            prefill_start: None,
            first_token_time: None,
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
    /// Policy (designed to prevent starvation of active sequences):
    /// - If both the active batch and prefill queue are empty, idle.
    /// - If the active batch has sequences, always decode first. This
    ///   ensures active sequences make progress every tick and are not
    ///   starved by a continuous stream of incoming prefill requests.
    /// - If the active batch is empty but the prefill queue is not, prefill
    ///   the next queued request.
    /// - If the batch is full and the queue is non-empty, decode (the queued
    ///   requests will be prefilled once a slot opens).
    fn decide_action(&self) -> BatchSchedulerAction {
        if self.active_batch.is_empty() && self.prefill_queue.is_empty() {
            return BatchSchedulerAction::Idle;
        }
        // Active sequences always get a decode step before admitting new
        // prefills. This prevents latency starvation under sustained load.
        if !self.active_batch.is_empty() {
            return BatchSchedulerAction::Decode(self.active_batch.sequence_ids());
        }
        // Batch is empty but queue has work -- prefill the next request.
        // The sentinel ID is informational; execute_prefill dequeues
        // from the front of the queue regardless.
        BatchSchedulerAction::Prefill(SequenceId::from_raw(0))
    }

    // ------------------------------------------------------------------
    // Prefill execution
    // ------------------------------------------------------------------

    /// Prefill a single sequence: run the full prompt through the model,
    /// sample the first token, and move the sequence to the active batch.
    fn execute_prefill(&mut self, _action_id: SequenceId) {
        let mut seq = match self.prefill_queue.dequeue() {
            Some(s) => s,
            None => return,
        };

        // Transition state
        if let Err(err) = seq.state.transition_to(SequenceState::Prefilling) {
            tracing::error!("State transition error: {err}");
            self.abort_sequence(seq, &err);
            return;
        }
        seq.prefill_start = Some(Instant::now());

        // Set up sampling RNG
        seed_rng_if_needed(&seq.sampling);

        // Build merged EOS list for this sequence
        let eos_tokens =
            merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);

        // Build token history for penalty sampling
        let needs_history = seq.sampling.needs_token_history();
        let mut token_history = initial_token_history(&seq.prompt_tokens, needs_history);

        // Get per-sequence caches
        let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
            Some(c) => c,
            None => {
                self.abort_sequence(seq, "Cache not found for sequence during prefill");
                return;
            }
        };

        // Prepare input tensor
        let prompt_len = seq.prompt_tokens.len() as i32;
        let input = mlxcel_core::from_slice_i32(&seq.prompt_tokens, &[1, prompt_len]);

        // Run prefill (with or without VLM embeddings)
        let logits = if let Some(ref embeddings) = seq.vlm_embeddings {
            match prepared_embedding_refs(embeddings) {
                Ok((input_embeds, mask_ref)) => {
                    let logits = self.model.forward_with_embeddings(
                        &input,
                        Some(input_embeds),
                        caches,
                        mask_ref,
                    );
                    // Force evaluation before after_prefill modifications
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
            self.model.forward(&input, caches, None)
        };

        // Clear intermediate prefill tensors
        mlxcel_core::clear_memory_cache();

        // Sample first token
        let (first_token_arr, _logprobs) =
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
            // Emit done immediately (no tokens generated)
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

        // Record the generated token
        seq.generated_tokens.push(first_token);
        if needs_history {
            token_history.push(first_token);
        }

        // Emit the first token via streaming
        if let Some(new_text) = seq.decode_state.on_token(first_token, &self.tokenizer) {
            let _ = seq.response_tx.send(GenerateEvent::Token(new_text));
        }

        // Check max_tokens (unlikely after just 1 token, but be correct)
        if seq.generated_tokens.len() >= seq.max_tokens {
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Length))
            {
                tracing::error!("State transition error: {err}");
            }
            // Will be cleaned up in finalize_completed via active batch
            // But since we haven't added to active batch yet, handle inline
            seq.decode_state.flush(&self.tokenizer);
            let result =
                seq.decode_state
                    .finish(seq.created_at, seq.prompt_tokens.len(), seq.max_tokens);
            let _ = seq.response_tx.send(GenerateEvent::Done(result));
            self.cache_pool.release(seq.seq_id);
            return;
        }

        // Transition to Decoding and add to active batch
        if let Err(err) = seq.state.transition_to(SequenceState::Decoding) {
            tracing::error!("State transition error: {err}");
            self.abort_sequence(seq, &err);
            return;
        }

        // Update cache metadata
        if let Some(cache_set) = self.cache_pool.get_mut(seq.seq_id) {
            cache_set.prompt_len = seq.prompt_tokens.len();
            cache_set.current_offset = prompt_len + 1; // prompt + first token
        }

        if let Err(err) = self.active_batch.add(seq) {
            tracing::error!("Failed to add sequence to active batch: {err}");
        }
    }

    // ------------------------------------------------------------------
    // Decode execution (Phase 1: loop-based, one forward per sequence)
    // ------------------------------------------------------------------

    /// Run one decode step for each active sequence.
    ///
    /// Phase 1: sequential loop. Each sequence gets its own `forward()` call.
    fn execute_decode_step(&mut self, seq_ids: &[SequenceId]) {
        for &seq_id in seq_ids {
            self.decode_single_step(seq_id);
        }
    }

    /// Decode one token for a single sequence.
    fn decode_single_step(&mut self, seq_id: SequenceId) {
        // We need to split borrows: get the caches from the pool first,
        // then the sequence from the batch. Since both need &mut self,
        // we extract what we need in stages.

        // Extract only cheap, Copy/non-owning data from the sequence to
        // avoid cloning SamplingConfig every decode step.
        let (last_token, needs_history) = {
            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => return,
            };
            let last = *seq.generated_tokens.last().unwrap_or(&0);
            (last, seq.sampling.needs_token_history())
        };

        // Build a token history snapshot. This is only needed when penalty
        // sampling is active (repetition, frequency, or presence penalties).
        // We build it only when required to avoid O(prompt+gen) allocation
        // on every step.
        let token_history = if needs_history {
            let seq = self.active_batch.get_mut(seq_id).unwrap();
            let mut history =
                Vec::with_capacity(seq.prompt_tokens.len() + seq.generated_tokens.len());
            history.extend_from_slice(&seq.prompt_tokens);
            history.extend_from_slice(&seq.generated_tokens);
            history
        } else {
            Vec::new()
        };

        // Get caches and run forward
        let caches = match self.cache_pool.get_caches_mut(seq_id) {
            Some(c) => c,
            None => {
                tracing::error!("Cache not found for {seq_id} during decode");
                return;
            }
        };

        // Build input: single token reshaped to [1, 1]
        let input = mlxcel_core::from_slice_i32(&[last_token], &[1, 1]);

        // Forward pass
        let logits = self.model.forward(&input, caches, None);

        // Sample -- borrow sampling by reference from the sequence to avoid
        // cloning the SamplingConfig (which contains Vecs and Strings).
        let (token_val, eos_tokens) = {
            let seq = self.active_batch.get_mut(seq_id).unwrap();
            let (token_arr, _logprobs) =
                sample_token_optimized(&logits, &seq.sampling, &token_history);
            mlxcel_core::eval(&token_arr);
            let val = mlxcel_core::item_i32(&token_arr);

            let eos =
                merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
            (val, eos)
        };

        // Update the sequence
        let seq = match self.active_batch.get_mut(seq_id) {
            Some(s) => s,
            None => return,
        };

        // Check EOS
        if eos_tokens.contains(&token_val) {
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Stop))
            {
                tracing::error!("State transition error: {err}");
            }
            return;
        }

        // Record token
        seq.generated_tokens.push(token_val);

        // Stream the decoded text
        if let Some(new_text) = seq.decode_state.on_token(token_val, &self.tokenizer) {
            let _ = seq.response_tx.send(GenerateEvent::Token(new_text));
        }

        // Check max_tokens
        if seq.generated_tokens.len() >= seq.max_tokens
            && let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Length))
        {
            tracing::error!("State transition error: {err}");
        }

        // Periodic cache clearing
        if seq.generated_tokens.len() % 512 == 0 {
            mlxcel_core::clear_memory_cache();
        }

        // Update cache pool metadata
        if let Some(cache_set) = self.cache_pool.get_mut(seq_id) {
            cache_set.current_offset += 1;
        }
    }

    // ------------------------------------------------------------------
    // Completion and cleanup
    // ------------------------------------------------------------------

    /// Collect and finalize all finished sequences.
    fn finalize_completed(&mut self) {
        // Collect finished IDs first to avoid borrow issues
        let finished_ids: Vec<SequenceId> = self
            .active_batch
            .sequence_ids()
            .into_iter()
            .filter(|id| {
                self.active_batch
                    .get_mut(*id)
                    .map(|s| s.state.is_finished())
                    .unwrap_or(false)
            })
            .collect();

        for id in finished_ids {
            if let Some(mut seq) = self.active_batch.remove(id) {
                // Flush any remaining buffered text
                seq.decode_state.flush(&self.tokenizer);

                // Build and send result
                let result = seq.decode_state.finish(
                    seq.created_at,
                    seq.prompt_tokens.len(),
                    seq.max_tokens,
                );
                let _ = seq.response_tx.send(GenerateEvent::Done(result));

                // Release cache
                self.cache_pool.release(id);

                tracing::debug!(
                    "Sequence {id} completed ({} tokens)",
                    seq.generated_tokens.len()
                );
            }
        }
    }

    /// Abort a sequence with an error, releasing resources.
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

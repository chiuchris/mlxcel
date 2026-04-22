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
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::{
    CachePool, DetachedCacheSet, PagedKvLayout, SequenceId, SequenceStateBackend,
    SequenceStateLayout,
};
use mlxcel_core::generate::{
    DecodeBatchContext, DecodeStorageBackend as CoreDecodeStorageBackend, LanguageModel,
};
use mlxcel_core::generation_policy::{
    initial_token_history, merged_eos_token_ids, seed_rng_if_needed,
};
use mlxcel_core::hardware;
use mlxcel_core::sampling::{TokenBiasMap, compute_logprobs, sample_token_optimized};
use mlxcel_core::streams::{install_default_stream, new_generation_stream};
use mlxcel_core::utils::{align_to_na_tile, create_padded_prefill_mask};
use mlxcel_core::{MlxStream, UniquePtr};

use crate::LoadedModel;
use crate::server::ServerGenerateOptions;
use crate::server::batch::observability::BatchObservability;
use crate::server::config::{
    DecodeStorageBackend, PreemptionPolicy, PromptCacheRequestContext, ReasoningBudgetOverride,
};
use crate::server::model_provider::model_worker::{
    StreamingDecodeState, build_generation_result, merge_config_stop_tokens,
    prepare_request_vlm_embeddings,
};
use crate::server::model_provider::{GenerateEvent, ModelRequest};
use crate::server::prompt_cache::key::PromptCacheKey;
use crate::server::prompt_cache::{CacheEntry, PromptCacheStore};
use crate::server::state::BatchMetrics;
use crate::server::thinking_budget::{
    ThinkingBudget, ThinkingDecision, ThinkingState, ThinkingTokenIds,
};
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::feature_cache::ModelVisionCaches;
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

const DEFAULT_PAGED_BLOCK_SIZE: usize = 32;

fn effective_decode_storage_backend(
    requested: DecodeStorageBackend,
    max_batch_size: usize,
    supports_batching: bool,
    supports_paged_decode_backend: bool,
) -> DecodeStorageBackend {
    let paged_available = max_batch_size > 1 && supports_batching && supports_paged_decode_backend;
    match requested {
        DecodeStorageBackend::Auto | DecodeStorageBackend::Paged if paged_available => {
            DecodeStorageBackend::Paged
        }
        DecodeStorageBackend::Auto | DecodeStorageBackend::Paged => DecodeStorageBackend::Dense,
        DecodeStorageBackend::Dense => DecodeStorageBackend::Dense,
    }
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
    /// Decode-time sequence-state backend used by this scheduler.
    decode_storage_backend: DecodeStorageBackend,

    // -- Vision feature cache --
    /// Per-model vision feature cache bundle. Contains LRU caches for
    /// post-projection image features so multi-turn VLM conversations can
    /// skip the vision tower when the same image is referenced across turns.
    ///
    /// Stored as `Rc<..>` because the scheduler is single-threaded (all MLX
    /// work runs on the worker thread). The cache is cleared automatically
    /// when the scheduler (and thus this loaded model) is dropped.
    vision_caches: Rc<ModelVisionCaches>,

    // -- Axis B / Epic #362 — language-bias token map --
    /// Cached per-scheduler `TokenBiasMap` resolved once from the server-level
    /// `LangBiasConfig` at worker startup.
    ///
    /// **Phase 1 limitation — single policy per batch**: every sequence in
    /// this scheduler's active batch receives the same bias, regardless of
    /// per-request preferences. Per-sequence override via the
    /// `/v1/chat/completions` request body is reserved for a follow-up
    /// issue (B12) tracked outside this Epic. The bias is attached to each
    /// queued sequence's [`SamplingConfig`] at `enqueue_request` time so
    /// per-step sampling (`sample_token_optimized`) observes it with no
    /// additional hot-path overhead beyond the existing
    /// [`mlxcel_core::sampling::apply_token_bias`] fast path.
    ///
    /// Empty map = bit-exact baseline path (no sampling change, no alloc).
    token_bias: TokenBiasMap,

    // -- Issue #409 — thinking-token budget --
    /// Server-wide default thinking-token budget. `None` means unrestricted.
    /// Per-request `thinking_budget_tokens` overrides this at enqueue time.
    reasoning_budget: Option<ThinkingBudget>,
    /// Cached `<think>` / `</think>` token id pair resolved once from the
    /// tokenizer at worker startup. `None` for non-thinking models; when
    /// `None`, every sequence's [`ThinkingState`] is constructed as disabled
    /// regardless of any budget configuration.
    thinking_token_ids: Option<ThinkingTokenIds>,

    // -- Epic #416 / issue #421: cross-request prompt-prefix KV cache --
    /// Shared store that hands out detached KV caches on prefix match and
    /// absorbs donated caches on sequence finish. `None` when the feature is
    /// disabled at config time so the hot path has zero overhead.
    prompt_cache: Option<Arc<PromptCacheStore>>,

    /// Parallel map indexed by `SequenceId`: remembers the
    /// [`PromptCacheRequestContext`] per in-flight sequence so the donate-back
    /// path on completion can rebuild the cache key without touching the HTTP
    /// request layer again. Dropped automatically when the sequence is
    /// removed from the map on finish / error.
    prompt_cache_seq_ctx: std::collections::HashMap<SequenceId, PromptCacheRequestContext>,
}

impl BatchScheduler {
    fn release_sequence_caches(&mut self, seq_id: SequenceId) {
        self.model.release_sequence_state_by_id(seq_id);
        if let Some(caches) = self.cache_pool.get_caches_mut(seq_id) {
            self.model.release_sequence_state(caches);
        }
        self.cache_pool.release(seq_id);
    }

    fn begin_prefill(seq: &mut SequenceInfo) -> Result<(), String> {
        seq.state.transition_to(SequenceState::Prefilling)?;
        seq.prefill_start = Some(Instant::now());
        seed_rng_if_needed(&seq.sampling);
        Ok(())
    }

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
            DecodeStorageBackend::Dense,
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
        decode_storage_backend: DecodeStorageBackend,
    ) -> Self {
        let generation_stream = new_generation_stream();
        let max_batch_size = max_batch_size.max(1);
        let effective_decode_storage = effective_decode_storage_backend(
            decode_storage_backend,
            max_batch_size,
            model.supports_batching(),
            model.supports_paged_decode_backend(),
        );
        if decode_storage_backend == DecodeStorageBackend::Paged
            && effective_decode_storage != decode_storage_backend
        {
            tracing::info!(
                "Paged decode storage requested but unavailable for this worker; falling back to dense"
            );
            batch_observability.record_decode_storage_fallback();
        }
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
            decode_storage_backend: effective_decode_storage,
            vision_caches: Rc::new(ModelVisionCaches::new(
                crate::vision::feature_cache::DEFAULT_VISION_CACHE_SIZE,
            )),
            token_bias: TokenBiasMap::default(),
            reasoning_budget: None,
            thinking_token_ids: None,
            prompt_cache: None,
            prompt_cache_seq_ctx: std::collections::HashMap::new(),
        }
    }

    /// Replace the default vision feature cache with one sized per the server
    /// configuration.
    ///
    /// `max_size == 0` disables the cache entirely; non-zero values mirror
    /// the `--vision-cache-size` CLI flag. Callers that do not invoke this
    /// method get the default size from
    /// [`crate::vision::feature_cache::DEFAULT_VISION_CACHE_SIZE`].
    pub fn with_vision_cache_size(mut self, max_size: usize) -> Self {
        self.vision_caches = Rc::new(ModelVisionCaches::new(max_size));
        self
    }

    /// Attach a pre-resolved Axis B `TokenBiasMap` to this scheduler (B8).
    ///
    /// The bias is cached for the scheduler's lifetime and applied to every
    /// queued sequence's [`SamplingConfig`] at enqueue time (see the merge in
    /// [`Self::enqueue_request`]). An empty map is a zero-overhead no-op on
    /// the hot sampling path — [`sample_token_optimized`] still short-circuits
    /// via the existing `config.token_bias.is_empty()` branch.
    ///
    /// **Phase 1 limitation**: one policy per batch (scheduler-wide).
    /// Per-sequence overrides via request-body `lang_bias` are reserved for
    /// the B12 follow-up outside this Epic.
    pub fn with_token_bias(mut self, bias: TokenBiasMap) -> Self {
        self.token_bias = bias;
        self
    }

    /// Returns a reference to the cached token-bias map (for tests).
    pub fn token_bias(&self) -> &TokenBiasMap {
        &self.token_bias
    }

    /// Attach the server-wide thinking-token budget and resolved
    /// `<think>` / `</think>` token ids (issue #409).
    ///
    /// `token_ids == None` means the model is non-thinking; the budget is
    /// then silently ignored for every sequence. Callers resolve the token
    /// ids once via
    /// [`crate::server::thinking_budget::resolve_thinking_token_ids`] after
    /// the tokenizer is loaded.
    pub fn with_reasoning_budget(
        mut self,
        budget: Option<ThinkingBudget>,
        token_ids: Option<ThinkingTokenIds>,
    ) -> Self {
        self.reasoning_budget = budget;
        self.thinking_token_ids = token_ids;
        self
    }

    /// Attach the shared prompt-prefix KV cache store
    /// (epic #416 / issue #421).
    ///
    /// When `Some(..)`, the scheduler:
    /// * Looks up a longest-prefix match on each new request and calls
    ///   [`CachePool::adopt`] on hit to skip re-prefill of the shared prefix.
    /// * Donates the sequence's full cache back to the store on a healthy
    ///   finish (normal stop / length / cancelled without error).
    /// * Never donates back on OOM, transition errors, or
    ///   `Finished(FinishReason::Error(..))`.
    ///
    /// When `None` every hot path short-circuits on the `is_some()` check
    /// before any store access so the bit-exact baseline is preserved.
    pub fn with_prompt_cache(mut self, store: Option<Arc<PromptCacheStore>>) -> Self {
        self.prompt_cache = store;
        self
    }

    /// Whether the installed prompt-cache store is currently accepting
    /// lookups and inserts (scheduler-level gate for #424).
    #[inline]
    fn prompt_cache_active(&self) -> bool {
        self.prompt_cache
            .as_ref()
            .map(|s| s.is_enabled())
            .unwrap_or(false)
    }

    /// Build a [`PromptCacheKey`] bound to the per-request metadata the
    /// scheduler captured at enqueue time. Returns `None` when the request
    /// carried no [`PromptCacheRequestContext`] (e.g. non-chat endpoints).
    fn compose_prompt_cache_key<'a>(
        ctx: &'a PromptCacheRequestContext,
        tokens: &'a [i32],
    ) -> PromptCacheKey<'a> {
        PromptCacheKey::new_full(
            ctx.model_id.as_str(),
            ctx.lora_id.as_deref(),
            ctx.template_sig.as_str(),
            Some(ctx.session_key.as_str()),
            tokens,
        )
    }

    /// Attempt to adopt a cached prefix for a freshly tokenized request,
    /// returning the adopted `SequenceId` together with the matched-prefix
    /// length on success.
    ///
    /// The caller invokes this **before** [`Self::allocate_sequence_state`]
    /// so the adopted id becomes the sequence's canonical id and no
    /// seq_id rebinding dance is required. On any miss path the caller
    /// proceeds with a fresh allocation under a brand-new id.
    ///
    /// Gating (all of these yield `None`, which maps to a cold prefill):
    /// * feature disabled at config time,
    /// * request carried no [`PromptCacheRequestContext`] (non-chat endpoint),
    /// * scheduler configured with the paged decode backend (paged adoption
    ///   through the store is deferred — the store's [`CacheEntry`] type
    ///   currently only carries dense detached sets),
    /// * store miss / match shorter than `min_prefix_tokens`,
    /// * race with another worker that already consumed the entry,
    /// * empty-tensor detached set (e.g. stored against an aborted seq),
    /// * [`CachePool::adopt`] error (capacity, unsupported backend, …).
    fn try_adopt_cached_prefix(
        &mut self,
        ctx: &PromptCacheRequestContext,
        tokens: &[i32],
    ) -> Option<(SequenceId, usize)> {
        if !self.prompt_cache_active() {
            return None;
        }
        if self.decode_storage_backend == DecodeStorageBackend::Paged {
            return None;
        }

        let store = self.prompt_cache.as_ref()?.clone();
        let key = Self::compose_prompt_cache_key(ctx, tokens);
        let (entry, matched_len) = store.lookup_longest_prefix(&key, tokens)?;
        // `take_detached` is one-shot: it returns `None` if a racing lookup
        // already consumed this entry. The miss path is safe — the current
        // sequence just does a fresh prefill.
        let detached = entry.take_detached()?;
        if detached.caches.is_empty() || detached.caches.iter().all(|c| c.is_empty()) {
            return None;
        }

        match self
            .cache_pool
            .adopt(&self.model as &dyn LanguageModel, detached)
        {
            Ok(adopted_id) => {
                tracing::debug!(
                    seq_id = %adopted_id,
                    matched = matched_len,
                    total = tokens.len(),
                    "prompt-cache hit: adopted {matched_len}/{} tokens",
                    tokens.len()
                );
                self.batch_observability
                    .record_prompt_cache_hit(matched_len);
                Some((adopted_id, matched_len))
            }
            Err(err) => {
                tracing::debug!("prompt-cache adopt failed ({err}); falling back to cold prefill");
                None
            }
        }
    }

    /// Donate a finished sequence's KV cache back to the store so future
    /// requests sharing a prefix can adopt it.
    ///
    /// The caller must invoke this **before** calling
    /// [`Self::release_sequence_caches`] — once release runs the underlying
    /// tensors are gone. Safe to call unconditionally; all the gating checks
    /// (feature enabled, healthy finish, context present, dense backend)
    /// live inside this method so the caller can keep its hot-path code
    /// simple.
    fn donate_finished_sequence_cache(
        &mut self,
        seq_id: SequenceId,
        prompt_tokens: &[i32],
        generated_tokens: &[i32],
        healthy_finish: bool,
    ) {
        if !healthy_finish {
            return;
        }
        if !self.prompt_cache_active() {
            return;
        }
        // Remove the context regardless of whether the donate-back succeeds
        // so the map doesn't grow unbounded across sequences that never
        // qualified for a donate-back.
        let ctx = match self.prompt_cache_seq_ctx.remove(&seq_id) {
            Some(c) => c,
            None => return,
        };

        // Only dense sequences can travel through the store's
        // `DetachedCacheSet` (paged follow-up work).
        let backend = self
            .cache_pool
            .get_mut(seq_id)
            .map(|s| s.backend)
            .unwrap_or(SequenceStateBackend::ModelOwned);
        if backend != SequenceStateBackend::DenseKvCache {
            return;
        }

        let detached: DetachedCacheSet = match self.cache_pool.detach(seq_id) {
            Some(d) => d,
            None => return,
        };
        if detached.caches.is_empty() || detached.caches.iter().all(|c| c.is_empty()) {
            // Nothing to cache: aborted before any prefill completed, or
            // the model never populated the KV tensors.
            return;
        }

        // Tokens stored against the entry are the full prompt + generated
        // tail, so the next turn's `prompt + new user turn` can match at
        // least up through the previous assistant reply.
        let mut tokens = Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
        tokens.extend_from_slice(prompt_tokens);
        tokens.extend_from_slice(generated_tokens);

        let store = match self.prompt_cache.as_ref() {
            Some(s) => s.clone(),
            None => return,
        };
        // The `CacheEntry` takes ownership of `tokens` and the key borrows
        // from the same buffer. Build the entry first, then form the key
        // against `entry.tokens` so both reference the same contiguous
        // allocation without copying the vector.
        let entry = CacheEntry::new(tokens, detached);
        let key_tokens = entry.tokens.clone();
        let key = Self::compose_prompt_cache_key(&ctx, &key_tokens);
        match store.insert(&key, entry) {
            Ok(()) => {
                self.batch_observability.record_prompt_cache_insert();
            }
            Err(err) => {
                // Oversized / disabled / prefix-too-short — drop the entry
                // so the detached buffers are freed.
                tracing::debug!(
                    seq_id = %seq_id,
                    "prompt-cache donate-back skipped: {err:?}"
                );
                self.batch_observability.record_prompt_cache_insert_reject();
            }
        }
    }

    /// Apply issue #409 thinking-budget enforcement to a freshly sampled
    /// token for a single sequence.
    ///
    /// Returns the final token id to commit to the sequence (either the
    /// sampled value, or the forced `</think>` id when the budget fires).
    /// Caller is responsible for using the returned id for the remainder of
    /// the decode step (EOS check, streaming emission, history update).
    ///
    /// The state advances with the final id so subsequent steps see the
    /// post-close phase.
    ///
    /// # Notes on bypass of sampling knobs
    ///
    /// When the budget fires the forced id bypasses the sampler's logits
    /// pipeline for that step. No retroactive re-penalization happens because
    /// - `token_history` is only appended once per step (caller uses the
    ///   returned id),
    /// - `merged_eos` checks use the returned id,
    /// - the next step samples fresh logits from the underlying model.
    fn apply_thinking_budget(seq_thinking: &mut ThinkingState, sampled: i32) -> i32 {
        if seq_thinking.is_disabled() {
            return sampled;
        }
        let final_id = match seq_thinking.decide_override(sampled) {
            ThinkingDecision::NoOverride => sampled,
            ThinkingDecision::ForceClose(close_id) => close_id,
        };
        seq_thinking.observe(final_id);
        final_id
    }

    /// Effective thinking-budget for a single sequence.
    ///
    /// Combines the server default with any per-request override attached to
    /// the request's `ServerGenerateOptions`. Returns a [`ThinkingState`]
    /// ready to be stored on `SequenceInfo`.
    ///
    /// `enter_block_on_start` is passed through to the [`ThinkingState`].
    /// Chat endpoints set `true` (the Qwen3 chat template primes `<think>\n`);
    /// raw text endpoints (`/v1/completions`, `/completion`) set `false` so
    /// the model must emit `<think>` before any in-block counting begins.
    fn build_thinking_state(
        &self,
        override_: ReasoningBudgetOverride,
        enter_block_on_start: bool,
    ) -> ThinkingState {
        // No thinking tokens -> always disabled regardless of config.
        let Some(token_ids) = self.thinking_token_ids else {
            return ThinkingState::disabled();
        };
        let effective = match override_ {
            ReasoningBudgetOverride::InheritServerDefault => self.reasoning_budget,
            ReasoningBudgetOverride::Explicit(v) => v,
        };
        ThinkingState::new(Some(token_ids), effective, enter_block_on_start)
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
        let paged_stats = self.cache_pool.paged_stats();
        let paged_block_size = self.cache_pool.paged_block_size().unwrap_or(0);
        self.batch_metrics.set_active_count(active);
        self.batch_metrics.set_queue_depth(queued);
        self.batch_observability.update_gauges(
            active,
            queued,
            self.cache_pool.active_count(),
            self.cache_pool.memory_usage_bytes() as u64,
            paged_block_size,
            paged_stats,
        );
    }

    fn allocate_sequence_state(&mut self) -> Result<SequenceId, String> {
        let layout_override = self.sequence_state_layout_override();
        let seq_id = self
            .cache_pool
            .allocate_with_layout(&self.model, layout_override)?;
        self.model.prepare_sequence_state(seq_id);
        Ok(seq_id)
    }

    fn sequence_state_layout_override(&self) -> Option<SequenceStateLayout> {
        if self.decode_storage_backend != DecodeStorageBackend::Paged {
            return None;
        }

        let num_layers = self.model.num_layers();
        let paged_layout = PagedKvLayout::uniform(
            num_layers,
            DEFAULT_PAGED_BLOCK_SIZE,
            DEFAULT_PAGED_BLOCK_SIZE,
        )
        .expect("valid paged decode layout");
        Some(SequenceStateLayout::paged_kv_cache(paged_layout))
    }

    fn sync_sequence_storage(&mut self, seq_id: SequenceId) {
        if let Err(err) = self
            .model
            .sync_sequence_storage(seq_id, &mut self.cache_pool)
        {
            tracing::warn!("Failed to sync paged state for {seq_id}: {err}");
        }
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
                audio,
                response_tx,
                cancelled,
            } => {
                self.enqueue_request(prompt, options, images, audio, response_tx, cancelled);
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
        audio: Vec<Vec<u8>>,
        response_tx: mpsc::Sender<GenerateEvent>,
        cancelled: Arc<AtomicBool>,
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

        // Empty-prompt guard (null/empty-cache safety):
        //
        // A zero-token prompt cannot be prefilled — the forward pass would
        // run with a `[1, 0]` input and the per-sequence KV cache would
        // remain in the `keys is None, offset == 0` state. Admitting such a
        // request into the batch could later crash the scheduler when the
        // cache is used alongside populated caches in `execute_batched_*`.
        // Mirrors the upstream `mlx-lm` `BatchKVCache.extend` null-guard
        // that refuses to pad/concatenate a cache with no tensors. VLM
        // requests may legitimately start with an empty token list (image
        // tokens are injected later by `prepare_request_vlm_embeddings`),
        // so this guard only applies to pure-text requests without images
        // or audio.
        if prompt_tokens.is_empty() && images.is_empty() && audio.is_empty() {
            let _ = response_tx.send(GenerateEvent::Error(
                "Empty prompt: request has no input tokens to process".to_string(),
            ));
            return;
        }

        let mut sampling = merge_config_stop_tokens(options.sampling.clone(), &self.config_eos);

        // Axis B (B8): attach the scheduler-wide token bias to each sequence's
        // sampling config when no per-request override is present. Empty
        // cached bias = bit-exact baseline (the `is_empty()` short-circuit in
        // `sample_token_optimized` keeps hot-path cost at zero).
        //
        // Phase 1 limitation: one policy per batch. Per-request overrides
        // via `/v1/chat/completions` request body are deferred to B12.
        if !self.token_bias.is_empty() && sampling.token_bias.is_empty() {
            sampling.token_bias = self.token_bias.clone();
        }

        // Epic #416 / issue #421: before allocating a fresh KV-cache slot,
        // probe the prompt-prefix cache for a reusable detached set. On a
        // hit, adopt under a brand-new SequenceId and record how many
        // leading tokens the prefill can skip. On a miss (which includes
        // feature-disabled, no ctx, and race paths), fall through to the
        // cold-allocation path below.
        //
        // VLM / audio requests opt out of the cache path entirely: their
        // pre-injection token stream is not self-describing (image token
        // placeholders expand later inside `prepare_request_vlm_embeddings`),
        // so matching against it risks reusing a KV slice built for a
        // different media payload. Support for image-aware cache keys is
        // tracked separately in issue #425.
        let is_multimodal = !images.is_empty() || !audio.is_empty();
        let ctx_ref = if is_multimodal {
            None
        } else {
            options.prompt_cache_ctx.as_ref()
        };
        let (seq_id, prefill_start_offset, already_cached_tokens) =
            match ctx_ref.and_then(|ctx| self.try_adopt_cached_prefix(ctx, &prompt_tokens)) {
                Some((adopted_id, matched_len)) => (adopted_id, matched_len, matched_len),
                None => {
                    // Miss or feature disabled → regular allocate.
                    let seq_id = match self.allocate_sequence_state() {
                        Ok(id) => id,
                        Err(err) => {
                            tracing::warn!("Cache pool allocation failed: {err}");
                            let _ = response_tx
                                .send(GenerateEvent::Error(format!("Server busy: {err}")));
                            return;
                        }
                    };
                    (seq_id, 0, 0)
                }
            };

        let vlm_embeddings = match prepare_request_vlm_embeddings(
            &self.model,
            &self.tokenizer,
            &prompt,
            &mut prompt_tokens,
            &images,
            &audio,
            Some(self.vision_caches.as_ref()),
        ) {
            Ok(emb) => emb,
            Err(err) => {
                // Clean up the context map so a donate-back won't fire for
                // a sequence that never reached a healthy finish.
                self.prompt_cache_seq_ctx.remove(&seq_id);
                self.release_sequence_caches(seq_id);
                let _ = response_tx.send(GenerateEvent::Error(err.to_string()));
                return;
            }
        };

        let decode_state = StreamingDecodeState::new(&self.tokenizer, &prompt_tokens);

        // Issue #409: resolve the effective thinking-token budget for this
        // sequence from the per-request override + server default. The route
        // layer supplies `thinking_enter_block_on_start` as `true` when the
        // rendered prompt primes `<think>` (chat endpoints) and `false` for
        // raw text endpoints.
        let thinking = self.build_thinking_state(
            options.reasoning_budget,
            options.thinking_enter_block_on_start,
        );

        // Record the per-request prompt-cache context so the donate-back
        // path can compose the insert key without reaching back into the
        // HTTP layer. Only stored when the feature is active and the
        // request actually carried a context — otherwise the map stays
        // empty and the donate-back short-circuits. Multimodal requests
        // opt out of the cache altogether (see above).
        if self.prompt_cache_active()
            && !is_multimodal
            && let Some(ctx) = options.prompt_cache_ctx.clone()
        {
            self.prompt_cache_seq_ctx.insert(seq_id, ctx);
        }

        // Guard against a degenerate cache hit where the adopted prefix
        // covers the entire tokenized prompt. This can legitimately happen
        // when a client replays an identical prompt. Back off one token so
        // the prefill path still runs and the sampler sees fresh logits.
        let prefill_start_offset =
            if prefill_start_offset >= prompt_tokens.len() && !prompt_tokens.is_empty() {
                tracing::debug!(
                    seq_id = %seq_id,
                    "prompt-cache hit covered the entire prompt; re-running the \
                     last token through prefill to produce a sampling logit"
                );
                prompt_tokens.len() - 1
            } else {
                prefill_start_offset
            };

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
            audio,
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            decode_state,
            prefill_offset: 0,
            prefill_start_offset,
            already_cached_tokens,
            response_tx,
            cancelled,
            created_at: Instant::now(),
            prefill_start: None,
            first_token_time: None,
            token_history: Vec::new(),
            merged_eos: Vec::new(),
            thinking,
        };

        if let Err(rejected) = self.prefill_queue.enqueue(seq) {
            self.prompt_cache_seq_ctx.remove(&rejected.seq_id);
            self.release_sequence_caches(rejected.seq_id);
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

        if let Err(err) = Self::begin_prefill(&mut seq) {
            tracing::error!("State transition error: {err}");
            self.abort_sequence(seq, &err);
            return;
        }

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

        // Most models only implement batched decode (`[B, 1]`) and do not
        // support full-sequence prompt prefill via `forward_batched()`.
        // Keep those on the single-sequence prefill path so correctness and
        // the standard NAX-friendly prefill route are preserved.
        if !self.model.supports_batched_prefill() {
            tracing::debug!(
                "batched prefill: falling back to sequential (model lacks full batched prefill)"
            );
            for mut seq in seqs {
                if let Err(err) = Self::begin_prefill(&mut seq) {
                    tracing::error!("Batched prefill state transition error: {err}");
                    self.abort_sequence(seq, &err);
                    continue;
                }
                self.execute_full_prefill(seq);
            }
            return;
        }

        // Any VLM request cannot currently be batched (embeddings are
        // per-sequence and would need separate handling). Fall back for the
        // whole batch when any request carries VLM embeddings.
        if seqs.iter().any(|s| s.vlm_embeddings.is_some()) {
            tracing::debug!("batched prefill: falling back to sequential (VLM request in batch)");
            for mut seq in seqs {
                if let Err(err) = Self::begin_prefill(&mut seq) {
                    tracing::error!("Batched prefill state transition error: {err}");
                    self.abort_sequence(seq, &err);
                    continue;
                }
                self.execute_full_prefill(seq);
            }
            return;
        }

        // Epic #416 / issue #421: any sequence that adopted a cached prefix
        // cannot participate in the padded batched prefill path because the
        // KV-history offsets differ across sequences. Take the single-
        // sequence path for those so their `prefill_start_offset` is
        // honored correctly; batched-prefill continues for the rest of
        // this batch in the normal padded pipeline below.
        if seqs.iter().any(|s| s.prefill_start_offset > 0) {
            tracing::debug!(
                "batched prefill: falling back to sequential (adopted prompt-cache prefix in batch)"
            );
            for mut seq in seqs {
                if let Err(err) = Self::begin_prefill(&mut seq) {
                    tracing::error!("Batched prefill state transition error: {err}");
                    self.abort_sequence(seq, &err);
                    continue;
                }
                self.execute_full_prefill(seq);
            }
            return;
        }

        let b = seqs.len();
        let max_len = seqs.iter().map(|s| s.prompt_tokens.len()).max().unwrap();
        let can_pad_prefill = self.model.supports_padded_prefill();
        if !can_pad_prefill && seqs.iter().any(|s| s.prompt_tokens.len() != max_len) {
            tracing::debug!(
                "batched prefill: falling back to sequential (model requires equal prompt lengths)"
            );
            for seq in seqs {
                self.execute_full_prefill(seq);
            }
            return;
        }

        let padded_len = if can_pad_prefill && should_align_prefill() {
            align_to_na_tile(max_len)
        } else {
            max_len
        };

        tracing::debug!("batched prefill: {} requests, padded to {}", b, padded_len);

        // Transition all sequences to Prefilling.
        for seq in &mut seqs {
            if let Err(err) = Self::begin_prefill(seq) {
                tracing::error!("Batched prefill state transition error: {err}");
            }
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
        // Each mask has shape [padded_len, padded_len]. Stacking on axis 0
        // produces [B, padded_len, padded_len], which model batched-prefill
        // paths slice per sequence into [padded_len, padded_len].
        let stacked_mask = if seqs.iter().any(|s| s.prompt_tokens.len() != padded_len) {
            let mut batch_masks: Vec<UniquePtr<mlxcel_core::MlxArray>> = Vec::with_capacity(b);
            for seq in &seqs {
                let actual = seq.prompt_tokens.len() as i32;
                let padded = padded_len as i32;
                let mask = create_padded_prefill_mask(actual, padded, 0);
                batch_masks.push(mask);
            }
            Some(mlxcel_core::stack_owned(&batch_masks, 0))
        } else {
            None
        };

        let batch_ids: Vec<SequenceId> = seqs.iter().map(|seq| seq.seq_id).collect();
        let mut batch_caches = match self.cache_pool.get_batch_caches_mut(&batch_ids) {
            Ok(caches) => caches,
            Err(err) => {
                tracing::warn!("batched prefill: {err}, falling back");
                // Re-queue all sequences for sequential processing.
                for seq in seqs {
                    self.execute_full_prefill(seq);
                }
                return;
            }
        };

        if batch_caches.len() != b {
            // Re-queue all sequences for sequential processing.
            for seq in seqs {
                self.execute_full_prefill(seq);
            }
            return;
        }

        // Single batched forward pass: [B, padded_len] → [B, padded_len, vocab]
        let raw_logits = self.model.forward_batched_with_context_and_ids(
            &input,
            Some(&batch_ids),
            &mut batch_caches,
            stacked_mask.as_deref(),
            None,
        );

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

            self.sync_sequence_storage(seq.seq_id);

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
    ///
    /// Epic #416 / issue #421: when `seq.prefill_start_offset > 0`, a
    /// prompt-cache hit has installed the first `prefill_start_offset` tokens
    /// of KV state on this sequence. Only the suffix tokens are fed to the
    /// model. The VLM-prefix path deliberately opts out of cache adoption at
    /// the enqueue site, so this branch never has to mix the two.
    fn execute_full_prefill(&mut self, mut seq: SequenceInfo) {
        let _span = tracing::info_span!(
            "prefill",
            seq_id = %seq.seq_id,
            prompt_len = seq.prompt_tokens.len(),
            cached = seq.prefill_start_offset,
        )
        .entered();
        // Only the suffix enters the prefill counters — the first
        // `prefill_start_offset` tokens were resolved from the adopted
        // detached cache with zero model work.
        let suffix_len = seq.prompt_tokens.len() - seq.prefill_start_offset;
        self.batch_observability.record_prefill_start(suffix_len);

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

        // Feed only the suffix tokens to the model when a cached prefix was
        // adopted. For cold prefills `start == 0` and this is identical to
        // the legacy behavior.
        let suffix_tokens: Vec<i32> = seq.prompt_tokens[seq.prefill_start_offset..].to_vec();

        // Run prefill (with or without VLM embeddings).
        // On M5+ hardware pad the prompt to a 32-token tile boundary for
        // optimal Neural Accelerator throughput.
        let actual_len = suffix_tokens.len();
        let (effective_tokens, pad_mask_opt) = if should_align_prefill() {
            let padded_len = align_to_na_tile(actual_len);
            if padded_len > actual_len {
                let mut padded = suffix_tokens.clone();
                padded.resize(padded_len, 0);
                // The padding mask anchors to the adopted cache offset so
                // the newly-prefilled positions see the correct KV-history
                // positions on M5+ hardware.
                let mask = create_padded_prefill_mask(
                    actual_len as i32,
                    padded_len as i32,
                    seq.prefill_start_offset as i32,
                );
                (padded, Some(mask))
            } else {
                (suffix_tokens.clone(), None)
            }
        } else {
            (suffix_tokens.clone(), None)
        };

        let eff_len = effective_tokens.len() as i32;
        let input = mlxcel_core::from_slice_i32(&effective_tokens, &[1, eff_len]);
        let logits = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found for sequence during prefill");
                    return;
                }
            };

            let raw_logits = if let Some(ref embeddings) = seq.vlm_embeddings {
                // VLM path: apply provided mask or the tile-alignment mask.
                match prepared_embedding_refs(embeddings) {
                    Ok((input_embeds, caller_mask)) => {
                        // Caller-supplied mask takes precedence; tile-alignment mask
                        // is used only when the caller does not provide one.
                        let effective_mask =
                            caller_mask.or(pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()));
                        let logits = self.model.forward_with_embeddings_and_sequence_id(
                            &input,
                            Some(input_embeds),
                            Some(seq.seq_id),
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
                self.model.forward_with_sequence_id(
                    &input,
                    Some(seq.seq_id),
                    caches,
                    pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
                )
            };

            // Extract logits at the last real token position and trim padding from
            // KV caches so the decode phase begins with the correct cache offset.
            if pad_mask_opt.is_some() && effective_tokens.len() > actual_len {
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
            }
        };
        self.sync_sequence_storage(seq.seq_id);

        mlxcel_core::clear_memory_cache();
        // `prefill_offset` is a cursor into `prompt_tokens`, so it must
        // include the adopted prefix even though those tokens bypassed the
        // forward pass.
        seq.prefill_offset = seq.prefill_start_offset + actual_len;

        self.finish_prefill(seq, logits, eos_tokens, token_history, needs_history);
    }

    /// Begin a chunked prefill: process the first chunk and store the
    /// sequence for continuation on subsequent ticks.
    ///
    /// Epic #416 / issue #421: `seq.prefill_start_offset` skips over the
    /// leading tokens that the adopted prompt-cache entry already covers,
    /// so the first chunk starts *after* the cached prefix.
    fn start_chunked_prefill(&mut self, mut seq: SequenceInfo) {
        let _span = tracing::info_span!(
            "chunked_prefill_start",
            seq_id = %seq.seq_id,
            prompt_len = seq.prompt_tokens.len(),
            chunk_size = self.prefill_chunk_size,
            cached = seq.prefill_start_offset,
        )
        .entered();
        // Counter reflects only the work the model actually runs.
        let suffix_len = seq.prompt_tokens.len() - seq.prefill_start_offset;
        self.batch_observability.record_prefill_start(suffix_len);

        // Reset internal caches for non-batching models (same as execute_full_prefill).
        if !self.model.supports_batching() {
            let _ = self.model.make_caches();
        }

        let chunk_size = self.prefill_chunk_size;
        let start = seq.prefill_start_offset;
        let end = (start + chunk_size).min(seq.prompt_tokens.len());
        let chunk = &seq.prompt_tokens[start..end];

        // Align the first chunk to a 32-token tile boundary on M5+ hardware.
        let actual_chunk_len = chunk.len();
        let (eff_chunk, pad_mask_opt) = if should_align_prefill() {
            let padded_len = align_to_na_tile(actual_chunk_len);
            if padded_len > actual_chunk_len {
                let mut padded = chunk.to_vec();
                padded.resize(padded_len, 0);
                // Mask anchored to the KV offset the adopted prefix already
                // installed (starts at zero for cold prefills).
                let mask = create_padded_prefill_mask(
                    actual_chunk_len as i32,
                    padded_len as i32,
                    start as i32,
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
        {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found for sequence during chunked prefill");
                    return;
                }
            };

            // VLM embeddings are applied only on the first chunk.
            if let Some(ref embeddings) = seq.vlm_embeddings {
                match prepared_embedding_refs(embeddings) {
                    Ok((input_embeds, caller_mask)) => {
                        let effective_mask =
                            caller_mask.or(pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()));
                        let logits = self.model.forward_with_embeddings_and_sequence_id(
                            &input,
                            Some(input_embeds),
                            Some(seq.seq_id),
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
                let logits = self.model.forward_with_sequence_id(
                    &input,
                    Some(seq.seq_id),
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
        }
        self.sync_sequence_storage(seq.seq_id);

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

        // Align each continuation chunk to a 32-token tile boundary on M5+.
        let actual_chunk_len = chunk.len();
        // For non-batching models the scheduler's dummy caches always have
        // offset=0.  Use the prefill_offset (number of tokens already
        // processed) as the KV offset instead, which is accurate regardless
        // of whether the model uses internal or scheduler-managed caches.
        let kv_offset = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found during chunked prefill continuation");
                    return;
                }
            };
            if self.model.supports_batching() {
                caches.first().map_or(0, |c| c.offset)
            } else {
                offset as i32
            }
        };
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
        let logits = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found during chunked prefill continuation");
                    return;
                }
            };

            let logits = self.model.forward_with_sequence_id(
                &input,
                Some(seq.seq_id),
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
            logits
        };
        self.sync_sequence_storage(seq.seq_id);

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
        let sampled_first_token = mlxcel_core::item_i32(&first_token_arr);

        // Issue #409: thinking-budget override. Qwen3 chat templates prime
        // `<think>\n`, so the first prefill-completion token is already
        // inside the reasoning block when `enter_block_on_start == true`.
        let first_token = Self::apply_thinking_budget(&mut seq.thinking, sampled_first_token);

        seq.first_token_time = Some(Instant::now());

        // Issue #409: if the budget fired and substituted the first token,
        // drop the logprob below (computed against the sampled token) so the
        // streamed metadata stays consistent with the emitted token text.
        let override_fired = first_token != sampled_first_token;

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
            // Prefill produced a valid KV cache (EOS on turn 1 is a healthy
            // stop). Donate it back so the next turn can reuse the prompt
            // prefix. `generated_tokens` is empty here by construction.
            self.donate_finished_sequence_cache(seq.seq_id, &seq.prompt_tokens, &[], true);
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            return;
        }

        // Optionally compute logprobs for the first token. When the override
        // fired, the sampled token differs from the emitted `first_token`;
        // suppress logprob emission in that case to keep token text and
        // logprob metadata consistent (issue #409).
        let token_lp = if override_fired {
            None
        } else {
            compute_logprobs(&adjusted_logits, first_token, &seq.logprobs_config)
        };

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
            self.donate_finished_sequence_cache(
                seq.seq_id,
                &seq.prompt_tokens,
                &seq.generated_tokens,
                true,
            );
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
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
            self.release_sequence_caches(victim.seq_id);

            // Reset the sequence for re-prefill: clear generated tokens,
            // reset decode state, and re-allocate a cache slot.
            //
            // Preemption discards the adopted prefix cache as well — the
            // victim must re-prefill from scratch to stay consistent with
            // the fresh `allocate_sequence_state` that follows.
            victim.generated_tokens.clear();
            victim.generated_text.clear();
            victim.prefill_offset = 0;
            victim.prefill_start_offset = 0;
            victim.already_cached_tokens = 0;
            victim.decode_state = StreamingDecodeState::new(&self.tokenizer, &victim.prompt_tokens);
            victim.token_history.clear();
            victim.merged_eos.clear();

            // Allocate a fresh cache slot
            match self.allocate_sequence_state() {
                Ok(new_id) => {
                    victim.seq_id = new_id;
                    if let Err(err) = victim.state.transition_to(SequenceState::Queued) {
                        tracing::error!("Eviction state transition error: {err}");
                        self.release_sequence_caches(new_id);
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
                self.release_sequence_caches(rejected.seq_id);
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
        // Filter-to-empty guard: a zero-sized decode step is a no-op, not a
        // failure. The observability counter already reflects length 0 for
        // caller-side traceability, so we still record it, then skip the
        // dispatch entirely. This matches the null-guard pattern upstream
        // `mlx-lm` added to `BatchKVCache.filter` when the filtered index
        // list is empty.
        if seq_ids.is_empty() {
            self.batch_observability.record_decode_step(0);
            return;
        }

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
    ///
    /// # Null/empty-cache safety
    ///
    /// Early-exits on `seq_ids.is_empty()`. Though the scheduler's current
    /// [`Self::decide_action`] never produces a `Decode(ids)` action with an
    /// empty list (it returns [`BatchSchedulerAction::Idle`] first), this
    /// guard makes the method robust against future policy changes and any
    /// direct caller. Dispatching a zero-batch forward pass would otherwise
    /// materialize an empty `[0, 1]` input tensor and invoke the model
    /// kernel with no work to do, which is both wasteful and potentially
    /// undefined behavior in downstream MLX kernels.
    ///
    /// This mirrors the upstream `mlx-lm` `BatchKVCache.filter` / `extend`
    /// null-guards that prevent cache operations from crashing when all
    /// sequences have been filtered out of the batch.
    fn execute_batched_decode(&mut self, seq_ids: &[SequenceId]) {
        if seq_ids.is_empty() {
            // Filter-to-empty case: nothing to do. Bookkeeping is handled by
            // the caller (`execute_decode_step`) via its own length guard.
            return;
        }

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

        let mut batch_caches = match self.cache_pool.get_batch_caches_mut(seq_ids) {
            Ok(caches) => caches,
            Err(err) => {
                tracing::error!("{err} during batched decode");
                return;
            }
        };

        let decode_context = match self.decode_storage_backend {
            DecodeStorageBackend::Auto | DecodeStorageBackend::Dense => {
                debug_assert_ne!(
                    self.decode_storage_backend,
                    DecodeStorageBackend::Auto,
                    "scheduler should normalize decode storage backend before decode dispatch"
                );
                DecodeBatchContext::dense()
            }
            DecodeStorageBackend::Paged => DecodeBatchContext {
                storage_backend: CoreDecodeStorageBackend::Paged,
                paged_block_size: DEFAULT_PAGED_BLOCK_SIZE as i32,
                use_native_paged_kernel: true,
            },
        };
        let logits = self.model.forward_batched_with_context_and_ids(
            &input,
            Some(seq_ids),
            &mut batch_caches,
            None,
            Some(&decode_context),
        );
        drop(batch_caches);

        for &seq_id in seq_ids {
            self.sync_sequence_storage(seq_id);
        }

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
                let sampled = mlxcel_core::item_i32(&token_arr);
                // Issue #409: apply the thinking-budget override first so that
                // when the override fires (sampled != final_id) we can skip
                // the log-softmax work entirely. The logprob metadata would
                // be dropped anyway because the emitted `</think>` differs
                // from the token the logits describe, so computing it first
                // is wasted GPU work on the decode hot path.
                let final_id = Self::apply_thinking_budget(&mut seq.thinking, sampled);
                let lp = if final_id == sampled {
                    compute_logprobs(&adjusted_logits, sampled, &seq.logprobs_config)
                } else {
                    // Override fired; token text and logprob metadata must
                    // stay consistent, so drop the logprob for this step.
                    None
                };
                (final_id, lp)
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

        let input = mlxcel_core::from_slice_i32(&[last_token], &[1, 1]);
        let logits = {
            let caches = match self.cache_pool.get_caches_mut(seq_id) {
                Some(c) => c,
                None => {
                    tracing::error!("Cache not found for {seq_id} during decode");
                    return;
                }
            };
            self.model
                .forward_with_sequence_id(&input, Some(seq_id), caches, None)
        };
        self.sync_sequence_storage(seq_id);

        // Use cached token_history from SequenceInfo (incrementally maintained)
        // and cached merged_eos (computed once during prefill) to avoid
        // per-step allocation and reconstruction overhead.
        let (token_val, token_lp) = {
            let seq = self.active_batch.get_mut(seq_id).unwrap();
            let (token_arr, adjusted_logits) =
                sample_token_optimized(&logits, &seq.sampling, &seq.token_history);
            mlxcel_core::eval(&token_arr);
            let sampled = mlxcel_core::item_i32(&token_arr);
            // Issue #409: apply the thinking-budget override first so that
            // when the override fires the log-softmax work is skipped — the
            // logprob metadata for the sampled token would be dropped anyway
            // (token text and logprob `token_id` must stay consistent), so
            // computing it up-front wastes GPU time on every override step.
            let final_id = Self::apply_thinking_budget(&mut seq.thinking, sampled);
            let lp = if final_id == sampled {
                compute_logprobs(&adjusted_logits, sampled, &seq.logprobs_config)
            } else {
                None
            };
            (final_id, lp)
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
        // First, transition any cancelled sequences to Finished(Cancelled).
        // This must happen before the finished-ID scan so that newly cancelled
        // sequences are collected in the same pass.
        let cancelled_ids: Vec<SequenceId> = self
            .active_batch
            .iter_sequences()
            .filter(|s| !s.state.is_finished() && s.cancelled.load(Ordering::Relaxed))
            .map(|s| s.seq_id)
            .collect();

        for id in &cancelled_ids {
            if let Some(seq) = self.active_batch.get_mut(*id) {
                if let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Cancelled))
                {
                    tracing::warn!("Failed to cancel sequence {id}: {err}");
                } else {
                    tracing::info!("Sequence {id} cancelled (client disconnected)");
                }
            }
        }

        // Cancel a chunked-prefill-in-progress sequence if client disconnected.
        if let Some(ref seq) = self.chunked_prefill_seq
            && seq.cancelled.load(Ordering::Relaxed)
        {
            let seq = self.chunked_prefill_seq.take().unwrap();
            tracing::info!(
                "Chunked-prefill sequence {} cancelled (client disconnected)",
                seq.seq_id
            );
            let _ = seq.response_tx.send(GenerateEvent::Error(
                "Request cancelled: client disconnected".to_string(),
            ));
            // Cancellation during prefill means the KV cache is only
            // partially populated; skip donate-back and just release. The
            // context map still needs cleanup so no dangling entries leak.
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            self.batch_observability.record_sequence_completed();
        }

        // Also cancel queued sequences whose client has already disconnected,
        // so they never enter the active batch.
        self.cancel_queued_disconnected();

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

                // Epic #416 / issue #421: donate the full KV cache back to
                // the prompt-cache store on *healthy* finishes (Stop /
                // Length / Cancelled) so the next turn of the same
                // conversation can adopt it. `Finished(Error)` paths bypass
                // this branch — their cache is assumed tainted.
                let healthy = matches!(
                    seq.state,
                    SequenceState::Finished(
                        FinishReason::Stop | FinishReason::Length | FinishReason::Cancelled,
                    )
                );
                self.donate_finished_sequence_cache(
                    id,
                    &seq.prompt_tokens,
                    &seq.generated_tokens,
                    healthy,
                );
                // `donate_finished_sequence_cache` already removed the
                // context from `prompt_cache_seq_ctx` on donate; drop it
                // defensively on the non-donate paths so the map cannot
                // grow unbounded across long-lived workers.
                self.prompt_cache_seq_ctx.remove(&id);

                self.release_sequence_caches(id);
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

    /// Remove queued sequences whose client has already disconnected.
    ///
    /// This prevents cancelled requests from ever entering the active batch,
    /// freeing the prefill queue slot immediately.
    fn cancel_queued_disconnected(&mut self) {
        let drained: Vec<SequenceInfo> = self.prefill_queue.drain_cancelled();
        for seq in drained {
            tracing::info!(
                "Queued sequence {} cancelled before prefill (client disconnected)",
                seq.seq_id
            );
            let _ = seq.response_tx.send(GenerateEvent::Error(
                "Request cancelled: client disconnected".to_string(),
            ));
            // No prefill ran → no valid cache to donate. Clear the
            // context entry so it cannot linger.
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            self.batch_observability.record_sequence_completed();
        }
    }

    fn abort_sequence(&mut self, seq: SequenceInfo, error: &str) {
        let _ = seq
            .response_tx
            .send(GenerateEvent::Error(error.to_string()));
        // Abort paths produce an error outcome (OOM / transition failure /
        // invalid cache); the KV cache is untrustworthy and must not be
        // donated back. Dropping the context entry prevents a future
        // finalize pass from trying.
        self.prompt_cache_seq_ctx.remove(&seq.seq_id);
        self.release_sequence_caches(seq.seq_id);
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;

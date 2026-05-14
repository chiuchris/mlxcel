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

//! MTP target adapter for Gemma 4 (issue #666).
//!
//! Glue layer that wires the binary-side
//! [`crate::models::gemma4::Gemma4Wrapper`] (and its VLM variant
//! [`crate::vision::Gemma4VLModel`]) to the
//! [`mlxcel_core::speculative::mtp::target::MtpTarget`] trait defined in
//! `mlxcel-core` (issue #662). The trait sits between the round-loop
//! driver [`mlxcel_core::speculative::mtp::MtpGenerator`] and the concrete
//! Gemma 4 wrapper so the driver can stay in the core crate without
//! pulling in `mlxcel`-binary types.
//!
//! ## Why a separate adapter struct?
//!
//! The trait methods need to address a **specific per-sequence cache
//! slot** on the wrapper (the scheduler allocates a [`SequenceId`] per
//! request and routes through
//! [`Gemma4Wrapper::sequence_state`](crate::models::gemma4::Gemma4Wrapper)).
//! The trait itself takes `&self` only — it has no `seq_id` parameter —
//! so we pair the wrapper reference with the sequence id in an adapter
//! struct ([`Gemma4MtpTargetAdapter`]). The round-loop driver holds the
//! adapter by value and the adapter delegates each call to the matching
//! `forward_with_speculative_sinks` /
//! `rollback_speculative_cache` hook with the captured `seq_id`.
//!
//! ## Scope (B = 1 today)
//!
//! This adapter implements the B = 1 trait methods (`prefill_and_seed`,
//! `verify_forward`, `verify_finalize`, `embed_token`, `num_layers`,
//! `eos_token_ids`). The B > 1 batched methods (`prefill_and_seed_batched`,
//! `verify_forward_batched`, `verify_finalize_batched`) fall back to the
//! trait's default `DrafterError::DraftFailed` implementation, which the
//! batched round-loop driver surfaces as a clear "not yet wired" error.
//! Batched MTP integration with the continuous-batching scheduler lands
//! in a peer follow-up — see the worker-startup warning in
//! `src/server/model_worker.rs`.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::sampling::sample_token_optimized;
use mlxcel_core::speculative::mtp::target::{
    MtpTarget, MtpVerifyOutput, VerifyCaptured, VerifyForwardOutput,
};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::gemma4::{Gemma4SpeculativeSinks, Gemma4Wrapper};

/// MTP target adapter binding a [`Gemma4Wrapper`] to a specific
/// per-sequence cache slot.
///
/// Constructed by the server-scheduler dispatch hook at the start of a
/// speculative request (after prefill) and consumed by
/// [`mlxcel_core::speculative::mtp::MtpGenerator::generate`] over the
/// course of the round loop. The adapter does NOT own the wrapper — it
/// only borrows it for the lifetime of the round-loop driver, which is
/// strictly within a single decode tick on the worker thread.
pub struct Gemma4MtpTargetAdapter<'a> {
    /// Borrowed target wrapper. The wrapper owns the per-sequence cache
    /// slot identified by `seq_id`; this adapter resolves into it on every
    /// trait call without copying state.
    wrapper: &'a Gemma4Wrapper,
    /// Per-sequence cache slot identifier. `None` selects the wrapper's
    /// internal fallback slot (used by the CLI / single-row tests). The
    /// server scheduler always passes `Some(seq_id)`.
    seq_id: Option<SequenceId>,
}

impl<'a> Gemma4MtpTargetAdapter<'a> {
    /// Construct an adapter that routes every trait call through the
    /// per-sequence cache slot at `seq_id`.
    ///
    /// The scheduler calls this once per speculative request and hands
    /// the adapter to
    /// [`mlxcel_core::speculative::mtp::MtpGenerator::new`].
    pub fn new(wrapper: &'a Gemma4Wrapper, seq_id: Option<SequenceId>) -> Self {
        Self { wrapper, seq_id }
    }

    /// Slice the captured shared K/V tensors along the seq-len axis by
    /// `rejected = block_size - accepted - 1` so the post-rollback K/V
    /// matches the trimmed cache.
    ///
    /// Mirrors the upstream Python sequence:
    ///
    /// ```python
    /// rejected = block_size - (accepted + 1)
    /// for k, kv in verify_out.shared_kv_states.items():
    ///     K, V = kv
    ///     next_shared_kv[k] = (K[..., :K.shape[-2] - rejected, :], ...)
    /// ```
    ///
    /// `tensors` is the slab vector captured by [`Self::verify_forward`]
    /// (`[k_full, v_full, k_swa, v_swa]` ordering). `rejected == 0`
    /// short-circuits and returns the unchanged tensors so the
    /// full-accept case pays zero overhead.
    pub(crate) fn slice_shared_kv(
        tensors: Vec<UniquePtr<MlxArray>>,
        rejected: usize,
    ) -> Vec<UniquePtr<MlxArray>> {
        if rejected == 0 {
            return tensors;
        }
        tensors
            .into_iter()
            .map(|ptr| {
                let array = ptr.as_ref().expect("shared K/V slab must be non-null");
                let shape = mlxcel_core::array_shape(array);
                // Shape is `[B, num_kv_heads, kv_len, head_dim]`. We
                // crop along axis 2 (kv_len) by `rejected` tokens.
                debug_assert!(
                    shape.len() == 4,
                    "shared K/V slab must be 4-D, got shape {:?}",
                    shape
                );
                let kv_len = shape[2];
                let new_kv_len = kv_len - rejected as i32;
                debug_assert!(
                    new_kv_len >= 0,
                    "slice_shared_kv: rejected ({rejected}) exceeds kv_len ({kv_len})",
                );
                // mlxcel_core::slice takes &[start_0, start_1, ...] and
                // &[stop_0, stop_1, ...] for every axis.
                let starts: Vec<i32> = vec![0, 0, 0, 0];
                let stops: Vec<i32> = vec![shape[0], shape[1], new_kv_len, shape[3]];
                mlxcel_core::slice(array, &starts, &stops)
            })
            .collect()
    }

    /// Compute the argmax along the last axis for each row of a
    /// `[1, block_size, vocab]` logits tensor and return the
    /// per-position token ids.
    ///
    /// At temperature > 0 the caller is expected to override this with
    /// a real sampler; this helper handles only the greedy-parity path
    /// (`temperature == 0`). The MTP greedy-parity invariant
    /// (referenced in `references/mlx-vlm`) requires the
    /// target tokens to match the target's own argmax extension, so for
    /// `temperature == 0` this is the load-bearing choice.
    ///
    /// Mirrors `argmax_logits_to_vec` in
    /// [`mlxcel_core::drafter::dflash::round_loop`] — duplicating the
    /// helper rather than re-exporting because it is a single-call
    /// utility and the DFlash module's version is private. Future
    /// refactor: lift this into `mlxcel_core::utils` once a second
    /// adapter (Qwen 3.5 MTP variant) lands and needs the same shape.
    pub(crate) fn argmax_per_position(logits: &MlxArray) -> Vec<i32> {
        let shape = mlxcel_core::array_shape(logits);
        debug_assert!(shape.len() == 3, "expected [1, block_size, vocab] logits");
        let block_size = shape[1];

        // `argmax_last_axis` reduces over the trailing axis, producing
        // `[1, block_size]`.
        let argmax = mlxcel_core::argmax_last_axis(logits);
        mlxcel_core::eval(&argmax);

        // Materialize per-position scalars. The buffer is now tiny
        // (block_size i32 cells), so the per-cell extraction is cheap.
        let mut out: Vec<i32> = Vec::with_capacity(block_size as usize);
        for s in 0..block_size {
            let cell = mlxcel_core::slice(&argmax, &[0, s], &[1, s + 1]);
            let scalar = mlxcel_core::reshape(&cell, &[]);
            out.push(mlxcel_core::item_i32(&scalar));
        }
        out
    }

    /// Compute per-position [`mlxcel_core::sampling::TokenLogprobData`]
    /// for a `[1, block_size, vocab]` verify-logits tensor, one entry
    /// per position aligned 1:1 with `target_tokens` (issue #678).
    ///
    /// The verify forward is greedy-only today (`temperature == 0`,
    /// pure argmax — see [`Self::argmax_per_position`]), so the
    /// per-position logits ARE the penalty-adjusted logits the classic
    /// decode path would feed `compute_logprobs`: no temperature
    /// scaling, no history-dependent penalty applied during verify.
    /// That makes the resulting logprobs byte-identical to the classic
    /// path's decode-token logprobs for greedy sampling.
    ///
    /// Returns `None` when `logprobs_config.enabled` is false (the
    /// zero-overhead path — no slicing, no log-softmax).
    fn per_position_logprobs(
        logits: &MlxArray,
        target_tokens: &[i32],
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> Option<Vec<mlxcel_core::sampling::TokenLogprobData>> {
        if !logprobs_config.enabled {
            return None;
        }
        let shape = mlxcel_core::array_shape(logits);
        debug_assert!(shape.len() == 3, "expected [1, block_size, vocab] logits");
        let vocab = shape[2];
        let mut out: Vec<mlxcel_core::sampling::TokenLogprobData> =
            Vec::with_capacity(target_tokens.len());
        for (pos, &tok) in target_tokens.iter().enumerate() {
            // Slice position `pos` to a `[1, vocab]` tensor — the shape
            // `compute_logprobs` expects.
            let pos_i32 = pos as i32;
            let pos_logits_3d =
                mlxcel_core::slice(logits, &[0, pos_i32, 0], &[1, pos_i32 + 1, vocab]);
            let pos_logits = mlxcel_core::reshape(&pos_logits_3d, &[1, vocab]);
            // `compute_logprobs` returns `Some` here because
            // `logprobs_config.enabled` is true (checked above); the
            // `unwrap_or` is a defensive fallback that should never
            // fire.
            let lp = mlxcel_core::sampling::compute_logprobs(&pos_logits, tok, logprobs_config)
                .unwrap_or(mlxcel_core::sampling::TokenLogprobData {
                    token_id: tok,
                    logprob: 0.0,
                    top_alternatives: Vec::new(),
                });
            out.push(lp);
        }
        Some(out)
    }
}

impl<'a> MtpTarget for Gemma4MtpTargetAdapter<'a> {
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> (
        i32,
        MtpVerifyOutput,
        Option<mlxcel_core::sampling::TokenLogprobData>,
    ) {
        let prompt_arr =
            mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);

        // Capture last-layer hidden + last full/SWA shared K/V slabs.
        // Gemma 4 owns its own caches via `ModelOwnedSequenceState`;
        // the wrapper resolves `seq_id` to the matching slot internally
        // and the trait method does not take an external cache slice.
        let mut sinks = Gemma4SpeculativeSinks::with_hidden_and_shared_kv();
        let logits = self.wrapper.forward_with_speculative_sinks(
            &prompt_arr,
            None,
            None,
            None,
            self.seq_id,
            None,
            Some(&mut sinks),
        );

        // Sample the first bonus from the last-position logits.
        // `token_history` carries the history-dependent-penalty context
        // (repetition / frequency / presence / DRY) so the first bonus
        // is byte-identical to the classic decode path's first token
        // (issue #677). `sample_token_optimized` returns
        // `(token_arr, adjusted_logits)`; `adjusted_logits` is the
        // penalty-adjusted `[1, vocab]` slice the bonus was sampled
        // from, and feeds `compute_logprobs` so the first-bonus logprob
        // is byte-identical to the classic path's first-token logprob
        // (issue #678).
        let (token_arr, adjusted_logits) =
            sample_token_optimized(&logits, sampler, token_history);
        mlxcel_core::eval(&token_arr);
        let first_bonus = mlxcel_core::item_i32(&token_arr);
        let first_bonus_lp =
            mlxcel_core::sampling::compute_logprobs(&adjusted_logits, first_bonus, logprobs_config);

        // Build the seed MtpVerifyOutput from the captured sinks.
        // `next_hidden` is the last entry in the hidden-sink (since we
        // didn't pass `capture_layer_ids`, the sink has exactly one
        // entry: the last decoder layer's pre-norm hidden state).
        let hidden_sink = sinks
            .hidden_sink
            .expect("with_hidden_and_shared_kv installs the hidden sink");
        let next_hidden = hidden_sink
            .into_iter()
            .next_back()
            .expect("hidden sink must carry at least one entry after a forward pass");

        // Materialize the shared K/V vector in the canonical
        // `[k_full, v_full, k_swa, v_swa]` order. The wrapper's
        // `shared_kv_sink` is a HashMap keyed by attention kind name.
        let shared_kv_map = sinks
            .shared_kv_sink
            .expect("with_hidden_and_shared_kv installs the shared K/V sink");
        let mut next_shared_kv: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(4);
        if let Some((k_full, v_full)) = shared_kv_map.get("full_attention") {
            next_shared_kv.push(mlxcel_core::copy(k_full.as_ref().unwrap()));
            next_shared_kv.push(mlxcel_core::copy(v_full.as_ref().unwrap()));
        }
        if let Some((k_swa, v_swa)) = shared_kv_map.get("sliding_attention") {
            next_shared_kv.push(mlxcel_core::copy(k_swa.as_ref().unwrap()));
            next_shared_kv.push(mlxcel_core::copy(v_swa.as_ref().unwrap()));
        }

        // After prefill the cache holds `prompt_tokens.len()` entries;
        // the bonus token is NOT yet in the cache (it will be the first
        // slot of the first round's verify input). `kv_offset` therefore
        // equals `prompt_tokens.len()`.
        let kv_offset = prompt_tokens.len();
        let bonus_position = kv_offset;

        let seed = MtpVerifyOutput {
            next_hidden,
            next_shared_kv,
            kv_offset,
            bonus_position,
        };
        (first_bonus, seed, first_bonus_lp)
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        // Build a `[1, 1]` input_ids tensor and route through the
        // wrapper's `input_embeddings` accessor (which forwards to the
        // text model's `embed_tokens.forward(...)`).
        let input_ids = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        // `Gemma4Wrapper::input_embeddings` is `pub(crate)`, so we go
        // through the `LanguageModel::embed_tokens` trait method
        // (returns `Option<UniquePtr<MlxArray>>` for the LM contract;
        // Gemma 4 always returns `Some` because it owns its embedding
        // table).
        <Gemma4Wrapper as mlxcel_core::generate::LanguageModel>::embed_tokens(
            self.wrapper,
            &input_ids,
        )
        .expect("Gemma4Wrapper exposes its embed_tokens table")
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        _sampler: &SamplingConfig,
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> VerifyForwardOutput {
        // Sink-aware forward over `[bonus, draft_0, …, draft_{K-2}]`.
        let verify_arr = mlxcel_core::from_slice_i32(verify_input, &[1, verify_input.len() as i32]);
        let mut sinks = Gemma4SpeculativeSinks::with_hidden_and_shared_kv();
        let logits = self.wrapper.forward_with_speculative_sinks(
            &verify_arr,
            None,
            None,
            None,
            self.seq_id,
            None,
            Some(&mut sinks),
        );

        // Greedy-parity gate: pull the per-position argmax tokens from
        // the verify logits. At temperature == 0 this is byte-identical
        // to the drafter-less target's own argmax extension.
        //
        // At temperature > 0 a future enhancement plumbs the sampler
        // through per-position; the round-loop driver's perf-sensitive
        // path is greedy, so we keep argmax-only for now.
        let target_tokens = Self::argmax_per_position(&logits);

        // Per-position log-probability data, aligned 1:1 with
        // `target_tokens`. `None` (zero-overhead) when logprobs are
        // disabled; the round loop forwards the entries for accepted
        // positions on to `finalize_burst_success` (issue #678).
        let target_logprobs =
            Self::per_position_logprobs(&logits, &target_tokens, logprobs_config);

        // Capture the hidden + pre-slice shared K/V for the finalize step.
        let hidden_sink = sinks
            .hidden_sink
            .expect("with_hidden_and_shared_kv installs the hidden sink");
        let hidden_full = hidden_sink
            .into_iter()
            .next_back()
            .expect("hidden sink must carry at least one entry");

        let shared_kv_map = sinks
            .shared_kv_sink
            .expect("with_hidden_and_shared_kv installs the shared K/V sink");
        let mut captured_tensors: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(5);
        captured_tensors.push(hidden_full);
        if let Some((k_full, v_full)) = shared_kv_map.get("full_attention") {
            captured_tensors.push(mlxcel_core::copy(k_full.as_ref().unwrap()));
            captured_tensors.push(mlxcel_core::copy(v_full.as_ref().unwrap()));
        }
        if let Some((k_swa, v_swa)) = shared_kv_map.get("sliding_attention") {
            captured_tensors.push(mlxcel_core::copy(k_swa.as_ref().unwrap()));
            captured_tensors.push(mlxcel_core::copy(v_swa.as_ref().unwrap()));
        }

        VerifyForwardOutput {
            target_tokens,
            target_logprobs,
            captured: VerifyCaptured {
                tensors: captured_tensors,
                // No per-step scalar metadata used today; the per-row
                // batched path will populate these with kv_offset_pre /
                // bonus_position_pre.
                scalars: Vec::new(),
            },
        }
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        // Pop the hidden out of the captured tensors. The convention
        // documented on `VerifyCaptured` is index 0 = hidden_full,
        // indices 1.. = the shared K/V slabs in
        // `[k_full, v_full, k_swa, v_swa]` order.
        let mut tensors = captured.tensors;
        assert!(
            !tensors.is_empty(),
            "VerifyCaptured must carry at least the hidden tensor at index 0"
        );
        let next_hidden = tensors.remove(0);

        // Per-row tail-zero rollback. For B = 1 the accept slice is a
        // single-element view; the rotating-cache zeroing inside
        // `rollback_speculative_cache` is a no-op when `accepted.len() == 1`.
        let accepted_i32 = accepted as i32;
        let block_size_i32 = block_size as i32;
        let _ =
            self.wrapper
                .rollback_speculative_cache(self.seq_id, &[accepted_i32], block_size_i32);

        // Slice the captured shared K/V to the post-rollback length.
        let rejected = block_size - accepted - 1;
        let next_shared_kv = Self::slice_shared_kv(tensors, rejected);

        // After rollback, the cache offset advanced from the pre-verify
        // offset by `accepted + 1` (one per accepted token + the bonus
        // position). The round-loop driver tracks the absolute offset
        // separately via the sequence of returned `MtpVerifyOutput`s; the
        // value we return here is only used to RoPE-rotate the drafter's
        // cross-attention queries, so we report the new offset directly.
        //
        // We cannot read the cache's offset back out without exposing a
        // new accessor; the upstream Python carries the value via
        // `prompt_cache[0].offset`. As a load-bearing fix, the round-loop
        // driver itself advances `kv_offset` between calls — this
        // implementation reports `accepted + 1` as the **delta** which
        // the driver adds to the pre-call offset.
        //
        // Issue #666 follow-up: surface a `cache_offset(seq_id) -> usize`
        // accessor on `Gemma4Wrapper` so the kv_offset is read directly
        // rather than reconstructed. Until then the driver layer
        // accounts for the delta — see `MtpGenerator::set_shared_kv_from_verify`.
        let kv_offset = accepted + 1;
        let bonus_position = kv_offset;

        MtpVerifyOutput {
            next_hidden,
            next_shared_kv,
            kv_offset,
            bonus_position,
        }
    }

    fn num_layers(&self) -> usize {
        self.wrapper.num_layers_value()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.wrapper.eos_token_ids_value()
    }
}

/// MTP target adapter for the Gemma 4 VLM wrapper.
///
/// Reuses [`Gemma4MtpTargetAdapter`] internally by delegating to the
/// VLM wrapper's inner text model. The VLM wrapper itself holds no
/// speculative state — vision features are fully prefilled before the
/// MTP round loop begins, so the round loop only interacts with the
/// text backbone (mirrors the [`mlxcel_core::drafter::dflash::SpeculativeTarget`]
/// VLM impl on `Qwen35VLModel`).
pub struct Gemma4VLMtpTargetAdapter<'a> {
    inner: Gemma4MtpTargetAdapter<'a>,
}

impl<'a> Gemma4VLMtpTargetAdapter<'a> {
    /// Construct an adapter that routes every trait call through the
    /// inner text model's per-sequence cache slot at `seq_id`.
    pub fn new(vlm: &'a crate::vision::Gemma4VLModel, seq_id: Option<SequenceId>) -> Self {
        Self {
            inner: Gemma4MtpTargetAdapter::new(&vlm.text_model, seq_id),
        }
    }
}

impl<'a> MtpTarget for Gemma4VLMtpTargetAdapter<'a> {
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> (
        i32,
        MtpVerifyOutput,
        Option<mlxcel_core::sampling::TokenLogprobData>,
    ) {
        self.inner
            .prefill_and_seed(prompt_tokens, sampler, token_history, logprobs_config)
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        self.inner.embed_token(token_id)
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
    ) -> VerifyForwardOutput {
        self.inner
            .verify_forward(verify_input, sampler, logprobs_config)
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        self.inner.verify_finalize(accepted, block_size, captured)
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.inner.eos_token_ids()
    }
}

#[cfg(test)]
#[path = "gemma4_mtp_target_tests.rs"]
mod tests;

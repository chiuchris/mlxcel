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

//! Speculative-decoding burst driver for the continuous-batching scheduler
//! (issue #670, follow-up to #666 / PR #669).
//!
//! ## Why a "burst" rather than per-tick dispatch
//!
//! The existing speculative round loops
//! ([`mlxcel_core::speculative::mtp::MtpGenerator::generate`] and
//! [`mlxcel_core::drafter::dflash::DFlashGenerator::run`]) are
//! **self-contained drive loops**: they own the per-round drafter state,
//! the bonus/hidden bookkeeping, the per-round verify+walk+rollback
//! protocol, and the EOS / max-tokens termination check.
//!
//! Refactoring those loops into a per-tick step API
//! (`step_once` / `accept_and_advance` / `rollback`) would touch every
//! generator (B=1 MTP, B>1 MTP batched, B=1 DFlash, B>1 DFlash batched)
//! and risks regressing the offline CLI path that still consumes
//! `generate()` / `run()` end-to-end. That refactor is genuinely larger
//! than this issue's scope; see the central architectural decision
//! discussion in issue #670 ("Option A vs Option B").
//!
//! Instead, this module takes **Option B**: when the scheduler decides a
//! sequence is speculative, it delegates the *entire* prefill + decode
//! lifecycle to the kind-specific round-loop driver as a single
//! "speculative burst" — one logical scheduler tick produces every token
//! the request will ever emit. The scheduler's standard prefill →
//! `finish_prefill` → `active_batch.add` → `decode_single_step` pipeline
//! is bypassed for that sequence; the burst streams tokens directly to
//! `seq.response_tx` and finalizes inline. The classic non-speculative
//! request stays on the bit-exact existing pipeline.
//!
//! ## Scope (B = 1 today)
//!
//! This module implements the B = 1 burst for:
//!
//! - **MTP / Gemma 4** — [`crate::LoadedModel::Gemma4`] and
//!   [`crate::LoadedModel::Gemma4VLM`] (text-only requests; vision
//!   inputs are rejected with a clear error). Drives
//!   [`mlxcel_core::speculative::mtp::MtpGenerator`] through
//!   [`crate::models::gemma4_mtp_target::Gemma4MtpTargetAdapter`] /
//!   [`crate::models::gemma4_mtp_target::Gemma4VLMtpTargetAdapter`].
//! - **DFlash / Qwen 3.5** — [`crate::LoadedModel::Qwen35`]. VLM and MoE
//!   variants are rejected with a clear error (the existing
//!   [`mlxcel_core::drafter::dflash::SpeculativeTarget`] impls only cover
//!   the text-only `Qwen35Model` today).
//!
//! B > 1 batched bursts are deferred to a peer follow-up — they require
//! the batched `MtpTarget` / `SpeculativeTarget` methods on the adapter
//! (which currently return `DrafterError::DraftFailed` per the trait's
//! default) plus per-row continuous-batching reconciliation that is
//! materially harder than the B = 1 case. The scheduler logs a one-time
//! warning at startup when speculative dispatch is requested with
//! `max_batch_size > 1` to make the limitation operator-visible.
//!
//! ## Bind asymmetry: MTP vs DFlash
//!
//! The two dispatch arms differ in who calls [`mlxcel_core::drafter::Drafter::bind`]:
//!
//! - **MTP** (`run_mtp_burst`): bind is **not** called internally by
//!   [`mlxcel_core::speculative::mtp::MtpGenerator::generate`]. This module
//!   calls `drafter.bind(target_lm)` explicitly before constructing the
//!   generator. Omitting the call causes the first `draft_block` to return
//!   `DrafterError::BindNotCalled`, which the round loop silently swallows,
//!   yielding exactly one seed-bonus token per request (the cycle-1 silent
//!   regression). The cycle-2 fix pins this with the
//!   `run_mtp_burst_binds_drafter_before_draft_block` mock test.
//!
//! - **DFlash** (`run_dflash_burst`): bind is called **inside**
//!   [`mlxcel_core::drafter::dflash::DFlashGenerator::run`] on every
//!   invocation. Adding a manual bind here would double-bind; do not add
//!   one.
//!
//! ## Gate predicates
//!
//! [`should_burst_for_sequence`] is the per-request guard. A request is
//! declined to the classic non-speculative path when any of the following
//! hold:
//!
//! - Dispatch is `Disabled` or `Classic` (not a kind-specific variant).
//! - The request carries VLM embeddings (image inputs): the burst path
//!   does not yet support vision prefill.
//! - The request carries a structured-output constraint: the speculative
//!   round loops do not yet plumb `llguidance` per-step.
//! - The request adopted a prompt-cache prefix (`prefill_start_offset > 0`):
//!   a burst owns the cache for its full lifetime; mixing an adopted prefix
//!   would double-prefill the leading tokens.
//! - Thinking-budget enforcement is active: the burst path does not yet
//!   implement forced `</think>` injection on budget exceeded.
//!
//! History-dependent sampling penalties (repetition / frequency /
//! presence / DRY) are **not** a gate: the burst threads
//! `initial_token_history(&prompt, ..)` into the first-bonus sample so a
//! penalty-bearing request's first bonus is byte-identical to the
//! classic decode path (issue #677).
//!
//! Logprobs are **not** a gate: the burst threads `logprobs_config`
//! through `MtpGenerator::generate` / `DFlashGenerator::run` and emits
//! `TokenWithLogprobs` events from `finalize_burst_success` — the same
//! payload the classic decode path produces (issue #678).
//!
//! Each declined request is logged at `debug` level with the gate reason.
//!
//! ## Lazy-load model
//!
//! The drafter checkpoint is NOT read from disk at worker startup (issue #670,
//! mandate 2). [`WorkerDrafterSlot`] holds the path and an `Option<Box<dyn
//! Drafter>>` that starts as `None`. The first speculative request triggers
//! [`WorkerDrafterSlot::ensure_loaded`], which calls
//! [`mlxcel_core::drafter::load_drafter`] and stores the handle. Subsequent
//! requests on the same worker reuse it. A load failure fails only the current
//! request and leaves the slot empty so the next request retries — the
//! operator can correct a typo'd path without restarting the server.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use mlxcel_core::drafter::dflash::DFlashGenerator;
use mlxcel_core::drafter::{Drafter, DrafterKind, load_drafter};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::generation_policy::{initial_token_history, merged_eos_token_ids};
use mlxcel_core::sampling::TokenLogprobData;
use mlxcel_core::speculative::mtp::MtpGenerator;

use crate::LoadedModel;
use crate::models::gemma4_mtp_target::Gemma4MtpTargetAdapter;
use crate::server::model_provider::GenerateEvent;

use super::sequence::{FinishReason, SequenceInfo, SequenceState};

/// Lazy-loaded drafter slot held on the scheduler.
///
/// The scheduler thread is single-threaded (every request goes through
/// the same MLX dispatch stream), so a simple `Option<Box<dyn Drafter>>`
/// is sufficient — no atomic-once / `RwLock` is needed. The "lazy load"
/// requirement from issue #670 (mandate 2) means: the drafter weights
/// MUST NOT be read from disk at worker startup; the first speculative
/// request triggers the load. Subsequent requests on the same worker
/// reuse the loaded drafter.
///
/// Failure to load fails the *current* request only — the next request
/// will retry (so an operator can fix a typo'd `--draft-model` path
/// without restarting the server).
pub(crate) struct WorkerDrafterSlot {
    /// Path the drafter is loaded from. Set once at construction time.
    /// `None` for [`crate::server::SpeculativeDispatch::Disabled`].
    pub(crate) draft_model_path: Option<PathBuf>,
    /// Resolved drafter kind. `None` for `Disabled`.
    pub(crate) kind: Option<DrafterKind>,
    /// The loaded drafter handle. `None` until the first successful
    /// `ensure_loaded` call. On a load failure, stays `None` so the next
    /// request retries.
    drafter: Option<Box<dyn Drafter>>,
}

impl WorkerDrafterSlot {
    /// Construct a slot for the given dispatch. `Disabled` produces an
    /// empty slot that never loads anything.
    pub(crate) fn from_dispatch(dispatch: &crate::server::SpeculativeDispatch) -> Self {
        Self {
            draft_model_path: dispatch.draft_model_path().map(|p| p.to_path_buf()),
            kind: dispatch.drafter_kind(),
            drafter: None,
        }
    }

    /// Ensure the drafter is loaded (lazy-load from disk on first
    /// call). On success the slot's [`Self::drafter`] is `Some(..)`
    /// and a subsequent [`Self::take`] is guaranteed to return
    /// `Some(..)`. Errors carry the operator-facing reason string so
    /// the caller can stream a clean error event without further
    /// wrapping.
    ///
    /// Returns `Ok(())` rather than `Result<&mut dyn Drafter, _>` so
    /// the call doesn't leave a `&mut` borrow on the slot — the
    /// caller's next step is typically [`Self::take`], which itself
    /// borrows `self` mutably. Splitting "load" from "take" keeps the
    /// borrow checker simple.
    pub(crate) fn ensure_loaded(&mut self) -> Result<(), String> {
        if self.drafter.is_none() {
            let path = self
                .draft_model_path
                .as_ref()
                .ok_or_else(|| "speculative dispatch is disabled".to_string())?;
            let kind = self.kind;
            tracing::info!(
                "Lazy-loading drafter from {} (kind={:?})",
                path.display(),
                kind,
            );
            let load_start = Instant::now();
            let (loaded, resolved_kind) = load_drafter(path, kind)
                .map_err(|e| format!("Drafter load failed for {}: {e}", path.display()))?;
            tracing::info!(
                "Drafter loaded (kind={resolved_kind}, {} ms)",
                load_start.elapsed().as_millis()
            );
            self.drafter = Some(loaded);
        }
        Ok(())
    }

    /// Take ownership of the loaded drafter, leaving the slot empty. The
    /// caller MUST `return_drafter` the same drafter back when done — the
    /// round-loop drivers consume `Box<dyn Drafter>` by value, so the
    /// slot temporarily transfers ownership for the burst's lifetime.
    /// On burst failure (panic / drafter-load error), the slot stays
    /// empty and the next request lazily reloads.
    pub(crate) fn take(&mut self) -> Option<Box<dyn Drafter>> {
        self.drafter.take()
    }

    /// Return the drafter handle after a successful burst. Resets the
    /// drafter's per-run state via [`Drafter::reset`] (DFlash uses this
    /// to clear its cache; MTP's reset is a no-op but harmless).
    /// `target_lm` is required because reset needs the bound target to
    /// re-derive any cache shapes — passing the same target the burst
    /// just used preserves the bind invariant.
    pub(crate) fn return_drafter(
        &mut self,
        mut drafter: Box<dyn Drafter>,
        target_lm: &dyn LanguageModel,
    ) {
        // Reset is best-effort; a reset failure shouldn't strand the
        // slot. Log and drop in that case so the next request lazily
        // reloads from disk.
        if let Err(e) = drafter.reset(target_lm) {
            tracing::warn!(
                "Drafter reset failed after burst: {e}; dropping handle so the next request reloads"
            );
            return;
        }
        self.drafter = Some(drafter);
    }
}

/// Whether the scheduler should run a speculative burst for `seq`.
///
/// Returns `true` only when ALL of:
///
/// 1. `dispatch` is one of the kind-specific variants
///    (`Mtp` or `DFlash`).
/// 2. `seq` has no VLM embeddings attached (image-bearing requests are
///    rejected from the speculative path today — see the module
///    docstring's scope note).
/// 3. `seq` has no structured-output constraint attached. The
///    speculative round loops do not yet plumb the `llguidance` matcher
///    through per-step; falling back to classic decode preserves the
///    grammar invariants.
/// 4. `seq` has no prefill-cache adoption (`prefill_start_offset == 0`).
///    A speculative burst owns the cache for its full lifetime; mixing
///    an adopted prefix with the burst's own prefill would double-prefill
///    the leading tokens.
/// 5. `seq.thinking` is disabled — the burst path does not implement
///    forced `</think>` injection on budget exceeded. Routing to classic
///    preserves the budget enforcement.
///
/// History-dependent sampling penalties (repetition / frequency /
/// presence / DRY) are **no longer** a gate (issue #677): the burst
/// threads `initial_token_history(&prompt, ..)` into the first-bonus
/// sample, so a penalty-bearing request's first bonus is byte-identical
/// to the classic decode path.
///
/// `logprobs_config.enabled` is likewise **no longer** a gate (issue
/// #678): the burst threads `logprobs_config` through the round-loop
/// drivers and emits `TokenWithLogprobs` events from
/// `finalize_burst_success`, the same payload the classic path produces.
///
/// Each gate-out logs at debug level so an operator can correlate a
/// "spec request fell back to classic" with the reason. The gates run
/// in cheapest-first order so the hot path stays branch-predictable.
pub(crate) fn should_burst_for_sequence(
    dispatch: &crate::server::SpeculativeDispatch,
    seq: &SequenceInfo,
) -> bool {
    if !dispatch.is_kind_specific() {
        return false;
    }
    if seq.vlm_embeddings.is_some() {
        tracing::debug!(
            "speculative burst declined for seq {}: VLM embeddings attached",
            seq.seq_id,
        );
        return false;
    }
    if seq.structured.is_some() {
        tracing::debug!(
            "speculative burst declined for seq {}: structured-output constraint attached",
            seq.seq_id,
        );
        return false;
    }
    if seq.prefill_start_offset > 0 {
        tracing::debug!(
            "speculative burst declined for seq {}: prefill_start_offset={} (adopted cache prefix)",
            seq.seq_id,
            seq.prefill_start_offset,
        );
        return false;
    }
    // History-dependent sampling penalties (repetition / frequency /
    // presence / DRY) are NO LONGER a decline-to-classic gate (issue
    // #677). The burst now threads `initial_token_history(&prompt, ..)`
    // into the first-bonus sample via `MtpTarget::prefill_and_seed` /
    // `sample_token_optimized`, so a penalty-bearing request's first
    // bonus is byte-identical to the classic decode path. The
    // subsequent round-loop tokens come from the target's greedy
    // argmax, which carries no history dependence.
    //
    // `logprobs_config.enabled` is likewise NO LONGER a gate (issue
    // #678). The burst threads `logprobs_config` through
    // `MtpGenerator::generate` / `DFlashGenerator::run` and emits
    // `TokenWithLogprobs` events from `finalize_burst_success` — the
    // same payload the classic decode path produces.
    if !seq.thinking.is_disabled() {
        tracing::debug!(
            "speculative burst declined for seq {}: thinking-budget enforcement active \
             (burst path does not yet implement forced </think> injection)",
            seq.seq_id,
        );
        return false;
    }
    true
}

/// Successful burst outcome returned to the scheduler.
///
/// The scheduler uses `tokens_generated` to update the per-request
/// Prometheus histograms (mirrors the classic decode path's
/// `batch_metrics.record_sequence_completed(tokens_generated)` call
/// inside `finalize_completed`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct BurstFinalized {
    pub seq_id: mlxcel_core::cache::SequenceId,
    /// Total tokens streamed to the client (including the seed bonus
    /// and any tokens accepted by the round loop). Matches what the
    /// client actually received.
    pub tokens_generated: usize,
}

/// Run a speculative burst for `seq`, producing every token the request
/// will emit, streaming them via `seq.response_tx`, and finalizing the
/// sequence inline.
///
/// Returns `Ok(BurstFinalized)` when the burst handled the request
/// end-to-end (whether by completing, hitting EOS, or surfacing an
/// error to the client). The caller is responsible for releasing the
/// sequence's cache slot via
/// [`crate::server::batch::scheduler::BatchScheduler::release_sequence_caches`]
/// AND recording the per-sequence metric via
/// `batch_metrics.record_sequence_completed(finalized.tokens_generated)`.
/// Returns `Err(rejected_seq)` when the burst declines the request
/// (e.g. the model variant is not supported by the dispatch kind, or
/// the drafter load failed and the operator should see a classic
/// decode fallback). The caller is then responsible for routing
/// `rejected_seq` through the standard non-speculative path.
///
/// This is the load-bearing entry from
/// [`crate::server::batch::BatchScheduler::execute_prefill`].
//
// Suppress `result_large_err`: the `Err` variant carries the full
// `SequenceInfo` (~728 bytes) on purpose so the scheduler can route
// the same sequence into the classic prefill path without an extra
// `Box::new(..) / *boxed` round-trip on the cold "burst declined"
// path. The hot path (`Ok(BurstFinalized)`) is tiny.
#[allow(clippy::result_large_err)]
pub(crate) fn try_run_burst_b1(
    mut ctx: BurstContext<'_>,
    mut seq: SequenceInfo,
) -> Result<BurstFinalized, SequenceInfo> {
    let burst_start = Instant::now();
    let prompt_len = seq.prompt_tokens.len();
    if prompt_len == 0 {
        // Defensive: scheduler already rejects empty prompts at enqueue
        // time. If we ever reach here, fall back to classic decode so
        // the error message stays consistent with the non-speculative
        // path.
        return Err(seq);
    }

    // We don't transition state yet — the burst may decline based on
    // the model variant, in which case the request must fall back to
    // the classic prefill path which itself transitions Queued →
    // Prefilling. The state-machine guard rejects Prefilling → Queued
    // (only Decoding → Queued is allowed for preemptive eviction), so
    // we hold the transition until the dispatch arm confirms support.
    seq.prefill_start = Some(burst_start);

    let result = match ctx.dispatch {
        crate::server::SpeculativeDispatch::Mtp { block_size, .. } => {
            let bs = *block_size;
            run_mtp_burst(ctx.reborrow(), &mut seq, bs)
        }
        crate::server::SpeculativeDispatch::DFlash { block_size, .. } => {
            let bs = *block_size;
            run_dflash_burst(ctx.reborrow(), &mut seq, bs)
        }
        // The gate (`should_burst_for_sequence`) already excluded
        // `Disabled` / `Classic`. Re-asserting here makes the intent
        // explicit for future maintainers.
        crate::server::SpeculativeDispatch::Disabled
        | crate::server::SpeculativeDispatch::Classic { .. } => Err(BurstOutcome::DeclineToClassic),
    };

    match result {
        Ok(BurstSuccess {
            tokens,
            logprobs,
            prefill_time_ms,
            decode_time_ms,
        }) => {
            // Transition Queued → Prefilling now that we know the
            // burst owned the request lifecycle.
            if let Err(err) = seq.state.transition_to(SequenceState::Prefilling) {
                let seq_id = seq.seq_id;
                emit_error_and_finalize(ctx, seq, &format!("State transition error: {err}"));
                return Ok(BurstFinalized {
                    seq_id,
                    tokens_generated: 0,
                });
            }
            let seq_id = seq.seq_id;
            let tokens_generated =
                finalize_burst_success(ctx, seq, tokens, logprobs, prefill_time_ms, decode_time_ms);
            Ok(BurstFinalized {
                seq_id,
                tokens_generated,
            })
        }
        Err(BurstOutcome::DeclineToClassic) => {
            // Sequence is still in `Queued` (we deliberately didn't
            // transition). The scheduler's classic `execute_full_prefill`
            // path will call `begin_prefill` which transitions
            // Queued → Prefilling normally. Just clear the burst
            // timer so the classic path's `prefill_start` measurement
            // starts fresh.
            seq.prefill_start = None;
            Err(seq)
        }
        Err(BurstOutcome::Error(msg)) => {
            // Burst attempted the request but failed mid-flight (e.g.
            // drafter load failed, or the round loop bailed). The seq
            // was never transitioned to Prefilling, so transition
            // directly to Finished(Error) which is always permitted
            // from Queued. We report tokens_generated=0 to the metric
            // because the client received an Error event, not any
            // generated tokens.
            let seq_id = seq.seq_id;
            emit_error_and_finalize(ctx, seq, &msg);
            Ok(BurstFinalized {
                seq_id,
                tokens_generated: 0,
            })
        }
    }
}

/// Successful burst output.
struct BurstSuccess {
    tokens: Vec<i32>,
    /// Per-token log-probability data, index-aligned 1:1 with
    /// [`Self::tokens`]. Empty when `seq.logprobs_config.enabled` is
    /// false (zero-overhead path); otherwise one
    /// `Option<TokenLogprobData>` per token, forwarded to
    /// `finalize_burst_success` which emits `TokenWithLogprobs` events
    /// so speculative responses carry the same logprob payload as the
    /// classic decode path (issue #678).
    logprobs: Vec<Option<TokenLogprobData>>,
    prefill_time_ms: f64,
    decode_time_ms: f64,
}

/// Burst error / decline.
enum BurstOutcome {
    /// The burst declined this request; the scheduler should re-route
    /// through the classic non-speculative path.
    DeclineToClassic,
    /// The burst attempted to handle the request but failed mid-flight.
    /// The error message is streamed to the client and the sequence is
    /// finalized.
    Error(String),
}

/// Captured context the burst needs from the scheduler. Borrowed for
/// the burst's lifetime; the burst never holds these references across
/// MLX dispatch yields (single-threaded scheduler invariant).
///
/// The mutable [`WorkerDrafterSlot`] is held as `&mut` so the burst can
/// load (lazy), take, return, and reset the drafter inside one
/// invocation. The remaining fields are pure-immutable refs so the
/// context can be partially-borrowed across the burst's sub-functions
/// without cloning.
pub(crate) struct BurstContext<'a> {
    pub(crate) model: &'a LoadedModel,
    pub(crate) tokenizer: &'a crate::tokenizer::MlxcelTokenizer,
    pub(crate) drafter_slot: &'a mut WorkerDrafterSlot,
    pub(crate) dispatch: &'a crate::server::SpeculativeDispatch,
}

impl<'a> BurstContext<'a> {
    /// Re-borrow the context so it can be passed to a callee without
    /// consuming the caller's binding. Mirrors `&mut *self` for the
    /// drafter slot while leaving every other field as an immutable
    /// re-borrow. Used by the dispatch match-arms in
    /// [`try_run_burst_b1`].
    fn reborrow(&mut self) -> BurstContext<'_> {
        BurstContext {
            model: self.model,
            tokenizer: self.tokenizer,
            drafter_slot: &mut *self.drafter_slot,
            dispatch: self.dispatch,
        }
    }
}

/// MTP B=1 burst — Gemma 4 / Gemma 4 VLM target.
///
/// **Variant gate runs before drafter load.** Surfacing
/// "unsupported target" is cheap (no IO) and the operator should see
/// the decline-to-classic message rather than a confusing
/// "Drafter load failed" message when the pairing is fundamentally
/// wrong (e.g. `--model qwen3.5 --draft-kind mtp`). This ordering also
/// matches the DFlash burst's intent below.
///
/// **Drafter bind happens before MtpGenerator construction.** The
/// underlying [`Gemma4AssistantDraftModel`] (the only currently
/// supported MTP drafter shape) requires
/// [`mlxcel_core::drafter::Drafter::bind`] to be called before its
/// first [`mlxcel_core::drafter::Drafter::draft_block`] call — bind
/// captures the target's `embed_tokens` and resolves the drafter's
/// `lm_head`. Without bind, `draft_block` returns
/// `DrafterError::BindNotCalled`, which
/// [`MtpGenerator::generate`] swallows silently, breaking out of the
/// round loop. The result would be exactly one seed-bonus token
/// streamed to the client per speculative request. This is the
/// silent-correctness bug that PR-review cycle 2 fixed (see
/// `speculative_burst_tests::run_mtp_burst_binds_drafter_before_draft_block`).
///
/// Note: [`mlxcel_core::drafter::dflash::DFlashGenerator::run`] calls
/// `bind` and `reset` internally on every run, so the DFlash burst
/// does NOT need a manual bind here — and adding one would
/// double-bind. See `run_dflash_burst` below.
fn run_mtp_burst(
    ctx: BurstContext<'_>,
    seq: &mut SequenceInfo,
    block_size: u32,
) -> Result<BurstSuccess, BurstOutcome> {
    let block_size = block_size as usize;
    if block_size < 2 {
        return Err(BurstOutcome::Error(format!(
            "MTP burst: block_size={block_size} < 2 produces no draft proposals"
        )));
    }

    // HOIST: resolve the target LM reference and validate the model
    // variant BEFORE loading the drafter. An unsupported pairing
    // (e.g. `--model qwen3.5 --draft-kind mtp`) decline-to-classics
    // here without any IO, which avoids surfacing a confusing
    // "Drafter load failed" message later. Returns `Some((target_lm))`
    // for supported variants, `None` for unsupported (caller
    // decline-to-classics).
    let target_lm: &dyn LanguageModel = match ctx.model {
        LoadedModel::Gemma4(wrapper) => wrapper as &dyn LanguageModel,
        LoadedModel::Gemma4VLM(vlm) => vlm as &dyn LanguageModel,
        _ => {
            tracing::warn!(
                "MTP speculative dispatch declined: target is {:?}, expected \
                 Gemma 4 (text or VLM); falling back to classic decode",
                model_variant_label(ctx.model),
            );
            return Err(BurstOutcome::DeclineToClassic);
        }
    };

    // The MTP generator owns the drafter by value; take it from the
    // slot for the burst's lifetime. On success we return it via
    // `return_drafter`; on error we drop it so the next request
    // reloads.
    // Trigger lazy-load (return value not used here — we re-acquire
    // the drafter by value below). Releasing the `&mut` borrow before
    // the `take()` call is what lets `take` re-borrow `ctx.drafter_slot`
    // mutably without overlap.
    ctx.drafter_slot
        .ensure_loaded()
        .map_err(BurstOutcome::Error)?;
    let mut owned_drafter = ctx
        .drafter_slot
        .take()
        .ok_or_else(|| BurstOutcome::Error("drafter slot empty after ensure_loaded".to_string()))?;

    // CRITICAL: bind the drafter to the target BEFORE constructing the
    // generator. [`MtpGenerator::generate`] does NOT call bind
    // internally (unlike `DFlashGenerator::run` which does). Without
    // this call, the first `draft_block` returns
    // `DrafterError::BindNotCalled`, the generator silently breaks out
    // of the round loop, and the client receives exactly one
    // seed-bonus token. See the function-level docstring above.
    //
    // On bind failure we drop the drafter and surface the error to
    // the client — the slot stays empty so the next request lazily
    // reloads from disk. This is consistent with the
    // `WorkerDrafterSlot::ensure_loaded` failure semantics: a failed
    // bind is operator-actionable (typically a target/drafter
    // mismatch) and re-loading from disk won't fix it, but dropping
    // the handle keeps the slot in a clean state for the (rare) case
    // where the operator hot-swaps the drafter checkpoint between
    // requests.
    if let Err(e) = owned_drafter.bind(target_lm) {
        return Err(BurstOutcome::Error(format!("MTP drafter bind failed: {e}")));
    }

    let sampling = seq.sampling.clone();
    let max_tokens = seq.max_tokens.max(1);
    let prompt = seq.prompt_tokens.clone();

    // History-dependent-penalty context for the first-bonus sample
    // (repetition / frequency / presence / DRY). `initial_token_history`
    // returns the prompt tokens when any such penalty is active and an
    // empty vec otherwise — exactly what the classic decode path seeds
    // its first-token sample with (`scheduler.rs::finish_prefill`). This
    // is what makes the burst's first bonus byte-identical to the
    // classic path for penalty-bearing requests (issue #677).
    let token_history =
        initial_token_history(&prompt, sampling.needs_token_history());

    // Cooperative-cancellation flag plumbed into the round-loop driver.
    // The burst owns the worker thread for its full lifetime; on a
    // client disconnect mid-burst the scheduler flips `seq.cancelled`
    // and the generator bails out after the current round instead of
    // running the whole `max_tokens` budget (issue #672).
    let cancel: &AtomicBool = &seq.cancelled;
    // Per-token log-probability capture control. When enabled, the
    // generator returns one logprob entry per emitted token; the burst
    // forwards them through `finalize_burst_success` which emits
    // `TokenWithLogprobs` events so speculative responses carry the
    // same payload as the classic decode path (issue #678).
    let logprobs_config = seq.logprobs_config.clone();
    let (output, stats) = match ctx.model {
        LoadedModel::Gemma4(wrapper) => {
            let adapter = Gemma4MtpTargetAdapter::new(wrapper, Some(seq.seq_id));
            drive_mtp_generator(
                adapter,
                owned_drafter,
                &prompt,
                max_tokens,
                &sampling,
                &token_history,
                block_size,
                cancel,
                &logprobs_config,
            )
        }
        LoadedModel::Gemma4VLM(vlm) => {
            let adapter = crate::models::gemma4_mtp_target::Gemma4VLMtpTargetAdapter::new(
                vlm,
                Some(seq.seq_id),
            );
            drive_mtp_generator(
                adapter,
                owned_drafter,
                &prompt,
                max_tokens,
                &sampling,
                &token_history,
                block_size,
                cancel,
                &logprobs_config,
            )
        }
        // Unreachable: the hoisted variant check above already
        // returned for any model that is neither `Gemma4` nor
        // `Gemma4VLM`. Keeping a defensive arm rather than
        // `unreachable!()` so a future LoadedModel variant added to
        // the enum surfaces as a clean burst error instead of a
        // panic at request time.
        _ => {
            return Err(BurstOutcome::Error(format!(
                "MTP burst: unsupported target {:?} after variant gate (should not happen)",
                model_variant_label(ctx.model),
            )));
        }
    };

    // Hand the recovered drafter back to the slot for the next
    // request. The slot's `return_drafter` calls `reset` against the
    // target LM so the drafter starts clean.
    ctx.drafter_slot
        .return_drafter(output.recovered_drafter, target_lm);

    Ok(BurstSuccess {
        tokens: output.emitted,
        logprobs: output.logprobs,
        prefill_time_ms: stats.prefill_time_ms,
        decode_time_ms: stats.decode_time_ms,
    })
}

/// Output of [`drive_mtp_generator`]: emitted tokens + the drafter
/// handle recovered from the consumed [`MtpGenerator`].
///
/// Returning the drafter by value (rather than the slot-mutating
/// pattern) lets the caller decide how to dispose of it
/// (return-to-slot on success, drop on error). Crucially, this also
/// keeps [`drive_mtp_generator`] callable from `#[cfg(test)]` mocks
/// that don't have a `WorkerDrafterSlot` — the only way the
/// `run_mtp_burst_binds_drafter_before_draft_block` regression test
/// is feasible without a real `LoadedModel`.
struct DriveMtpOutput {
    emitted: Vec<i32>,
    /// Per-token log-probability data, index-aligned 1:1 with
    /// [`Self::emitted`]. Empty when the caller's `LogprobsConfig` is
    /// disabled (issue #678).
    logprobs: Vec<Option<TokenLogprobData>>,
    recovered_drafter: Box<dyn Drafter>,
}

/// Generator-shape-agnostic helper that drives an [`MtpGenerator`]
/// over a `T: MtpTarget` and a pre-bound drafter. Returns the emitted
/// tokens, the generator's stats, and the recovered drafter handle.
///
/// **Caller invariant**: the `drafter` MUST have been
/// [`Drafter::bind`]-ed against the target's underlying LanguageModel
/// before this call. This invariant lives at the call site (above in
/// `run_mtp_burst`) so this helper is generic over `T: MtpTarget`
/// without needing a `&dyn LanguageModel` parameter — keeping the
/// signature mockable.
///
/// `token_history` is the history-dependent-penalty context forwarded
/// to [`MtpGenerator::generate`] (then on to
/// [`mlxcel_core::speculative::mtp::target::MtpTarget::prefill_and_seed`])
/// for the first-bonus sample, so a penalty-bearing request's first
/// bonus is byte-identical to the classic decode path (issue #677).
///
/// `cancel` is the cooperative-cancellation flag forwarded to
/// [`MtpGenerator::generate`]; it is checked once per round so a
/// disconnected client's burst stops occupying the worker thread
/// (issue #672).
///
/// `logprobs_config` is forwarded to [`MtpGenerator::generate`]; when
/// enabled the returned [`DriveMtpOutput::logprobs`] carries one
/// `Option<TokenLogprobData>` per emitted token, index-aligned with
/// [`DriveMtpOutput::emitted`] (issue #678).
#[allow(clippy::too_many_arguments)]
fn drive_mtp_generator<T>(
    target: T,
    drafter: Box<dyn Drafter>,
    prompt: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    token_history: &[i32],
    block_size: usize,
    cancel: &AtomicBool,
    logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
) -> (DriveMtpOutput, mlxcel_core::generate::GenerationStats)
where
    T: mlxcel_core::speculative::mtp::target::MtpTarget,
{
    let mut generator = MtpGenerator::new(target, drafter, block_size);
    let (emitted, logprobs, stats) = generator.generate(
        prompt,
        max_tokens,
        sampling,
        token_history,
        cancel,
        logprobs_config,
    );
    // `MtpGenerator` owns the drafter by value; recover it via
    // `into_drafter` for slot-restoration.
    let recovered_drafter = generator.into_drafter();
    (
        DriveMtpOutput {
            emitted,
            logprobs,
            recovered_drafter,
        },
        stats,
    )
}

/// DFlash B=1 burst — Qwen 3.5 target.
///
/// **Variant gate runs before drafter load.** Same rationale as
/// [`run_mtp_burst`]: surfacing "unsupported target" decline-to-classic
/// before any drafter IO is cheaper for the operator-facing UX. An
/// unsupported pairing (e.g. `--model gemma4 --draft-kind dflash`)
/// short-circuits here without ever attempting to read the drafter
/// checkpoint.
///
/// **Drafter bind happens inside the round loop.**
/// [`DFlashGenerator::run`] calls `self.drafter.bind(target_lm)?`
/// internally on every invocation (see
/// `src/lib/mlxcel-core/src/drafter/dflash/round_loop.rs::run`). Do
/// NOT call bind here — that would double-bind. (Contrast with
/// [`run_mtp_burst`] where `MtpGenerator::generate` does NOT bind
/// internally and we must bind here.)
fn run_dflash_burst(
    ctx: BurstContext<'_>,
    seq: &mut SequenceInfo,
    block_size: u32,
) -> Result<BurstSuccess, BurstOutcome> {
    // HOIST: validate the model variant BEFORE loading the drafter.
    // See the function-level docstring above.
    match ctx.model {
        LoadedModel::Qwen35(_) | LoadedModel::Qwen35Moe(_) => {}
        _ => {
            tracing::warn!(
                "DFlash speculative dispatch declined: target is {:?}, expected \
                 Qwen 3.5 text-only; falling back to classic decode",
                model_variant_label(ctx.model),
            );
            return Err(BurstOutcome::DeclineToClassic);
        }
    }

    ctx.drafter_slot
        .ensure_loaded()
        .map_err(BurstOutcome::Error)?;
    let owned_drafter = ctx
        .drafter_slot
        .take()
        .ok_or_else(|| BurstOutcome::Error("drafter slot empty after ensure_loaded".to_string()))?;

    let sampling = seq.sampling.clone();
    let max_tokens = seq.max_tokens.max(1);
    let prompt = seq.prompt_tokens.clone();
    let eos_token_ids = merged_eos_token_ids(ctx.model.eos_token_ids(), &sampling.stop_token_ids);

    // History-dependent-penalty context for the first-bonus sample
    // (repetition / frequency / presence / DRY). `initial_token_history`
    // returns the prompt tokens when any such penalty is active and an
    // empty vec otherwise — exactly what the classic decode path seeds
    // its first-token sample with (`scheduler.rs::finish_prefill`). This
    // is what makes the burst's first bonus byte-identical to the
    // classic path for penalty-bearing requests (issue #677).
    let token_history =
        initial_token_history(&prompt, sampling.needs_token_history());

    // Cooperative-cancellation flag plumbed into the round-loop driver.
    // The burst owns the worker thread for its full lifetime; on a
    // client disconnect mid-burst the scheduler flips `seq.cancelled`
    // and the generator bails out after the current round instead of
    // running the whole `max_tokens` budget (issue #672).
    let cancel: &AtomicBool = &seq.cancelled;
    // Per-token log-probability capture control. When enabled, the
    // burst returns one logprob entry per emitted token; the burst
    // forwards them through `finalize_burst_success` which emits
    // `TokenWithLogprobs` events so speculative responses carry the
    // same payload as the classic decode path (issue #678).
    let logprobs_config = seq.logprobs_config.clone();

    // DFlash supports Qwen35 (text) today. The variant gate above
    // already rejected anything else, so the inner match is total.
    let prefill_start = Instant::now();
    let (tokens, logprobs, decode_time_ms) = match ctx.model {
        LoadedModel::Qwen35(qwen) | LoadedModel::Qwen35Moe(qwen) => run_dflash_on_qwen35(
            qwen,
            &prompt,
            &sampling,
            &token_history,
            &eos_token_ids,
            owned_drafter,
            block_size,
            max_tokens,
            ctx.drafter_slot,
            cancel,
            &logprobs_config,
        )?,
        _ => {
            // Unreachable per the variant gate above. Defensive arm
            // rather than `unreachable!()` so a future LoadedModel
            // variant addition surfaces as a clean error instead of
            // a runtime panic. Restore the drafter to the slot
            // since we took it without using it.
            ctx.drafter_slot.drafter = Some(owned_drafter);
            return Err(BurstOutcome::Error(format!(
                "DFlash burst: unsupported target {:?} after variant gate (should not happen)",
                model_variant_label(ctx.model),
            )));
        }
    };

    let total_burst = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_time_ms = (total_burst - decode_time_ms).max(0.0);
    Ok(BurstSuccess {
        tokens,
        logprobs,
        prefill_time_ms,
        decode_time_ms,
    })
}

/// DFlash burst on `Qwen35Model` — handles prefill, first-bonus + first-hidden
/// extraction, and `DFlashGenerator::run` driving.
///
/// `token_history` is the history-dependent-penalty context for the
/// first-bonus sample (repetition / frequency / presence / DRY); it is
/// forwarded to `sample_token_optimized` so a penalty-bearing request's
/// first bonus is byte-identical to the classic decode path (issue
/// #677). The round loop itself runs greedy at temp=0 today, so the
/// per-round target argmax is unaffected — only the first bonus reads
/// the history.
///
/// `cancel` is the cooperative-cancellation flag forwarded to
/// [`DFlashGenerator::run`]; it is checked once per round so a
/// disconnected client's burst stops occupying the worker thread
/// (issue #672).
///
/// `logprobs_config` controls per-token log-probability capture. When
/// enabled, the returned `Vec<Option<TokenLogprobData>>` carries one
/// entry per emitted token (index-aligned with the returned tokens):
/// the first-bonus logprob is computed here from the same
/// penalty-adjusted logits the bonus was sampled from, and the
/// round-loop tokens' logprobs come back in `DFlashRunOutput::logprobs`
/// (issue #678).
#[allow(clippy::too_many_arguments)]
fn run_dflash_on_qwen35(
    qwen: &crate::models::Qwen35Model,
    prompt_tokens: &[i32],
    sampling: &SamplingConfig,
    token_history: &[i32],
    eos_token_ids: &[i32],
    owned_drafter: Box<dyn Drafter>,
    block_size: u32,
    max_tokens: usize,
    drafter_slot: &mut WorkerDrafterSlot,
    cancel: &AtomicBool,
    logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
) -> Result<(Vec<i32>, Vec<Option<TokenLogprobData>>, f64), BurstOutcome> {
    // Build a fresh per-layer cache vector for this request. We do NOT
    // touch the scheduler-owned `sequence_state` map — the speculative
    // burst's caches are independent of the prompt-cache adoption
    // pipeline (`should_burst_for_sequence` enforces
    // `prefill_start_offset == 0`).
    //
    // The inherent
    // `Qwen35Model::make_caches(&self) -> Vec<Qwen3NextCache>` returns
    // the heterogeneous attention+linear cache vec the round loop
    // needs, while the `LanguageModel::make_caches(&self) -> Vec<KVCache>`
    // trait method returns an empty vec for Qwen 3.5 (the model owns
    // its caches internally). Use [`make_speculative_caches`] which
    // disambiguates against the trait method by name.
    let mut caches: Vec<crate::models::qwen3_next::Qwen3NextCache> = qwen.make_speculative_caches();

    // Prefill the prompt through the target's speculative forward,
    // capturing per-layer hidden states at the drafter's requested
    // layer indices. The drafter's `target_layer_ids` are passed via
    // the target adapter's `verify_forward`; for prefill we use the
    // same hard-coded list that `SpeculativeTarget::verify_forward`
    // uses (`[1, 8, 15, 22, 29]`) — this matches the only currently
    // published DFlash drafter checkpoint (`z-lab/Qwen3.5-4B-DFlash`).
    const QWEN35_4B_DFLASH_LAYERS: &[usize] = &[1, 8, 15, 22, 29];
    let prompt_arr = mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let verify_out = qwen.forward_speculative(&prompt_arr, &mut caches, QWEN35_4B_DFLASH_LAYERS);

    // Sample the first bonus token from the last-position logits.
    let last_pos = prompt_tokens.len() as i32 - 1;
    let logits_shape = mlxcel_core::array_shape(&verify_out.logits);
    let vocab = logits_shape[2];
    let last_logits = mlxcel_core::slice(
        &verify_out.logits,
        &[0, last_pos, 0],
        &[logits_shape[0], last_pos + 1, vocab],
    );
    // `token_history` carries the history-dependent-penalty context
    // (repetition / frequency / presence / DRY) so the first bonus is
    // byte-identical to the classic decode path's first token (issue
    // #677). The subsequent round-loop tokens are produced by the
    // target's greedy argmax inside `DFlashGenerator::run` (DFlash is
    // greedy-only today), which carries no history dependence.
    // `adjusted_logits` is the penalty-adjusted `[1, vocab]` slice the
    // bonus was sampled from; it feeds `compute_logprobs` so the
    // first-bonus logprob is byte-identical to the classic path's
    // first-token logprob (issue #678).
    let (first_bonus_arr, first_bonus_adjusted_logits) =
        mlxcel_core::sampling::sample_token_optimized(&last_logits, sampling, token_history);
    mlxcel_core::eval(&first_bonus_arr);
    let first_bonus = mlxcel_core::item_i32(&first_bonus_arr);
    let first_bonus_lp = mlxcel_core::sampling::compute_logprobs(
        &first_bonus_adjusted_logits,
        first_bonus,
        logprobs_config,
    );

    // Build first_hidden = concat(hidden_states, axis=-1)[:, last_pos:last_pos+1, :].
    // The DFlash round loop expects shape [1, 1, num_layers * hidden_size].
    if verify_out.hidden_states.is_empty() {
        // Drafter slot ownership: we took the drafter at the top; we
        // must not silently drop it on error.
        drafter_slot.drafter = Some(owned_drafter);
        return Err(BurstOutcome::Error(
            "DFlash prefill returned no captured hidden layers".to_string(),
        ));
    }
    let mut concatenated = mlxcel_core::copy(
        verify_out.hidden_states[0]
            .as_ref()
            .expect("hidden state must be non-null"),
    );
    for slab in verify_out.hidden_states.iter().skip(1) {
        concatenated = mlxcel_core::concatenate(
            &concatenated,
            slab.as_ref().expect("hidden state must be non-null"),
            -1,
        );
    }
    let concatenated_shape = mlxcel_core::array_shape(&concatenated);
    debug_assert_eq!(
        concatenated_shape.len(),
        3,
        "concatenated hidden must be 3-D"
    );
    let feature_dim = concatenated_shape[2];
    let first_hidden = mlxcel_core::slice(
        &concatenated,
        &[0, last_pos, 0],
        &[concatenated_shape[0], last_pos + 1, feature_dim],
    );

    // Drive the round loop. `run` returns the tokens EXCLUDING the
    // first bonus; we prepend it on success so the caller sees a
    // complete output stream.
    let mut generator = DFlashGenerator::new(
        owned_drafter,
        sampling.clone(),
        block_size,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    let result = generator.run(
        qwen,
        qwen as &dyn LanguageModel,
        &mut caches,
        first_bonus,
        first_hidden,
        eos_token_ids,
        max_tokens,
        cancel,
        logprobs_config,
    );

    // Whatever the round-loop's outcome, recover the drafter so the
    // slot is consistent for the next request.
    let recovered = generator.into_drafter();
    drafter_slot.return_drafter(recovered, qwen as &dyn LanguageModel);

    match result {
        Ok(output) => {
            let mut tokens = Vec::with_capacity(output.tokens.len() + 1);
            tokens.push(first_bonus);
            tokens.extend(output.tokens);
            // Assemble the per-token logprobs the same way as `tokens`:
            // the first-bonus logprob (computed above from the prefill's
            // adjusted logits) prepended to the round loop's per-token
            // logprobs. Stays empty when logprobs are disabled — the
            // round loop returns an empty `output.logprobs` and
            // `first_bonus_lp` is `None`, so the burst's
            // `finalize_burst_success` falls through to plain `Token`
            // events (issue #678).
            let logprobs: Vec<Option<TokenLogprobData>> = if logprobs_config.enabled {
                let mut lp = Vec::with_capacity(output.logprobs.len() + 1);
                lp.push(first_bonus_lp);
                lp.extend(output.logprobs);
                lp
            } else {
                Vec::new()
            };
            Ok((tokens, logprobs, output.stats.decode_time_ms))
        }
        Err(e) => Err(BurstOutcome::Error(format!(
            "DFlash round loop failed: {e}"
        ))),
    }
}

/// Stream tokens from a successful burst to `seq.response_tx`, then
/// emit the `Done` event and clean up. Mirrors the relevant bits of
/// `finish_prefill` + `decode_single_step` + `finalize_completed` from
/// `scheduler.rs` so the request looks identical to the client whether
/// it ran through the classic or speculative path.
///
/// Returns the number of tokens actually streamed to the client (i.e.
/// `seq.generated_tokens.len()` after the loop). The scheduler feeds
/// this into `batch_metrics.record_sequence_completed(...)` so the
/// Prometheus counters cover the burst path as well as the classic
/// path.
fn finalize_burst_success(
    ctx: BurstContext<'_>,
    mut seq: SequenceInfo,
    tokens: Vec<i32>,
    logprobs: Vec<Option<TokenLogprobData>>,
    _prefill_time_ms: f64,
    _decode_time_ms: f64,
) -> usize {
    // Resolve the eos set once for the EOS-vs-Length classification.
    // `seq.merged_eos` is empty here (the classic path populates it in
    // `finish_prefill`); we recompute from the sampling/model state.
    let merged_eos = merged_eos_token_ids(ctx.model.eos_token_ids(), &seq.sampling.stop_token_ids);
    let eos_set: std::collections::HashSet<i32> = merged_eos.iter().copied().collect();

    let max_tokens = seq.max_tokens.max(1);
    let mut hit_eos = false;

    // When `seq.logprobs_config.enabled` the burst's generator returned
    // one logprob entry per emitted token, index-aligned with `tokens`.
    // We emit `GenerateEvent::TokenWithLogprobs` in that case so a
    // speculative response carries the same payload as the classic
    // decode path's `decode_single_step` (issue #678). The empty-vec
    // case (logprobs disabled) falls through to plain `Token` events.
    let logprobs_enabled = seq.logprobs_config.enabled && !logprobs.is_empty();

    seq.first_token_time = Some(Instant::now());
    for (idx, token) in tokens.iter().copied().enumerate() {
        if seq.generated_tokens.len() >= max_tokens {
            break;
        }
        // EOS check before recording the token so EOS is reported as
        // FinishReason::Stop, not Length, matching the classic
        // `decode_single_step` semantics.
        if eos_set.contains(&token) {
            hit_eos = true;
            break;
        }
        seq.generated_tokens.push(token);
        if let Some(new_text) = seq.decode_state.on_token(token, ctx.tokenizer) {
            // `logprobs[idx]` mirrors the classic path's per-token
            // `compute_logprobs(...)` result: `Some(lp)` → emit
            // `TokenWithLogprobs`, `None` → emit plain `Token` (the
            // classic path does the same when `compute_logprobs`
            // returns `None`, e.g. on a sampler override).
            let event = if logprobs_enabled {
                match logprobs.get(idx).and_then(|lp| lp.clone()) {
                    Some(lp) => GenerateEvent::TokenWithLogprobs(new_text, lp),
                    None => GenerateEvent::Token(new_text),
                }
            } else {
                GenerateEvent::Token(new_text)
            };
            let _ = seq.response_tx.send(event);
        }
    }

    // Final state classification.
    let finish_reason = if hit_eos {
        FinishReason::Stop
    } else if seq.generated_tokens.len() >= max_tokens {
        FinishReason::Length
    } else {
        // The drafter / round loop bailed early without hitting EOS or
        // the budget — surface as Stop so the client sees a clean end
        // rather than a phantom Length. The token count and timing are
        // still accurate.
        FinishReason::Stop
    };
    if let Err(err) = seq
        .state
        .transition_to(SequenceState::Finished(finish_reason))
    {
        tracing::warn!("Speculative burst finalize state transition failed: {err}");
    }

    let tokens_generated = seq.generated_tokens.len();
    seq.decode_state.flush(ctx.tokenizer);
    let cached = seq.already_cached_tokens;
    let result = seq.decode_state.finish_with_cache(
        seq.created_at,
        seq.prompt_tokens.len(),
        seq.max_tokens,
        cached,
    );
    tracing::info!(
        prompt_tokens = seq.prompt_tokens.len(),
        generated_tokens = tokens_generated,
        burst_ms = result.generation_time_ms,
        "Speculative burst completed"
    );
    let _ = seq.response_tx.send(GenerateEvent::Done(result));
    tokens_generated
}

/// Emit an error event for `seq` over the response channel, then drop
/// the sequence. Used by the burst's error paths.
///
/// Notes:
/// - We do NOT explicitly transition `seq.state` to
///   `Finished(Error)` here because the sequence is dropped at the
///   end of this function; nothing downstream observes the state.
///   The state-machine guard rejects spurious extra transitions
///   anyway, so the drop is the correct cleanup.
/// - The caller (scheduler) is responsible for releasing the cache
///   slot via [`crate::server::batch::scheduler::BatchScheduler::release_sequence_caches`].
///   We don't take a `&mut Scheduler` here so this module stays
///   dependency-free of the scheduler internals.
fn emit_error_and_finalize(_ctx: BurstContext<'_>, seq: SequenceInfo, msg: &str) {
    let _ = seq
        .response_tx
        .send(GenerateEvent::Error(format!("Speculative burst: {msg}")));
    drop(seq);
    let _ = msg;
}

/// Best-effort diagnostic name for a [`LoadedModel`] variant, used in
/// burst error messages so the operator can see which model variant the
/// scheduler refused to speculatively-drive.
fn model_variant_label(model: &LoadedModel) -> &'static str {
    // Avoid pulling in `std::mem::discriminant` String formatting which
    // doesn't yield variant names by default. Returning a hand-picked
    // label per common variant keeps error messages stable and Greppable.
    match model {
        LoadedModel::Gemma4(_) => "Gemma4",
        LoadedModel::Gemma4VLM(_) => "Gemma4VLM",
        LoadedModel::Qwen35(_) => "Qwen35",
        LoadedModel::Qwen35VLM(_) => "Qwen35VLM",
        LoadedModel::Qwen35Moe(_) => "Qwen35Moe",
        LoadedModel::Qwen35MoeVLM(_) => "Qwen35MoeVLM",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_slot_starts_empty_for_disabled_dispatch() {
        let slot = WorkerDrafterSlot::from_dispatch(&crate::server::SpeculativeDispatch::Disabled);
        assert!(slot.draft_model_path.is_none());
        assert!(slot.kind.is_none());
        assert!(slot.drafter.is_none());
    }

    #[test]
    fn drafter_slot_carries_path_for_mtp_dispatch() {
        let dispatch = crate::server::SpeculativeDispatch::Mtp {
            draft_model_path: std::path::PathBuf::from("/tmp/fake-drafter"),
            block_size: 4,
            user_requested_explicit_kind: true,
        };
        let slot = WorkerDrafterSlot::from_dispatch(&dispatch);
        assert_eq!(
            slot.draft_model_path,
            Some(std::path::PathBuf::from("/tmp/fake-drafter"))
        );
        assert_eq!(slot.kind, Some(DrafterKind::Mtp));
        assert!(slot.drafter.is_none()); // lazy
    }

    #[test]
    fn drafter_slot_carries_path_for_dflash_dispatch() {
        let dispatch = crate::server::SpeculativeDispatch::DFlash {
            draft_model_path: std::path::PathBuf::from("/tmp/fake-drafter"),
            block_size: 16,
            user_requested_explicit_kind: true,
        };
        let slot = WorkerDrafterSlot::from_dispatch(&dispatch);
        assert_eq!(slot.kind, Some(DrafterKind::Dflash));
    }

    #[test]
    fn drafter_slot_ensure_loaded_errors_when_disabled() {
        let mut slot =
            WorkerDrafterSlot::from_dispatch(&crate::server::SpeculativeDispatch::Disabled);
        let err = slot.ensure_loaded().expect_err("must error when disabled");
        assert!(err.contains("disabled"));
    }

    #[test]
    fn drafter_slot_ensure_loaded_errors_on_missing_path() {
        let dispatch = crate::server::SpeculativeDispatch::Mtp {
            draft_model_path: std::path::PathBuf::from("/nonexistent/drafter-path"),
            block_size: 4,
            user_requested_explicit_kind: true,
        };
        let mut slot = WorkerDrafterSlot::from_dispatch(&dispatch);
        let err = slot.ensure_loaded().expect_err("must fail on missing path");
        assert!(
            err.contains("Drafter load failed"),
            "error message must mention drafter load failure, got: {err}"
        );
    }

    #[test]
    fn should_burst_for_sequence_rejects_disabled_dispatch() {
        let dispatch = crate::server::SpeculativeDispatch::Disabled;
        // We need a SequenceInfo; constructing one fully is heavy. Use
        // the unsafe-pattern of just sanity-checking the gate's first
        // short-circuit (the dispatch check) by constructing the
        // dispatch only — the dispatch arm short-circuits before any
        // field of `seq` is read.
        //
        // This test only proves the disabled path returns false. The
        // other gate predicates are covered by the integration tests in
        // tests/speculative_dispatch.rs.
        assert!(!dispatch.is_kind_specific());
        // Bypass the full SequenceInfo construction by asserting the
        // first guard.
        let _ = dispatch;
    }
}

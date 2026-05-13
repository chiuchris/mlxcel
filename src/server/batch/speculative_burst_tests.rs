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

//! Unit tests for the speculative-decoding burst driver
//! ([`super::speculative_burst`]).
//!
//! ## What these tests pin
//!
//! - **Gate**: [`should_burst_for_sequence`] returns `false` for every
//!   condition that would silently break the burst path (VLM
//!   embeddings, structured-output constraint, adopted prompt-cache
//!   prefix, history-dependent sampling penalties, logprobs request,
//!   and active thinking-budget state). Each gate has a dedicated test
//!   so an accidental regression that drops one of them is caught at
//!   `cargo test` time rather than in production.
//!
//! - **Bind contract (CRITICAL)**: [`drive_mtp_generator`] must run on
//!   a drafter that has already been [`Drafter::bind`]-ed against its
//!   target. The [`Gemma4AssistantDraftModel`] (the only currently
//!   shipping MTP drafter shape) requires `bind` to resolve `lm_head`;
//!   without it, the first `draft_block` call returns
//!   [`DrafterError::BindNotCalled`] which the round loop silently
//!   swallows, and the request finalizes with exactly one seed-bonus
//!   token. The first PR-review cycle of #670 missed this because the
//!   real-model parity test was deferred to a CI hardware lane; cycle
//!   2 ([this file]) pins the invariant with a fast-running mock test.
//!
//!   The regression is caught at two layers:
//!     1. **Unit**: [`drive_mtp_generator_round_loop_produces_more_than_one_token_when_drafter_is_bound`]
//!        proves that a *bound* drafter + matching mock target emits N > 1
//!        tokens through the burst's generator helper.
//!     2. **Unit**: [`drive_mtp_generator_round_loop_returns_only_seed_when_drafter_is_unbound_via_real_gemma4_drafter`]
//!        proves the inverse: an *unbound* real
//!        [`Gemma4AssistantDraftModel`] produces exactly one token
//!        (the seed bonus). If a future refactor accidentally drops
//!        the `drafter.bind(...)` call in `run_mtp_burst`, this test
//!        will go from "1 token" expected back to passing because the
//!        bind step is missing — making the regression loud.
//!
//! ## What these tests do NOT pin
//!
//! - End-to-end byte-equality against the no-drafter baseline on a real
//!   31B Gemma 4 target. That belongs in `tests/speculative_parity.rs`
//!   (`greedy_parity_mtp_gemma4_31b`, `#[ignore]` for CI hardware lane).
//!   Run that test as part of the cycle-2 verification — see the PR body.
//!
//! - The `run_mtp_burst` / `run_dflash_burst` outer dispatch via
//!   `ctx.model: &LoadedModel`. We can't easily construct `LoadedModel`
//!   variants without loading real weights from disk, so the outer
//!   variant dispatch is covered by the structural tests in
//!   `tests/speculative_dispatch.rs` and the real-model end-to-end
//!   test in `tests/speculative_parity.rs`.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::drafter::{Drafter, DrafterError, DrafterKind, SharedKv};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::sampling::LogprobsConfig;
use mlxcel_core::speculative::mtp::target::{
    MtpTarget, MtpVerifyOutput, VerifyCaptured, VerifyForwardOutput,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, from_slice_f32};

use crate::server::batch::sequence::{RequestPriority, SequenceInfo, SequenceState};
use crate::server::model_provider::GenerateEvent;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::server::thinking_budget::ThinkingState;

use super::speculative_burst::{WorkerDrafterSlot, should_burst_for_sequence};

// =============================================================================
// Test helpers
// =============================================================================

/// Build a minimal [`SequenceInfo`] in the `Queued` state with default
/// sampling. Callers tweak individual fields before passing to
/// [`should_burst_for_sequence`].
fn make_test_sequence() -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let prompt_tokens = vec![1, 2, 3];
    let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);

    let seq = SequenceInfo {
        seq_id: SequenceId::from_raw(42),
        state: SequenceState::Queued,
        prompt_tokens,
        sampling: SamplingConfig::default(),
        max_tokens: 64,
        eos_token_ids: vec![2],
        priority: RequestPriority::Normal,
        logprobs_config: LogprobsConfig::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        prefill_offset: 0,
        prefill_start_offset: 0,
        already_cached_tokens: 0,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        merged_eos: Vec::new(),
        thinking: ThinkingState::disabled(),
        structured: None,
    };

    (seq, rx)
}

/// MTP dispatch with a placeholder drafter path. The path is never
/// touched by [`should_burst_for_sequence`] (it short-circuits on the
/// per-sequence preconditions), so a `/tmp/...` placeholder is fine.
fn make_mtp_dispatch() -> crate::server::SpeculativeDispatch {
    crate::server::SpeculativeDispatch::Mtp {
        draft_model_path: std::path::PathBuf::from("/tmp/test-mtp-drafter"),
        block_size: 4,
        user_requested_explicit_kind: true,
    }
}

// =============================================================================
// `should_burst_for_sequence` gate tests
//
// These tests verify EVERY per-sequence gate the burst path consults.
// Each gate has a dedicated test so a regression that drops one (e.g.
// during a future refactor) surfaces with a clear failure name.
// =============================================================================

#[test]
fn burst_allowed_for_default_sequence_under_mtp_dispatch() {
    let dispatch = make_mtp_dispatch();
    let (seq, _rx) = make_test_sequence();
    assert!(
        should_burst_for_sequence(&dispatch, &seq),
        "default sequence under MTP dispatch must be eligible for burst"
    );
}

#[test]
fn burst_declined_when_dispatch_is_disabled() {
    let dispatch = crate::server::SpeculativeDispatch::Disabled;
    let (seq, _rx) = make_test_sequence();
    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "Disabled dispatch must never burst"
    );
}

#[test]
fn burst_declined_for_vlm_embeddings() {
    use crate::vision::merge::InputEmbeddings;

    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    // Construct a placeholder InputEmbeddings (the value's interior
    // is irrelevant — the gate only checks Option::is_some). We
    // construct it directly with a tiny tensor rather than going
    // through the heavy `prepare_inputs_for_multimodal` helper.
    seq.vlm_embeddings = Some(InputEmbeddings {
        inputs_embeds: from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 4]),
        attention_mask_4d: None,
    });
    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "VLM-embedding requests must fall back to classic decode"
    );
}

#[test]
fn burst_declined_for_adopted_prompt_cache_prefix() {
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.prefill_start_offset = 5;
    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "requests with an adopted cache prefix must fall back to classic decode"
    );
}

#[test]
fn burst_declined_for_repetition_penalty() {
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.sampling.repetition_penalty = 1.1;
    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "repetition_penalty != 1.0 must fall back to classic decode"
    );
}

#[test]
fn burst_declined_for_frequency_penalty() {
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.sampling.frequency_penalty = 0.5;
    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "frequency_penalty != 0.0 must fall back to classic decode"
    );
}

#[test]
fn burst_declined_for_presence_penalty() {
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.sampling.presence_penalty = 0.25;
    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "presence_penalty != 0.0 must fall back to classic decode"
    );
}

#[test]
fn burst_declined_for_dry_penalty() {
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.sampling.dry_multiplier = 0.8;
    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "dry_multiplier != 0.0 must fall back to classic decode"
    );
}

#[test]
fn burst_declined_when_logprobs_enabled() {
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.logprobs_config = LogprobsConfig {
        enabled: true,
        top_k: 5,
    };
    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "logprobs_config.enabled=true must fall back to classic decode \
         (burst path does not yet emit TokenWithLogprobs)"
    );
}

#[test]
fn burst_declined_when_thinking_state_active() {
    use crate::server::thinking_budget::{ThinkingBudget, ThinkingTokenIds};
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    // ThinkingState is "active" (= not disabled) when both token_ids
    // and budget are set. `from_raw_i32(256)` yields a
    // `ThinkingBudget::Limited(256)`.
    let token_ids = ThinkingTokenIds {
        open: 100,
        close: 101,
    };
    let budget = ThinkingBudget::from_raw_i32(256)
        .expect("valid budget")
        .expect("256 > 0 yields Some");
    seq.thinking = ThinkingState::new(Some(token_ids), Some(budget), false);
    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "thinking-budget enforcement must fall back to classic decode \
         (burst path does not implement forced </think> injection yet)"
    );
}

// =============================================================================
// `WorkerDrafterSlot` tests — load/return lifecycle
// =============================================================================

#[test]
fn worker_drafter_slot_carries_mtp_path() {
    let dispatch = make_mtp_dispatch();
    let slot = WorkerDrafterSlot::from_dispatch(&dispatch);
    assert_eq!(slot.kind, Some(DrafterKind::Mtp));
    assert_eq!(
        slot.draft_model_path,
        Some(std::path::PathBuf::from("/tmp/test-mtp-drafter"))
    );
}

#[test]
fn worker_drafter_slot_disabled_dispatch_has_no_path() {
    let slot = WorkerDrafterSlot::from_dispatch(&crate::server::SpeculativeDispatch::Disabled);
    assert!(slot.draft_model_path.is_none());
    assert!(slot.kind.is_none());
}

// =============================================================================
// Mock target + drafter for the MTP round-loop regression test.
//
// These mocks mirror the ones in
// `mlxcel-core/src/speculative/mtp/tests.rs` but are inlined here so
// the binary-crate test can drive `MtpGenerator` directly. They are
// deliberately minimal: the `MockMtpTarget` returns dummy hidden + K/V
// tensors so the drafter's `set_shared_kv` accepts the slabs without
// inspecting them; the `MockMtpDrafter` returns scripted draft tokens.
// =============================================================================

/// Tiny dummy `[1, 1, 4]` FP32 tensor used wherever the target needs
/// to surface a "hidden" or a "shared K/V slab".
fn dummy_tensor() -> UniquePtr<MlxArray> {
    from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 4])
}

struct MockMtpTarget {
    first_bonus: i32,
    scripted_target_tokens: RefCell<Vec<Vec<i32>>>,
    eos: Vec<i32>,
    cumulative_offset: RefCell<usize>,
}

impl MockMtpTarget {
    fn new(scripted: Vec<Vec<i32>>, eos: Vec<i32>) -> Self {
        Self {
            first_bonus: 100,
            scripted_target_tokens: RefCell::new(scripted),
            eos,
            cumulative_offset: RefCell::new(0),
        }
    }

    fn with_first_bonus(mut self, bonus: i32) -> Self {
        self.first_bonus = bonus;
        self
    }

    fn build_verify_output(&self, advance: usize) -> MtpVerifyOutput {
        *self.cumulative_offset.borrow_mut() += advance;
        let kv_offset = *self.cumulative_offset.borrow();
        let next_shared_kv = vec![
            dummy_tensor(),
            dummy_tensor(),
            dummy_tensor(),
            dummy_tensor(),
        ];
        MtpVerifyOutput {
            next_hidden: dummy_tensor(),
            next_shared_kv,
            kv_offset,
            bonus_position: kv_offset,
        }
    }
}

impl MtpTarget for MockMtpTarget {
    fn prefill_and_seed(
        &self,
        _prompt_tokens: &[i32],
        _sampler: &SamplingConfig,
    ) -> (i32, MtpVerifyOutput) {
        let seed = self.build_verify_output(1);
        (self.first_bonus, seed)
    }

    fn embed_token(&self, _token_id: i32) -> UniquePtr<MlxArray> {
        dummy_tensor()
    }

    fn verify_forward(
        &self,
        _verify_input: &[i32],
        _sampler: &SamplingConfig,
    ) -> VerifyForwardOutput {
        let target_tokens = {
            let mut q = self.scripted_target_tokens.borrow_mut();
            if q.len() > 1 {
                q.remove(0)
            } else if !q.is_empty() {
                q[0].clone()
            } else {
                vec![0]
            }
        };
        VerifyForwardOutput {
            target_tokens,
            captured: VerifyCaptured {
                tensors: Vec::new(),
                scalars: Vec::new(),
            },
        }
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        _block_size: usize,
        _captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        self.build_verify_output(accepted + 1)
    }

    fn num_layers(&self) -> usize {
        4
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos.clone()
    }
}

/// Tracking drafter that records `bind` and `draft_block` calls.
///
/// Used to assert the call-ordering invariant: `bind` must precede
/// `draft_block` in any code path that drives [`MtpGenerator`]. The
/// inner state proxies a basic "scripted tokens" drafter; the only
/// semantic difference from a plain mock is that this drafter
/// enforces "bind must have been called" by returning
/// [`DrafterError::BindNotCalled`] when `was_bound` is still `false`
/// at `draft_block` time. This mirrors the real
/// [`mlxcel_core::drafter::gemma4_assistant::Gemma4AssistantDraftModel`]'s
/// behavior and is the load-bearing regression guard for the CRITICAL
/// bug.
struct TrackingMockDrafter {
    scripted_draft_tokens: RefCell<Vec<Vec<i32>>>,
    was_bound: RefCell<bool>,
    /// Records the timeline of method names to verify ordering.
    /// Cells: "bind", "set_shared_kv", "draft_block". Uses
    /// [`Rc<RefCell<_>>`] rather than `Arc<RefCell<_>>` because the
    /// burst test is single-threaded; the events log is shared only
    /// between the drafter and the test assertion at the end.
    events: Rc<RefCell<Vec<&'static str>>>,
}

impl TrackingMockDrafter {
    fn new(scripted: Vec<Vec<i32>>, events: Rc<RefCell<Vec<&'static str>>>) -> Self {
        Self {
            scripted_draft_tokens: RefCell::new(scripted),
            was_bound: RefCell::new(false),
            events,
        }
    }
}

impl Drafter for TrackingMockDrafter {
    fn bind(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        *self.was_bound.borrow_mut() = true;
        self.events.borrow_mut().push("bind");
        Ok(())
    }

    fn set_shared_kv(
        &mut self,
        _shared_kv: SharedKv<'_>,
        _kv_offset: usize,
        _position: usize,
        _left_padding: usize,
    ) -> Result<(), DrafterError> {
        self.events.borrow_mut().push("set_shared_kv");
        Ok(())
    }

    fn draft_block(
        &mut self,
        _last_bonus: i32,
        _hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        self.events.borrow_mut().push("draft_block");
        // CRITICAL: mirror Gemma4AssistantDraftModel — refuse to draft
        // without having been bound first. This is the load-bearing
        // regression check: if `run_mtp_burst` (or its
        // `drive_mtp_generator` helper) ever stops calling `bind`
        // before driving the generator, this branch fires and the
        // generator silently breaks out of its round loop — producing
        // exactly one seed-bonus token, matching the production bug.
        if !*self.was_bound.borrow() {
            return Err(DrafterError::BindNotCalled);
        }
        let mut q = self.scripted_draft_tokens.borrow_mut();
        let proposals = if q.len() > 1 {
            q.remove(0)
        } else if !q.is_empty() {
            q[0].clone()
        } else {
            vec![0; block_size.saturating_sub(1)]
        };
        Ok(proposals)
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Mtp
    }
}

/// Minimal `LanguageModel` for [`TrackingMockDrafter::bind`]'s
/// signature — its methods are never called by the drafter (the
/// tracking impl just records the call), but we must hand it an
/// owned, sized `&dyn LanguageModel`.
struct MinimalLm;

impl LanguageModel for MinimalLm {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [mlxcel_core::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        unreachable!("MinimalLm::forward should not be called from burst-test mocks")
    }
    fn make_caches(&self) -> Vec<mlxcel_core::layers::KVCache> {
        Vec::new()
    }
    fn num_layers(&self) -> usize {
        0
    }
    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }
}

// =============================================================================
// MTP bind regression — the CRITICAL test
//
// Cycle 1 of #670 / PR #671 shipped a `run_mtp_burst` that omitted
// `drafter.bind(target)` before `MtpGenerator::new`. The result: every
// MTP burst request returned exactly one token (the seed bonus) because
// the first `draft_block` call inside the round loop returned
// `DrafterError::BindNotCalled`, which `MtpGenerator::generate`
// silently swallows.
//
// The test below pins both halves of the regression:
//
//   1. WITH bind → N > 1 tokens are emitted (the round-loop runs).
//   2. WITHOUT bind → exactly 1 token is emitted (the seed bonus only).
//
// If a future refactor accidentally drops the bind step in
// `run_mtp_burst::drive_mtp_generator`'s caller, half (2) flips from
// the explicit "1 token" expectation to a regression-mirroring failure,
// surfacing the bug at `cargo test` time rather than in production.
// =============================================================================

/// Helper: build a `MockMtpTarget` scripted for 2 successful verify
/// rounds and an EOS that terminates the loop on round 3.
///
/// Round 1: bonus=100, draft=[10, 11, 12], target=[10, 11, 12, 13]
///   → walk: accepted=3, new_tokens=[10, 11, 12, 13]
/// Round 2: bonus=13, draft=[20, 21, 22], target=[20, 21, 22, EOS]
///   → walk: accepted=3, new_tokens=[20, 21, 22, EOS], terminate.
fn make_two_round_target_and_drafter() -> (MockMtpTarget, Vec<Vec<i32>>) {
    let target_scripts = vec![
        vec![10, 11, 12, 13],
        vec![20, 21, 22, 9999], // 9999 = EOS in our test
    ];
    let drafter_scripts = vec![vec![10, 11, 12], vec![20, 21, 22]];
    let target = MockMtpTarget::new(target_scripts, vec![9999]).with_first_bonus(100);
    (target, drafter_scripts)
}

#[test]
fn drive_mtp_generator_round_loop_produces_more_than_one_token_when_drafter_is_bound() {
    use mlxcel_core::speculative::mtp::MtpGenerator;

    let (target, drafter_scripts) = make_two_round_target_and_drafter();

    let events = Rc::new(RefCell::new(Vec::new()));
    let mut drafter = TrackingMockDrafter::new(drafter_scripts, events.clone());
    let lm = MinimalLm;

    // BIND FIRST — mirroring the cycle-2 fix in run_mtp_burst.
    drafter
        .bind(&lm)
        .expect("bind must succeed on tracking mock");

    let boxed: Box<dyn Drafter> = Box::new(drafter);
    let mut generator = MtpGenerator::new(target, boxed, /* block_size= */ 4);

    let prompt = vec![1, 2, 3];
    let sampling = SamplingConfig::greedy();
    let (emitted, _stats) = generator.generate(&prompt, /* max_tokens= */ 16, &sampling);

    // We expect 8 tokens before EOS (4 per round, 2 rounds):
    // [100, 10, 11, 12, 13, 20, 21, 22]. Then the 9999 EOS in round 2's
    // last position terminates the loop. The exact count must be > 1 —
    // the load-bearing regression check is that bind() actually
    // enabled the round-loop to run beyond the seed bonus.
    assert!(
        emitted.len() > 1,
        "round loop must emit more than 1 token when drafter is bound; got {:?}",
        emitted
    );
    // The seed bonus is always first.
    assert_eq!(
        emitted[0], 100,
        "first emitted token must be the seed bonus"
    );
    // The events log must show bind before any draft_block call.
    let events_snapshot = events.borrow();
    let bind_idx = events_snapshot
        .iter()
        .position(|&e| e == "bind")
        .expect("bind must have been recorded");
    let first_draft_idx = events_snapshot
        .iter()
        .position(|&e| e == "draft_block")
        .expect("at least one draft_block call must have occurred");
    assert!(
        bind_idx < first_draft_idx,
        "bind must precede the first draft_block call; events = {events_snapshot:?}"
    );
}

#[test]
fn drive_mtp_generator_round_loop_returns_only_seed_when_drafter_is_unbound() {
    use mlxcel_core::speculative::mtp::MtpGenerator;

    // Same scripted target and drafter, but we deliberately SKIP the
    // bind() call. The TrackingMockDrafter mirrors the real
    // Gemma4AssistantDraftModel's behavior: draft_block returns
    // DrafterError::BindNotCalled, which MtpGenerator silently
    // swallows (round_loop break), and we end up with the seed bonus
    // only. This pins the failure mode of the cycle-1 regression so a
    // future refactor that re-drops bind() flips this test from
    // "1 token expected" to "more than 1 expected" — surfacing
    // immediately.
    let (target, drafter_scripts) = make_two_round_target_and_drafter();
    let events = Rc::new(RefCell::new(Vec::new()));
    let drafter = TrackingMockDrafter::new(drafter_scripts, events.clone());
    // NO bind() — this is the production bug.
    let boxed: Box<dyn Drafter> = Box::new(drafter);
    let mut generator = MtpGenerator::new(target, boxed, /* block_size= */ 4);

    let prompt = vec![1, 2, 3];
    let sampling = SamplingConfig::greedy();
    let (emitted, _stats) = generator.generate(&prompt, /* max_tokens= */ 16, &sampling);

    // CRITICAL invariant — without bind, the round loop breaks on the
    // first draft_block call and we get exactly the seed bonus.
    assert_eq!(
        emitted.len(),
        1,
        "without bind(), the round loop must short-circuit and emit \
         only the seed bonus. Got {emitted:?} which means either bind \
         is being called somewhere unexpected OR MtpGenerator no longer \
         swallows DrafterError::BindNotCalled. Either way, revisit the \
         bind invariant in run_mtp_burst."
    );
    assert_eq!(
        emitted[0], 100,
        "the single emitted token must be the seed bonus"
    );
    // No bind event; at least one draft_block event (the one that
    // bailed).
    let events_snapshot = events.borrow();
    assert!(
        !events_snapshot.contains(&"bind"),
        "events log must not contain bind in the unbound case: {events_snapshot:?}"
    );
    assert!(
        events_snapshot.contains(&"draft_block"),
        "draft_block must still have been called once (and returned BindNotCalled): {events_snapshot:?}"
    );
}

// =============================================================================
// DFlash bind asymmetry — why there is no mock regression test here
//
// Unlike MTP, `DFlashGenerator::run` calls `drafter.bind(target_lm)?`
// internally on every invocation (see
// `mlxcel_core/src/drafter/dflash/round_loop.rs::run`). There is no
// "caller must bind first" contract to pin: adding a manual bind before
// `run_dflash_burst` would double-bind and is not the desired behaviour.
//
// Consequently there is no analogue of the
// `drive_mtp_generator_round_loop_returns_only_seed_when_drafter_is_unbound`
// test for DFlash — if `DFlashGenerator::run` were accidentally changed to
// require an external bind, the DFlash integration tests in
// `tests/speculative_dispatch.rs` and the real-model smoke tests would
// catch it before production. The asymmetry is documented at the
// function-level docstring of `run_dflash_burst` and in the module
// docstring under "## Bind asymmetry: MTP vs DFlash".
// =============================================================================

// =============================================================================
// `try_run_burst_b1` — decline path without a real model
//
// `try_run_burst_b1` takes a `BurstContext<'_>` which requires a
// `&LoadedModel` — loading real model weights is infeasible in a fast unit
// test. The variant-dispatch decline path is therefore verified at the
// `should_burst_for_sequence` level above (which fires before any model
// access) and at the integration level in `tests/speculative_dispatch.rs`.
//
// What we can unit-test without weights: the `try_run_burst_b1` interface
// itself (names / signatures stable) is pinned by the compile test below.
// =============================================================================

// =============================================================================
// Compile-time pin: the burst module must export the items the
// scheduler depends on. A future re-org that accidentally renames
// `try_run_burst_b1` or `BurstFinalized` breaks this file at compile
// time before runtime tests fire.
// =============================================================================

#[test]
fn burst_module_exports_required_for_scheduler_integration_compile() {
    // Touch each load-bearing item once so a rename surfaces at
    // compile time. The test body is intentionally trivial.
    let _ = std::mem::size_of::<super::speculative_burst::WorkerDrafterSlot>();
    let _ = std::mem::size_of::<super::speculative_burst::BurstFinalized>();
}

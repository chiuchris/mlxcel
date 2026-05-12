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

//! DFlash speculative-decoding round-loop driver (B=1).
//!
//! Rust port of upstream
//! `references/mlx-vlm/mlx_vlm/generate.py::_dflash_rounds`
//! (`mlx-vlm` HEAD, lines 856–942). Issue #636 / epic #633 sub-12.
//!
//! ## What this module does
//!
//! Once the caller has prefilled the target's KV / GDN caches and sampled
//! the first bonus token from the target's last logits, this module runs
//! the **draft → verify → walk → rollback** round loop:
//!
//! 1. The drafter ([`crate::drafter::Drafter`] trait, `DrafterKind::Dflash`)
//!    produces `block_size - 1` proposal tokens in a single masked forward
//!    that takes the current bonus token plus a `[1, hidden_dim]` row
//!    sliced from the target's multi-layer captured hidden concatenation.
//! 2. The target ([`SpeculativeTarget`] trait) verifies the candidate
//!    block in a single batched forward, returning per-position logits,
//!    fresh per-layer captured hidden states, and GDN rollback snapshots
//!    (Qwen 3.5 hybrid linear-attention state).
//! 3. The walk accepts the longest prefix where the target's greedy
//!    argmax matches the drafter's proposal, plus one **bonus token** at
//!    the divergence position from the target.
//! 4. On partial acceptance, the target's `rollback_speculative_cache`
//!    trims both KV caches (attention layers) AND replays the GDN state
//!    to the accepted position (linear-attention layers). This dual
//!    rollback is what distinguishes DFlash from the Gemma 4 MTP path
//!    (sub-6 / #629) and from the classic `SpeculativeGenerator`.
//!
//! ## Scope
//!
//! **B=1 only.** The batched DFlash round loop (B>1) lands in sub-13
//! (#637); this module deliberately omits the per-row tail-zeroing and
//! continuous-batching plumbing that the upstream
//! `_dflash_rounds_batch` carries.
//!
//! ## Why a target trait?
//!
//! The concrete target type for DFlash today is `Qwen35Model` /
//! `Qwen35VLModel`, which live in the **binary** crate (`src/models/`,
//! `src/vision/`) and therefore cannot be named from `mlxcel-core`. The
//! [`SpeculativeTarget`] trait defined here captures exactly the four
//! hooks the round loop needs (verify forward, rollback, accepted-prefix
//! hidden slice, last-token sampling), with associated types for the
//! target's cache slice (`Cache`) and verify-output (`VerifyOut`). The
//! binary side implements this trait once on `Qwen35Model` (#634 lands
//! the `forward_speculative` / `rollback_speculative_cache` methods that
//! the impl wraps) and the round loop becomes target-agnostic.
//!
//! ## Greedy parity gate
//!
//! At `temperature = 0.0`, the round loop is byte-identical to a
//! drafter-less greedy generator: the target's greedy argmax determines
//! both the accept threshold and the bonus token, so every token the
//! loop emits is exactly what the target would have emitted on its own.
//! This is pinned by the synthetic-target unit test
//! [`tests::greedy_parity_with_synthetic_target_for_thirty_two_tokens`].

use crate::drafter::{Drafter, DrafterError};
use crate::ffi::{self, MlxArray};
use crate::generate::{GenerationStats, SamplingConfig};
use cxx::UniquePtr;
use std::time::{Duration, Instant};

/// Default DFlash draft-block length. Matches the upstream
/// `Qwen3.5-4B-DFlash` checkpoint's `config.json::block_size`.
///
/// Used as the fallback when the caller does not override
/// `DFlashGenerator::block_size`.
pub const DEFAULT_BLOCK_SIZE: u32 = 16;

/// Default mask-token id placed in the drafter's masked forward block at
/// positions `[1, block_size)`. Mirrors upstream
/// `DFlashConfig::mask_token_id = 248_070`, which lives in the Qwen 3.5
/// tokenizer's reserved-token range.
///
/// **Not used by the round loop directly** — the drafter
/// ([`crate::drafter::dflash::DFlashDrafter::draft_block`]) consumes
/// `config.mask_token_id` from its own loaded `DFlashConfig`. This
/// constant exists so binary-side CLI plumbing can plumb an override
/// through the same numeric path the round loop would honor on a future
/// `--draft-mask-id` flag (sub-7 / #630).
pub const DEFAULT_MASK_TOKEN_ID: i32 = 248_070;

/// Trait the round loop calls into for the target side of speculative
/// decoding.
///
/// The trait deliberately exposes only the four hooks
/// [`_dflash_rounds`](https://github.com/Blaizzy/mlx-vlm) needs:
///
/// - [`verify_forward`](Self::verify_forward) — one batched forward over
///   `[bonus, d_0, ..., d_{K-1}]` returning per-position logits + the
///   captured multi-layer hiddens + the GDN rollback snapshot.
/// - [`rollback_partial`](Self::rollback_partial) — invoked only when
///   `accepted < block_size - 1`. Trims both KV (attention) and GDN
///   (linear-attention) state to the accepted prefix.
/// - [`concat_hidden_for_drafter`](Self::concat_hidden_for_drafter) —
///   builds the drafter's per-step input by concatenating the captured
///   hidden states from `target_layer_ids` along the feature axis and
///   slicing to row `accepted` (the divergence position).
/// - [`capture_layer_ids`](Self::capture_layer_ids) — returns the
///   target-side layer indices the drafter wants captured. The drafter
///   carries this list in its config (`target_layer_ids`); the target
///   surfaces it back to the round loop so the verify call passes the
///   right indices.
///
/// ## Associated types
///
/// - `Cache` — the target's heterogeneous cache slice element type. For
///   Qwen 3.5 this is `Qwen3NextCache` (attention KV + linear-attention
///   GDN state). The round loop holds `&mut [Self::Cache]` and never
///   inspects its variants directly.
/// - `VerifyOut` — the target's verify-output bundle (logits + captured
///   hidden states + GDN rollback snapshots). For Qwen 3.5 this is
///   `crate::models::qwen3_5::VerifyOutput` from the binary side.
///
/// ## Implementations
///
/// The intended impl lives on `Qwen35Model` / `Qwen35VLModel` in the
/// binary crate. Tests in this module use a small in-crate
/// `SyntheticTarget` (see `tests` submodule) so the round-loop logic can
/// be pinned without dragging the FFI-backed target into the test crate.
pub trait SpeculativeTarget {
    /// The target's per-layer cache slice element type. Heterogeneous
    /// for Qwen 3.5 (attention KV + linear-attention GDN); the trait
    /// stays parametric so the round loop is target-agnostic.
    type Cache;

    /// The target's verify-pass output. Carries the logits the round
    /// loop samples from, the captured per-layer hidden states the
    /// drafter consumes, and the GDN rollback snapshots the rollback
    /// path requires.
    type VerifyOut;

    /// Indices of the target layers whose post-block hidden states the
    /// drafter wants captured. Typically `[1, 8, 15, 22, 29]` for the
    /// Qwen 3.5 4B DFlash checkpoint.
    fn capture_layer_ids(&self) -> &[usize];

    /// Run one verify forward over the candidate block.
    ///
    /// Input `verify_input` has shape `[1, block_size]` and carries
    /// `[bonus, d_0, ..., d_{block_size - 2}]`. The implementation MUST
    /// capture per-layer hidden states for the layers returned by
    /// [`Self::capture_layer_ids`], capture GDN rollback snapshots for
    /// every linear-attention layer, and advance the target's caches
    /// in `caches` over all `block_size` tokens.
    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut;

    /// Rewind the target's caches to the accepted-prefix position.
    ///
    /// Called only when `accepted < block_size - 1`. The implementation
    /// MUST:
    ///
    /// 1. Trim the attention-layer KV caches by `block_size - (accepted + 1)`.
    /// 2. Replay each linear-attention layer's GDN state over the first
    ///    `accepted + 1` positions of the verify-block, starting from
    ///    the snapshot captured during [`Self::verify_forward`].
    ///
    /// For Qwen 3.5 this delegates to
    /// `Qwen35Model::rollback_speculative_cache(caches, verify_out.gdn_states, &[accepted], block_size)`.
    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    );

    /// Rewind the target's caches to per-row accepted-prefix positions
    /// (B > 1 path).
    ///
    /// Mirrors `rollback_partial` but accepts a `&[i32]` of length `B`
    /// where `accepted[r]` is the per-row accept count for row `r`. The
    /// implementation MUST:
    ///
    /// 1. Trim the attention-layer KV caches by
    ///    `block_size - (max(accepted) + 1)` (the global trim amount).
    /// 2. Per-row tail-zero each attention KV row whose `accepted[r] <
    ///    max(accepted)` (so its K/V positions past `accepted[r] + 1`
    ///    are zeroed in that row only — sibling rows keep their tails).
    /// 3. Replay each linear-attention layer's GDN state per row, using
    ///    the per-row `accepted[r]` to determine the replay length.
    ///
    /// For Qwen 3.5 this delegates directly to
    /// `Qwen35Model::rollback_speculative_cache(caches, verify_out.gdn_states, accepted, block_size)`,
    /// which already supports a per-row `accepted` slice (issue #634).
    ///
    /// The default implementation delegates to [`Self::rollback_partial`]
    /// when `accepted.len() == 1` so single-row targets don't need a
    /// separate impl. For B > 1, every implementation MUST override.
    fn rollback_partial_batched(
        &self,
        caches: &mut [Self::Cache],
        verify_out: &Self::VerifyOut,
        accepted: &[i32],
        block_size: i32,
    ) {
        debug_assert!(
            !accepted.is_empty(),
            "rollback_partial_batched must be called with B >= 1"
        );
        if accepted.len() == 1 {
            self.rollback_partial(caches, verify_out, accepted[0], block_size);
        } else {
            panic!(
                "rollback_partial_batched: target must override for B > 1 (got B = {})",
                accepted.len()
            );
        }
    }

    /// Build the drafter's per-round hidden input by concatenating the
    /// captured per-layer hidden states along the feature axis.
    ///
    /// Returned tensor has shape `[B, block_size, num_capture_layers * hidden_size]`.
    /// For B = 1 (and the B = 1 round loop) this is `[1, bs, dim]`.
    /// For the Qwen 3.5 4B DFlash drafter with
    /// `target_layer_ids = [1, 8, 15, 22, 29]` and `hidden_size = 2560`,
    /// the full tensor is `[B, bs, 12800]`.
    ///
    /// The round loop slices this tensor to `[B, max(accepted) + 1, dim]`
    /// on partial-accept rounds (mirroring upstream
    /// `hidden = hidden[:, :max_a + 1, :]`); on full-accept rounds the
    /// loop forwards the full block. The target trait stays simple and
    /// the slice logic lives in one place.
    fn concat_hidden_for_drafter(
        &self,
        verify_out: &Self::VerifyOut,
    ) -> UniquePtr<MlxArray>;

    /// Read the per-position logits out of `verify_out` for use by the
    /// round loop's argmax pass. Returned tensor has shape
    /// `[B, block_size, vocab]`. For the B = 1 path this is
    /// `[1, block_size, vocab]`.
    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray;
}

/// B=1 DFlash speculative-decoding round-loop driver.
///
/// Construction owns the boxed drafter and the runtime parameters; the
/// target is borrowed at call time via [`Self::run`]. This split keeps
/// the generator reusable across multiple prompts (the drafter cache
/// resets between runs via [`Drafter::reset`]) while still letting the
/// caller hold a borrow of the target.
///
/// See the module docstring for the algorithm and the
/// [`SpeculativeTarget`] trait for the target-side contract.
pub struct DFlashGenerator {
    drafter: Box<dyn Drafter>,
    sampler: SamplingConfig,
    block_size: u32,
    /// Mask-token id forwarded for documentation. The drafter consumes
    /// its own `DFlashConfig::mask_token_id` directly; this field
    /// exists so a future CLI override can be plumbed through a single
    /// numeric path without touching the drafter signature. Stored here
    /// because a `--draft-mask-id` flag (planned in sub-7 / #630) would
    /// configure the generator, not the drafter, to keep the drafter
    /// itself stateless wrt round-loop runtime options.
    #[allow(dead_code)]
    mask_token_id: i32,
    /// Tokens emitted across the round-loop body. Kept on the generator
    /// so the caller can inspect the accept-lens / token sequence after
    /// a run (for tests and benchmark instrumentation).
    generated_tokens: Vec<i32>,
    /// Per-round accept counts (in `0..=block_size - 1`). Index `i` is
    /// the accept count of round `i + 1` (round 0 is the prefill-side
    /// first-bonus emission, which has no walk). Used by tests to pin
    /// the upstream `accept_lens` parity.
    accept_lens: Vec<u32>,
}

impl DFlashGenerator {
    /// Construct a new generator with the given drafter and sampler.
    ///
    /// `block_size` is the draft-block length the round loop will request
    /// from the drafter every round. Defaults to [`DEFAULT_BLOCK_SIZE`]
    /// when constructing with [`Self::with_drafter`].
    ///
    /// `mask_token_id` is informational on the generator side (the
    /// drafter uses its own `DFlashConfig::mask_token_id` internally);
    /// stored so a future `--draft-mask-id` CLI flag can wire through
    /// without reshaping the drafter signature.
    pub fn new(
        drafter: Box<dyn Drafter>,
        sampler: SamplingConfig,
        block_size: u32,
        mask_token_id: i32,
    ) -> Self {
        Self {
            drafter,
            sampler,
            block_size,
            mask_token_id,
            generated_tokens: Vec::new(),
            accept_lens: Vec::new(),
        }
    }

    /// Convenience constructor that uses the [`DEFAULT_BLOCK_SIZE`] and
    /// [`DEFAULT_MASK_TOKEN_ID`] defaults.
    pub fn with_drafter(drafter: Box<dyn Drafter>, sampler: SamplingConfig) -> Self {
        Self::new(drafter, sampler, DEFAULT_BLOCK_SIZE, DEFAULT_MASK_TOKEN_ID)
    }

    /// Tokens emitted by the round loop. Excludes the first bonus token
    /// the caller passes into [`Self::run`] (that token is the caller's
    /// own and the caller is expected to have already emitted it).
    pub fn tokens(&self) -> &[i32] {
        &self.generated_tokens
    }

    /// Per-round accept counts. `accept_lens()[i]` is the count for the
    /// `i`-th round the loop executed. Pinned by tests.
    pub fn accept_lens(&self) -> &[u32] {
        &self.accept_lens
    }

    /// Drafter handle for tests that need to inspect or fault-inject
    /// drafter state. Production callers typically need only
    /// [`Self::run`].
    pub fn drafter(&self) -> &dyn Drafter {
        self.drafter.as_ref()
    }

    /// Drafter mutable handle for tests / advanced callers.
    pub fn drafter_mut(&mut self) -> &mut dyn Drafter {
        self.drafter.as_mut()
    }

    /// Resets per-run state on the generator (token buffer + accept-len
    /// history). Does NOT reset the drafter — that is the caller's
    /// responsibility via [`Drafter::reset`] because reset needs a
    /// concrete `&dyn LanguageModel` target the round loop does not
    /// itself hold.
    fn reset_run_state(&mut self) {
        self.generated_tokens.clear();
        self.accept_lens.clear();
    }
}

/// Greedy speculative walk for a single batch.
///
/// Rust port of upstream
/// `references/mlx-vlm/mlx_vlm/generate.py::_speculative_walk`.
///
/// Given `draft_tokens` (the drafter's `K-1` proposals) and `target_tokens`
/// (the target's per-position argmax across all `K` verify positions),
/// accepts the longest prefix where the target's greedy choice matches
/// the drafter's proposal, then takes the target's choice at the
/// divergence position as a **bonus token**.
///
/// Returns `(accepted, new_tokens)` where:
/// - `accepted` ∈ `[0, draft_tokens.len()]` is the prefix length.
/// - `new_tokens` = `draft_tokens[..accepted] + [target_tokens[accepted]]`,
///   truncated to `budget`. Length is `min(accepted + 1, budget)`.
///
/// **At `temperature = 0.0`**, this is the greedy-parity gate: every
/// token emitted by the loop is the target's own argmax extension of
/// the previously emitted prefix, so the loop is byte-identical to a
/// drafter-less generator.
pub(crate) fn speculative_walk(
    draft_tokens: &[i32],
    target_tokens: &[i32],
    budget: usize,
) -> (usize, Vec<i32>) {
    // Compare position-by-position up to min(draft, target_excluding_last).
    // The target tensor has one more position than the drafter's proposals
    // (the trailing position is the post-block argmax, which the walk
    // takes as the bonus token only if the entire prefix matched).
    let n = draft_tokens.len().min(target_tokens.len().saturating_sub(0));
    let mut accepted = 0;
    while accepted < n {
        // Guard against target_tokens being shorter than expected (degenerate
        // case the synthetic target tests can trip — the FFI-backed verify
        // forward will always return block_size tokens).
        if accepted >= target_tokens.len() {
            break;
        }
        if draft_tokens[accepted] != target_tokens[accepted] {
            break;
        }
        accepted += 1;
    }

    // The bonus token is the target's choice at the divergence position
    // (or at the end of the block if every proposal was accepted).
    // Mirrors upstream `t[accepted]` indexing.
    let bonus_idx = accepted.min(target_tokens.len().saturating_sub(1));
    let mut new_tokens: Vec<i32> = draft_tokens[..accepted].to_vec();
    if bonus_idx < target_tokens.len() {
        new_tokens.push(target_tokens[bonus_idx]);
    }

    // Truncate to the caller's budget.
    if new_tokens.len() > budget {
        new_tokens.truncate(budget);
    }
    (accepted, new_tokens)
}

/// Round-loop run output: emitted tokens (excludes the caller's first
/// bonus) plus generation timing statistics.
pub struct DFlashRunOutput {
    /// Tokens emitted by the round loop, not including the first bonus
    /// token the caller passed into [`DFlashGenerator::run`].
    pub tokens: Vec<i32>,
    /// Per-round accept counts for diagnostic logging and tests.
    pub accept_lens: Vec<u32>,
    /// Wall-clock decode statistics. Prefill stats are the caller's
    /// responsibility (the loop runs after prefill).
    pub stats: GenerationStats,
}

impl DFlashGenerator {
    /// Run the round loop.
    ///
    /// # Arguments
    ///
    /// - `target` — implements [`SpeculativeTarget`] over the target
    ///   model's cache type.
    /// - `caches` — the target's per-layer cache slice, already filled
    ///   by the caller's prefill. The round loop will advance and
    ///   selectively rewind these caches.
    /// - `first_bonus` — the first bonus token sampled from the
    ///   target's last logits after prefill. The round loop does NOT
    ///   emit this token (the caller is expected to have already
    ///   delivered it to whatever stream consumes generated tokens);
    ///   it is used only to seed the first verify-input.
    /// - `first_hidden` — the multi-layer hidden concatenation at the
    ///   last position of the prefill (shape
    ///   `[1, 1, num_capture_layers * hidden_size]`). The caller is
    ///   responsible for building this from the target's prefill
    ///   `forward_speculative` output; the round loop only consumes it.
    /// - `eos_token_ids` — token ids that should stop generation. The
    ///   loop stops on the first emitted token that matches any id in
    ///   this slice. Empty slice = no EOS stop (the loop runs to
    ///   `max_tokens`).
    /// - `max_tokens` — generation budget INCLUDING the first bonus
    ///   token. The loop emits at most `max_tokens - 1` further tokens.
    ///
    /// # Returns
    ///
    /// A [`DFlashRunOutput`] carrying the emitted tokens (excludes
    /// `first_bonus`) and per-round accept counts.
    ///
    /// # Errors
    ///
    /// Propagates [`DrafterError`] from the drafter's `draft_block` /
    /// `reset` / `bind` calls. Other failure modes are not currently
    /// surfaced because the target trait's methods are infallible by
    /// design (failures inside `verify_forward` panic, matching the
    /// rest of mlxcel-core's FFI-backed model paths).
    // Each argument here represents a distinct piece of generation
    // state (target, target-LM, caches, first bonus, first hidden,
    // EOS, budget) that cannot be folded together without losing
    // clarity at the only call site (the binary's `generate` command).
    // The classic `SpeculativeGenerator::generate` similarly carries
    // 6 args; an 8-arg DFlash-side `run` is the same shape with the
    // added `target_lm` + `first_hidden` that DFlash specifically
    // needs.
    #[allow(clippy::too_many_arguments)]
    pub fn run<T: SpeculativeTarget>(
        &mut self,
        target: &T,
        target_lm: &dyn crate::generate::LanguageModel,
        caches: &mut [T::Cache],
        first_bonus: i32,
        first_hidden: UniquePtr<MlxArray>,
        eos_token_ids: &[i32],
        max_tokens: usize,
    ) -> Result<DFlashRunOutput, DrafterError> {
        self.reset_run_state();

        // Bind + reset the drafter against the target. `bind` is a
        // capability smoke-test (target must expose embed_tokens);
        // `reset` clears the drafter's own KV cache between runs.
        self.drafter.bind(target_lm)?;
        self.drafter.reset(target_lm)?;

        if max_tokens <= 1 {
            // First bonus is the caller's own; we have no further work.
            return Ok(DFlashRunOutput {
                tokens: Vec::new(),
                accept_lens: Vec::new(),
                stats: build_zero_stats(),
            });
        }

        let decode_start = Instant::now();

        let block_size_cfg = self.block_size as usize;
        let mut bonus = first_bonus;
        let mut hidden: UniquePtr<MlxArray> = first_hidden;
        // `emitted` counts ALL tokens the caller will see, including
        // `first_bonus`. The round-loop body has already advanced
        // `emitted` from 0 → 1 conceptually (the first bonus is
        // emitted before we enter the loop).
        let mut emitted: usize = 1;

        loop {
            if emitted >= max_tokens {
                break;
            }
            // Upstream: bs = min(block_total, max_tokens - emitted + 1)
            let remaining_plus_one = max_tokens - emitted + 1;
            let bs = block_size_cfg.min(remaining_plus_one);
            if bs <= 1 {
                break;
            }

            // ---- Draft ----
            let draft_tokens = self.drafter.draft_block(
                bonus,
                Some(hidden.as_ref().expect("hidden must be Some")),
                bs,
                &self.sampler,
            )?;
            // `draft_tokens.len()` is `bs - 1` for the DFlash drafter.
            debug_assert_eq!(
                draft_tokens.len(),
                bs - 1,
                "DFlash drafter must produce bs - 1 proposals"
            );

            // ---- Verify ----
            // Build the verify input `[bonus, d_0, ..., d_{bs - 2}]`.
            let mut verify_tokens: Vec<i32> = Vec::with_capacity(bs);
            verify_tokens.push(bonus);
            verify_tokens.extend_from_slice(&draft_tokens);
            let verify_input = ffi::from_slice_i32(&verify_tokens, &[1, bs as i32]);
            let verify_out = target.verify_forward(&verify_input, caches);

            // ---- Argmax sample of target's per-position logits ----
            // Greedy at temp=0.0 / top_k=1 — the only mode this sub-issue
            // gates parity on. Stochastic sampling for DFlash is sub-9 / #632.
            let target_tokens =
                argmax_logits_to_vec(target.verify_logits(&verify_out), bs as i32);

            // ---- Walk ----
            let budget = max_tokens.saturating_sub(emitted);
            let (accepted, new_tokens) =
                speculative_walk(&draft_tokens, &target_tokens, budget);
            self.accept_lens.push(accepted as u32);

            // ---- Emit ----
            // Track EOS in the same loop body so we can stop early
            // exactly the way the upstream `for tok in new_tokens` yields
            // tokens one at a time.
            let mut hit_eos = false;
            for tok in &new_tokens {
                self.generated_tokens.push(*tok);
                emitted += 1;
                if eos_token_ids.contains(tok) {
                    hit_eos = true;
                    break;
                }
                if emitted >= max_tokens {
                    break;
                }
            }
            if hit_eos {
                break;
            }

            // ---- Update bonus + next-round hidden ----
            //
            // Upstream:
            //   if accepted < bs - 1:
            //       hidden = hidden[:, : accepted + 1, :]
            //   b = new_tokens[-1] if new_tokens else b
            //
            // Whether `accepted < bs - 1` or not, the next round's
            // drafter input is the captured hidden row at position
            // `accepted`. We materialize that row via
            // `concat_hidden_for_drafter` so the round loop never has to
            // know the per-layer hidden topology.
            if let Some(last) = new_tokens.last() {
                bonus = *last;
            }

            if emitted >= max_tokens {
                break;
            }

            // Build the next round's hidden input. Upstream:
            //   hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
            //   if accepted < bs - 1:
            //       hidden = hidden[:, : accepted + 1, :]
            //
            // Full-accept keeps the [1, bs, dim] tensor; partial-accept
            // slices to [1, accepted+1, dim] so the drafter's cross-
            // attention only sees the validated context.
            let full_hidden = target.concat_hidden_for_drafter(&verify_out);
            hidden = if (accepted as i32) < (bs as i32) - 1 {
                let full_shape = ffi::array_shape(&full_hidden);
                debug_assert_eq!(
                    full_shape.len(),
                    3,
                    "concat_hidden_for_drafter must return a 3D [1, bs, dim] tensor"
                );
                ffi::slice(
                    &full_hidden,
                    &[0, 0, 0],
                    &[full_shape[0], accepted as i32 + 1, full_shape[2]],
                )
            } else {
                full_hidden
            };

            // ---- Rollback (only on partial acceptance) ----
            //
            // The verify forward advanced the target's caches by `bs`
            // tokens. We accepted `accepted` drafter proposals plus one
            // bonus → keep `accepted + 1` positions; the remaining
            // `bs - (accepted + 1)` cache positions must be rolled back.
            //
            // For Qwen 3.5 (hybrid Mamba+Transformer), rollback is
            // dual: KV trim for attention layers + GDN state replay for
            // linear-attention layers. The target trait method handles
            // both.
            if accepted < bs - 1 {
                target.rollback_partial(caches, &verify_out, accepted as i32, bs as i32);
            }

            // Periodic memory cache clear, mirroring
            // `if emitted % 256 == 0: mx.clear_cache()`. Bound by 256
            // tokens like the upstream loop.
            if emitted.is_multiple_of(256) {
                ffi::clear_memory_cache();
            }
        }

        let stats = GenerationStats {
            prompt_tokens: 0,
            generated_tokens: self.generated_tokens.len(),
            prefill_time_ms: 0.0,
            decode_time_ms: decode_start.elapsed().as_secs_f64() * 1000.0,
            prefill_tok_per_sec: 0.0,
            decode_tok_per_sec: tokens_per_second(self.generated_tokens.len(), decode_start.elapsed()),
        };

        Ok(DFlashRunOutput {
            tokens: std::mem::take(&mut self.generated_tokens),
            accept_lens: std::mem::take(&mut self.accept_lens),
            stats,
        })
    }
}

/// Per-position argmax over `logits` of shape `[1, seq_len, vocab]`,
/// materialized to a `Vec<i32>` of length `seq_len`.
///
/// Equivalent to upstream `sampler(verify_out.logits)` with the greedy
/// `sampler = argmax(axis=-1)`. Stochastic samplers are out of scope
/// for this sub-issue (sub-9 / #632 covers stochastic DFlash parity).
fn argmax_logits_to_vec(logits: &MlxArray, seq_len: i32) -> Vec<i32> {
    let shape = ffi::array_shape(logits);
    debug_assert!(shape.len() == 3, "expected [1, seq_len, vocab] logits");
    debug_assert_eq!(shape[1], seq_len, "logits seq dim must match block_size");

    let argmax = ffi::argmax_last_axis(logits);
    ffi::eval(&argmax);

    // `argmax_last_axis` reduces over the trailing axis, producing
    // `[1, seq_len]`. Materialize each entry via per-position scalar
    // extraction (a single eval was already done above so the data is
    // resident).
    let mut out: Vec<i32> = Vec::with_capacity(seq_len as usize);
    for s in 0..seq_len {
        let cell = ffi::slice(&argmax, &[0, s], &[1, s + 1]);
        // `item_i32` requires a scalar — `cell` has shape `[1, 1]` so
        // reshape it to `[]` (i.e. `&[]`-shaped scalar).
        let scalar = ffi::reshape(&cell, &[]);
        out.push(ffi::item_i32(&scalar));
    }
    out
}

/// Zero generation stats, used when [`DFlashGenerator::run`] short-circuits
/// (e.g. `max_tokens <= 1`).
fn build_zero_stats() -> GenerationStats {
    GenerationStats {
        prompt_tokens: 0,
        generated_tokens: 0,
        prefill_time_ms: 0.0,
        decode_time_ms: 0.0,
        prefill_tok_per_sec: 0.0,
        decode_tok_per_sec: 0.0,
    }
}

/// Compute decode throughput in tokens/sec, guarding against
/// divide-by-zero when the loop runs for less than a measurable
/// interval.
fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 {
        tokens as f64 / secs
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drafter::{DrafterKind, SharedKv};
    use crate::generate::LanguageModel;
    use crate::layers::KVCache;
    use crate::weights::WeightMap;
    use std::cell::Cell;
    use std::rc::Rc;

    /// Sanity: walk accepts the full prefix when drafter matches target
    /// position-by-position, and the bonus is the target's choice at
    /// position `accepted = K - 1` (i.e. one past the last proposal).
    #[test]
    fn speculative_walk_accepts_full_prefix() {
        let draft = vec![10, 20, 30];
        // Target has one more position than draft (the verify pass
        // outputs K logit positions over K input tokens).
        let target = vec![10, 20, 30, 99];
        let (accepted, new_tokens) = speculative_walk(&draft, &target, 100);
        assert_eq!(accepted, 3, "all proposals match");
        assert_eq!(new_tokens, vec![10, 20, 30, 99]);
    }

    /// Walk truncates `new_tokens` to `budget`. This pins the
    /// `(d[:acc] + [t[acc]])[:budget]` upstream slice semantics.
    #[test]
    fn speculative_walk_truncates_to_budget() {
        let draft = vec![10, 20, 30];
        let target = vec![10, 20, 30, 99];
        let (_, new_tokens) = speculative_walk(&draft, &target, 2);
        assert_eq!(new_tokens.len(), 2, "must respect budget");
        assert_eq!(new_tokens, vec![10, 20]);
    }

    /// Walk stops at the first mismatch and takes the target's choice
    /// at that position. This is the load-bearing greedy semantics.
    #[test]
    fn speculative_walk_stops_at_first_mismatch() {
        let draft = vec![10, 20, 30];
        let target = vec![10, 25, 30, 99];
        let (accepted, new_tokens) = speculative_walk(&draft, &target, 100);
        assert_eq!(accepted, 1, "mismatch at position 1");
        // Accepted prefix + bonus from target at divergence position.
        assert_eq!(new_tokens, vec![10, 25]);
    }

    /// Zero acceptance: bonus is the target's choice at position 0.
    #[test]
    fn speculative_walk_zero_accept_returns_single_bonus() {
        let draft = vec![10, 20, 30];
        let target = vec![99, 25, 30, 42];
        let (accepted, new_tokens) = speculative_walk(&draft, &target, 100);
        assert_eq!(accepted, 0);
        assert_eq!(new_tokens, vec![99]);
    }

    /// Empty draft is a degenerate case (block_size == 1) that the round
    /// loop short-circuits on; we still pin walk's behavior for it.
    #[test]
    fn speculative_walk_empty_draft_returns_target_first_token() {
        let draft: Vec<i32> = Vec::new();
        let target = vec![42];
        let (accepted, new_tokens) = speculative_walk(&draft, &target, 100);
        assert_eq!(accepted, 0);
        assert_eq!(new_tokens, vec![42]);
    }

    // ============================================================
    // Synthetic target + drafter test fixtures
    //
    // The fixtures below pin the round-loop control flow without
    // dragging the FFI-backed `Qwen35Model` into the test crate. They
    // are deliberately small (no real attention or matmul) so the
    // tests run on every developer workstation regardless of MLX
    // availability beyond the basic FFI surface mlxcel-core already
    // links against.
    //
    // Greedy-parity invariant (pinned by `greedy_parity_*` tests):
    //   At temp = 0.0, the round loop is byte-identical to a
    //   drafter-less greedy generator over the same target — the
    //   target's argmax determines both the accept threshold and the
    //   bonus token at every position.
    // ============================================================

    /// Synthetic target cache: a single i32 offset that counts how many
    /// tokens have been "advanced" through the cache. Just enough state
    /// to verify the rollback hook trims by the expected amount.
    #[derive(Debug, Default)]
    struct SyntheticCache {
        offset: i32,
    }

    /// Verify-pass output for the synthetic target. Stores the verify
    /// input length so `rollback_partial` can recompute the trim
    /// amount, and the per-position argmax sequence the round loop
    /// will consume.
    struct SyntheticVerifyOut {
        /// Logits of shape `[1, K, vocab]` where K = verify input length.
        logits: UniquePtr<MlxArray>,
        /// The synthetic captured-hidden tensor of shape
        /// `[1, K, concat_hidden_dim]`. The synthetic target builds
        /// this deterministically from the verify input.
        captured_hidden: UniquePtr<MlxArray>,
        /// Verify input length (== block_size for the round loop's call).
        verify_len: i32,
    }

    /// Driver function shape: given a `position` index in the verify
    /// block (0-based: 0 is the bonus, 1..K are the masked proposal
    /// positions), return the target's argmax token at that position.
    /// The round-loop test seeds this so the accept pattern is
    /// deterministic.
    ///
    /// The closure also receives the current bonus token so synthetic
    /// targets can model "argmax depends on previous token" patterns
    /// (which is what a real causal LM does).
    type SyntheticArgmaxFn = dyn Fn(i32, i32) -> i32;

    /// Synthetic target implementing [`SpeculativeTarget`] over the
    /// trivial `SyntheticCache`.
    struct SyntheticTarget {
        capture_layer_ids: Vec<usize>,
        concat_hidden_dim: i32,
        argmax_fn: Box<SyntheticArgmaxFn>,
        /// Recorded rollback events: `(accepted, block_size)` pairs.
        /// Tests inspect this slice to confirm rollback_partial is
        /// called exactly on the partial-acceptance rounds.
        rollback_events: Rc<Cell<Vec<(i32, i32)>>>,
        /// Recorded verify-input lengths. Tests inspect this to confirm
        /// the round loop calls `verify_forward` with the expected `K`.
        verify_call_lens: Rc<Cell<Vec<i32>>>,
        /// Current bonus that was sent into the last verify call. Used
        /// to derive per-position target tokens deterministically.
        last_bonus: Rc<Cell<i32>>,
    }

    impl SyntheticTarget {
        fn new<F: Fn(i32, i32) -> i32 + 'static>(
            capture_layer_ids: Vec<usize>,
            concat_hidden_dim: i32,
            argmax_fn: F,
        ) -> Self {
            Self {
                capture_layer_ids,
                concat_hidden_dim,
                argmax_fn: Box::new(argmax_fn),
                rollback_events: Rc::new(Cell::new(Vec::new())),
                verify_call_lens: Rc::new(Cell::new(Vec::new())),
                last_bonus: Rc::new(Cell::new(0)),
            }
        }

        fn rollback_events(&self) -> Vec<(i32, i32)> {
            let v = self.rollback_events.take();
            self.rollback_events.set(v.clone());
            v
        }

        fn verify_call_lens(&self) -> Vec<i32> {
            let v = self.verify_call_lens.take();
            self.verify_call_lens.set(v.clone());
            v
        }
    }

    impl SpeculativeTarget for SyntheticTarget {
        type Cache = SyntheticCache;
        type VerifyOut = SyntheticVerifyOut;

        fn capture_layer_ids(&self) -> &[usize] {
            &self.capture_layer_ids
        }

        fn verify_forward(
            &self,
            verify_input: &MlxArray,
            caches: &mut [Self::Cache],
        ) -> Self::VerifyOut {
            let shape = ffi::array_shape(verify_input);
            let k = shape[1];
            // Record the call length for test inspection.
            let mut lens = self.verify_call_lens.take();
            lens.push(k);
            self.verify_call_lens.set(lens);

            // Advance the synthetic caches by K positions.
            for c in caches.iter_mut() {
                c.offset += k;
            }

            // Build the target's per-position argmax. We need to know
            // the verify-input token at every position to feed the
            // synthetic `argmax_fn`. Read the input back via per-cell
            // scalar extraction.
            let mut target_tokens: Vec<i32> = Vec::with_capacity(k as usize);
            for s in 0..k {
                let cell = ffi::slice(verify_input, &[0, s], &[1, s + 1]);
                let scalar = ffi::reshape(&cell, &[]);
                target_tokens.push(ffi::item_i32(&scalar));
            }

            // Synthesize argmax tokens position-by-position. Position
            // index `s` is fed the verify-input token at that position
            // as "previous token" context.
            let argmax: Vec<i32> = (0..k as usize)
                .map(|s| (self.argmax_fn)(s as i32, target_tokens[s]))
                .collect();

            // Build a one-hot logits tensor: at position s, the channel
            // index `argmax[s]` is +10.0, all others are -10.0. The
            // vocab is generously sized so synthetic argmax chains and
            // mismatch sentinels both fit without overflow.
            const VOCAB: usize = 1024;
            let mut buf = vec![-10.0f32; k as usize * VOCAB];
            for s in 0..k as usize {
                let id = argmax[s];
                debug_assert!(
                    (0..VOCAB as i32).contains(&id),
                    "synthetic test token out of vocab: got {id}, vocab = {VOCAB}"
                );
                buf[s * VOCAB + id as usize] = 10.0;
            }
            let logits = ffi::from_slice_f32(&buf, &[1, k, VOCAB as i32]);

            // Record the bonus the loop sent (verify position 0).
            self.last_bonus.set(target_tokens[0]);

            // Synthetic captured hidden: a deterministic ramp tensor of
            // shape `[1, K, concat_hidden_dim]` that any later
            // `concat_hidden_for_drafter` slice can reproduce.
            let hidden = ffi::zeros(&[1, k, self.concat_hidden_dim], crate::dtype::FLOAT32);

            SyntheticVerifyOut {
                logits,
                captured_hidden: hidden,
                verify_len: k,
            }
        }

        fn rollback_partial(
            &self,
            caches: &mut [Self::Cache],
            verify_out: &Self::VerifyOut,
            accepted: i32,
            block_size: i32,
        ) {
            // The round loop guarantees `accepted < block_size - 1`
            // when rollback_partial is called. Trim by the same
            // formula as the real target.
            let n = accepted + 1;
            let trim = block_size - n;
            for c in caches.iter_mut() {
                c.offset -= trim;
            }
            // Record the event for test inspection. Drop the verify_out
            // reference; we only use its `verify_len` field as a sanity
            // probe.
            debug_assert_eq!(verify_out.verify_len, block_size);
            let mut ev = self.rollback_events.take();
            ev.push((accepted, block_size));
            self.rollback_events.set(ev);
        }

        fn concat_hidden_for_drafter(
            &self,
            verify_out: &Self::VerifyOut,
        ) -> UniquePtr<MlxArray> {
            // Return the full captured hidden tensor; the round loop
            // does its own axis-1 slice on partial accept.
            ffi::slice(
                &verify_out.captured_hidden,
                &[0, 0, 0],
                &[1, verify_out.verify_len, self.concat_hidden_dim],
            )
        }

        fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
            verify_out.logits.as_ref().expect("logits must be Some")
        }
    }

    /// A `LanguageModel` shim used only by `Drafter::bind`'s
    /// `embed_tokens` capability smoke-test. Forwards a trivial one-hot
    /// embedding so the bind path returns `Ok(())` without exercising
    /// any real attention or matmul.
    struct EmbedOnlyLm;

    impl LanguageModel for EmbedOnlyLm {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::zeros(&[1, 1, 1], crate::dtype::FLOAT32)
        }
        fn make_caches(&self) -> Vec<KVCache> {
            Vec::new()
        }
        fn num_layers(&self) -> usize {
            0
        }
        fn eos_token_ids(&self) -> Vec<i32> {
            Vec::new()
        }
        fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
            let shape = ffi::array_shape(input_ids);
            Some(ffi::zeros(&[shape[0], shape[1], 1], crate::dtype::FLOAT32))
        }
    }

    /// Synthetic drafter that returns a fixed-length proposal sequence
    /// at every `draft_block` call. The proposals are generated by a
    /// caller-supplied closure so each test can pin a specific
    /// accept pattern.
    type SyntheticProposeFn = dyn FnMut(i32, usize) -> Vec<i32>;

    struct SyntheticDrafter {
        propose: Box<SyntheticProposeFn>,
        bind_calls: u32,
        reset_calls: u32,
    }

    impl SyntheticDrafter {
        fn new<F: FnMut(i32, usize) -> Vec<i32> + 'static>(propose: F) -> Self {
            Self {
                propose: Box::new(propose),
                bind_calls: 0,
                reset_calls: 0,
            }
        }
    }

    impl Drafter for SyntheticDrafter {
        fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
            // Trigger the same `embed_tokens` smoke-test the real
            // DFlashDrafter performs.
            let dummy = ffi::from_slice_i32(&[0_i32], &[1, 1]);
            let embedded = target.embed_tokens(&dummy);
            if embedded.is_none() {
                return Err(DrafterError::BindFailed {
                    reason: "embed_tokens missing on test target".to_string(),
                });
            }
            self.bind_calls += 1;
            Ok(())
        }

        fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
            self.bind(target)?;
            self.reset_calls += 1;
            Ok(())
        }

        fn set_shared_kv(
            &mut self,
            _shared_kv: SharedKv<'_>,
            _kv_offset: usize,
            _position: usize,
            _left_padding: usize,
        ) -> Result<(), DrafterError> {
            // DFlash does not use shared_kv.
            Ok(())
        }

        fn draft_block(
            &mut self,
            last_bonus: i32,
            _hidden: Option<&MlxArray>,
            block_size: usize,
            _sampler: &SamplingConfig,
        ) -> Result<Vec<i32>, DrafterError> {
            // DFlash drafter returns `block_size - 1` proposals.
            let proposals = (self.propose)(last_bonus, block_size);
            debug_assert_eq!(
                proposals.len(),
                block_size - 1,
                "synthetic drafter must produce block_size - 1 proposals"
            );
            Ok(proposals)
        }

        fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
            Ok(())
        }

        fn kind(&self) -> DrafterKind {
            DrafterKind::Dflash
        }
    }

    /// Helper: drive the round loop with a synthetic target whose
    /// argmax pattern is a deterministic function of position, and a
    /// synthetic drafter whose proposals partially match the target.
    /// Returns the emitted tokens and per-round accept counts.
    fn run_synthetic_round_loop(
        block_size: u32,
        max_tokens: usize,
        first_bonus: i32,
        argmax_fn: impl Fn(i32, i32) -> i32 + 'static,
        propose_fn: impl FnMut(i32, usize) -> Vec<i32> + 'static,
    ) -> (DFlashRunOutput, Vec<(i32, i32)>, Vec<i32>) {
        let target = SyntheticTarget::new(vec![1, 8, 15, 22, 29], 5 * 8, argmax_fn);
        let mut caches: Vec<SyntheticCache> = (0..3)
            .map(|_| SyntheticCache::default())
            .collect();

        let drafter = SyntheticDrafter::new(propose_fn);
        let lm = EmbedOnlyLm;
        let mut gen = DFlashGenerator::with_drafter(
            Box::new(drafter),
            SamplingConfig::greedy(),
        );
        // Round loop pulls block_size from the generator.
        gen.block_size = block_size;

        // Build the initial hidden as a `[1, 1, concat_hidden_dim]` tensor.
        // The synthetic target's concat_hidden_for_drafter slices the
        // captured-hidden tensor every round; the very first hidden is
        // a free-form ramp the caller supplies (the prefill produced it
        // in the real pipeline).
        let first_hidden = ffi::zeros(&[1, 1, 5 * 8], crate::dtype::FLOAT32);

        let out = gen
            .run(&target, &lm, &mut caches, first_bonus, first_hidden, &[], max_tokens)
            .expect("synthetic round loop must not fail");
        let rollback_events = target.rollback_events();
        let verify_lens = target.verify_call_lens();
        (out, rollback_events, verify_lens)
    }

    // --------------------------------------------------------------
    // Round-loop control-flow tests
    // --------------------------------------------------------------

    /// All proposals match: every round accepts `block_size - 1` and
    /// rollback is NEVER invoked. Pins the "full-accept" hot path.
    ///
    /// Synthetic LM convention: argmax at verify position `s` is "the
    /// next token after the verify-input token at position s". We use
    /// an increment model: `argmax(s, prev) = prev + 1`. Then a perfect
    /// drafter starting from bonus `b` proposes `[b+1, b+2, ..., b+K-1]`
    /// for block_size K, and the target argmax over `[b, b+1, ..., b+K-1]`
    /// produces `[b+1, b+2, ..., b+K]`. Walk: drafter's positions 0..K-1
    /// each match target's positions 0..K-1 → accepted = K-1; the bonus
    /// for the next round is target[K-1] = b+K.
    #[test]
    fn round_loop_full_accept_every_round_skips_rollback() {
        let argmax_fn = |_s: i32, prev_token: i32| prev_token + 1;
        let propose_fn = |bonus: i32, bs: usize| -> Vec<i32> {
            (1..bs as i32).map(|s| bonus + s).collect()
        };

        let (out, rollback_events, verify_lens) = run_synthetic_round_loop(
            8,
            /*max_tokens=*/ 24,
            /*first_bonus=*/ 100,
            argmax_fn,
            propose_fn,
        );

        // Each round must have accepted exactly `block_size - 1 = 7`.
        for (i, acc) in out.accept_lens.iter().enumerate() {
            assert_eq!(
                *acc, 7,
                "round {i} should accept all 7 proposals at full match"
            );
        }
        // No rollback was invoked.
        assert!(
            rollback_events.is_empty(),
            "full-accept must never call rollback_partial, got {rollback_events:?}"
        );
        // Verify-call lengths are all `block_size = 8` (until the loop
        // tightens `bs` to fit the remaining budget at the tail).
        assert!(
            verify_lens.iter().all(|&k| (2..=8).contains(&k)),
            "verify lens must be in [2, 8]; got {verify_lens:?}"
        );
    }

    /// Drafter intentionally mismatches at proposal position 4: every
    /// round must accept exactly 3 (the matching prefix) and call
    /// rollback with `(accepted=3, block_size=8)`. Pins the
    /// partial-acceptance code path and the rollback formula
    /// (`trim = block_size - (accepted + 1)`).
    ///
    /// Convention reminder: drafter's proposal at index `i` (i ∈ [0, K-1))
    /// is compared against target's argmax at verify position `i`.
    /// For the increment model, target argmax at position `i` (with
    /// input `bonus + i`) is `bonus + i + 1`. So a "matching" drafter
    /// proposal at index `i` must be `bonus + i + 1`, which equals
    /// `bonus + (i + 1)`. With bonus=100 and a perfect drafter for
    /// indices 0..3 then mismatches: `[101, 102, 103, 999, 999, 999, 999]`.
    #[test]
    fn round_loop_partial_accept_invokes_rollback_with_correct_args() {
        let argmax_fn = |_s: i32, prev_token: i32| prev_token + 1;
        let propose_fn = |bonus: i32, bs: usize| -> Vec<i32> {
            // Proposals are indexed 0..K-1; proposal i must match
            // bonus + i + 1 to be accepted. Match for i ∈ [0, 3),
            // mismatch from i = 3.
            let mut out = Vec::with_capacity(bs - 1);
            for i in 0..bs - 1 {
                if i < 3 {
                    out.push(bonus + i as i32 + 1);
                } else {
                    out.push(999); // wildly wrong → mismatch
                }
            }
            out
        };

        let (out, rollback_events, _) = run_synthetic_round_loop(
            8,
            /*max_tokens=*/ 32,
            /*first_bonus=*/ 100,
            argmax_fn,
            propose_fn,
        );

        for (i, acc) in out.accept_lens.iter().enumerate() {
            assert_eq!(
                *acc, 3,
                "round {i} should accept the 3-token matching prefix"
            );
        }
        // Rollback was invoked with (accepted=3, block_size=8) every round.
        assert!(
            !rollback_events.is_empty(),
            "partial accept must invoke rollback_partial"
        );
        for ev in &rollback_events {
            assert_eq!(*ev, (3, 8), "rollback args must be (3, 8); got {ev:?}");
        }
    }

    /// Pins the expected accept-len progression for 3 rounds at
    /// block_size = 8 (the issue #636 acceptance criterion).
    #[test]
    fn round_loop_accept_lens_progress_pins_three_rounds_at_block_size_8() {
        // Round 1: drafter matches the first 1 position (accept 1).
        // Round 2: drafter matches the first 4 positions (accept 4).
        // Round 3: drafter matches all positions (accept 7).
        let round = Rc::new(Cell::new(0u32));
        let round_clone = round.clone();
        let argmax_fn = |_s: i32, prev_token: i32| prev_token + 1;
        let propose_fn = move |bonus: i32, bs: usize| -> Vec<i32> {
            let r = round_clone.get();
            round_clone.set(r + 1);
            let want_accept = match r {
                0 => 1usize,
                1 => 4usize,
                _ => bs - 1, // accept all
            };
            (0..bs - 1)
                .map(|i| {
                    if i < want_accept {
                        bonus + i as i32 + 1
                    } else {
                        999
                    }
                })
                .collect()
        };

        let (out, rollback_events, _) = run_synthetic_round_loop(
            8,
            /*max_tokens=*/ 24,
            /*first_bonus=*/ 100,
            argmax_fn,
            propose_fn,
        );

        assert!(
            out.accept_lens.len() >= 3,
            "must run at least 3 rounds, got {:?}",
            out.accept_lens
        );
        assert_eq!(out.accept_lens[0], 1, "round 1 accept count");
        assert_eq!(out.accept_lens[1], 4, "round 2 accept count");
        assert_eq!(out.accept_lens[2], 7, "round 3 accept count");

        // Rollback called on rounds 1 and 2 (partial accept) but NOT
        // on round 3 (full accept). Count only partial-accept events
        // (`accepted < block_size - 1`).
        let partial_count = rollback_events
            .iter()
            .filter(|(a, b)| *a < *b - 1)
            .count();
        assert!(
            partial_count >= 2,
            "rollback should fire on partial-accept rounds 1 and 2; got {rollback_events:?}"
        );
    }

    /// The trait-method matrix says `rollback_partial` MUST NOT be
    /// called on full-accept rounds. Pin that boundary explicitly
    /// (this is a separate gate from "partial calls rollback" — it
    /// guards the wasted GDN replay on the hot path).
    #[test]
    fn round_loop_skips_rollback_on_full_accept_round() {
        // First and third rounds fully accept; second round mismatches
        // at index 3.
        let round = Rc::new(Cell::new(0u32));
        let round_clone = round.clone();
        let argmax_fn = |_s: i32, prev_token: i32| prev_token + 1;
        let propose_fn = move |bonus: i32, bs: usize| -> Vec<i32> {
            let r = round_clone.get();
            round_clone.set(r + 1);
            (0..bs - 1)
                .map(|i| {
                    if r == 1 && i >= 3 {
                        999 // round 2: mismatch from index 3
                    } else {
                        bonus + i as i32 + 1
                    }
                })
                .collect()
        };

        let (_out, rollback_events, _) = run_synthetic_round_loop(
            8,
            /*max_tokens=*/ 24,
            /*first_bonus=*/ 100,
            argmax_fn,
            propose_fn,
        );

        // Exactly one rollback event (round 2), not two or three.
        assert_eq!(
            rollback_events.len(),
            1,
            "only the partial-accept round must invoke rollback, got {rollback_events:?}"
        );
        assert_eq!(rollback_events[0], (3, 8));
    }

    // --------------------------------------------------------------
    // Greedy-parity gate
    // --------------------------------------------------------------

    /// Greedy parity at `temp = 0.0`: the round loop's emitted token
    /// stream is byte-identical to what a drafter-less greedy generator
    /// would produce given the same target. Tests this for ≥32 tokens.
    ///
    /// We don't need a real Qwen 3.5 model for this test — the
    /// invariant is a property of `_dflash_rounds`'s control flow:
    /// the target's argmax at every verify position determines what
    /// the round loop emits, regardless of what the drafter proposed.
    /// So any synthetic argmax function `argmax_fn` defines a unique
    /// "drafter-less" reference sequence, and the round loop with
    /// ANY drafter (matching or not) must reproduce that sequence
    /// at temp=0.
    ///
    /// Reference model: `argmax(prev) = (prev.rem_euclid(prime) + delta)
    /// % 200 + 30`. Deterministic positive output bounded in [30, 230)
    /// regardless of input sign, so the vocab-bound check in the
    /// synthetic target's logits assembly never trips.
    #[test]
    fn greedy_parity_with_synthetic_target_for_thirty_two_tokens() {
        fn chain_next(prev: i32) -> i32 {
            (prev.rem_euclid(101) * 7 + 13).rem_euclid(200) + 30
        }
        let argmax_fn = |_s: i32, prev_token: i32| chain_next(prev_token);

        let first_bonus = 100i32;
        let max_tokens = 33; // 1 first_bonus + 32 round-loop emissions
        // Build the reference greedy sequence.
        let mut reference: Vec<i32> = Vec::with_capacity(max_tokens);
        reference.push(first_bonus);
        for _ in 1..max_tokens {
            let prev = *reference.last().unwrap();
            reference.push(chain_next(prev));
        }

        // Variant 1: drafter that ALWAYS mismatches (accepts 0 per round).
        // Sentinel 0 is in-vocab and distinct from any value the chain
        // can produce ([30, 230)).
        let propose_always_wrong =
            |_bonus: i32, bs: usize| -> Vec<i32> { (1..bs as i32).map(|_| 0).collect() };
        let (out, _, _) = run_synthetic_round_loop(
            8,
            max_tokens,
            first_bonus,
            argmax_fn,
            propose_always_wrong,
        );

        let reference_tail = &reference[1..];
        assert_eq!(
            out.tokens.len(),
            reference_tail.len(),
            "round loop must emit exactly max_tokens - 1 tokens (got {}); reference tail len = {}",
            out.tokens.len(),
            reference_tail.len()
        );
        for (i, (got, want)) in out.tokens.iter().zip(reference_tail.iter()).enumerate() {
            assert_eq!(
                got, want,
                "token {i} diverged from greedy reference: got {got}, want {want}"
            );
        }

        // Variant 2: drafter that proposes the correct chain locally
        // for the first half of each block, then a constant sentinel.
        // At temp=0 the loop's output MUST still be byte-identical.
        let propose_partial = |bonus: i32, bs: usize| -> Vec<i32> {
            let mut out = Vec::with_capacity(bs - 1);
            let mut prev = bonus;
            for i in 0..bs - 1 {
                let next = chain_next(prev);
                if i < bs / 2 {
                    out.push(next);
                } else {
                    // Distinct in-vocab sentinel not in the chain image.
                    out.push(1);
                }
                prev = next;
            }
            out
        };
        let (out2, _, _) =
            run_synthetic_round_loop(8, max_tokens, first_bonus, argmax_fn, propose_partial);

        assert_eq!(
            out2.tokens, out.tokens,
            "byte-identical output with a different drafter at temp=0"
        );
        for (i, (got, want)) in out2.tokens.iter().zip(reference_tail.iter()).enumerate() {
            assert_eq!(
                got, want,
                "variant-2 token {i} diverged from greedy reference: got {got}, want {want}"
            );
        }

        // Variant 3: oracle drafter that always proposes correctly.
        // Should produce the same byte sequence with full acceptance.
        let propose_oracle = |bonus: i32, bs: usize| -> Vec<i32> {
            let mut out = Vec::with_capacity(bs - 1);
            let mut prev = bonus;
            for _ in 0..bs - 1 {
                let next = chain_next(prev);
                out.push(next);
                prev = next;
            }
            out
        };
        let (out3, _, _) =
            run_synthetic_round_loop(8, max_tokens, first_bonus, argmax_fn, propose_oracle);
        assert_eq!(
            out3.tokens, out.tokens,
            "byte-identical output with an oracle drafter (full-accept hot path)"
        );
    }

    /// EOS handling: when an emitted token equals an EOS id, the round
    /// loop stops emitting further tokens.
    ///
    /// Design: target uses the increment model `argmax = prev + 1`,
    /// `first_bonus = 100`, EOS id = 104. The drafter perfectly
    /// proposes `[101, 102, 103, 104, ...]`. Walk accepts all matching
    /// proposals up to 104, emits {101, 102, 103, 104}, stops on 104.
    #[test]
    fn round_loop_stops_on_eos_emission() {
        let argmax_fn = |_s: i32, prev: i32| prev + 1;
        let propose_fn = |bonus: i32, bs: usize| -> Vec<i32> {
            (1..bs as i32).map(|s| bonus + s).collect()
        };
        let target = SyntheticTarget::new(vec![1, 8, 15, 22, 29], 5 * 8, argmax_fn);
        let mut caches: Vec<SyntheticCache> =
            (0..3).map(|_| SyntheticCache::default()).collect();
        let drafter = SyntheticDrafter::new(propose_fn);
        let lm = EmbedOnlyLm;
        let mut gen = DFlashGenerator::with_drafter(Box::new(drafter), SamplingConfig::greedy());
        gen.block_size = 8;
        let first_hidden = ffi::zeros(&[1, 1, 5 * 8], crate::dtype::FLOAT32);
        let out = gen
            .run(
                &target,
                &lm,
                &mut caches,
                /*first_bonus=*/ 100,
                first_hidden,
                /*eos_token_ids=*/ &[104],
                /*max_tokens=*/ 100,
            )
            .expect("synthetic round loop");

        // Emitted: {101, 102, 103, 104}. Stop after EOS.
        assert_eq!(out.tokens, vec![101, 102, 103, 104]);
        // Single round, accepted = 7 (full match prefix; the EOS at
        // index 3 of new_tokens is inside the accepted-prefix vector).
        assert_eq!(out.accept_lens, vec![7]);
    }

    /// max_tokens=1: round loop emits nothing (the caller has already
    /// emitted the first bonus).
    #[test]
    fn round_loop_max_tokens_one_emits_nothing() {
        let argmax_fn = |_s: i32, _prev: i32| 99;
        let propose_fn = |_bonus: i32, bs: usize| -> Vec<i32> {
            (1..bs as i32).map(|_| 0).collect()
        };
        let (out, rollback_events, verify_lens) =
            run_synthetic_round_loop(8, /*max_tokens=*/ 1, /*first_bonus=*/ 100, argmax_fn, propose_fn);

        assert!(out.tokens.is_empty(), "max_tokens=1 must emit no further tokens");
        assert!(out.accept_lens.is_empty());
        assert!(rollback_events.is_empty());
        assert!(verify_lens.is_empty());
    }
}

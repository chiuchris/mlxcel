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

//! [`MtpGenerator`] — round-loop driver for Gemma 4 MTP speculative decoding
//! (issue #629).
//!
//! Top-level lifecycle (single-batch path):
//!
//! 1. **Prefill.** Call [`MtpTarget::forward_prefill`] over the whole prompt
//!    to populate the target's KV cache. Sample the first bonus token from
//!    the prefill's last-position logits (handled by the caller's
//!    sampler).
//! 2. **Seed.** Call [`MtpTarget::seed_verify`] with the bonus token to
//!    capture the initial hidden + shared K/V slabs. Bind the drafter and
//!    arm its `set_shared_kv` with the seed slabs.
//! 3. **Round loop.** While more tokens remain:
//!    a. Drafter produces `K-1` proposals via
//!       [`crate::drafter::Drafter::draft_block`] (autoregressive, with
//!       RoPE queries frozen at the bonus token's absolute position).
//!    b. Target verify on `[bonus, draft_0, …, draft_{K-2}]` via
//!       [`MtpTarget::verify`] — produces `target_tokens`, the next
//!       hidden, and the re-sliced shared K/V.
//!    c. Compare draft vs target via [`super::speculative_walk`].
//!    d. Emit `new_tokens`. Update `bonus` to the last emitted token.
//!    e. Rebind the drafter against the new shared K/V via
//!       [`crate::drafter::Drafter::set_shared_kv`].
//! 4. **Termination.** Loop exits when an emitted token is in the
//!    target's `eos_token_ids()` or `emitted >= max_tokens`.

use crate::drafter::{Drafter, SharedKv};
use crate::generate::{GenerationStats, SamplingConfig};
use crate::generation_policy::merged_eos_token_ids;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use super::target::{MtpTarget, MtpVerifyOutput};
use super::walk::speculative_walk;

/// Round-loop driver for Gemma 4 MTP speculative decoding (B=1).
///
/// Generic over `T: MtpTarget` so we get static dispatch into the target's
/// sink-aware forward / rollback hooks (one v-table hop per call would
/// otherwise show up as measurable noise at K=4 and decode-bound rounds).
/// The drafter is held as `Box<dyn Drafter>` so the call site can swap in
/// Gemma 4 assistant / DFlash / future MTP shapes without touching the
/// generator type.
///
/// ## Construction
///
/// Holds the target and drafter by value, plus a sampler and block size.
/// The drafter MUST already have been loaded from disk (e.g. via
/// [`crate::drafter::load_drafter`]) AND bound to the target via
/// [`crate::drafter::Drafter::bind`]. The generator does not own the
/// load or bind path — both happen at the call site before
/// constructing the generator. This avoids cross-trait coupling
/// between [`MtpTarget`] (concrete model wrapper) and the
/// [`crate::generate::LanguageModel`] trait expected by
/// [`crate::drafter::Drafter::bind`].
///
/// ## Lifecycle
///
/// [`MtpGenerator::generate`] is the only public entrypoint. It takes a
/// prompt and `max_tokens`, returns the emitted tokens and stats.
///
/// ## Threading
///
/// Single-threaded by design. The MTP round-loop's drafter ↔ target
/// dependence is too tight for the classic generator's
/// `install_thread_local_default_stream` pattern to help. Future
/// concurrency lands in #631 (batched MTP).
pub struct MtpGenerator<T: MtpTarget> {
    target: T,
    drafter: Box<dyn Drafter>,
    block_size: usize,
}

impl<T: MtpTarget> MtpGenerator<T> {
    /// Construct a new generator.
    ///
    /// `block_size` is `K` — the draft block length. The drafter
    /// produces `K-1` proposals per round; the verify pass takes
    /// `K` tokens (`[bonus, draft_0, …, draft_{K-2}]`). `K=4` is the
    /// upstream Gemma 4 default.
    pub fn new(target: T, drafter: Box<dyn Drafter>, block_size: usize) -> Self {
        assert!(
            block_size >= 2,
            "MtpGenerator: block_size must be >= 2 (block_size=1 produces no draft proposals)",
        );
        Self {
            target,
            drafter,
            block_size,
        }
    }

    /// Block size (K). Test/diagnostic accessor.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Target accessor. Test/diagnostic.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// Drafter accessor (read-only). Test/diagnostic.
    pub fn drafter(&self) -> &dyn Drafter {
        self.drafter.as_ref()
    }

    /// Consume the generator and return the boxed drafter handle.
    ///
    /// Used by the server-side speculative burst path (issue #670) so a
    /// loaded drafter can be reused across multiple requests on the same
    /// worker thread without re-loading from disk. The caller is
    /// expected to [`Drafter::reset`] the returned handle before the
    /// next burst so per-run drafter state (KV cache, accept counters)
    /// is cleared.
    pub fn into_drafter(self) -> Box<dyn Drafter> {
        self.drafter
    }

    /// Run the full MTP generate cycle.
    ///
    /// # Arguments
    ///
    /// - `prompt_tokens`: prompt token ids (non-empty). The generator
    ///   runs prefill through the target's
    ///   [`MtpTarget::prefill_and_seed`] hook, samples the first bonus,
    ///   and immediately enters the round loop.
    /// - `max_tokens`: total cap on emitted tokens, including the first
    ///   bonus.
    /// - `sampling`: sampling config. **Greedy parity requires
    ///   `temperature == 0`** — this is the load-bearing correctness
    ///   gate of the MTP path (see issue #629).
    /// - `cancel`: cooperative-cancellation flag. Checked **once per
    ///   round** (not per token) at the top of the round loop; when set,
    ///   the generator returns early with whatever tokens it has already
    ///   emitted (at minimum the first bonus). The server-side burst
    ///   path passes `&seq.cancelled` so a client disconnect mid-burst
    ///   stops occupying the worker thread (issue #672). The offline CLI
    ///   path passes `&AtomicBool::new(false)`.
    ///
    /// # Returns
    ///
    /// `(tokens, stats)` where `tokens` is the emitted sequence
    /// (including the first bonus at index 0) and `stats` contains
    /// the prefill + decode timing breakdown.
    pub fn generate(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        cancel: &AtomicBool,
    ) -> (Vec<i32>, GenerationStats) {
        assert!(
            !prompt_tokens.is_empty(),
            "MtpGenerator: prompt_tokens must be non-empty",
        );

        let prompt_len = prompt_tokens.len();
        let eos_tokens =
            merged_eos_token_ids(self.target.eos_token_ids(), &sampling.stop_token_ids);

        let mut emitted: Vec<i32> = Vec::with_capacity(max_tokens);
        if max_tokens == 0 {
            return (
                emitted,
                Self::build_stats(prompt_len, 0, std::time::Duration::ZERO, std::time::Duration::ZERO),
            );
        }

        // PREFILL + SEED.
        //
        // Combined: prefill the prompt through the target with sinks
        // armed, sample the first bonus from the last-position logits,
        // and capture hidden + shared K/V for the drafter's first
        // round.
        let prefill_start = Instant::now();
        let (first_bonus, mut verify_out) =
            self.target.prefill_and_seed(prompt_tokens, sampling);
        let prefill_time = prefill_start.elapsed();

        // Emit the first bonus and short-circuit if it's EOS or
        // max_tokens=1.
        emitted.push(first_bonus);
        if eos_tokens.contains(&first_bonus) || max_tokens == 1 {
            let gen_count = emitted.len();
            return (
                emitted,
                Self::build_stats(prompt_len, gen_count, prefill_time, std::time::Duration::ZERO),
            );
        }

        let decode_start = Instant::now();
        let mut bonus = first_bonus;

        // Arm the drafter's shared K/V from the seed capture.
        // The drafter MUST already have been bound (see struct docs).
        self.set_shared_kv_from_verify(&verify_out);

        // ROUND LOOP.
        loop {
            if emitted.len() >= max_tokens {
                break;
            }
            // Cooperative cancellation: checked once per round (cheap —
            // a single relaxed atomic load), not per token. On a client
            // disconnect mid-burst the server flips `seq.cancelled` and
            // this loop bails out with the tokens emitted so far rather
            // than running the full `max_tokens` budget and
            // head-of-line-blocking the next request (issue #672).
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            // Bound the block size by the remaining budget: per upstream
            // `bs = min(block_total, max_tokens - emitted + 1)`. The `+1`
            // is because the verify input is `[bonus, draft_0, …,
            // draft_{K-2}]` — one prefix bonus position that the
            // round-loop already counts as emitted.
            let bs = self.block_size.min(max_tokens - emitted.len() + 1);
            if bs <= 1 {
                break;
            }

            // Drafter produces K-1 proposals. Pass the bonus + last
            // hidden captured from the previous verify pass (or the
            // seed verify on the first iteration).
            let hidden = verify_out.next_hidden.as_ref();
            let draft_tokens = match self.drafter.draft_block(bonus, hidden, bs, sampling) {
                Ok(t) => t,
                Err(e) => {
                    // Drafter failed — bail out cleanly. We've already
                    // emitted at least the seed bonus, so return what
                    // we have rather than panicking. Future hardening
                    // (#632) can surface this through GenerationStats.
                    let _ = e;
                    break;
                }
            };

            // Verify input = [bonus, draft_0, ..., draft_{K-2}].
            // If the drafter returned fewer than bs-1 proposals (e.g.
            // it short-circuited on a non-greedy path), `bs` shrinks
            // accordingly so the verify shape stays consistent.
            let actual_bs = draft_tokens.len() + 1;
            let mut verify_input = Vec::with_capacity(actual_bs);
            verify_input.push(bonus);
            verify_input.extend_from_slice(&draft_tokens);

            // Phase 1: sink-aware forward. The target's KV cache is now
            // `bs` longer than before the call. We have target_tokens
            // for the walk; the captured state holds hidden + shared
            // K/V slabs for the finalize step.
            let forward_out = self.target.verify_forward(&verify_input, sampling);

            // Walk the draft against the target's argmax tokens.
            let budget = max_tokens - emitted.len();
            let walk = speculative_walk(&draft_tokens, &forward_out.target_tokens, budget);

            // Phase 2: rollback the cache + slice shared K/V based on
            // the walk's accepted count. This consumes the captured
            // state from phase 1.
            verify_out = self.target.verify_finalize(
                walk.accepted,
                actual_bs,
                forward_out.captured,
            );

            // Emit accepted tokens.
            for &tok in &walk.new_tokens {
                emitted.push(tok);
                if eos_tokens.contains(&tok) {
                    let decode_time = decode_start.elapsed();
                    let gen_count = emitted.len();
                    return (
                        emitted,
                        Self::build_stats(prompt_len, gen_count, prefill_time, decode_time),
                    );
                }
                if emitted.len() >= max_tokens {
                    let decode_time = decode_start.elapsed();
                    let gen_count = emitted.len();
                    return (
                        emitted,
                        Self::build_stats(prompt_len, gen_count, prefill_time, decode_time),
                    );
                }
            }

            // Next round's bonus is the last emitted token.
            bonus = match emitted.last() {
                Some(&t) => t,
                None => break,
            };

            // Re-arm the drafter against the (now post-rollback) shared
            // K/V the verify call produced. The drafter's
            // `set_shared_kv` will read these slabs at the start of the
            // next `draft_block`.
            self.set_shared_kv_from_verify(&verify_out);
        }

        let decode_time = decode_start.elapsed();
        let gen_count = emitted.len();
        (
            emitted,
            Self::build_stats(prompt_len, gen_count, prefill_time, decode_time),
        )
    }

    /// Wire the verify output's `next_shared_kv` into the drafter's
    /// `set_shared_kv`.
    ///
    /// Best-effort: the round-loop continues even if the drafter
    /// rejects the call (the next `draft_block` will fail closed and
    /// the outer loop bails out). Tests assert the slabs reach the
    /// drafter; real-model integration (#632) gates on a stricter
    /// error path via `GenerationStats.errors` or similar.
    fn set_shared_kv_from_verify(&mut self, verify_out: &MtpVerifyOutput) {
        let refs = verify_out.shared_kv_refs();
        let shared_kv = SharedKv::new(&refs);
        let _ = self.drafter.set_shared_kv(
            shared_kv,
            verify_out.kv_offset,
            verify_out.bonus_position,
            /* left_padding */ 0,
        );
    }

    fn build_stats(
        prompt_count: usize,
        gen_count: usize,
        prefill_time: std::time::Duration,
        decode_time: std::time::Duration,
    ) -> GenerationStats {
        let prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        let decode_ms = decode_time.as_secs_f64() * 1000.0;
        GenerationStats {
            prompt_tokens: prompt_count,
            generated_tokens: gen_count,
            prefill_time_ms: prefill_ms,
            decode_time_ms: decode_ms,
            prefill_tok_per_sec: if prefill_ms > 0.0 {
                prompt_count as f64 / (prefill_ms / 1000.0)
            } else {
                0.0
            },
            decode_tok_per_sec: if decode_ms > 0.0 {
                gen_count as f64 / (decode_ms / 1000.0)
            } else {
                0.0
            },
        }
    }
}

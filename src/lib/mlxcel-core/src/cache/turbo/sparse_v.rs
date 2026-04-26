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

//! Sparse-V dequant — attention-gated V-side dequantization (B8, issue #480,
//! epic #458).
//!
//! At long context the post-softmax attention distribution is sparse: most
//! KV positions receive a near-zero attention weight. The TurboQuant+ paper
//! [`docs/papers/sparse-v-dequant.md`] reports `~90%` sparsity at 32 K
//! context on a 35 B MoE model. Dequantizing those V vectors is wasted work.
//! Skipping them on a fused Metal kernel yields `+22.8 %` decode at 32 K
//! with no measurable PPL change.
//!
//! This Rust module is the **graph-level scaffolding** for that
//! optimization. Path (a) — a fused Metal kernel that does
//! `(scores → softmax → mask → on-demand V dequant → weighted sum)` in one
//! pass — is the production speed path the paper validates and is **out of
//! scope for this PR**: it requires Metal sources under `src/lib/mlx-cpp/`
//! and a MLX upstream kernel addition. See "Limitations" below.
//!
//! What this module ships:
//!
//! 1. Threshold and enablement helpers driven by `MLXCEL_SPARSE_V_THRESHOLD`
//!    (default `1e-6`, set to `0.0` to disable).
//! 2. [`compute_alive_mask`] — given attention scores, build the per-(KV
//!    head, KV token) "alive" mask. Aggregates across the Q axis (any
//!    Q-position with `attn > threshold` keeps the KV slot alive) and across
//!    Q-head groups for grouped attention (any Q head in the group keeps it
//!    alive, the same way SDPA's `repeat_kv` works).
//! 3. [`attention_sparse_v_turbo4`] — split-SDPA reference path: compute
//!    scores in FP32, threshold, dequantize V (full path), zero out dead
//!    rows, then `attn @ V_masked`. Numerically equivalent to the full
//!    Turbo4 dequant path within FP16 round-off when `threshold = 0`.
//!
//! What this module does **not** ship:
//!
//! - **Per-position lazy dequant.** The graph-level `where(alive, V_dq, 0)`
//!   still pays the full V dequant cost; the dequant kernel runs over the
//!   complete `[B, H, T, D]` tensor regardless of the mask. The actual speed
//!   gate (`+15 %` decode at 32 K) requires the fused Metal kernel — the
//!   `continue` in the unrolled inner loop, exactly as the paper describes
//!   in §3.1.
//! - **Integration into model attention call sites.** Every model file calls
//!   `cache.update_and_fetch(...)` then `attention(q, k, v, ...)`. Wiring
//!   sparse-V through that contract requires either (i) a new attention API
//!   that accepts `(q, k, v_packed, v_norms, params, ...)` and is opted into
//!   per cache mode, or (ii) intercepting the cache return type to lazy-V.
//!   Both are large surface-area changes deferred to a follow-up sub-issue.
//!
//! # Design rationale: graph-level vs Metal kernel
//!
//! The paper's §5 documents 14 alternative dequant implementations, all of
//! which fail to beat the constant-memory LUT on Apple Silicon. The
//! conclusion: the only path forward is *operation elimination*, not
//! *operation acceleration*. At the MLX graph level we do not have a
//! per-position skip mechanism — `where(alive, dequant(...), 0)` runs
//! `dequant(...)` for every position because MLX evaluates both branches
//! eagerly. The skip can only happen inside the Metal kernel where the
//! `if (attn_weight < 1e-6f) continue;` predicate gates a single thread's
//! work.
//!
//! Therefore this module's split-SDPA path is **correctness-only**: it
//! observes the threshold, masks out dead rows so they contribute zero to
//! the output, and validates that the masked attention output matches the
//! full attention output to FP16 precision. The PPL quality gate (`B3`,
//! issue #475) will thus pass with the threshold enabled because every
//! "skipped" position has near-zero contribution anyway. The speed gate
//! (`+15 %` at 32 K) is **deferred to the Metal kernel follow-up**.
//!
//! Used by: `KVCache::update_and_fetch` (Turbo4Asym, Turbo4Delegated modes)
//! when `sparse_v::is_enabled()` is `true`. (Integration TBD per the
//! "What this module does not ship" note above.)

use std::sync::OnceLock;

use cxx::UniquePtr;

use super::TurboQuantParams;
use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;

/// Default attention-weight threshold below which a V position is skipped.
///
/// `1e-6` is the value validated in the TurboQuant+ paper §3.2 / §4.3.
/// Their threshold ablation (§4.8) shows that any value in `[1e-8, 1e-4]`
/// produces identical PPL across wikitext-2 (8K, 16K, 32K) and wikitext-103
/// (50 chunks at 32K). We pick the same conservative default `1e-6` as the
/// paper's Metal kernel.
pub const DEFAULT_THRESHOLD: f32 = 1e-6;

/// Environment variable for runtime threshold override.
///
/// - Unset (default): use [`DEFAULT_THRESHOLD`] (`1e-6`).
/// - `0` or `0.0`: sparse-V disabled (every V position is "alive").
/// - Any positive float: use that value.
/// - Invalid (non-numeric or negative): warn and fall back to default.
pub const ENV_VAR: &str = "MLXCEL_SPARSE_V_THRESHOLD";

/// Resolved threshold (cached after first read of [`ENV_VAR`]).
static THRESHOLD: OnceLock<f32> = OnceLock::new();

/// Read the configured sparse-V threshold.
///
/// The first call resolves [`ENV_VAR`] and caches the result; subsequent
/// calls are lock-free reads. A value of exactly `0.0` disables sparse-V
/// (every position is treated as alive).
///
/// Used by: [`is_enabled`] and [`attention_sparse_v_turbo4`].
pub fn threshold() -> f32 {
    *THRESHOLD.get_or_init(|| match std::env::var(ENV_VAR) {
        Ok(s) => match s.trim().parse::<f32>() {
            Ok(v) if v >= 0.0 && v.is_finite() => v,
            Ok(_) | Err(_) => {
                eprintln!(
                    "[mlxcel] WARN: {ENV_VAR}={s:?} is not a non-negative finite float; \
                     falling back to default {DEFAULT_THRESHOLD}",
                );
                DEFAULT_THRESHOLD
            }
        },
        Err(_) => DEFAULT_THRESHOLD,
    })
}

/// Returns `true` iff sparse-V is enabled at the configured threshold.
///
/// Sparse-V is **disabled** when [`threshold`] returns exactly `0.0`. This
/// gives users an explicit kill switch (`MLXCEL_SPARSE_V_THRESHOLD=0`) for
/// A/B comparisons and bisecting regressions.
pub fn is_enabled() -> bool {
    threshold() > 0.0
}

/// Compute the per-(KV head, KV token) "alive" mask from attention scores.
///
/// # Inputs
///
/// - `attn_weights`: `[B, Hq, Tq, Tk]` — post-softmax attention weights, FP32
///   or FP16. The function casts internally to FP32 for a stable comparison.
/// - `kv_heads`: number of KV heads (`Hkv`). For grouped attention,
///   `Hq = n_rep * Hkv`. For non-grouped attention, `Hkv == Hq`.
/// - `threshold_value`: the alive cutoff. Positions strictly greater than
///   this stay alive.
///
/// # Output
///
/// `[B, Hkv, 1, Tk]` boolean (FP32 0.0 / 1.0) — per (B, KV head, KV token)
/// "is this slot alive on at least one Q-head and Q-position". The leading
/// `1` axis is kept so the mask broadcasts cleanly against `[B, Hkv, Tk, D]`
/// V tensors after a transpose.
///
/// # Aggregation rules
///
/// 1. **Across Q axis** — any Q position whose attention exceeds threshold
///    keeps the KV slot alive. Without this we would skip a slot that some
///    Q tokens still attend to.
/// 2. **Across grouped Q heads** — for `Hq > Hkv` we reshape Q heads to
///    `[B, Hkv, n_rep, Tq, Tk]` and reduce `n_rep` away by `max`. Any Q
///    head in the group attending to a slot keeps it alive.
///
/// Used by: [`attention_sparse_v_turbo4`].
pub fn compute_alive_mask(
    attn_weights: &MlxArray,
    kv_heads: i32,
    threshold_value: f32,
) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(attn_weights);
    debug_assert_eq!(shape.len(), 4, "attn_weights must be 4-D [B, Hq, Tq, Tk]");
    let b = shape[0];
    let hq = shape[1];
    let tq = shape[2];
    let tk = shape[3];
    debug_assert!(kv_heads > 0, "kv_heads must be positive");
    debug_assert!(
        hq % kv_heads == 0,
        "Hq ({hq}) must be a multiple of Hkv ({kv_heads})"
    );

    // Cast to FP32 to make the comparison stable regardless of the input
    // dtype (softmax outputs may be FP16 from the upstream graph).
    let attn_f32 = ffi::astype(attn_weights, dtype::FLOAT32);

    // Reduce across the Q-token axis (axis=2): any Q token attending above
    // threshold keeps this KV slot alive.
    let max_over_q = ffi::max_axis(&attn_f32, 2, true); // [B, Hq, 1, Tk]

    // For grouped attention, reduce Q-heads to KV-heads. We reshape `Hq`
    // into `[Hkv, n_rep]` and reduce `n_rep` (axis=2 in the new layout) by
    // max — any Q head in the group keeps the slot alive.
    let aggregated = if hq == kv_heads {
        max_over_q
    } else {
        let n_rep = hq / kv_heads;
        // [B, Hq, 1, Tk] → [B, Hkv, n_rep, 1, Tk]
        let reshaped = ffi::reshape(&max_over_q, &[b, kv_heads, n_rep, 1, tk]);
        // Reduce axis=2 (the n_rep axis) by max. Result: [B, Hkv, 1, Tk]
        ffi::max_axis(&reshaped, 2, false)
    };

    // Threshold and cast to FP32 mask (1.0 alive / 0.0 dead). MLX `greater`
    // returns a boolean array; we cast to FP32 so it can be `multiply`'d
    // against an FP16 V tensor (the V tensor is cast to FP32 before the
    // mask multiply for numerical sanity, then cast back at the end).
    let threshold_arr = ffi::full_f32(&[1], threshold_value, dtype::FLOAT32);
    let alive_bool = ffi::greater(&aggregated, &threshold_arr); // [B, Hkv, 1, Tk] bool
    let _ = tq; // tq is intentionally unused: aggregation already collapses it.
    ffi::astype(&alive_bool, dtype::FLOAT32)
}

/// Split-SDPA reference path with attention-gated V-side masking.
///
/// **Correctness scaffolding only.** Computes attention scores explicitly,
/// builds an alive mask, dequantizes V (full path), zeros out the dead rows,
/// and returns `attn_weights @ V_masked`. When `threshold` is `0.0` this is
/// numerically equivalent to the standard full-dequant attention path
/// within FP16 round-off.
///
/// This does **not** save the V dequant cost — the dequant kernel runs over
/// the complete `[B, H, T, D]` tensor. The speed-gate version of this
/// optimization requires a fused Metal kernel that gates the per-thread
/// dequant work on `attn_weight < threshold`. See the module-level
/// documentation for the rationale.
///
/// # Inputs
///
/// - `q`: `[B, Hq, Tq, D]` query tensor (FP16 or FP32)
/// - `k`: `[B, Hkv, Tk, D]` key tensor (FP16) — already dequantized for
///   `Turbo4Asym` (K stays FP16) or already dequantized by the caller for
///   `Turbo4` symmetric mode.
/// - `v_packed`: `[B, Hkv, Tk, D/2]` u8 — packed V indices.
/// - `v_norms`: `[B, Hkv, Tk, 1]` FP16 — per-token V norms.
/// - `params`: `TurboQuantParams` used at quantize time.
/// - `scale`: attention scale (typically `1 / sqrt(d)`).
/// - `mask`: optional additive attention mask (e.g. causal). `None` for
///   maskless attention.
/// - `threshold_value`: alive cutoff. `0.0` disables masking and matches the
///   non-sparse path bit-for-bit modulo FP32/FP16 round-off.
///
/// # Output
///
/// `[B, Hq, Tq, D]` FP16 attention output, the same shape and dtype as the
/// standard SDPA path returns.
///
/// Used by: tests under `cache/turbo_tests.rs` (issue #480 unit tests).
/// Production attention call sites continue to use the standard
/// `attention()` path until the Metal kernel lands.
pub fn attention_sparse_v_turbo4(
    q: &MlxArray,
    k: &MlxArray,
    v_packed: &MlxArray,
    v_norms: &MlxArray,
    params: &TurboQuantParams,
    scale: f32,
    mask: Option<&MlxArray>,
    threshold_value: f32,
) -> UniquePtr<MlxArray> {
    let q_shape = ffi::array_shape(q);
    let k_shape = ffi::array_shape(k);
    debug_assert_eq!(q_shape.len(), 4, "q must be 4-D [B, Hq, Tq, D]");
    debug_assert_eq!(k_shape.len(), 4, "k must be 4-D [B, Hkv, Tk, D]");
    let b = q_shape[0];
    let hq = q_shape[1];
    let kv_heads = k_shape[1];
    debug_assert!(kv_heads > 0, "Hkv must be positive");
    debug_assert!(
        hq % kv_heads == 0,
        "Hq ({hq}) must be a multiple of Hkv ({kv_heads})"
    );
    let n_rep = hq / kv_heads;

    // Repeat KV heads to match Q heads for the matmul. MLX has no `repeat_kv`
    // primitive exposed — we tile manually. Matches the pattern in
    // `compiled_softcap_sdpa_gqa` upstream.
    let k_for_q = if n_rep == 1 {
        ffi::contiguous(k, false)
    } else {
        let kt = k_shape[2];
        let kd = k_shape[3];
        // [B, Hkv, Tk, D] → [B, Hkv, 1, Tk, D] → [B, Hkv, n_rep, Tk, D] → [B, Hq, Tk, D]
        let k_exp = ffi::expand_dims(k, 2);
        let k_tiled = ffi::broadcast_to(&k_exp, &[b, kv_heads, n_rep, kt, kd]);
        ffi::reshape(&k_tiled, &[b, hq, kt, kd])
    };

    // Compute attention scores: scores = Q @ K^T * scale. MLX does not
    // expose a "matmul transpose-b" primitive in this FFI; we transpose K
    // along its last two axes before the matmul.
    let k_for_q_t = ffi::transpose_axes(&k_for_q, &[0, 1, 3, 2]);
    let q_f32 = ffi::astype(q, dtype::FLOAT32);
    let k_t_f32 = ffi::astype(&k_for_q_t, dtype::FLOAT32);
    let qk = ffi::matmul(&q_f32, &k_t_f32);
    let scale_arr = ffi::full_f32(&[1], scale, dtype::FLOAT32);
    let mut scores = ffi::multiply(&qk, &scale_arr);

    if let Some(m) = mask {
        let m_f32 = ffi::astype(m, dtype::FLOAT32);
        scores = ffi::add(&scores, &m_f32);
    }

    // Stable softmax along the K axis.
    let attn_weights = ffi::softmax_precise(&scores, -1); // [B, Hq, Tq, Tk] f32

    // Build alive mask if threshold > 0. When threshold == 0 every position
    // is alive and we skip the mask construction (zero overhead beyond the
    // explicit dequant + matmul).
    let v_dequant =
        super::quant::dequantize_v_turbo4(v_packed, v_norms, params); // [B, Hkv, Tk, D] f16

    let v_for_q = if threshold_value > 0.0 {
        let alive = compute_alive_mask(&attn_weights, kv_heads, threshold_value);
        // alive shape: [B, Hkv, 1, Tk] — broadcast against V [B, Hkv, Tk, D]
        // by unsqueeze → [B, Hkv, Tk, 1] then multiply.
        let alive_for_v = {
            let alive_t = ffi::transpose_axes(&alive, &[0, 1, 3, 2]); // [B, Hkv, Tk, 1]
            ffi::astype(&alive_t, dtype::FLOAT32)
        };
        let v_dq_f32 = ffi::astype(&v_dequant, dtype::FLOAT32);
        let v_masked = ffi::multiply(&v_dq_f32, &alive_for_v);

        // Now repeat-tile to match Q heads.
        if n_rep == 1 {
            ffi::astype(&v_masked, dtype::FLOAT32)
        } else {
            let vs = ffi::array_shape(&v_masked);
            let vt = vs[2];
            let vd = vs[3];
            let v_exp = ffi::expand_dims(&v_masked, 2);
            let v_tiled = ffi::broadcast_to(&v_exp, &[b, kv_heads, n_rep, vt, vd]);
            ffi::reshape(&v_tiled, &[b, hq, vt, vd])
        }
    } else {
        // No masking — just repeat-tile and matmul.
        let v_dq_f32 = ffi::astype(&v_dequant, dtype::FLOAT32);
        if n_rep == 1 {
            v_dq_f32
        } else {
            let vs = ffi::array_shape(&v_dq_f32);
            let vt = vs[2];
            let vd = vs[3];
            let v_exp = ffi::expand_dims(&v_dq_f32, 2);
            let v_tiled = ffi::broadcast_to(&v_exp, &[b, kv_heads, n_rep, vt, vd]);
            ffi::reshape(&v_tiled, &[b, hq, vt, vd])
        }
    };

    // Final attention output: attn @ V_masked. Cast back to FP16 to match
    // the standard SDPA contract.
    let out_f32 = ffi::matmul(&attn_weights, &v_for_q);
    ffi::astype(&out_f32, dtype::FLOAT16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Threshold env-var parsing — empty / unset → default.
    #[test]
    fn threshold_default_when_unset() {
        // Note: this test runs in the same process as other env-var tests,
        // so we must not depend on the global `THRESHOLD` cache. We test the
        // parse logic indirectly by re-creating it inline.
        let parse = |s: &str| -> f32 {
            match s.trim().parse::<f32>() {
                Ok(v) if v >= 0.0 && v.is_finite() => v,
                _ => DEFAULT_THRESHOLD,
            }
        };
        assert_eq!(parse(""), DEFAULT_THRESHOLD);
        assert_eq!(parse("0"), 0.0);
        assert_eq!(parse("0.0"), 0.0);
        assert_eq!(parse("1e-6"), 1e-6);
        assert_eq!(parse("0.001"), 0.001);
        assert_eq!(parse("not-a-number"), DEFAULT_THRESHOLD);
        assert_eq!(parse("-1"), DEFAULT_THRESHOLD);
        assert_eq!(parse("inf"), DEFAULT_THRESHOLD);
    }

    /// Default threshold matches the value validated in the paper.
    #[test]
    fn default_threshold_value() {
        assert_eq!(DEFAULT_THRESHOLD, 1e-6);
    }
}

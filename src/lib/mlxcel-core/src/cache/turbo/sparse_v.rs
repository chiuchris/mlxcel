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

//! Sparse-V dequant — attention-gated V-side dequantization (issues #498,
//! #505, epic #458).
//!
//! At long context the post-softmax attention distribution is sparse: most
//! KV positions receive a near-zero attention weight. The TurboQuant+ paper
//! [`docs/papers/sparse-v-dequant.md`] reports `~90%` sparsity at 32 K
//! context on a 35 B MoE model. Dequantizing those V vectors is wasted work.
//! Skipping them on a fused Metal kernel yields `+22.8 %` decode at 32 K
//! with no measurable PPL change.
//!
//! This module provides two execution paths for the sparse-V optimization:
//!
//! 1. **Graph-level reference path** (issue #498, [`attention_sparse_v_turbo4`]).
//!    Computes attention scores, builds an alive mask, dequantizes V in full,
//!    zeros dead rows, then runs `attn @ V_masked`. The V dequant kernel still
//!    runs over the complete `[B, H, T, D]` tensor — this path is
//!    correctness-only and does not deliver the `+22.8 %` throughput benefit.
//! 2. **Fused Metal kernel path** (issue #505, [`attention_sparse_v_turbo4_fused`]).
//!    Dispatches the JIT-compiled `turbo_sparse_v_weighted_sum` Metal kernel,
//!    which does the per-thread `if (attn_weight <= threshold) continue;` skip
//!    inside the SDPA inner loop. This is the production speed path validated
//!    in the paper. Active by default on macOS when
//!    `MLXCEL_SPARSE_V_THRESHOLD > 0`. Falls back to path (1) automatically
//!    for non-power-of-2 `head_dim` values (Gemma 4's 192-dim heads). Use
//!    `MLXCEL_SPARSE_V_KERNEL=0` to force the graph fallback for A/B testing.
//!
//! The kernel sources live under `src/lib/mlx-cpp/turbo/` and use MLX's
//! runtime `mlx::core::fast::metal_kernel` JIT path.
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
//! work. Issue #498 established the graph scaffold and correctness gate;
//! issue #505 landed the fused kernel that delivers the actual throughput
//! benefit.
//!
//! Used by: `KVCache::update_and_sparse_v_attention` (Turbo4Asym mode) when
//! `sparse_v::is_enabled()` is `true`. `Turbo4Delegated` is intentionally
//! excluded from `sparse_v_available()` because that mode splits the
//! visible token range across cold-packed V and hot-FP16 V; wiring sparse-V
//! through that split requires a hot+cold composition pass and is deferred.

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

/// Environment variable that disables the fused Metal kernel path even when
/// sparse-V is otherwise enabled. Useful for A/B testing the kernel against
/// the graph reference, or for falling back when the kernel is suspected of
/// numerical regression on a new MLX version. Default: kernel is used.
///
/// - Unset / any value other than the false literals below: kernel ON.
/// - `0`, `false`, `off`, `no`: kernel OFF (graph fallback).
pub const KERNEL_ENV_VAR: &str = "MLXCEL_SPARSE_V_KERNEL";

/// Cached kernel-enable flag (resolved on first read of [`KERNEL_ENV_VAR`]).
static KERNEL_ENABLED: OnceLock<bool> = OnceLock::new();

/// Returns `true` iff the fused Metal kernel path is allowed at runtime.
///
/// The kernel itself only links and dispatches on `target_os = "macos"`; on
/// other platforms this returns `false` unconditionally. On macOS the env var
/// gate gives a runtime kill switch for A/B testing without recompiling.
///
/// Used by: [`attention_sparse_v_turbo4`] (kernel preference) and the
/// kernel-vs-graph correctness test in `sparse_v_tests.rs`.
pub fn kernel_enabled() -> bool {
    if !cfg!(target_os = "macos") {
        // On non-Apple builds the cxx FFI symbol may be present (the bridge
        // compiles on Linux/CUDA via MLX upstream's no-op Metal stubs) but
        // dispatching the JIT kernel without a Metal device hangs. Force the
        // graph fallback there.
        return false;
    }
    *KERNEL_ENABLED.get_or_init(|| match std::env::var(KERNEL_ENV_VAR) {
        Ok(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    })
}

/// Environment variable that opts callers into the fused Turbo4Delegated
/// dequant + SDPA kernel path (issue #528). Default: kernel path is **off**;
/// callers route through the legacy `update_and_fetch + attention()` pair
/// until follow-up work brings the fused pipeline inside the steel-attention
/// envelope on M5 Max at the gate contexts.
///
/// - Unset / any value other than the truthy literals below: kernel OFF.
/// - `1`, `true`, `on`, `yes` (any ASCII case): kernel ON.
pub const TURBO4_DELEGATED_FUSED_ENV_VAR: &str = "MLXCEL_TURBO4_DELEGATED_FUSED";

/// Cached opt-in flag for the Turbo4Delegated fused kernel path (resolved on
/// first read of [`TURBO4_DELEGATED_FUSED_ENV_VAR`]).
static TURBO4_DELEGATED_FUSED_ENABLED: OnceLock<bool> = OnceLock::new();

/// Returns `true` iff the caller has opted into the fused Turbo4Delegated
/// dequant + SDPA kernel path via [`TURBO4_DELEGATED_FUSED_ENV_VAR`].
///
/// Mirrors the [`kernel_enabled`] caching pattern: the env var is parsed once
/// per process and cached in a `OnceLock<bool>` so per-token, per-layer
/// attention forwards pay only an atomic load instead of an env-table lookup
/// (`std::env::var` allocates a fresh `String` and takes a process-wide lock,
/// which is ~3,200 calls/sec/sequence on a 32-layer model — see the security
/// review on PR #530).
///
/// Used by: per-model attention call sites (Llama 3, Qwen 3, ...) that want
/// to opt into [`KVCache::update_and_turbo4_delegated_attention`] over the
/// standard `update_and_fetch + attention()` pair.
pub fn turbo4_delegated_fused_enabled() -> bool {
    *TURBO4_DELEGATED_FUSED_ENABLED.get_or_init(|| {
        match std::env::var(TURBO4_DELEGATED_FUSED_ENV_VAR) {
            Ok(s) => matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            ),
            Err(_) => false,
        }
    })
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
    let v_dequant = super::quant::dequantize_v_turbo4(v_packed, v_norms, params); // [B, Hkv, Tk, D] f16

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

/// Fused-skip Sparse-V SDPA path (issue #505, optimized by issue #520).
///
/// Drops the explicit `dequantize_v_turbo4` step in favour of the fused
/// Metal kernel `turbo_sparse_v_weighted_sum`, which does the per-thread
/// `if (attn_weight <= threshold) continue;` skip in the unrolled SDPA
/// inner loop. The kernel returns the *unrotated* per-head weighted sum
/// `Σ_t attn[t] · y_hat_unit[t] · norms[t]`; this function applies the
/// inverse Turbo4 rotation (`signs1 · WHT · signs2 ·`) outside, on the
/// smaller `[B, Hq, Tq, D]` output. That moves the rotation cost from
/// `O(B · Hkv · Tk · D log D)` (graph path) to `O(B · Hq · Tq · D log D)`
/// (kernel path), a Tk-vs-Tq factor — the source of the long-context speedup.
///
/// # Inputs
///
/// - `q`: `[B, Hq, Tq, D]` query tensor (FP16 or FP32).
/// - `k`: `[B, Hkv, Tk, D]` key tensor (FP16) — Turbo4Asym keeps K in FP16.
/// - `v_packed`: `[B, Hkv, Tk, D/2]` u8 — packed V indices.
/// - `v_rescale`: `[B, Hkv, Tk, 1]` FP16 — precomputed per-token kernel
///   rescale `norm[t] / max(|y_hat[t]|, 1e-10)`. Issue #520 moved this
///   computation from a per-token threadgroup tree reduction inside the
///   kernel to a one-time host-side precompute at quantize time. The
///   value matches the kernel's previous on-the-fly `vn / yh_safe` exactly,
///   so the kernel output is bit-for-bit unchanged within FP16 round-off.
/// - `params`: `TurboQuantParams` used at quantize time (centroids + sign
///   vectors + head_dim).
/// - `scale`: attention scale (`1 / sqrt(D)` typically).
/// - `mask`: optional additive attention mask.
/// - `threshold_value`: alive cutoff. `0.0` disables skipping.
///
/// # Output
///
/// `Some([B, Hq, Tq, D])` FP16 attention output, identical shape and dtype
/// contract to [`attention_sparse_v_turbo4`].
///
/// # Platform gating
///
/// Returns `None` when [`kernel_enabled`] is `false` (non-macOS or
/// `MLXCEL_SPARSE_V_KERNEL=0`). Callers should fall back to
/// [`attention_sparse_v_turbo4`] (which still consumes `v_norms`) in that
/// case.
///
/// Used by: [`KVCache::sparse_v_attention`] (issue #505).
pub fn attention_sparse_v_turbo4_fused(
    q: &MlxArray,
    k: &MlxArray,
    v_packed: &MlxArray,
    v_rescale: &MlxArray,
    params: &TurboQuantParams,
    scale: f32,
    mask: Option<&MlxArray>,
    threshold_value: f32,
) -> Option<UniquePtr<MlxArray>> {
    if !kernel_enabled() {
        return None;
    }
    let q_shape = ffi::array_shape(q);
    let k_shape = ffi::array_shape(k);
    debug_assert_eq!(q_shape.len(), 4, "q must be 4-D [B, Hq, Tq, D]");
    debug_assert_eq!(k_shape.len(), 4, "k must be 4-D [B, Hkv, Tk, D]");
    let b = q_shape[0];
    let hq = q_shape[1];
    let tq = q_shape[2];
    let head_dim = q_shape[3];
    let kv_heads = k_shape[1];
    let tk = k_shape[2];
    debug_assert!(kv_heads > 0, "Hkv must be positive");
    debug_assert!(
        hq % kv_heads == 0,
        "Hq ({hq}) must be a multiple of Hkv ({kv_heads})"
    );
    let n_rep = hq / kv_heads;

    // Kernel-friendly precondition: head_dim must be a power of 2 (the
    // threadgroup-memory tree reduction in the kernel halves stride per
    // round, so non-power-of-2 D would over-/under-shoot). All production
    // text models use head_dim ∈ {64, 128, 192, 256} — the 192 case (Gemma
    // 4) is the lone non-power-of-2 outlier in the field; for that we fall
    // back to the graph path.
    if !(head_dim as u32).is_power_of_two() {
        return None;
    }

    // 1. Compute attention scores via the standard graph path. We keep this
    //    in MLX so the score matrix benefits from steel-attention SDPA and
    //    NAX-accelerated matmul where available; the kernel's job is only
    //    the V-side weighted-sum + sparse-V skip.
    //
    //    Repeat KV heads to match Q heads. Identical to the graph path's
    //    `k_for_q` construction in `attention_sparse_v_turbo4`.
    let k_for_q = if n_rep == 1 {
        ffi::contiguous(k, false)
    } else {
        let kt = k_shape[2];
        let kd = k_shape[3];
        let k_exp = ffi::expand_dims(k, 2);
        let k_tiled = ffi::broadcast_to(&k_exp, &[b, kv_heads, n_rep, kt, kd]);
        ffi::reshape(&k_tiled, &[b, hq, kt, kd])
    };

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
    let attn_weights = ffi::softmax_precise(&scores, -1); // [B, Hq, Tq, Tk] f32

    // 2. Pre-flatten the kernel inputs. The kernel expects (B*Hq, Tq, Tk),
    //    (B*Hkv, Tk, D/2), (B*Hkv, Tk), and a 1-D codebook.
    let bhq = b * hq;
    let bhkv = b * kv_heads;
    let attn_flat = ffi::reshape(&attn_weights, &[bhq, tq, tk]);
    let v_packed_flat = ffi::reshape(v_packed, &[bhkv, tk, head_dim / 2]);
    // v_rescale graph shape is `[B, Hkv, Tk, 1]`. The kernel expects
    // `[B*Hkv, Tk]` — drop the trailing axis and flatten the first two.
    // (Same memory layout as the previous `v_norms` plumbing; only the
    // semantic content changed — see issue #520.)
    let v_rescale_flat = ffi::reshape(v_rescale, &[bhkv, tk]);
    let codebook_vec: Vec<f32> = params.codebook.centroids.as_ref().to_vec();
    let codebook_arr = ffi::from_slice_f32(&codebook_vec, &[codebook_vec.len() as i32]);

    // 3. Dispatch the fused-skip kernel.
    let out_pre_flat = ffi::turbo_sparse_v_weighted_sum(
        &attn_flat,
        &v_packed_flat,
        &v_rescale_flat,
        &codebook_arr,
        head_dim,
        n_rep,
        threshold_value,
    );

    // 4. Reshape back to [B, Hq, Tq, D] and apply the inverse Turbo4
    //    rotation: out = signs1 * WHT(signs2 * out_pre).
    let out_pre = ffi::reshape(&out_pre_flat, &[b, hq, tq, head_dim]);
    let signs1_arr = ffi::from_slice_f32(&params.signs1, &[1, 1, 1, head_dim]);
    let signs2_arr = ffi::from_slice_f32(&params.signs2, &[1, 1, 1, head_dim]);
    let pre_h = ffi::multiply(&out_pre, &signs2_arr);
    let post_h = crate::ops::wht(&pre_h);
    let post_d1 = ffi::multiply(&post_h, &signs1_arr);

    Some(ffi::astype(&post_d1, dtype::FLOAT16))
}

/// Fused dequant + SDPA path for `KVCacheMode::Turbo4Delegated` (issue #528).
///
/// The Turbo4Delegated cache stores V in two regions: a packed cold body
/// (`[B, Hkv, T_cold, D/2]` u8 + `[B, Hkv, T_cold, 1]` fp16 rescale) and an
/// FP16 hot ring (`[B, Hkv, T_hot, D]`). Pre-#528 the read path materialised
/// the full FP16 cold body via `dequantize_v_turbo4` (memoised by PR #525)
/// and ran a graph-level `concat(cold_v_dequant, hot_v, axis=2)` plus
/// standard SDPA. The memo's working set was `[B, Hkv, cold_offset, D]` fp16
/// — at 4K context that is 50–100 MB per layer per sequence and undoes the
/// 4-bit V compression that defines the mode.
///
/// This function replaces that path. Steps:
///
/// 1. Compute attention scores `Q · K^T` against the unified K (length
///    `T_total = T_cold + T_hot`), apply the optional mask, run softmax to
///    get `attn` of shape `[B, Hq, Tq, T_total]`.
/// 2. Slice `attn` into the cold range `[..., :T_cold]` and the hot range
///    `[..., T_cold:]`.
/// 3. **Cold contribution**: dispatch the fused kernel
///    [`ffi::turbo4_delegated_cold_weighted_sum`] which reads the packed cold
///    V directly. Returns the *unrotated* cold weighted sum
///    `[B*Hq, Tq, D]` fp32. Apply the inverse Turbo4 rotation
///    `signs1 · WHT(signs2 · ·)` on the host to produce the rotated cold
///    contribution `[B, Hq, Tq, D]` fp32.
/// 4. **Hot contribution**: standard MLX matmul `attn_hot @ hot_v`, using
///    the unrotated FP16 hot V. The host graph keeps this on the steel
///    attention / NAX matmul path so the small T_hot batch (≤ 1024 tokens at
///    `DELEGATED_HOT_MAX`) stays cheap.
/// 5. Sum the two contributions and cast to FP16.
///
/// At no point does the dequantised cold V exist as a tensor in global
/// memory. The dequant is computed inside the kernel into thread-local
/// registers and consumed by the weighted sum in the same dispatch — that
/// is the whole point of issue #528.
///
/// # Inputs
///
/// - `q`: `[B, Hq, Tq, D]` query tensor (FP16 or FP32). Must already have
///   RoPE / Q-norm applied (matches the standard attention call contract).
/// - `unified_k`: `[B, Hkv, T_total, D]` FP16 — the full unified K buffer
///   (issue #527 unified the K side, dropping the cold/hot split for K).
/// - `v_packed`: `[B, Hkv, T_cold, D/2]` u8 — packed cold V indices. May be
///   empty (`T_cold == 0`) on the very first decode step before any fold.
/// - `v_rescale`: `[B, Hkv, T_cold, 1]` FP16 — precomputed per-token kernel
///   rescale `norm[t] / max(|y_hat[t]|, 1e-10)` (issue #520). Same lockstep
///   shape as the Turbo4Asym path.
/// - `hot_v`: `[B, Hkv, T_hot, D]` FP16 — plain FP16 hot V tokens. May be
///   empty (`T_hot == 0`) immediately after a full fold.
/// - `params`: `TurboQuantParams` used at quantize time (V-side sign vectors
///   + codebook).
/// - `scale`: attention scale (`1 / sqrt(D)` typically).
/// - `mask`: optional additive attention mask.
/// - `threshold_value`: alive cutoff for the cold-V kernel skipping (gated
///   on `--turbo-sparse-v-threshold`). `0.0` runs the full cold sweep
///   without skipping. Inert hot tokens are unaffected — the hot matmul
///   does not participate in the threshold path.
///
/// # Output
///
/// `Some([B, Hq, Tq, D])` FP16 attention output, or `None` when the kernel
/// path is gated off (non-macOS, non-power-of-2 head dim, or
/// `MLXCEL_SPARSE_V_KERNEL=0`). Callers should fall back to
/// [`KVCache::update_and_fetch`] + `attention()` in that case.
///
/// Used by: `KVCache::update_and_turbo4_delegated_attention` (issue #528).
#[allow(clippy::too_many_arguments)]
pub fn attention_turbo4_delegated_fused(
    q: &MlxArray,
    unified_k: &MlxArray,
    v_packed: Option<&MlxArray>,
    v_rescale: Option<&MlxArray>,
    hot_v: Option<&MlxArray>,
    params: &TurboQuantParams,
    cold_offset: i32,
    hot_offset: i32,
    scale: f32,
    mask: Option<&MlxArray>,
    threshold_value: f32,
) -> Option<UniquePtr<MlxArray>> {
    if !kernel_enabled() {
        return None;
    }
    let q_shape = ffi::array_shape(q);
    let k_shape = ffi::array_shape(unified_k);
    debug_assert_eq!(q_shape.len(), 4, "q must be 4-D [B, Hq, Tq, D]");
    debug_assert_eq!(k_shape.len(), 4, "unified_k must be 4-D [B, Hkv, T_total, D]");
    let b = q_shape[0];
    let hq = q_shape[1];
    let tq = q_shape[2];
    let head_dim = q_shape[3];
    let kv_heads = k_shape[1];
    let t_total = k_shape[2];
    debug_assert!(kv_heads > 0, "Hkv must be positive");
    debug_assert!(
        hq % kv_heads == 0,
        "Hq ({hq}) must be a multiple of Hkv ({kv_heads})"
    );
    let n_rep = hq / kv_heads;
    debug_assert_eq!(
        t_total,
        cold_offset + hot_offset,
        "unified_k length ({t_total}) must equal cold_offset ({cold_offset}) + hot_offset ({hot_offset})"
    );

    // Kernel-friendly precondition: head_dim must be a power of 2. All
    // production text models use head_dim ∈ {64, 128, 192, 256}; the 192
    // case (Gemma 4) is the lone non-power-of-2 outlier. For that we fall
    // back to the graph path (caller uses `update_and_fetch`).
    if !(head_dim as u32).is_power_of_two() {
        return None;
    }

    // 1. Compute attention scores via the standard graph path. The unified K
    //    buffer is FP16; we cast to FP32 for a stable softmax. Repeat KV
    //    heads to match Q heads (grouped attention).
    let k_for_q = if n_rep == 1 {
        ffi::contiguous(unified_k, false)
    } else {
        let kt = k_shape[2];
        let kd = k_shape[3];
        let k_exp = ffi::expand_dims(unified_k, 2);
        let k_tiled = ffi::broadcast_to(&k_exp, &[b, kv_heads, n_rep, kt, kd]);
        ffi::reshape(&k_tiled, &[b, hq, kt, kd])
    };
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
    // Single softmax over the full T_total range so the cold and hot
    // contributions stay normalised against the same denominator. Slicing
    // softmax outputs into cold/hot halves preserves bit-equivalence with a
    // dense `attn @ V_full` computation.
    let attn_full = ffi::softmax_precise(&scores, -1); // [B, Hq, Tq, T_total] f32

    // 2. Cold and hot contributions. Each may be empty depending on cache
    //    state (cold is empty pre-fold; hot is empty immediately after a
    //    full fold).
    let bhq = b * hq;
    let bhkv = b * kv_heads;

    let cold_contrib_pre_rotate = if cold_offset > 0 {
        let v_packed = v_packed.expect("v_packed must exist when cold_offset > 0");
        let v_rescale = v_rescale.expect("v_rescale must exist when cold_offset > 0");

        // Slice the cold range out of the full attention. attn_full shape:
        // [B, Hq, Tq, T_total]. We need [B, Hq, Tq, T_cold].
        let attn_cold = ffi::slice(
            &attn_full,
            &[0, 0, 0, 0],
            &[b, hq, tq, cold_offset],
        );
        let attn_cold_flat = ffi::reshape(&attn_cold, &[bhq, tq, cold_offset]);
        // v_packed graph shape: [B, Hkv, T_cold, D/2]; flatten to
        // [B*Hkv, T_cold, D/2].
        let v_packed_flat = ffi::reshape(v_packed, &[bhkv, cold_offset, head_dim / 2]);
        // v_rescale graph shape: [B, Hkv, T_cold, 1]; drop the trailing axis
        // and flatten to [B*Hkv, T_cold].
        let v_rescale_flat = ffi::reshape(v_rescale, &[bhkv, cold_offset]);
        let codebook_vec: Vec<f32> = params.codebook.centroids.as_ref().to_vec();
        let codebook_arr =
            ffi::from_slice_f32(&codebook_vec, &[codebook_vec.len() as i32]);

        let out_cold_pre_flat = ffi::turbo4_delegated_cold_weighted_sum(
            &attn_cold_flat,
            &v_packed_flat,
            &v_rescale_flat,
            &codebook_arr,
            head_dim,
            n_rep,
            threshold_value,
        );
        let out_cold_pre = ffi::reshape(&out_cold_pre_flat, &[b, hq, tq, head_dim]);
        // Apply inverse Turbo4 rotation `signs1 · WHT(signs2 · ·)` on the
        // small `[B, Hq, Tq, D]` output. This is the exact rotation
        // `dequantize_v_turbo4` applies per-token; doing it once on the
        // weighted sum is mathematically equivalent (linearity) and runs
        // O(Hq · Tq · D log D) instead of O(Hkv · T_cold · D log D).
        let signs1_arr =
            ffi::from_slice_f32(&params.signs1, &[1, 1, 1, head_dim]);
        let signs2_arr =
            ffi::from_slice_f32(&params.signs2, &[1, 1, 1, head_dim]);
        let pre_h = ffi::multiply(&out_cold_pre, &signs2_arr);
        let post_h = crate::ops::wht(&pre_h);
        let post_d1 = ffi::multiply(&post_h, &signs1_arr);
        Some(post_d1)
    } else {
        None
    };

    let hot_contrib = if hot_offset > 0 {
        let hot_v = hot_v.expect("hot_v must exist when hot_offset > 0");
        // Slice the hot range out of the full attention. attn_full shape:
        // [B, Hq, Tq, T_total]; hot range is [..., cold_offset:].
        let attn_hot = ffi::slice(
            &attn_full,
            &[0, 0, 0, cold_offset],
            &[b, hq, tq, t_total],
        );
        // Repeat hot V along Hkv → Hq for grouped attention. attn_hot is
        // [B, Hq, Tq, T_hot]; hot_v is [B, Hkv, T_hot, D]. After the repeat
        // we get [B, Hq, T_hot, D] for the matmul.
        let hot_v_for_q = if n_rep == 1 {
            ffi::contiguous(hot_v, false)
        } else {
            let hv_exp = ffi::expand_dims(hot_v, 2);
            let hv_tiled = ffi::broadcast_to(&hv_exp, &[b, kv_heads, n_rep, hot_offset, head_dim]);
            ffi::reshape(&hv_tiled, &[b, hq, hot_offset, head_dim])
        };
        let hot_v_f32 = ffi::astype(&hot_v_for_q, dtype::FLOAT32);
        // matmul([B, Hq, Tq, T_hot], [B, Hq, T_hot, D]) = [B, Hq, Tq, D] f32.
        Some(ffi::matmul(&attn_hot, &hot_v_f32))
    } else {
        None
    };

    // 3. Sum cold and hot contributions. Both are FP32; cast to FP16 at the
    //    end to match the public attention contract.
    let combined = match (cold_contrib_pre_rotate, hot_contrib) {
        (Some(c), Some(h)) => ffi::add(&c, &h),
        (Some(c), None) => c,
        (None, Some(h)) => h,
        (None, None) => {
            // Empty cache (offset == 0). Should not occur in practice: a
            // decode step always sees at least the just-appended token. We
            // return zeros of the right shape for total safety.
            ffi::zeros(&[b, hq, tq, head_dim], dtype::FLOAT32)
        }
    };
    Some(ffi::astype(&combined, dtype::FLOAT16))
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

    /// Turbo4Delegated fused-kernel env-var parsing — truthy / falsy / unset.
    ///
    /// Same OnceLock caveat as `threshold_default_when_unset`: we cannot
    /// poke the global `TURBO4_DELEGATED_FUSED_ENABLED` OnceLock from a test
    /// because other tests may have already populated it. We exercise the
    /// parse closure inline instead.
    #[test]
    fn turbo4_delegated_fused_parse_logic() {
        let parse = |s: &str| -> bool {
            matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        };
        // Truthy values
        assert!(parse("1"));
        assert!(parse("true"));
        assert!(parse("True"));
        assert!(parse("TRUE"));
        assert!(parse("on"));
        assert!(parse("ON"));
        assert!(parse("yes"));
        assert!(parse("YES"));
        // Falsy / unrecognised → default off
        assert!(!parse("0"));
        assert!(!parse("false"));
        assert!(!parse("off"));
        assert!(!parse("no"));
        assert!(!parse(""));
        assert!(!parse("enabled"));
    }
}

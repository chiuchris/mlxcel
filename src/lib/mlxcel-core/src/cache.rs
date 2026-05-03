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

//! Attention cache state machines shared by text and VLM families.
//!
//! These types keep cache growth, rewinding, and sliding-window semantics in
//! one place so `layers.rs` can focus on layer math while models continue to
//! import the same cache types via `mlxcel_core::layers`.
//!
//! # KV Cache Quantization
//!
//! `KVCache` optionally stores keys/values in INT8 to reduce memory by ~50%.
//! Enable via `KVCacheMode::Int8` at construction time. The `update_and_fetch`
//! method always returns FP16 tensors (dequantized on read), so the attention
//! computation is unaffected.
//!
//! `KVCacheMode::Turbo4Asym` (issue #474, epic #458) keeps the K side as FP16
//! and compresses the V side to 4-bit PolarQuant indices plus per-token norms,
//! reducing total KV memory by ~26% at long context. The compressed V buffers
//! live in dedicated sidecar fields (`v_packed`, `v_norms`) and the
//! quantize/dequantize helpers in [`turbo::quant`] handle the math.
//!
//! `KVCacheMode::Turbo3Asym` (issue #477, epic #458) is the 3-bit sibling of
//! `Turbo4Asym` — same Fp16-K + asymmetric layout, but the V side uses the
//! 8-centroid (3-bit) Lloyd-Max codebook and the 24-bit-grouped packing
//! layout from [`turbo::pack3`]. Compression climbs to ~5.1× total KV
//! savings (vs ~3.8× for `Turbo4Asym`) at the cost of a slightly larger
//! V-reconstruction error. Symmetric Turbo3 is an explicit non-goal of
//! this PR — see [`turbo::quant3`] for rationale.
//!
//! `KVCacheMode::Turbo4` (issue #476, epic #458) extends Turbo4Asym to a
//! **symmetric** 4-bit K + 4-bit V layout, reducing KV memory by ~73% at
//! long context. The K side mirrors the V side bit-for-bit but uses an
//! independent pair of sign vectors (see [`turbo::quant::K_SEED_OFFSET`]).
//! Symmetric Turbo4 is **dangerous** on dense Q4_K_M weights — see the
//! [`turbo::allowlist`] module for the per-model allowlist that gates
//! end-user opt-in.
//!
//! `KVCacheMode::Turbo4Delegated` (issue #479, epic #458) extends the
//! asymmetric mode with a hybrid hot/cold split that recovers 97–100% of FP16
//! decode speed at long context. During prefill tokens accumulate in the
//! standard FP16 `keys`/`values` buffers (zero overhead). On the first decode
//! step the prefilled body is "folded" into cold storage: K is moved to
//! `cold_keys` (still FP16) and V is quantized to packed Turbo4 in
//! `v_packed`/`v_norms`. Subsequent decode tokens flow through the small
//! pre-allocated FP16 `keys`/`values` hot tail with zero-alloc slice-update.
//! When the hot tail crosses [`turbo::DELEGATED_HOT_THRESHOLD`] tokens, the
//! oldest hot block is folded into cold storage. SDPA always reads FP16:
//! reads return `concat(cold_keys, hot_K)` for K and
//! `concat(dequant(v_packed), hot_V)` for V. See `references/turboquant_plus/
//! README.md` §"MLX Framework Port" for the original architecture.

mod detach;
mod paged;
mod paged_detach;
#[cfg(test)]
#[path = "cache/paged_turbo_tests.rs"]
mod paged_turbo_tests;
#[cfg(test)]
#[path = "cache/sparse_v_tests.rs"]
mod sparse_v_tests;
pub mod turbo;
#[cfg(test)]
#[path = "cache/turbo_tests.rs"]
mod turbo_tests;

pub use detach::{DetachedCacheSet, DetachedHandle, DetachedKVCache, DetachedRotatingKVCache};
pub use paged::{
    PagedBlockId, PagedBlockPool, PagedCacheStats, PagedKvLayout, PagedLayerState,
    PagedSequenceState,
};
pub use paged_detach::DetachedPagedCacheSet;

use crate::concatenate;
use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::ops::divide_scalar;
use cxx::UniquePtr;

fn direct_prefill_cache_store_enabled() -> bool {
    std::env::var("MLXCEL_ENABLE_DIRECT_PREFILL_CACHE_STORE").is_ok()
}

/// Storage mode for KV cache tensors.
///
/// Controls the on-device representation of accumulated key/value tensors.
/// The public `update_and_fetch` interface always returns FP16 regardless of
/// the chosen mode, so attention kernels are unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KVCacheMode {
    /// Standard half-precision storage (default). No quantization overhead.
    #[default]
    Fp16,
    /// Per-token INT8 absmax quantization. Reduces KV cache memory by ~50%
    /// at the cost of small quantization error per token.
    Int8,
    /// Asymmetric Fp16-K + Turbo4-V. K side stays in FP16; V side uses 4-bit
    /// PolarQuant with Walsh–Hadamard rotation for ~26% net KV memory savings
    /// at long context. See issue #474 / epic #458.
    Turbo4Asym,
    /// Asymmetric Fp16-K + Turbo3-V. K side stays in FP16; V side uses 3-bit
    /// PolarQuant with Walsh–Hadamard rotation for ~5.1× total KV memory
    /// savings (vs ~3.8× for `Turbo4Asym`) at the cost of a slightly higher
    /// V-reconstruction error. The 3-bit packing is awkward (8 coords share
    /// 3 bytes / 24 bits — see [`turbo::pack3`]), so the dequant path runs
    /// a host round-trip rather than the 4-bit path's pure on-device unpack.
    /// Symmetric Turbo3 is **not** offered in this mode (epic #458's
    /// "Quality–compression tradeoff control" explicitly defers it). See
    /// issue #477 / epic #458.
    Turbo3Asym,
    /// Symmetric Turbo4-K + Turbo4-V. Both K and V use 4-bit PolarQuant with
    /// independent Walsh–Hadamard rotations for ~73% net KV memory savings
    /// at long context. **Dangerous on dense Q4_K_M weights** — gated by
    /// the per-model allowlist in [`turbo::allowlist`]. See issue #476 /
    /// epic #458.
    Turbo4,
    /// Delegated hot/cold split on top of `Turbo4Asym`. Prefill stores raw
    /// FP16; on the first decode step the prefilled body is folded into cold
    /// storage (FP16 cold K + Turbo4 packed cold V); subsequent decode tokens
    /// flow through a small FP16 hot tail with zero-alloc slice-update. SDPA
    /// always reads FP16. Targets ≥97%-of-FP16 decode speed at 4K and ≥95%
    /// at 16K on M5 Max. See issue #479 / epic #458.
    Turbo4Delegated,
}

impl std::str::FromStr for KVCacheMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "fp16" | "float16" => Ok(Self::Fp16),
            "int8" | "i8" => Ok(Self::Int8),
            // Both spellings are accepted: the canonical user-facing string
            // ("fp16+turbo4") makes the asymmetric K/V split explicit, while
            // "turbo4-asym" is a shorter alias for scripts and tests.
            "turbo4-asym" | "fp16+turbo4" => Ok(Self::Turbo4Asym),
            // Turbo3Asym (3-bit V, asymmetric only — issue #477). Same alias
            // pattern as Turbo4Asym: "fp16+turbo3" is the canonical string,
            // "turbo3-asym" / "turbo3" are shorter forms used in scripts and
            // tests. There is intentionally no symmetric "turbo3" alias —
            // symmetric 3-bit on dense Q4_K_M weights is catastrophic and
            // explicitly out of scope for this PR (see [`turbo::quant3`]).
            "turbo3-asym" | "fp16+turbo3" | "turbo3" => Ok(Self::Turbo3Asym),
            // Symmetric Turbo4 (4-bit K + 4-bit V). The canonical user-facing
            // string is "turbo4"; "turbo4-sym" is an explicit alias used in
            // tests/scripts when readers benefit from the K/V symmetry being
            // spelled out.
            "turbo4" | "turbo4-sym" => Ok(Self::Turbo4),
            // Delegated hot/cold split (B7, issue #479). Same K=FP16 +
            // V=Turbo4 contract as Turbo4Asym but uses the delegated KVCache
            // architecture for decode speed at long context.
            "turbo4-delegated" | "fp16+turbo4-delegated" => Ok(Self::Turbo4Delegated),
            other => Err(format!(
                "unknown kv-cache-mode \"{other}\"; expected one of \
                 \"fp16\", \"int8\", \"fp16+turbo4\" (alias \"turbo4-asym\"), \
                 \"turbo4\" (alias \"turbo4-sym\"), \
                 \"turbo4-delegated\" (alias \"fp16+turbo4-delegated\"), \
                 \"fp16+turbo3\" (aliases \"turbo3-asym\" / \"turbo3\")"
            )),
        }
    }
}

impl std::fmt::Display for KVCacheMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp16 => f.write_str("fp16"),
            Self::Int8 => f.write_str("int8"),
            Self::Turbo4Asym => f.write_str("fp16+turbo4"),
            Self::Turbo4 => f.write_str("turbo4"),
            Self::Turbo4Delegated => f.write_str("turbo4-delegated"),
            Self::Turbo3Asym => f.write_str("fp16+turbo3"),
        }
    }
}

// ---------------------------------------------------------------------------
// INT8 quantization helpers
// ---------------------------------------------------------------------------

/// Quantize a tensor to INT8 using per-token absmax scaling.
///
/// `x` has shape `[B, H, T, D]` where T is typically 1 (one new token).
/// Returns `(x_int8, scale)` where:
/// - `x_int8`: `[B, H, T, D]` INT8
/// - `scale`:  `[B, H, T, 1]` FP16 — the absmax / 127.0 for each token
///
/// Used by: QuantizedKVCache (INT8 mode of KVCache)
fn quantize_per_token(x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Compute per-token absmax: reduce over last dim (head_dim), keepdims
    let abs_x = ffi::abs(x);
    let absmax = ffi::max_axis(&abs_x, -1, true); // [B, H, T, 1]

    // scale = absmax / 127.0  (FP16 to match cache dtype)
    let scale = divide_scalar(&absmax, 127.0); // [B, H, T, 1]

    // Avoid divide-by-zero: replace zero scales with 1.0
    let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT16);
    let safe_scale = ffi::maximum(&scale, &one);

    // x_int8 = round(x / safe_scale).clamp(-128, 127)
    let x_div = ffi::divide(x, &safe_scale);
    let x_rounded = ffi::round(&x_div);
    let lo = ffi::full_f32(&[1], -128.0, ffi::array_dtype(x));
    let hi = ffi::full_f32(&[1], 127.0, ffi::array_dtype(x));
    let x_clipped = ffi::clip(&x_rounded, &lo, &hi);
    let x_int8 = ffi::astype(&x_clipped, dtype::INT8);

    (x_int8, safe_scale)
}

/// Dequantize INT8 tensor back to FP16 for attention computation.
///
/// `x_int8`: `[B, H, L, D]` INT8
/// `scale`:  `[B, H, L, 1]` FP16
/// Returns:  `[B, H, L, D]` FP16
///
/// Used by: QuantizedKVCache (INT8 mode of KVCache)
fn dequantize(x_int8: &MlxArray, scale: &MlxArray) -> UniquePtr<MlxArray> {
    let x_fp16 = ffi::astype(x_int8, dtype::FLOAT16);
    ffi::multiply(&x_fp16, scale)
}

/// Default Turbo4Asym sign-vector seed when the caller does not supply one.
///
/// Hard-coded so two cache instances built independently with the same V
/// `head_dim` yield identical rotations — important for reproducibility and
/// for the detach/adopt round-trip when the Turbo4 sidecars travel without
/// the originating cache.
pub(crate) const TURBO_DEFAULT_SEED: u32 = 0x7B4_70404; // "TUR" 0x474 + B2 issue 474

/// KV Cache for attention layers.
///
/// Uses pre-allocated buffers with slice_update for O(1) per-token updates,
/// matching Python mlx-lm's KVCache implementation. Buffers grow by `step`
/// slots at a time (default 256) to amortize allocation cost.
///
/// When `mode` is `KVCacheMode::Int8`, keys and values are stored as INT8
/// tensors with per-token scale factors. `update_and_fetch` always returns
/// FP16 (dequantized) so attention kernels see standard tensors.
///
/// When `mode` is `KVCacheMode::Turbo4Asym`, the keys buffer stays FP16
/// while the values buffer is replaced by `v_packed` (nibble-packed 4-bit
/// PolarQuant indices) plus `v_norms` (per-token fp16 L2 norms). The
/// per-cache `turbo_params` field carries the deterministic sign vectors
/// and shared codebook used by the quantize/dequantize helpers in
/// [`turbo::quant`]. The standard `values` field stays `None` in this mode.
///
/// When `mode` is `KVCacheMode::Turbo4` (symmetric, issue #476), the K
/// buffers are *also* replaced by `k_packed` + `k_norms` sidecars; the
/// `keys` field stays `None`. The same `turbo_params` instance carries an
/// independent K-side sign-vector pair so the K and V quantization noise
/// is uncorrelated.
///
/// Used by: All transformer models (Llama, Qwen, Gemma, etc.). The shared
/// `update`, `update_and_fetch`, `trim`, `nbytes`, and
/// `bytes_per_reserved_token` methods dispatch over `mode` and support
/// `KVCacheMode::Fp16`, `KVCacheMode::Int8`, `KVCacheMode::Turbo4Asym`
/// (epic #458, issue #474), and `KVCacheMode::Turbo4` (epic #458,
/// issue #476) without per-model branching.
pub struct KVCache {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    pub offset: i32,
    step: i32,
    /// Quantization mode for stored keys/values.
    pub mode: KVCacheMode,
    // INT8-mode scale factors: [B, H, L, 1] FP16, None when mode == Fp16
    pub(crate) key_scales: Option<UniquePtr<MlxArray>>,
    pub(crate) val_scales: Option<UniquePtr<MlxArray>>,
    // Turbo4Asym/Turbo4-mode V-side packed indices: [B, H, L, head_dim/2] u8
    pub(crate) v_packed: Option<UniquePtr<MlxArray>>,
    // Turbo4Asym/Turbo4-mode V-side per-token norms: [B, H, L, 1] fp16
    pub(crate) v_norms: Option<UniquePtr<MlxArray>>,
    // Turbo4-V precomputed kernel rescale `norm[t] / |y_hat[t]|` (issue #520).
    // Same `[B, H, L, 1]` fp16 shape as `v_norms` and slice/concat/trimmed
    // lockstep with it. Populated only on Turbo4Asym / Turbo4 / Turbo4Delegated
    // updates — Turbo3Asym leaves this `None` because the 3-bit V kernel does
    // not exist. Consumed by `attention_sparse_v_turbo4_fused` to skip the
    // per-cache-token threadgroup tree reduction that previously dominated
    // decode latency at 4 K context (PR #519 A/B; issue #520).
    pub(crate) v_rescale: Option<UniquePtr<MlxArray>>,
    // Turbo4-mode (symmetric) K-side packed indices: [B, H, L, head_dim/2] u8
    pub(crate) k_packed: Option<UniquePtr<MlxArray>>,
    // Turbo4-mode (symmetric) K-side per-token norms: [B, H, L, 1] fp16
    pub(crate) k_norms: Option<UniquePtr<MlxArray>>,
    /// Cached PolarQuant params (sign vectors + codebook) for Turbo4* modes
    /// (Turbo4Asym / Turbo4 symmetric / Turbo4Delegated).
    ///
    /// Lazily initialised on the first Turbo4 update once the V `head_dim` is
    /// known. The deterministic seed is derived from the cache's `turbo_seed`
    /// so detach/adopt and re-construction reproduce the same rotation. In
    /// symmetric `Turbo4` mode the same params carry both V and K sign
    /// vectors (`signs1`/`signs2` for V, `k_signs1`/`k_signs2` for K).
    pub(crate) turbo_params: Option<turbo::TurboQuantParams>,
    /// Cached PolarQuant params for `Turbo3Asym` (issue #477). The 3-bit
    /// codebook (8 centroids) is incompatible with the 4-bit `turbo_params`
    /// codebook so a separate field is required. Lazily initialised on the
    /// first `Turbo3Asym` update once the V `head_dim` is known. Stays
    /// `None` for non-Turbo3 modes.
    pub(crate) turbo3_params: Option<turbo::quant3::TurboQuantParams3>,
    /// Deterministic seed for the Turbo4 sign vectors. Set at construction
    /// time so detach/adopt round-trip without recomputing rotations.
    pub(crate) turbo_seed: u32,
    /// Turbo4Delegated cold-side FP16 K buffer: `[B, H, cold_capacity, K_dim]`.
    ///
    /// Holds the tokens that have been folded out of the hot tail. Pre-allocated
    /// in `step`-sized increments just like the standard FP16 `keys` buffer.
    /// `None` until the first fold (i.e. the prefill→decode transition).
    pub(crate) cold_keys: Option<UniquePtr<MlxArray>>,
    /// Number of tokens currently held in cold storage (Turbo4Delegated only).
    ///
    /// Invariant: `cold_offset >= 0 && cold_offset <= offset`. The hot tail
    /// length is `offset - cold_offset`. In all non-delegated modes
    /// `cold_offset` stays at 0 and `cold_keys`/`v_packed`/`v_norms` are
    /// either `None` or behave per the per-mode invariants documented above.
    pub(crate) cold_offset: i32,
    /// Hot-tail fold threshold (Turbo4Delegated only). Mutable so test harness
    /// and benchmarks can configure it without going through the env var.
    pub(crate) hot_threshold: i32,
}

impl KVCache {
    /// Create a new empty KV cache with default step size (256) and FP16 mode.
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            step: 256,
            mode: KVCacheMode::Fp16,
            key_scales: None,
            val_scales: None,
            v_packed: None,
            v_norms: None,
            v_rescale: None,
            k_packed: None,
            k_norms: None,
            turbo_params: None,
            turbo3_params: None,
            turbo_seed: TURBO_DEFAULT_SEED,
            cold_keys: None,
            cold_offset: 0,
            hot_threshold: turbo::DELEGATED_HOT_THRESHOLD,
        }
    }

    /// Create a new empty KV cache with the specified quantization mode.
    ///
    /// Use `KVCacheMode::Int8` to store accumulated keys/values in INT8 format.
    /// Use `KVCacheMode::Turbo4Asym` for asymmetric Fp16-K + Turbo4-V
    /// compression (issue #474). Use `KVCacheMode::Turbo4` for symmetric
    /// Turbo4-K + Turbo4-V compression (issue #476) — note that this mode
    /// is dangerous on dense Q4_K_M weights and must be gated by the
    /// per-model allowlist in [`turbo::allowlist`] before being exposed
    /// to end users. The `update_and_fetch` method will transparently
    /// quantize incoming tensors and dequantize them on read, so callers
    /// always receive FP16.
    pub fn new_with_mode(mode: KVCacheMode) -> Self {
        Self::new_with_mode_and_seed(mode, TURBO_DEFAULT_SEED)
    }

    /// Like [`KVCache::new_with_mode`] but lets the caller pin a specific
    /// Turbo4 sign-vector seed.
    ///
    /// Production callers should prefer `new_with_mode` and let the cache
    /// pool layer disambiguate seeds across layers (see `make_caches` paths
    /// per model). The explicit seed is exposed for tests, benchmarks, and
    /// the detach/adopt path which must preserve the original seed.
    pub fn new_with_mode_and_seed(mode: KVCacheMode, turbo_seed: u32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            step: 256,
            mode,
            key_scales: None,
            val_scales: None,
            v_packed: None,
            v_norms: None,
            v_rescale: None,
            k_packed: None,
            k_norms: None,
            turbo_params: None,
            turbo3_params: None,
            turbo_seed,
            cold_keys: None,
            cold_offset: 0,
            hot_threshold: turbo::DELEGATED_HOT_THRESHOLD,
        }
    }

    /// Override the Turbo4Delegated hot-tail fold threshold for this cache.
    ///
    /// Test-only entry point for the unit/integration tests in `turbo_tests.rs`
    /// and the speed benchmarks. Production callers should leave the default
    /// [`turbo::DELEGATED_HOT_THRESHOLD`] in place; tuning this changes the
    /// fold cadence and therefore the speed/quality trade-off documented in
    /// `references/turboquant_plus/README.md`. Setting `threshold <= 0` is
    /// rejected so the fold path stays well-defined.
    pub fn set_hot_threshold(&mut self, threshold: i32) {
        if threshold > 0 {
            self.hot_threshold = threshold;
        }
    }

    /// Read the configured hot-tail fold threshold (Turbo4Delegated only;
    /// returns the default for other modes).
    pub fn hot_threshold(&self) -> i32 {
        self.hot_threshold
    }

    /// Number of tokens currently held in cold packed storage. Always 0 unless
    /// `mode == Turbo4Delegated`.
    pub fn cold_offset(&self) -> i32 {
        self.cold_offset
    }

    /// Number of tokens currently held in the FP16 hot tail. Equals
    /// `offset - cold_offset` and matches `seq_len()` when no folds have
    /// happened (i.e. during prefill or before the prefill→decode transition).
    pub fn hot_offset(&self) -> i32 {
        self.offset - self.cold_offset
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        // Symmetric Turbo4 has no `keys` buffer (K is packed into
        // `k_packed`), so check both. The cache is empty iff neither the
        // dense K buffer nor the packed K buffer is allocated.
        self.keys.is_none() && self.k_packed.is_none()
    }

    /// Get current sequence length in cache
    pub fn seq_len(&self) -> i32 {
        self.offset
    }

    /// Get the allocated buffer size (sequence dimension)
    fn buffer_seq_len(&self) -> i32 {
        // In symmetric Turbo4 mode the `keys` field stays None and the
        // step-grown buffer lives in `k_packed`; consult that instead. All
        // other modes (Fp16, Int8, Turbo4Asym) keep the buffer in `keys`.
        let buf = self.keys.as_ref().or(self.k_packed.as_ref());
        match buf {
            Some(k) => {
                let shape = ffi::array_shape(k);
                if shape.len() >= 3 {
                    shape[2]
                } else {
                    0
                }
            }
            None => 0,
        }
    }

    /// Update cache with new key/value using pre-allocated buffer + slice_update.
    ///
    /// In `KVCacheMode::Int8` the incoming tensors are quantized to INT8 before
    /// storage; scale factors are accumulated in a parallel `[B, H, L, 1]`
    /// buffer. In `KVCacheMode::Turbo4Asym` the K side stays FP16 while the V
    /// side is quantized to packed 4-bit indices plus per-token norms via
    /// [`turbo::quant::quantize_v_turbo4`]. In `KVCacheMode::Turbo4` (issue
    /// #476) **both** K and V are quantized via the symmetric Turbo4 path
    /// using independent sign-vector pairs. In `KVCacheMode::Fp16` this
    /// behaves identically to the original implementation.
    pub fn update(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
        match self.mode {
            KVCacheMode::Int8 => self.update_int8(new_keys, new_values),
            KVCacheMode::Turbo4Asym => self.update_turbo4_asym(new_keys, new_values),
            KVCacheMode::Turbo4 => self.update_turbo4_sym(new_keys, new_values),
            KVCacheMode::Turbo4Delegated => self.update_turbo4_delegated(new_keys, new_values),
            KVCacheMode::Turbo3Asym => self.update_turbo3_asym(new_keys, new_values),
            KVCacheMode::Fp16 => self.update_fp16(new_keys, new_values),
        }
    }

    /// FP16 (standard) update path — original pre-allocated buffer logic.
    fn update_fp16(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
        let key_shape = ffi::array_shape(&new_keys);
        let new_seq_len = key_shape[2];
        let prev = self.offset;

        if prev == 0 && self.keys.is_none() && direct_prefill_cache_store_enabled() {
            self.keys = Some(ffi::contiguous(&new_keys, false));
            self.values = Some(ffi::contiguous(&new_values, false));
            self.offset = new_seq_len;
            return;
        }

        if self.keys.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_head_dim = key_shape[3];
            let val_shape = ffi::array_shape(&new_values);
            let v_head_dim = val_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let k_dtype = ffi::array_dtype(&new_keys);
            let v_dtype = ffi::array_dtype(&new_values);
            let new_k = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], k_dtype);
            let new_v = ffi::zeros(&[b, n_kv_heads, buf_size, v_head_dim], v_dtype);

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_head_dim],
                    ));
                    self.values = Some(ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_head_dim],
                    ));
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k, 2));
                self.values = Some(concatenate(self.values.as_ref().unwrap(), &new_v, 2));
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        self.offset += new_seq_len;

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &new_keys,
            &[0, 0, prev, 0],
            &[k_shape[0], k_shape[1], self.offset, k_shape[3]],
        ));
        self.values = Some(ffi::slice_update(
            self.values.as_ref().unwrap(),
            &new_values,
            &[0, 0, prev, 0],
            &[v_shape[0], v_shape[1], self.offset, v_shape[3]],
        ));
    }

    /// INT8 update path — quantizes incoming K/V tokens and accumulates into
    /// INT8 key/value buffers alongside FP16 per-token scale buffers.
    ///
    /// Layout of stored buffers (step-aligned, grown lazily):
    /// - `keys`/`values`: `[B, H, capacity, D]` INT8
    /// - `key_scales`/`val_scales`: `[B, H, capacity, 1]` FP16
    fn update_int8(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
        // Cast incoming tensors to FP16 before quantization so scale
        // computation operates in a consistent dtype.
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        let (k_int8, k_scale) = quantize_per_token(&new_keys_f16);
        let (v_int8, v_scale) = quantize_per_token(&new_values_f16);

        let key_shape = ffi::array_shape(&k_int8);
        let new_seq_len = key_shape[2];
        let prev = self.offset;

        if prev == 0 && self.keys.is_none() && direct_prefill_cache_store_enabled() {
            self.keys = Some(ffi::contiguous(&k_int8, false));
            self.values = Some(ffi::contiguous(&v_int8, false));
            self.key_scales = Some(ffi::contiguous(&k_scale, false));
            self.val_scales = Some(ffi::contiguous(&v_scale, false));
            self.offset = new_seq_len;
            return;
        }

        if self.keys.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_head_dim = key_shape[3];
            let val_shape = ffi::array_shape(&v_int8);
            let v_head_dim = val_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let new_k_buf = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], dtype::INT8);
            let new_v_buf = ffi::zeros(&[b, n_kv_heads, buf_size, v_head_dim], dtype::INT8);
            let new_ks_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            let new_vs_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_head_dim],
                    ));
                    self.values = Some(ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_head_dim],
                    ));
                    self.key_scales = Some(ffi::slice(
                        self.key_scales.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                    self.val_scales = Some(ffi::slice(
                        self.val_scales.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k_buf, 2));
                self.values = Some(concatenate(self.values.as_ref().unwrap(), &new_v_buf, 2));
                self.key_scales = Some(concatenate(
                    self.key_scales.as_ref().unwrap(),
                    &new_ks_buf,
                    2,
                ));
                self.val_scales = Some(concatenate(
                    self.val_scales.as_ref().unwrap(),
                    &new_vs_buf,
                    2,
                ));
            } else {
                self.keys = Some(new_k_buf);
                self.values = Some(new_v_buf);
                self.key_scales = Some(new_ks_buf);
                self.val_scales = Some(new_vs_buf);
            }
        }

        self.offset += new_seq_len;

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
        let ks_shape = ffi::array_shape(self.key_scales.as_ref().unwrap());
        let vs_shape = ffi::array_shape(self.val_scales.as_ref().unwrap());

        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &k_int8,
            &[0, 0, prev, 0],
            &[k_shape[0], k_shape[1], self.offset, k_shape[3]],
        ));
        self.values = Some(ffi::slice_update(
            self.values.as_ref().unwrap(),
            &v_int8,
            &[0, 0, prev, 0],
            &[v_shape[0], v_shape[1], self.offset, v_shape[3]],
        ));
        self.key_scales = Some(ffi::slice_update(
            self.key_scales.as_ref().unwrap(),
            &k_scale,
            &[0, 0, prev, 0],
            &[ks_shape[0], ks_shape[1], self.offset, 1],
        ));
        self.val_scales = Some(ffi::slice_update(
            self.val_scales.as_ref().unwrap(),
            &v_scale,
            &[0, 0, prev, 0],
            &[vs_shape[0], vs_shape[1], self.offset, 1],
        ));
    }

    /// Turbo4Asym update path — keeps the K side FP16 (mirroring `update_fp16`)
    /// and quantizes the V side via PolarQuant + Walsh–Hadamard rotation.
    ///
    /// Layout of stored buffers (step-aligned, grown lazily):
    /// - `keys`:    `[B, H, capacity, K_dim]` FP16, identical to the FP16 path.
    /// - `v_packed`: `[B, H, capacity, V_dim/2]` UINT8 (nibble-packed indices).
    /// - `v_norms`:  `[B, H, capacity, 1]` FP16 (per-token L2 of original V).
    ///
    /// `values` stays `None` in this mode — the FP16 values tensor is
    /// reconstructed lazily on `update_and_fetch` via
    /// [`turbo::quant::dequantize_v_turbo4`].
    ///
    /// Used by: `KVCache::update` dispatch when `mode == KVCacheMode::Turbo4Asym`
    /// (epic #458, issue #474).
    fn update_turbo4_asym(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) {
        // Cast incoming K/V to FP16 to match the cache contract.
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        // Lazy-init TurboQuantParams once we know the V head_dim.
        if self.turbo_params.is_none() {
            let v_shape = ffi::array_shape(&new_values_f16);
            let v_head_dim = v_shape[3] as u32;
            self.turbo_params = Some(turbo::TurboQuantParams::new(v_head_dim, self.turbo_seed));
        }
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params just initialised");

        let (v_packed_new, v_norms_new, v_rescale_new) =
            turbo::quant::quantize_v_turbo4(&new_values_f16, params);

        let key_shape = ffi::array_shape(&new_keys_f16);
        let new_seq_len = key_shape[2];
        let prev = self.offset;
        let v_packed_shape = ffi::array_shape(&v_packed_new);

        if prev == 0 && self.keys.is_none() && direct_prefill_cache_store_enabled() {
            self.keys = Some(ffi::contiguous(&new_keys_f16, false));
            self.v_packed = Some(ffi::contiguous(&v_packed_new, false));
            self.v_norms = Some(ffi::contiguous(&v_norms_new, false));
            self.v_rescale = Some(ffi::contiguous(&v_rescale_new, false));
            self.offset = new_seq_len;
            return;
        }

        if self.keys.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_head_dim = key_shape[3];
            let v_packed_dim = v_packed_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let new_k_buf = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], dtype::FLOAT16);
            let new_vp_buf = ffi::zeros(&[b, n_kv_heads, buf_size, v_packed_dim], dtype::UINT8);
            let new_vn_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            let new_vr_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_head_dim],
                    ));
                    self.v_packed = Some(ffi::slice(
                        self.v_packed.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_packed_dim],
                    ));
                    self.v_norms = Some(ffi::slice(
                        self.v_norms.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                    if let Some(ref vr) = self.v_rescale {
                        self.v_rescale =
                            Some(ffi::slice(vr, &[0, 0, 0, 0], &[b, n_kv_heads, prev, 1]));
                    }
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k_buf, 2));
                self.v_packed = Some(concatenate(self.v_packed.as_ref().unwrap(), &new_vp_buf, 2));
                self.v_norms = Some(concatenate(self.v_norms.as_ref().unwrap(), &new_vn_buf, 2));
                self.v_rescale = match self.v_rescale.as_ref() {
                    Some(vr) => Some(concatenate(vr, &new_vr_buf, 2)),
                    None => Some(new_vr_buf),
                };
            } else {
                self.keys = Some(new_k_buf);
                self.v_packed = Some(new_vp_buf);
                self.v_norms = Some(new_vn_buf);
                self.v_rescale = Some(new_vr_buf);
            }
        }

        self.offset += new_seq_len;

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let vp_shape = ffi::array_shape(self.v_packed.as_ref().unwrap());
        let vn_shape = ffi::array_shape(self.v_norms.as_ref().unwrap());

        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &new_keys_f16,
            &[0, 0, prev, 0],
            &[k_shape[0], k_shape[1], self.offset, k_shape[3]],
        ));
        self.v_packed = Some(ffi::slice_update(
            self.v_packed.as_ref().unwrap(),
            &v_packed_new,
            &[0, 0, prev, 0],
            &[vp_shape[0], vp_shape[1], self.offset, vp_shape[3]],
        ));
        self.v_norms = Some(ffi::slice_update(
            self.v_norms.as_ref().unwrap(),
            &v_norms_new,
            &[0, 0, prev, 0],
            &[vn_shape[0], vn_shape[1], self.offset, 1],
        ));
        let vr_buf = self
            .v_rescale
            .as_ref()
            .expect("v_rescale lockstep with v_norms");
        let vr_shape = ffi::array_shape(vr_buf);
        self.v_rescale = Some(ffi::slice_update(
            vr_buf,
            &v_rescale_new,
            &[0, 0, prev, 0],
            &[vr_shape[0], vr_shape[1], self.offset, 1],
        ));
    }

    /// Turbo3Asym update path — keeps the K side FP16 (mirroring `update_fp16`)
    /// and quantizes the V side via 3-bit PolarQuant + Walsh–Hadamard rotation.
    ///
    /// Layout of stored buffers (step-aligned, grown lazily):
    /// - `keys`:    `[B, H, capacity, K_dim]` FP16, identical to the FP16 path.
    /// - `v_packed`: `[B, H, capacity, V_dim*3/8]` UINT8 (24-bit-grouped indices).
    /// - `v_norms`:  `[B, H, capacity, 1]` FP16 (per-token L2 of original V).
    ///
    /// `values` stays `None` in this mode — the FP16 values tensor is
    /// reconstructed lazily on `update_and_fetch` via
    /// [`turbo::quant3::dequantize_v_turbo3`].
    ///
    /// Mirrors `update_turbo4_asym` bit-for-bit except the V quantizer is
    /// `quantize_v_turbo3` (3-bit codebook + 24-bit packing) and the
    /// dedicated `turbo3_params` field caches the 3-bit codebook so the
    /// 4-bit `turbo_params` is not perturbed.
    ///
    /// Used by: `KVCache::update` dispatch when `mode == KVCacheMode::Turbo3Asym`
    /// (epic #458, issue #477).
    fn update_turbo3_asym(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) {
        // Cast incoming K/V to FP16 to match the cache contract.
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        // Lazy-init TurboQuantParams3 once we know the V head_dim.
        if self.turbo3_params.is_none() {
            let v_shape = ffi::array_shape(&new_values_f16);
            let v_head_dim = v_shape[3] as u32;
            self.turbo3_params = Some(turbo::quant3::TurboQuantParams3::new(
                v_head_dim,
                self.turbo_seed,
            ));
        }
        let params = self
            .turbo3_params
            .as_ref()
            .expect("turbo3_params just initialised");

        let (v_packed_new, v_norms_new) = turbo::quant3::quantize_v_turbo3(&new_values_f16, params);

        let key_shape = ffi::array_shape(&new_keys_f16);
        let new_seq_len = key_shape[2];
        let prev = self.offset;
        let v_packed_shape = ffi::array_shape(&v_packed_new);

        if prev == 0 && self.keys.is_none() && direct_prefill_cache_store_enabled() {
            self.keys = Some(ffi::contiguous(&new_keys_f16, false));
            self.v_packed = Some(ffi::contiguous(&v_packed_new, false));
            self.v_norms = Some(ffi::contiguous(&v_norms_new, false));
            self.offset = new_seq_len;
            return;
        }

        if self.keys.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_head_dim = key_shape[3];
            let v_packed_dim = v_packed_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let new_k_buf = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], dtype::FLOAT16);
            let new_vp_buf = ffi::zeros(&[b, n_kv_heads, buf_size, v_packed_dim], dtype::UINT8);
            let new_vn_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_head_dim],
                    ));
                    self.v_packed = Some(ffi::slice(
                        self.v_packed.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_packed_dim],
                    ));
                    self.v_norms = Some(ffi::slice(
                        self.v_norms.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k_buf, 2));
                self.v_packed = Some(concatenate(self.v_packed.as_ref().unwrap(), &new_vp_buf, 2));
                self.v_norms = Some(concatenate(self.v_norms.as_ref().unwrap(), &new_vn_buf, 2));
            } else {
                self.keys = Some(new_k_buf);
                self.v_packed = Some(new_vp_buf);
                self.v_norms = Some(new_vn_buf);
            }
        }

        self.offset += new_seq_len;

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let vp_shape = ffi::array_shape(self.v_packed.as_ref().unwrap());
        let vn_shape = ffi::array_shape(self.v_norms.as_ref().unwrap());

        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &new_keys_f16,
            &[0, 0, prev, 0],
            &[k_shape[0], k_shape[1], self.offset, k_shape[3]],
        ));
        self.v_packed = Some(ffi::slice_update(
            self.v_packed.as_ref().unwrap(),
            &v_packed_new,
            &[0, 0, prev, 0],
            &[vp_shape[0], vp_shape[1], self.offset, vp_shape[3]],
        ));
        self.v_norms = Some(ffi::slice_update(
            self.v_norms.as_ref().unwrap(),
            &v_norms_new,
            &[0, 0, prev, 0],
            &[vn_shape[0], vn_shape[1], self.offset, 1],
        ));
    }

    /// Symmetric Turbo4 update path — quantizes both K and V to 4-bit
    /// PolarQuant indices using independent sign-vector pairs.
    ///
    /// Layout of stored buffers (step-aligned, grown lazily):
    /// - `k_packed`: `[B, H, capacity, K_dim/2]` UINT8 (K-side nibble-packed indices).
    /// - `k_norms`:  `[B, H, capacity, 1]` FP16 (per-token L2 of original K).
    /// - `v_packed`: `[B, H, capacity, V_dim/2]` UINT8 (V-side nibble-packed indices).
    /// - `v_norms`:  `[B, H, capacity, 1]` FP16 (per-token L2 of original V).
    ///
    /// Both `keys` and `values` stay `None` in this mode — the FP16 K/V
    /// tensors are reconstructed lazily on `update_and_fetch` via
    /// [`turbo::quant::dequantize_k_turbo4`] / [`turbo::quant::dequantize_v_turbo4`].
    ///
    /// **Safety**: callers must consult [`turbo::is_symmetric_turbo_allowed`]
    /// before constructing a cache in this mode for an arbitrary model — see
    /// [`turbo::allowlist`] for the rationale.
    ///
    /// Used by: `KVCache::update` dispatch when `mode == KVCacheMode::Turbo4`
    /// (epic #458, issue #476).
    fn update_turbo4_sym(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) {
        // Cast incoming K/V to FP16 to match the cache contract.
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        // Lazy-init TurboQuantParams once we know the V head_dim. Both K and
        // V must share the same head_dim because attention requires Q·Kᵀ to
        // be a valid inner product — if the model used different head_dims
        // for K and V it would be a different architecture entirely. The
        // assertion below catches any future architecture that violates
        // this assumption.
        let v_shape = ffi::array_shape(&new_values_f16);
        let k_shape_in = ffi::array_shape(&new_keys_f16);
        debug_assert_eq!(
            k_shape_in[3], v_shape[3],
            "symmetric Turbo4 requires K and V head_dim to match; \
             got K head_dim={} V head_dim={} — this model likely uses \
             differently-sized K/V projections and is incompatible with \
             this mode.",
            k_shape_in[3], v_shape[3],
        );
        if self.turbo_params.is_none() {
            let v_head_dim = v_shape[3] as u32;
            self.turbo_params = Some(turbo::TurboQuantParams::new(v_head_dim, self.turbo_seed));
        }
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params just initialised");

        let (k_packed_new, k_norms_new) = turbo::quant::quantize_k_turbo4(&new_keys_f16, params);
        let (v_packed_new, v_norms_new, v_rescale_new) =
            turbo::quant::quantize_v_turbo4(&new_values_f16, params);

        let key_shape = ffi::array_shape(&new_keys_f16);
        let new_seq_len = key_shape[2];
        let prev = self.offset;
        let k_packed_shape = ffi::array_shape(&k_packed_new);
        let v_packed_shape = ffi::array_shape(&v_packed_new);

        if prev == 0 && self.is_empty() && direct_prefill_cache_store_enabled() {
            self.k_packed = Some(ffi::contiguous(&k_packed_new, false));
            self.k_norms = Some(ffi::contiguous(&k_norms_new, false));
            self.v_packed = Some(ffi::contiguous(&v_packed_new, false));
            self.v_norms = Some(ffi::contiguous(&v_norms_new, false));
            self.v_rescale = Some(ffi::contiguous(&v_rescale_new, false));
            self.offset = new_seq_len;
            return;
        }

        if self.k_packed.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_packed_dim = k_packed_shape[3];
            let v_packed_dim = v_packed_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let new_kp_buf = ffi::zeros(&[b, n_kv_heads, buf_size, k_packed_dim], dtype::UINT8);
            let new_kn_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            let new_vp_buf = ffi::zeros(&[b, n_kv_heads, buf_size, v_packed_dim], dtype::UINT8);
            let new_vn_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            let new_vr_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);

            if self.k_packed.is_some() {
                if prev % self.step != 0 {
                    self.k_packed = Some(ffi::slice(
                        self.k_packed.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_packed_dim],
                    ));
                    self.k_norms = Some(ffi::slice(
                        self.k_norms.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                    self.v_packed = Some(ffi::slice(
                        self.v_packed.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_packed_dim],
                    ));
                    self.v_norms = Some(ffi::slice(
                        self.v_norms.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                    if let Some(ref vr) = self.v_rescale {
                        self.v_rescale =
                            Some(ffi::slice(vr, &[0, 0, 0, 0], &[b, n_kv_heads, prev, 1]));
                    }
                }
                self.k_packed = Some(concatenate(self.k_packed.as_ref().unwrap(), &new_kp_buf, 2));
                self.k_norms = Some(concatenate(self.k_norms.as_ref().unwrap(), &new_kn_buf, 2));
                self.v_packed = Some(concatenate(self.v_packed.as_ref().unwrap(), &new_vp_buf, 2));
                self.v_norms = Some(concatenate(self.v_norms.as_ref().unwrap(), &new_vn_buf, 2));
                self.v_rescale = match self.v_rescale.as_ref() {
                    Some(vr) => Some(concatenate(vr, &new_vr_buf, 2)),
                    None => Some(new_vr_buf),
                };
            } else {
                self.k_packed = Some(new_kp_buf);
                self.k_norms = Some(new_kn_buf);
                self.v_packed = Some(new_vp_buf);
                self.v_norms = Some(new_vn_buf);
                self.v_rescale = Some(new_vr_buf);
            }
        }

        self.offset += new_seq_len;

        let kp_shape = ffi::array_shape(self.k_packed.as_ref().unwrap());
        let kn_shape = ffi::array_shape(self.k_norms.as_ref().unwrap());
        let vp_shape = ffi::array_shape(self.v_packed.as_ref().unwrap());
        let vn_shape = ffi::array_shape(self.v_norms.as_ref().unwrap());

        self.k_packed = Some(ffi::slice_update(
            self.k_packed.as_ref().unwrap(),
            &k_packed_new,
            &[0, 0, prev, 0],
            &[kp_shape[0], kp_shape[1], self.offset, kp_shape[3]],
        ));
        self.k_norms = Some(ffi::slice_update(
            self.k_norms.as_ref().unwrap(),
            &k_norms_new,
            &[0, 0, prev, 0],
            &[kn_shape[0], kn_shape[1], self.offset, 1],
        ));
        self.v_packed = Some(ffi::slice_update(
            self.v_packed.as_ref().unwrap(),
            &v_packed_new,
            &[0, 0, prev, 0],
            &[vp_shape[0], vp_shape[1], self.offset, vp_shape[3]],
        ));
        self.v_norms = Some(ffi::slice_update(
            self.v_norms.as_ref().unwrap(),
            &v_norms_new,
            &[0, 0, prev, 0],
            &[vn_shape[0], vn_shape[1], self.offset, 1],
        ));
        let vr_buf = self
            .v_rescale
            .as_ref()
            .expect("v_rescale lockstep with v_norms");
        let vr_shape = ffi::array_shape(vr_buf);
        self.v_rescale = Some(ffi::slice_update(
            vr_buf,
            &v_rescale_new,
            &[0, 0, prev, 0],
            &[vr_shape[0], vr_shape[1], self.offset, 1],
        ));
    }

    /// Turbo4Delegated update path — hybrid hot/cold split.
    ///
    /// **Phase 1: prefill / single-token append into hot tail.** Tokens land
    /// directly in the FP16 `keys`/`values` buffers, identical to the
    /// `update_fp16` path. No quantization runs.
    ///
    /// **Phase 2: prefill→decode transition.** Detected when `cold_offset == 0`
    /// and the incoming step is a single-token decode (`new_seq_len == 1`)
    /// against a populated cache (`offset > 0`). The current FP16 body is
    /// **folded** into cold storage: K is moved into `cold_keys` (still FP16)
    /// and V is quantized into `v_packed`/`v_norms` via
    /// [`turbo::quant::quantize_v_turbo4`]. The hot `keys`/`values` are then
    /// freshly allocated as a small ring (capacity =
    /// `step` × `ceil(hot_threshold / step)`) and the new decode token is
    /// appended.
    ///
    /// **Phase 3: subsequent decode.** Tokens append to the hot tail. When
    /// `hot_offset() > hot_threshold`, the oldest [`turbo::DELEGATED_FOLD_BLOCK`]-
    /// token block is folded into cold storage. The fold quantizes the V slice
    /// and slice-updates `cold_keys` / `v_packed` / `v_norms`. If the hot
    /// buffer ever exceeds [`turbo::DELEGATED_HOT_MAX`], an extra fold runs
    /// synchronously to bound the hot footprint.
    ///
    /// Layout of stored buffers in delegated mode:
    /// - `keys`/`values`: hot tail FP16, `[B, H, hot_capacity, K_dim]` /
    ///   `[B, H, hot_capacity, V_dim]`. Pre-allocated ring; the visible
    ///   length is `hot_offset()`.
    /// - `cold_keys`: cold body FP16, `[B, H, cold_capacity, K_dim]`. Visible
    ///   length is `cold_offset`.
    /// - `v_packed`: cold body Turbo4 indices, `[B, H, cold_capacity, V_dim/2]` u8.
    /// - `v_norms`: cold body per-token norms, `[B, H, cold_capacity, 1]` fp16.
    ///
    /// Used by: `KVCache::update` dispatch when `mode == KVCacheMode::Turbo4Delegated`
    /// (epic #458, issue #479).
    fn update_turbo4_delegated(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) {
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        let key_shape = ffi::array_shape(&new_keys_f16);
        let new_seq_len = key_shape[2];

        // Detect the prefill→decode transition: cold is empty, hot already has
        // tokens, and this is a single-token decode step. We fold the entire
        // FP16 body into cold storage *before* appending the new token so the
        // first decode step sees the fully compressed state.
        if self.cold_offset == 0
            && self.offset > 0
            && new_seq_len == 1
            && self.values.is_some()
            && self.keys.is_some()
        {
            self.fold_hot_to_cold_full();
            // After fold_hot_to_cold_full(), keys/values are reset to a fresh
            // hot buffer of capacity = ceil_step(hot_threshold). Append the
            // single new decode token to the hot tail below.
        }

        // Ensure the hot keys/values buffer can hold prev_hot + new_seq_len.
        let prev_hot = self.offset - self.cold_offset;
        let needed = prev_hot + new_seq_len;
        let mut allocate_fresh = false;
        if self.keys.is_none() {
            allocate_fresh = true;
        } else {
            let cap = self.buffer_seq_len();
            if needed > cap {
                allocate_fresh = true;
            }
        }
        if allocate_fresh {
            self.grow_hot_buffer(&new_keys_f16, &new_values_f16, needed);
        }

        // Slice-update new tokens into the hot buffers at position prev_hot.
        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
        let pos = prev_hot;
        let after = prev_hot + new_seq_len;
        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &new_keys_f16,
            &[0, 0, pos, 0],
            &[k_shape[0], k_shape[1], after, k_shape[3]],
        ));
        self.values = Some(ffi::slice_update(
            self.values.as_ref().unwrap(),
            &new_values_f16,
            &[0, 0, pos, 0],
            &[v_shape[0], v_shape[1], after, v_shape[3]],
        ));
        self.offset += new_seq_len;

        // Periodic fold: if hot tail exceeded the threshold (and cold is now
        // populated, so we are in steady-state decode), fold one block.
        // Repeat synchronously while we are still over the hard `HOT_MAX` cap;
        // a single fold of `DELEGATED_FOLD_BLOCK` tokens is normally enough to
        // bring us back inside the threshold.
        if self.cold_offset > 0 {
            while self.hot_offset() > self.hot_threshold {
                let block = turbo::DELEGATED_FOLD_BLOCK.min(self.hot_offset());
                self.fold_hot_block_to_cold(block);
                if self.hot_offset() <= turbo::DELEGATED_HOT_MAX {
                    break;
                }
            }
        }
    }

    /// Allocate (or grow) the FP16 hot tail buffer for Turbo4Delegated mode.
    ///
    /// Capacity is rounded up to a multiple of `step`. Existing visible
    /// `hot_offset()` tokens are copied into the new buffer so subsequent
    /// slice-updates land on the right tokens.
    fn grow_hot_buffer(
        &mut self,
        new_keys_f16: &MlxArray,
        new_values_f16: &MlxArray,
        needed_seq_len: i32,
    ) {
        let key_shape = ffi::array_shape(new_keys_f16);
        let val_shape = ffi::array_shape(new_values_f16);
        let b = key_shape[0];
        let n_kv_heads = key_shape[1];
        let k_head_dim = key_shape[3];
        let v_head_dim = val_shape[3];

        // Hot buffer capacity = next multiple of step that fits `needed`.
        // Anchor at hot_threshold so the first decode-time allocation is
        // sized to absorb roughly one fold-period of tokens before growing.
        let target = needed_seq_len.max(self.hot_threshold);
        let n_steps = (target + self.step - 1) / self.step;
        let buf_size = (n_steps * self.step).max(self.step);

        let new_k = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], dtype::FLOAT16);
        let new_v = ffi::zeros(&[b, n_kv_heads, buf_size, v_head_dim], dtype::FLOAT16);

        let prev_hot = self.offset - self.cold_offset;
        if prev_hot > 0 && self.keys.is_some() {
            // Copy existing hot tokens into the fresh buffer at position 0.
            let old_k = self.keys.as_ref().unwrap();
            let old_v = self.values.as_ref().unwrap();
            let old_k_shape = ffi::array_shape(old_k);
            let old_v_shape = ffi::array_shape(old_v);
            let old_k_slice = ffi::slice(
                old_k,
                &[0, 0, 0, 0],
                &[old_k_shape[0], old_k_shape[1], prev_hot, old_k_shape[3]],
            );
            let old_v_slice = ffi::slice(
                old_v,
                &[0, 0, 0, 0],
                &[old_v_shape[0], old_v_shape[1], prev_hot, old_v_shape[3]],
            );
            let copied_k = ffi::slice_update(
                &new_k,
                &old_k_slice,
                &[0, 0, 0, 0],
                &[b, n_kv_heads, prev_hot, k_head_dim],
            );
            let copied_v = ffi::slice_update(
                &new_v,
                &old_v_slice,
                &[0, 0, 0, 0],
                &[b, n_kv_heads, prev_hot, v_head_dim],
            );
            self.keys = Some(copied_k);
            self.values = Some(copied_v);
        } else {
            self.keys = Some(new_k);
            self.values = Some(new_v);
        }
    }

    /// Fold the entire FP16 hot body into cold storage.
    ///
    /// Used for the prefill→decode transition: every prefilled token is moved
    /// from `keys`/`values` into `cold_keys` (FP16) plus `v_packed`/`v_norms`
    /// (Turbo4). The hot buffer is then re-allocated empty so subsequent
    /// decode tokens see a fresh ring.
    ///
    /// Used by: `update_turbo4_delegated` on the prefill→decode transition.
    fn fold_hot_to_cold_full(&mut self) {
        let prev_hot = self.offset - self.cold_offset;
        if prev_hot <= 0 || self.keys.is_none() || self.values.is_none() {
            return;
        }
        // Pull the visible hot region out as fresh fp16 slices.
        let hot_k = self.keys.as_ref().unwrap();
        let hot_v = self.values.as_ref().unwrap();
        let hk_shape = ffi::array_shape(hot_k);
        let hv_shape = ffi::array_shape(hot_v);
        let hot_k_slice = ffi::slice(
            hot_k,
            &[0, 0, 0, 0],
            &[hk_shape[0], hk_shape[1], prev_hot, hk_shape[3]],
        );
        let hot_v_slice = ffi::slice(
            hot_v,
            &[0, 0, 0, 0],
            &[hv_shape[0], hv_shape[1], prev_hot, hv_shape[3]],
        );

        // Quantize V before consuming the slice. K stays FP16.
        if self.turbo_params.is_none() {
            let v_head_dim = hv_shape[3] as u32;
            self.turbo_params = Some(turbo::TurboQuantParams::new(v_head_dim, self.turbo_seed));
        }
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params just initialised");
        let (v_packed_new, v_norms_new, v_rescale_new) =
            turbo::quant::quantize_v_turbo4(&hot_v_slice, params);

        // Append the produced cold buffers (cold is empty before the full fold).
        self.append_cold_block(
            &hot_k_slice,
            &v_packed_new,
            &v_norms_new,
            &v_rescale_new,
            prev_hot,
        );

        // Reset the hot buffer to a freshly-allocated ring sized for the
        // configured hot_threshold (rounded up to step).
        let b = hk_shape[0];
        let n_kv_heads = hk_shape[1];
        let k_head_dim = hk_shape[3];
        let v_head_dim = hv_shape[3];
        let target = self.hot_threshold;
        let n_steps = (target + self.step - 1) / self.step;
        let buf_size = (n_steps * self.step).max(self.step);
        self.keys = Some(ffi::zeros(
            &[b, n_kv_heads, buf_size, k_head_dim],
            dtype::FLOAT16,
        ));
        self.values = Some(ffi::zeros(
            &[b, n_kv_heads, buf_size, v_head_dim],
            dtype::FLOAT16,
        ));
        // self.offset stays where it was (offset = cold_offset + 0).
    }

    /// Fold the oldest `block` tokens of the hot tail into cold storage and
    /// shift the remaining hot tokens left by `block` positions.
    ///
    /// Used by: steady-state decode in `update_turbo4_delegated` whenever
    /// `hot_offset() > hot_threshold`.
    fn fold_hot_block_to_cold(&mut self, block: i32) {
        if block <= 0 {
            return;
        }
        let prev_hot = self.offset - self.cold_offset;
        let block = block.min(prev_hot);
        if block == 0 {
            return;
        }

        let hot_k = self.keys.as_ref().expect("hot keys must exist for fold");
        let hot_v = self
            .values
            .as_ref()
            .expect("hot values must exist for fold");
        let hk_shape = ffi::array_shape(hot_k);
        let hv_shape = ffi::array_shape(hot_v);
        let b = hk_shape[0];
        let n_kv_heads = hk_shape[1];
        let k_head_dim = hk_shape[3];
        let v_head_dim = hv_shape[3];

        // Slice the [0:block] window — these are the oldest hot tokens.
        let hot_k_old = ffi::slice(hot_k, &[0, 0, 0, 0], &[b, n_kv_heads, block, k_head_dim]);
        let hot_v_old = ffi::slice(hot_v, &[0, 0, 0, 0], &[b, n_kv_heads, block, v_head_dim]);

        // Quantize V; K stays FP16. Inline the `turbo_params` lookup so the
        // immutable borrow ends before `append_cold_block` (which takes `&mut self`).
        let (v_packed_new, v_norms_new, v_rescale_new) = turbo::quant::quantize_v_turbo4(
            &hot_v_old,
            self.turbo_params
                .as_ref()
                .expect("turbo_params must be set after first fold_hot_to_cold_full"),
        );

        // Pre-compute the hot-tail keep slices BEFORE any mutable borrow on
        // self. Once `hot_k_keep`/`hot_v_keep` are owned, the immutable
        // borrows on `self.keys`/`self.values` (held via `hot_k`/`hot_v`)
        // can be released by NLL, freeing the path for `&mut self` calls.
        let remaining = prev_hot - block;
        let hot_keep = if remaining > 0 {
            let hot_k_keep = ffi::slice(
                hot_k,
                &[0, 0, block, 0],
                &[b, n_kv_heads, prev_hot, k_head_dim],
            );
            let hot_v_keep = ffi::slice(
                hot_v,
                &[0, 0, block, 0],
                &[b, n_kv_heads, prev_hot, v_head_dim],
            );
            Some((hot_k_keep, hot_v_keep))
        } else {
            None
        };
        // hot_k/hot_v are no longer used past this point — NLL releases the
        // immutable borrows on self.keys/self.values here.

        // Append into cold storage (which is non-empty here — prefix already
        // populated by the full fold).
        self.append_cold_block(
            &hot_k_old,
            &v_packed_new,
            &v_norms_new,
            &v_rescale_new,
            block,
        );

        // Shift the remaining hot tail [block..prev_hot] left into
        // [0..prev_hot-block] so subsequent slice-updates see contiguous
        // tokens at the front of the ring.
        if let Some((hot_k_keep, hot_v_keep)) = hot_keep {
            self.keys = Some(ffi::slice_update(
                self.keys.as_ref().unwrap(),
                &hot_k_keep,
                &[0, 0, 0, 0],
                &[b, n_kv_heads, remaining, k_head_dim],
            ));
            self.values = Some(ffi::slice_update(
                self.values.as_ref().unwrap(),
                &hot_v_keep,
                &[0, 0, 0, 0],
                &[b, n_kv_heads, remaining, v_head_dim],
            ));
        }
        // self.offset is unchanged: cold_offset += block, hot_len -= block.
        // The total length stays the same.
    }

    /// Append a `block`-token chunk to the cold storage buffers, growing them
    /// (in `step`-sized increments) when the existing capacity does not fit.
    ///
    /// Inputs are the freshly-prepared cold tensors:
    /// - `cold_k_block`: `[B, H, block, K_dim]` fp16 — to be appended to `cold_keys`.
    /// - `v_packed_block`:  `[B, H, block, V_dim/2]` u8   — appended to `v_packed`.
    /// - `v_norms_block`:   `[B, H, block, 1]`      fp16  — appended to `v_norms`.
    /// - `v_rescale_block`: `[B, H, block, 1]`      fp16  — appended to `v_rescale`
    ///   (issue #520; precomputed Sparse-V kernel rescale, lockstep with `v_norms`).
    ///
    /// Updates `cold_offset` by `block`. Used by both the full prefill→decode
    /// fold and the steady-state per-block fold.
    fn append_cold_block(
        &mut self,
        cold_k_block: &MlxArray,
        v_packed_block: &MlxArray,
        v_norms_block: &MlxArray,
        v_rescale_block: &MlxArray,
        block: i32,
    ) {
        let k_shape = ffi::array_shape(cold_k_block);
        let vp_shape = ffi::array_shape(v_packed_block);
        let _vn_shape = ffi::array_shape(v_norms_block);
        let b = k_shape[0];
        let n_kv_heads = k_shape[1];
        let k_head_dim = k_shape[3];
        let v_packed_dim = vp_shape[3];

        let prev_cold = self.cold_offset;
        let needed = prev_cold + block;

        // Grow cold buffers if needed.
        let cold_cap = match &self.cold_keys {
            Some(ck) => {
                let ck_shape = ffi::array_shape(ck);
                if ck_shape.len() >= 3 {
                    ck_shape[2]
                } else {
                    0
                }
            }
            None => 0,
        };
        if needed > cold_cap {
            let n_steps = (needed + self.step - 1) / self.step;
            let buf_size = n_steps * self.step;
            let new_ck = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], dtype::FLOAT16);
            let new_vp = ffi::zeros(&[b, n_kv_heads, buf_size, v_packed_dim], dtype::UINT8);
            let new_vn = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            let new_vr = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            if let Some(old_ck) = self.cold_keys.as_ref() {
                if prev_cold > 0 {
                    let old_ck_slice = ffi::slice(
                        old_ck,
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev_cold, k_head_dim],
                    );
                    let old_vp = self.v_packed.as_ref().unwrap();
                    let old_vn = self.v_norms.as_ref().unwrap();
                    let old_vp_slice = ffi::slice(
                        old_vp,
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev_cold, v_packed_dim],
                    );
                    let old_vn_slice =
                        ffi::slice(old_vn, &[0, 0, 0, 0], &[b, n_kv_heads, prev_cold, 1]);
                    let old_vr_slice = self
                        .v_rescale
                        .as_ref()
                        .map(|vr| ffi::slice(vr, &[0, 0, 0, 0], &[b, n_kv_heads, prev_cold, 1]));
                    self.cold_keys = Some(ffi::slice_update(
                        &new_ck,
                        &old_ck_slice,
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev_cold, k_head_dim],
                    ));
                    self.v_packed = Some(ffi::slice_update(
                        &new_vp,
                        &old_vp_slice,
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev_cold, v_packed_dim],
                    ));
                    self.v_norms = Some(ffi::slice_update(
                        &new_vn,
                        &old_vn_slice,
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev_cold, 1],
                    ));
                    self.v_rescale = Some(match old_vr_slice {
                        Some(s) => ffi::slice_update(
                            &new_vr,
                            &s,
                            &[0, 0, 0, 0],
                            &[b, n_kv_heads, prev_cold, 1],
                        ),
                        None => new_vr,
                    });
                } else {
                    self.cold_keys = Some(new_ck);
                    self.v_packed = Some(new_vp);
                    self.v_norms = Some(new_vn);
                    self.v_rescale = Some(new_vr);
                }
            } else {
                self.cold_keys = Some(new_ck);
                self.v_packed = Some(new_vp);
                self.v_norms = Some(new_vn);
                self.v_rescale = Some(new_vr);
            }
        }

        // Slice-update the new block at position prev_cold.
        let after = prev_cold + block;
        let ck_buf = self.cold_keys.as_ref().unwrap();
        let vp_buf = self.v_packed.as_ref().unwrap();
        let vn_buf = self.v_norms.as_ref().unwrap();
        let ck_buf_shape = ffi::array_shape(ck_buf);
        let vp_buf_shape = ffi::array_shape(vp_buf);
        let vn_buf_shape = ffi::array_shape(vn_buf);
        self.cold_keys = Some(ffi::slice_update(
            ck_buf,
            cold_k_block,
            &[0, 0, prev_cold, 0],
            &[ck_buf_shape[0], ck_buf_shape[1], after, ck_buf_shape[3]],
        ));
        self.v_packed = Some(ffi::slice_update(
            vp_buf,
            v_packed_block,
            &[0, 0, prev_cold, 0],
            &[vp_buf_shape[0], vp_buf_shape[1], after, vp_buf_shape[3]],
        ));
        self.v_norms = Some(ffi::slice_update(
            vn_buf,
            v_norms_block,
            &[0, 0, prev_cold, 0],
            &[vn_buf_shape[0], vn_buf_shape[1], after, 1],
        ));
        let vr_buf = self
            .v_rescale
            .as_ref()
            .expect("v_rescale lockstep with v_norms in append_cold_block");
        let vr_buf_shape = ffi::array_shape(vr_buf);
        self.v_rescale = Some(ffi::slice_update(
            vr_buf,
            v_rescale_block,
            &[0, 0, prev_cold, 0],
            &[vr_buf_shape[0], vr_buf_shape[1], after, 1],
        ));

        self.cold_offset += block;
    }

    /// Trim the last `n` entries from the cache.
    ///
    /// Returns the number of entries actually trimmed.
    /// In INT8 mode the corresponding scale buffers are also trimmed.
    /// In Turbo4Asym mode the V-packed and V-norms buffers are also trimmed.
    /// In Turbo4 (symmetric) mode both the K- and V-side packed sidecars are
    /// trimmed.
    /// In Turbo4Delegated mode the hot tail is shrunk first; only when the
    /// requested trim depth exceeds the hot tail does the cold storage get
    /// touched. This mirrors speculative decoding's "rewind one block" pattern
    /// from `update_turbo4_asym` and avoids paying for a re-quantize on the
    /// common short-rewind case.
    /// Used by: speculative decoding cache rewinds
    pub fn trim(&mut self, n: i32) -> i32 {
        let n = n.min(self.offset);
        if n <= 0 {
            return 0;
        }
        // Turbo4Delegated: hot-first trim. Tokens to remove from cold = max(0, n - hot_len).
        // We adjust cold_offset and offset, then fall through to the per-mode buffer slicing
        // logic below to keep the buffers consistent with the new offsets.
        let mut hot_trim = 0_i32;
        let mut cold_trim = 0_i32;
        if self.mode == KVCacheMode::Turbo4Delegated {
            let hot_len = self.offset - self.cold_offset;
            hot_trim = n.min(hot_len);
            cold_trim = (n - hot_trim).max(0);
        }
        self.offset -= n;
        if self.mode == KVCacheMode::Turbo4Delegated {
            self.cold_offset -= cold_trim;
        }
        if self.offset == 0 {
            self.keys = None;
            self.values = None;
            self.key_scales = None;
            self.val_scales = None;
            self.v_packed = None;
            self.v_norms = None;
            self.v_rescale = None;
            self.k_packed = None;
            self.k_norms = None;
            self.cold_keys = None;
            self.cold_offset = 0;
            // Clear turbo_params so the next quantize call rebuilds it from
            // scratch (required if the caller reuses this cache slot with a
            // different head_dim). LOW-1 fix (#474). The 3-bit
            // `turbo3_params` (issue #477) follows the same contract.
            self.turbo_params = None;
            self.turbo3_params = None;
        } else if self.mode == KVCacheMode::Turbo4Delegated {
            // Hot tail trims first (cheap), cold trims only on overflow.
            let new_hot_len = self.offset - self.cold_offset;
            // Re-slice the hot keys/values to the new hot length so the next
            // update sees a clean prefix in the ring buffer.
            if let Some(ref k) = self.keys {
                let k_shape = ffi::array_shape(k);
                if new_hot_len > 0 {
                    self.keys = Some(ffi::slice(
                        k,
                        &[0, 0, 0, 0],
                        &[k_shape[0], k_shape[1], new_hot_len, k_shape[3]],
                    ));
                }
            }
            if let Some(ref v) = self.values {
                let v_shape = ffi::array_shape(v);
                if new_hot_len > 0 {
                    self.values = Some(ffi::slice(
                        v,
                        &[0, 0, 0, 0],
                        &[v_shape[0], v_shape[1], new_hot_len, v_shape[3]],
                    ));
                }
            }
            // If the cold portion shrank (n exceeded the hot tail), re-slice
            // cold buffers to the new cold_offset.
            if cold_trim > 0 {
                if let Some(ref ck) = self.cold_keys {
                    let ck_shape = ffi::array_shape(ck);
                    self.cold_keys = Some(ffi::slice(
                        ck,
                        &[0, 0, 0, 0],
                        &[ck_shape[0], ck_shape[1], self.cold_offset, ck_shape[3]],
                    ));
                }
                if let Some(ref vp) = self.v_packed {
                    let vp_shape = ffi::array_shape(vp);
                    self.v_packed = Some(ffi::slice(
                        vp,
                        &[0, 0, 0, 0],
                        &[vp_shape[0], vp_shape[1], self.cold_offset, vp_shape[3]],
                    ));
                }
                if let Some(ref vn) = self.v_norms {
                    let vn_shape = ffi::array_shape(vn);
                    self.v_norms = Some(ffi::slice(
                        vn,
                        &[0, 0, 0, 0],
                        &[vn_shape[0], vn_shape[1], self.cold_offset, 1],
                    ));
                }
                if let Some(ref vr) = self.v_rescale {
                    let vr_shape = ffi::array_shape(vr);
                    self.v_rescale = Some(ffi::slice(
                        vr,
                        &[0, 0, 0, 0],
                        &[vr_shape[0], vr_shape[1], self.cold_offset, 1],
                    ));
                }
            }
            // Suppress unused-variable warnings in non-delegated branches.
            let _ = hot_trim;
        } else {
            // Non-delegated modes: simple buffer-prefix trim.
            // Trim keys (always present except in Turbo4Asym pre-init).
            if let Some(ref k) = self.keys {
                let k_shape = ffi::array_shape(k);
                self.keys = Some(ffi::slice(
                    k,
                    &[0, 0, 0, 0],
                    &[k_shape[0], k_shape[1], self.offset, k_shape[3]],
                ));
            }
            // Trim values for FP16/INT8 modes (Turbo4Asym keeps values None).
            if let Some(ref v) = self.values {
                let v_shape = ffi::array_shape(v);
                self.values = Some(ffi::slice(
                    v,
                    &[0, 0, 0, 0],
                    &[v_shape[0], v_shape[1], self.offset, v_shape[3]],
                ));
            }
            // Trim INT8 scale sidecars.
            if self.mode == KVCacheMode::Int8 {
                if let Some(ref ks) = self.key_scales {
                    let ks_shape = ffi::array_shape(ks);
                    self.key_scales = Some(ffi::slice(
                        ks,
                        &[0, 0, 0, 0],
                        &[ks_shape[0], ks_shape[1], self.offset, 1],
                    ));
                }
                if let Some(ref vs) = self.val_scales {
                    let vs_shape = ffi::array_shape(vs);
                    self.val_scales = Some(ffi::slice(
                        vs,
                        &[0, 0, 0, 0],
                        &[vs_shape[0], vs_shape[1], self.offset, 1],
                    ));
                }
            }
            // Trim Turbo4* V sidecars (per-token; speculative-decode rewinds
            // <1 block at a time so we never need to re-quantize a partial
            // block — the trimmed tail is already block-aligned in the buffer).
            // `Turbo4Asym`, `Turbo4`, and `Turbo3Asym` (issue #477) all carry
            // the V sidecars; only `Turbo4` (symmetric) additionally carries
            // the K-side sidecars.
            if matches!(
                self.mode,
                KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4 | KVCacheMode::Turbo3Asym
            ) {
                if let Some(ref vp) = self.v_packed {
                    let vp_shape = ffi::array_shape(vp);
                    self.v_packed = Some(ffi::slice(
                        vp,
                        &[0, 0, 0, 0],
                        &[vp_shape[0], vp_shape[1], self.offset, vp_shape[3]],
                    ));
                }
                if let Some(ref vn) = self.v_norms {
                    let vn_shape = ffi::array_shape(vn);
                    self.v_norms = Some(ffi::slice(
                        vn,
                        &[0, 0, 0, 0],
                        &[vn_shape[0], vn_shape[1], self.offset, 1],
                    ));
                }
                // v_rescale (issue #520): tracks v_norms lockstep. Turbo3Asym
                // never populates v_rescale (3-bit V kernel does not exist),
                // so the `if let Some(...)` guard handles that case.
                if let Some(ref vr) = self.v_rescale {
                    let vr_shape = ffi::array_shape(vr);
                    self.v_rescale = Some(ffi::slice(
                        vr,
                        &[0, 0, 0, 0],
                        &[vr_shape[0], vr_shape[1], self.offset, 1],
                    ));
                }
            }
            // K-side sidecars are present only in symmetric Turbo4.
            if self.mode == KVCacheMode::Turbo4 {
                if let Some(ref kp) = self.k_packed {
                    let kp_shape = ffi::array_shape(kp);
                    self.k_packed = Some(ffi::slice(
                        kp,
                        &[0, 0, 0, 0],
                        &[kp_shape[0], kp_shape[1], self.offset, kp_shape[3]],
                    ));
                }
                if let Some(ref kn) = self.k_norms {
                    let kn_shape = ffi::array_shape(kn);
                    self.k_norms = Some(ffi::slice(
                        kn,
                        &[0, 0, 0, 0],
                        &[kn_shape[0], kn_shape[1], self.offset, 1],
                    ));
                }
            }
        }
        n
    }

    /// Update cache and return view of filled portion.
    ///
    /// In `KVCacheMode::Fp16` returns sliced FP16 keys/values directly.
    /// In `KVCacheMode::Int8` dequantizes the accumulated INT8 buffers back to
    /// FP16 before returning, so attention kernels always receive FP16 tensors.
    /// In `KVCacheMode::Turbo4Asym` returns FP16 keys (untouched) and a V tensor
    /// reconstructed from the packed 4-bit indices + per-token norms.
    /// In `KVCacheMode::Turbo4` (symmetric) **both** K and V are reconstructed
    /// from packed 4-bit indices using independent sign-vector pairs.
    /// In `KVCacheMode::Turbo4Delegated` returns the concatenated cold+hot views:
    /// `[cold_keys; hot_keys[:hot_offset]]` for K and
    /// `[dequant(v_packed[:cold_offset]); hot_values[:hot_offset]]` for V. SDPA
    /// always sees FP16; the packed cold storage is internal.
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update(new_keys, new_values);

        match self.mode {
            KVCacheMode::Int8 => {
                // Dequantize the filled portion of the INT8 buffers
                let k_int8 = self.keys.as_ref().unwrap();
                let v_int8 = self.values.as_ref().unwrap();
                let k_scales = self.key_scales.as_ref().unwrap();
                let v_scales = self.val_scales.as_ref().unwrap();

                let ks = ffi::array_shape(k_int8);
                let vs = ffi::array_shape(v_int8);
                let kss = ffi::array_shape(k_scales);
                let vss = ffi::array_shape(v_scales);

                let k_slice =
                    ffi::slice(k_int8, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]);
                let v_slice =
                    ffi::slice(v_int8, &[0, 0, 0, 0], &[vs[0], vs[1], self.offset, vs[3]]);
                let ks_slice =
                    ffi::slice(k_scales, &[0, 0, 0, 0], &[kss[0], kss[1], self.offset, 1]);
                let vs_slice =
                    ffi::slice(v_scales, &[0, 0, 0, 0], &[vss[0], vss[1], self.offset, 1]);

                (
                    dequantize(&k_slice, &ks_slice),
                    dequantize(&v_slice, &vs_slice),
                )
            }
            KVCacheMode::Turbo4Asym => {
                let k = self.keys.as_ref().unwrap();
                let vp = self.v_packed.as_ref().unwrap();
                let vn = self.v_norms.as_ref().unwrap();
                let params = self
                    .turbo_params
                    .as_ref()
                    .expect("turbo_params must be initialised after first update_turbo4_asym");

                let ks = ffi::array_shape(k);
                let vps = ffi::array_shape(vp);
                let vns = ffi::array_shape(vn);

                let k_slice = ffi::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]);
                let vp_slice =
                    ffi::slice(vp, &[0, 0, 0, 0], &[vps[0], vps[1], self.offset, vps[3]]);
                let vn_slice = ffi::slice(vn, &[0, 0, 0, 0], &[vns[0], vns[1], self.offset, 1]);

                let v_dequantized = turbo::quant::dequantize_v_turbo4(&vp_slice, &vn_slice, params);
                (k_slice, v_dequantized)
            }
            KVCacheMode::Turbo4 => {
                let kp = self.k_packed.as_ref().unwrap();
                let kn = self.k_norms.as_ref().unwrap();
                let vp = self.v_packed.as_ref().unwrap();
                let vn = self.v_norms.as_ref().unwrap();
                let params = self
                    .turbo_params
                    .as_ref()
                    .expect("turbo_params must be initialised after first update_turbo4_sym");

                let kps = ffi::array_shape(kp);
                let kns = ffi::array_shape(kn);
                let vps = ffi::array_shape(vp);
                let vns = ffi::array_shape(vn);

                let kp_slice =
                    ffi::slice(kp, &[0, 0, 0, 0], &[kps[0], kps[1], self.offset, kps[3]]);
                let kn_slice = ffi::slice(kn, &[0, 0, 0, 0], &[kns[0], kns[1], self.offset, 1]);
                let vp_slice =
                    ffi::slice(vp, &[0, 0, 0, 0], &[vps[0], vps[1], self.offset, vps[3]]);
                let vn_slice = ffi::slice(vn, &[0, 0, 0, 0], &[vns[0], vns[1], self.offset, 1]);

                let k_dequantized = turbo::quant::dequantize_k_turbo4(&kp_slice, &kn_slice, params);
                let v_dequantized = turbo::quant::dequantize_v_turbo4(&vp_slice, &vn_slice, params);
                (k_dequantized, v_dequantized)
            }
            KVCacheMode::Turbo4Delegated => self.fetch_turbo4_delegated(),
            KVCacheMode::Turbo3Asym => {
                let k = self.keys.as_ref().unwrap();
                let vp = self.v_packed.as_ref().unwrap();
                let vn = self.v_norms.as_ref().unwrap();
                let params = self
                    .turbo3_params
                    .as_ref()
                    .expect("turbo3_params must be initialised after first update_turbo3_asym");

                let ks = ffi::array_shape(k);
                let vps = ffi::array_shape(vp);
                let vns = ffi::array_shape(vn);

                let k_slice = ffi::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]);
                let vp_slice =
                    ffi::slice(vp, &[0, 0, 0, 0], &[vps[0], vps[1], self.offset, vps[3]]);
                let vn_slice = ffi::slice(vn, &[0, 0, 0, 0], &[vns[0], vns[1], self.offset, 1]);

                let v_dequantized =
                    turbo::quant3::dequantize_v_turbo3(&vp_slice, &vn_slice, params);
                (k_slice, v_dequantized)
            }
            KVCacheMode::Fp16 => {
                let k = self.keys.as_ref().unwrap();
                let v = self.values.as_ref().unwrap();
                let ks = ffi::array_shape(k);
                let vs = ffi::array_shape(v);
                (
                    ffi::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]),
                    ffi::slice(v, &[0, 0, 0, 0], &[vs[0], vs[1], self.offset, vs[3]]),
                )
            }
        }
    }

    /// Read path for `KVCacheMode::Turbo4Delegated`.
    ///
    /// Concatenates the cold body (FP16 cold K + dequantized cold V from
    /// `v_packed`/`v_norms`) with the hot tail (FP16 keys/values prefix). Both
    /// outputs are FP16 of total length `self.offset`. When `cold_offset == 0`
    /// (still in prefill or at the boundary before any fold has happened) this
    /// degrades to a plain hot slice; when `hot_offset() == 0` (just after a
    /// full fold with no decode token yet) this returns just the cold body.
    fn fetch_turbo4_delegated(&self) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let hot_len = self.offset - self.cold_offset;
        // Hot slice: take `hot_len` tokens from the front of the hot ring.
        let hot_k_slice = if hot_len > 0 {
            let k = self.keys.as_ref().expect("hot keys must exist");
            let ks = ffi::array_shape(k);
            Some(ffi::slice(
                k,
                &[0, 0, 0, 0],
                &[ks[0], ks[1], hot_len, ks[3]],
            ))
        } else {
            None
        };
        let hot_v_slice = if hot_len > 0 {
            let v = self.values.as_ref().expect("hot values must exist");
            let vs = ffi::array_shape(v);
            Some(ffi::slice(
                v,
                &[0, 0, 0, 0],
                &[vs[0], vs[1], hot_len, vs[3]],
            ))
        } else {
            None
        };

        // Cold slice: dequantize the visible cold body.
        if self.cold_offset == 0 {
            // No cold body yet — just return the hot prefix. Caller (SDPA)
            // sees the same dtype/shape as the FP16 path.
            return (
                hot_k_slice.expect("hot must exist when cold_offset == 0 and offset > 0"),
                hot_v_slice.expect("hot must exist when cold_offset == 0 and offset > 0"),
            );
        }

        let ck = self.cold_keys.as_ref().expect("cold_keys must exist");
        let vp = self.v_packed.as_ref().expect("v_packed must exist");
        let vn = self.v_norms.as_ref().expect("v_norms must exist");
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params must be set after first fold");
        let ck_shape = ffi::array_shape(ck);
        let vp_shape = ffi::array_shape(vp);
        let vn_shape = ffi::array_shape(vn);
        let cold_k_slice = ffi::slice(
            ck,
            &[0, 0, 0, 0],
            &[ck_shape[0], ck_shape[1], self.cold_offset, ck_shape[3]],
        );
        let vp_slice = ffi::slice(
            vp,
            &[0, 0, 0, 0],
            &[vp_shape[0], vp_shape[1], self.cold_offset, vp_shape[3]],
        );
        let vn_slice = ffi::slice(
            vn,
            &[0, 0, 0, 0],
            &[vn_shape[0], vn_shape[1], self.cold_offset, 1],
        );
        let cold_v_dequant = turbo::quant::dequantize_v_turbo4(&vp_slice, &vn_slice, params);

        // Concatenate cold + hot along the seq axis. When hot_len == 0 the
        // cold view is the full result.
        match (hot_k_slice, hot_v_slice) {
            (Some(hot_k), Some(hot_v)) => {
                let full_k = concatenate(&cold_k_slice, &hot_k, 2);
                let full_v = concatenate(&cold_v_dequant, &hot_v, 2);
                (full_k, full_v)
            }
            _ => (cold_k_slice, cold_v_dequant),
        }
    }

    /// Get the total memory size of the cached keys and values in bytes.
    ///
    /// In INT8 mode this includes both the INT8 buffers and the scale tensors.
    /// In Turbo4Asym mode this counts the FP16 keys plus the packed-V and
    /// V-norm sidecars; the original FP16 values tensor is not stored.
    /// In Turbo4 (symmetric) mode this counts only the K- and V-side packed
    /// sidecars; both the FP16 keys and FP16 values tensors are absent.
    /// In Turbo4Delegated mode this also counts the cold-side FP16 K buffer.
    /// In Turbo3Asym mode this counts the FP16 keys plus the 3-bit packed-V
    /// and V-norm sidecars; the underlying `array_nbytes` already accounts
    /// for the smaller per-token byte count (`head_dim * 3 / 8` u8 vs the
    /// 4-bit path's `head_dim / 2`), so the headline number reflects the
    /// ~5.1× total compression vs FP16.
    pub fn nbytes(&self) -> usize {
        let k_bytes = self.keys.as_ref().map_or(0, |k| ffi::array_nbytes(k));
        let v_bytes = self.values.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let ks_bytes = self.key_scales.as_ref().map_or(0, |k| ffi::array_nbytes(k));
        let vs_bytes = self.val_scales.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let vp_bytes = self.v_packed.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let vn_bytes = self.v_norms.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let vr_bytes = self.v_rescale.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let kp_bytes = self.k_packed.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let kn_bytes = self.k_norms.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let ck_bytes = self.cold_keys.as_ref().map_or(0, |c| ffi::array_nbytes(c));
        k_bytes
            + v_bytes
            + ks_bytes
            + vs_bytes
            + vp_bytes
            + vn_bytes
            + vr_bytes
            + kp_bytes
            + kn_bytes
            + ck_bytes
    }

    /// Returns `true` iff this cache holds a packed V (the Turbo4 family)
    /// and the sparse-V threshold is enabled (see
    /// [`turbo::sparse_v::is_enabled`]).
    ///
    /// Sparse-V is currently supported only for `KVCacheMode::Turbo4Asym`.
    /// `Turbo4Delegated` is *not* yet wired through `sparse_v_attention`
    /// because that mode splits the visible token range across cold packed
    /// V (`v_packed[0..cold_offset]`) and hot FP16 V (`hot_values[..]`),
    /// while the current dispatcher in [`Self::sparse_v_attention`] slices a
    /// single contiguous `0..self.offset` range. Wiring delegated mode
    /// through sparse-V requires a hot+cold composition pass and is
    /// deferred to a follow-up issue. Symmetric `Turbo4` (K also packed) is
    /// also not yet wired through the split-SDPA path. Callers that want to
    /// opt into the attention-gated dequant path should check this accessor
    /// and, if true, use [`Self::sparse_v_attention`] instead of the
    /// standard [`Self::update_and_fetch`] + `attention()` pair.
    ///
    /// Used by: future model attention call sites (integration deferred —
    /// see `cache/turbo/sparse_v.rs` module docs).
    pub fn sparse_v_available(&self) -> bool {
        if !turbo::sparse_v::is_enabled() {
            return false;
        }
        matches!(self.mode, KVCacheMode::Turbo4Asym)
    }

    /// Borrow the packed V indices for sparse-V attention.
    ///
    /// Returns `None` when sparse-V is not active (see
    /// [`Self::sparse_v_available`]) or before the first
    /// `update_and_fetch` call has populated the V sidecars.
    ///
    /// Used by: [`Self::sparse_v_attention`].
    pub fn v_packed(&self) -> Option<&MlxArray> {
        self.v_packed.as_deref()
    }

    /// Borrow the per-token V norms paired with [`Self::v_packed`].
    pub fn v_norms(&self) -> Option<&MlxArray> {
        self.v_norms.as_deref()
    }

    /// Borrow the per-token V Sparse-V kernel rescale paired with
    /// [`Self::v_packed`] (issue #520).
    ///
    /// Stores `norm[t] / max(|y_hat[t]|, 1e-10)` in fp16 — the exact value the
    /// fused kernel previously derived per-token via a threadgroup tree
    /// reduction. Populated by Turbo4Asym / Turbo4 / Turbo4Delegated update
    /// paths; `None` for non-Turbo4 modes.
    pub fn v_rescale(&self) -> Option<&MlxArray> {
        self.v_rescale.as_deref()
    }

    /// Borrow the cached `TurboQuantParams` (sign vectors + codebook).
    ///
    /// Returns `None` until the first Turbo4 update has lazy-initialised
    /// the params (head_dim is known only at that point).
    pub fn turbo_params(&self) -> Option<&turbo::TurboQuantParams> {
        self.turbo_params.as_ref()
    }

    /// Attention-gated SDPA dispatch: routes to the fused Sparse-V Metal
    /// kernel when available, otherwise falls back to the graph reference
    /// [`turbo::sparse_v::attention_sparse_v_turbo4`] path. Returns `None`
    /// when sparse-V is inactive so the caller can use the standard
    /// `attention()` path.
    ///
    /// **Contract.** This preserves the full-dequant attention result within
    /// FP16 round-off at `threshold=0`. At positive thresholds, positions with
    /// attention weight below the configured cutoff are skipped. On macOS with
    /// a supported power-of-two head dimension the skip happens inside the
    /// fused Metal kernel; elsewhere the graph fallback remains a correctness
    /// reference and still pays full V dequant cost.
    ///
    /// # Inputs
    ///
    /// - `q`: `[B, Hq, Tq, D]` query tensor
    /// - `k`: `[B, Hkv, Tk, D]` key tensor (already FP16 — Turbo4Asym
    ///   keeps K in FP16, Turbo4Delegated returns FP16 from the read path)
    /// - `scale`: attention scale factor (typically `1 / sqrt(d)`)
    /// - `mask`: optional additive mask
    ///
    /// # Output
    ///
    /// `Some([B, Hq, Tq, D])` FP16 attention output when sparse-V was
    /// applied, `None` otherwise (caller falls back to standard SDPA).
    ///
    /// Used by: future model attention call sites that have been ported
    /// to the split-SDPA path. Standard call sites continue to use
    /// `cache.update_and_fetch(...)` followed by `attention(...)`.
    /// Combined update + sparse-V attention dispatch (issue #505).
    ///
    /// The standard `update_and_fetch + attention()` pair pays the full V
    /// dequant cost inside `update_and_fetch` even when sparse-V is enabled.
    /// This method side-steps that by:
    ///
    /// 1. Calling [`Self::update`] to fill the packed buffers (no V dequant).
    /// 2. Reading the K side (FP16 for `Turbo4Asym`).
    /// 3. Dispatching [`Self::sparse_v_attention`] directly with the packed
    ///    V buffers — the fused Metal kernel does the per-thread dequant +
    ///    skip in one pass.
    ///
    /// Returns:
    /// - `Some(attn_out)` when sparse-V is active and the kernel/graph path
    ///   handled the dispatch. The caller uses this output and skips the
    ///   standard `attention()` call.
    /// - `None` when sparse-V is *not* active. The caller falls back to
    ///   `update_and_fetch + attention()`.
    ///
    /// The Q tensor must already have RoPE / Q-norm applied; the caller is
    /// responsible for that, just as with the standard path.
    ///
    /// Used by: per-model attention call sites (Llama 3, Qwen 3, etc.)
    /// when the cache is in `Turbo4Asym` mode and
    /// `MLXCEL_SPARSE_V_THRESHOLD > 0`. `Turbo4Delegated` integration is
    /// deferred — see [`Self::sparse_v_available`].
    pub fn update_and_sparse_v_attention(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> Option<UniquePtr<MlxArray>> {
        if !self.sparse_v_available() {
            return None;
        }
        // Fill the packed buffers; this also advances `self.offset`.
        self.update(new_keys, new_values);

        // For Turbo4Asym K stays FP16 in `self.keys` and the visible token
        // range is contiguous (`0..self.offset`). Slice it directly.
        // (`Turbo4` symmetric mode would need a packed K dequant here, and
        // `Turbo4Delegated` would need a hot+cold composition; both are
        // excluded from `sparse_v_available()`.)
        let k_buf = self.keys.as_ref()?;
        let ks = ffi::array_shape(k_buf);
        let k_slice = ffi::slice(k_buf, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]);

        self.sparse_v_attention(q, &k_slice, scale, mask)
    }

    pub fn sparse_v_attention(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> Option<UniquePtr<MlxArray>> {
        if !self.sparse_v_available() {
            return None;
        }
        let v_packed_buf = self.v_packed()?;
        let v_norms_buf = self.v_norms()?;
        let v_rescale_buf = self.v_rescale()?;
        let params = self.turbo_params()?;
        let threshold = turbo::sparse_v::threshold();

        // Slice the packed V buffers down to the visible token range. The
        // raw `v_packed`/`v_norms`/`v_rescale` accessors return the full
        // buffer, which includes pre-allocated capacity beyond `self.offset`.
        // Without the slice the kernel would sweep `Tk = buffer_seq_len`,
        // paying for dead capacity slots and producing wrong contributions.
        //
        // For Turbo4Asym the packed shapes are:
        //   v_packed:  [B, H, capacity, D/2] u8
        //   v_norms:   [B, H, capacity, 1]   f16
        //   v_rescale: [B, H, capacity, 1]   f16  (issue #520)
        // We slice axis 2 (the token axis) to [0, self.offset).
        let vp_shape = ffi::array_shape(v_packed_buf);
        let vn_shape = ffi::array_shape(v_norms_buf);
        let vr_shape = ffi::array_shape(v_rescale_buf);
        let v_packed_owned = ffi::slice(
            v_packed_buf,
            &[0, 0, 0, 0],
            &[vp_shape[0], vp_shape[1], self.offset, vp_shape[3]],
        );
        let v_norms_owned = ffi::slice(
            v_norms_buf,
            &[0, 0, 0, 0],
            &[vn_shape[0], vn_shape[1], self.offset, 1],
        );
        let v_rescale_owned = ffi::slice(
            v_rescale_buf,
            &[0, 0, 0, 0],
            &[vr_shape[0], vr_shape[1], self.offset, 1],
        );

        // Prefer the fused Metal kernel path (issue #505 + #520) when
        // available. Falls through to the graph-level reference path on
        // non-macOS, when the kernel is disabled via
        // `MLXCEL_SPARSE_V_KERNEL=0`, or when the model uses a non-power-of-2
        // head_dim (Gemma 4 192-dim heads).
        //
        // The fused kernel reads the precomputed `v_rescale` (norm/|y_hat|)
        // directly, eliminating the per-token threadgroup tree reduction
        // that previously dominated decode latency on M5 Max at 4 K context
        // (issue #520). The graph fallback continues to use `v_norms` only.
        if let Some(out) = turbo::sparse_v::attention_sparse_v_turbo4_fused(
            q,
            k,
            &v_packed_owned,
            &v_rescale_owned,
            params,
            scale,
            mask,
            threshold,
        ) {
            return Some(out);
        }
        Some(turbo::sparse_v::attention_sparse_v_turbo4(
            q,
            k,
            &v_packed_owned,
            &v_norms_owned,
            params,
            scale,
            mask,
            threshold,
        ))
    }

    /// Estimated storage bytes per reserved token slot in the backing buffer.
    ///
    /// This uses the allocated buffer capacity rather than the visible offset
    /// so callers can mirror dense-cache physical storage into paged block
    /// accounting even when the buffer is step-allocated.
    pub fn bytes_per_reserved_token(&self) -> usize {
        let capacity = self.buffer_seq_len();
        if capacity <= 0 {
            return 0;
        }
        self.nbytes() / capacity as usize
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Rotating KV Cache for sliding window attention (e.g. Gemma 3, Ministral 3).
///
/// Maintains a fixed-size circular buffer for keys/values. Oversized prefill
/// is linearized before single-token decode so wraparound stays well-defined.
///
/// # KV Cache Quantization
///
/// `RotatingKVCache` supports the same `KVCacheMode` set as `KVCache`:
/// `Fp16` (default), `Int8`, and `Turbo4Asym`. The mode is selected at
/// construction time via [`RotatingKVCache::new_with_mode`].
///
/// In `KVCacheMode::Turbo4Asym` the V buffer is replaced by `v_packed`
/// (`[B, H, max_size, head_dim/2]` UINT8) plus `v_norms`
/// (`[B, H, max_size, 1]` FP16); the `values` field stays `None`. K stays
/// FP16 in the regular `keys` buffer.
///
/// ## Block alignment invariant (B9, issue #481, epic #458)
///
/// TurboQuant V quantization is **per-token**: each token's `head_dim` vector
/// is rotated and quantized independently along the last axis. There is no
/// cross-token shared state in the rotation pipeline (the `BLOCK_SIZE` = 32
/// constant only governs buffer growth granularity for the dense `KVCache`).
///
/// As a result, the ring-buffer wraparound at position `idx` is correct
/// **for any** `max_size` — single-token writes simply overwrite the packed
/// bytes and the per-token norm at `idx`, regardless of where `idx` falls.
///
/// However, we still require `max_size` to be a multiple of
/// [`turbo::BLOCK_SIZE`] (32) when `mode == Turbo4Asym` for two reasons:
///
/// 1. **Future-proofing**: B5 (3-bit packing, issue #477) introduces 24-bit
///    groups that span byte boundaries. A non-32-aligned ring would force
///    partial-block re-quantization on every wraparound. Locking the
///    invariant here means B5 only has to revisit pack/unpack, not the cache.
/// 2. **Trim path simplicity**: the dense `KVCache::trim` path already
///    exploits 32-alignment to avoid mid-block re-quantize work. Holding the
///    same invariant in `RotatingKVCache` keeps speculative-decode rewinds
///    bit-identical between dense and rotating caches.
///
/// All sliding-window models we ship today (Gemma 3 4 K, Gemma 4 8 K,
/// Ministral 3 8 K, GPT-OSS 4 K / 16 K, RecurrentGemma 2 K, Exaone 4 K)
/// already use 32-divisible window sizes, so this constraint is non-binding
/// in practice. The constructor enforces it with a clear error message
/// rather than silently accepting an invalid configuration.
///
/// Used by: Gemma3, Gemma4, Ministral3, GPT-OSS, RecurrentGemma, Exaone
pub struct RotatingKVCache {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    pub max_size: i32,
    pub offset: i32,
    /// Current write position in the buffer (separate from offset to handle trim correctly)
    idx: i32,
    step: i32,
    /// Quantization mode for stored keys/values.
    pub mode: KVCacheMode,
    /// INT8-mode per-token K scale buffer: `[B, H, max_size, 1]` FP16.
    pub(crate) key_scales: Option<UniquePtr<MlxArray>>,
    /// INT8-mode per-token V scale buffer: `[B, H, max_size, 1]` FP16.
    pub(crate) val_scales: Option<UniquePtr<MlxArray>>,
    /// Turbo4Asym-mode V-side packed indices: `[B, H, max_size, head_dim/2]` u8.
    pub(crate) v_packed: Option<UniquePtr<MlxArray>>,
    /// Turbo4Asym-mode V-side per-token norms: `[B, H, max_size, 1]` fp16.
    pub(crate) v_norms: Option<UniquePtr<MlxArray>>,
    /// Turbo4-V precomputed Sparse-V kernel rescale `norm[t] / |y_hat[t]|`
    /// (issue #520). Same shape and lockstep lifecycle as `v_norms`. See the
    /// dense `KVCache::v_rescale` docs for the rationale.
    pub(crate) v_rescale: Option<UniquePtr<MlxArray>>,
    /// Cached PolarQuant params (sign vectors + codebook) for Turbo4Asym.
    /// Lazily initialised on the first Turbo4 update once V `head_dim` is
    /// known; deterministic given `turbo_seed` so detach/adopt round-trips
    /// without recomputing rotations.
    pub(crate) turbo_params: Option<turbo::TurboQuantParams>,
    /// Deterministic seed for the Turbo4 sign vectors. Set at construction
    /// time so detach/adopt round-trip without recomputing rotations.
    pub(crate) turbo_seed: u32,
}

impl RotatingKVCache {
    /// Create a new rotating KV cache with specified maximum size (FP16 mode).
    pub fn new(max_size: i32) -> Self {
        Self::new_with_mode_and_seed(max_size, KVCacheMode::Fp16, TURBO_DEFAULT_SEED)
    }

    /// Create a new rotating KV cache with the specified maximum size and
    /// quantization mode.
    ///
    /// `Turbo4Asym` requires `max_size` to be a positive multiple of
    /// [`turbo::BLOCK_SIZE`] — see the type-level docs for rationale.
    ///
    /// # Panics
    ///
    /// Panics if `mode == Turbo4Asym` and `max_size` is not a positive
    /// multiple of `turbo::BLOCK_SIZE`. Misconfiguring this would leave the
    /// trim path silently broken; failing fast at construction is the only
    /// correct contract.
    pub fn new_with_mode(max_size: i32, mode: KVCacheMode) -> Self {
        Self::new_with_mode_and_seed(max_size, mode, TURBO_DEFAULT_SEED)
    }

    /// Like [`RotatingKVCache::new_with_mode`] but lets the caller pin a
    /// specific Turbo4 sign-vector seed.
    ///
    /// Production callers should prefer `new_with_mode`; the explicit seed is
    /// exposed for tests, benchmarks, and the detach/adopt path which must
    /// preserve the original seed.
    pub fn new_with_mode_and_seed(max_size: i32, mode: KVCacheMode, turbo_seed: u32) -> Self {
        if mode == KVCacheMode::Turbo4Asym {
            assert!(
                max_size > 0 && max_size % turbo::BLOCK_SIZE == 0,
                "RotatingKVCache::new_with_mode: max_size must be a positive multiple of \
                 turbo::BLOCK_SIZE ({}); got {max_size}. See type docs for rationale.",
                turbo::BLOCK_SIZE
            );
        }
        Self {
            keys: None,
            values: None,
            max_size,
            offset: 0,
            idx: 0,
            step: 256,
            mode,
            key_scales: None,
            val_scales: None,
            v_packed: None,
            v_norms: None,
            v_rescale: None,
            turbo_params: None,
            turbo_seed,
        }
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }

    /// Get current sequence length in cache
    pub fn seq_len(&self) -> i32 {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            if shape.len() >= 3 {
                shape[2]
            } else {
                0
            }
        } else {
            0
        }
    }

    /// Update cache with new key/value, rotating if necessary.
    ///
    /// Returns the full cached keys/values, always in FP16 — the public
    /// contract is mode-independent. The `Turbo4Asym` path quantizes V on
    /// write and dequantizes the visible window on read so attention kernels
    /// see standard FP16 tensors regardless of stored representation.
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self.mode {
            KVCacheMode::Fp16 => self.update_and_fetch_fp16(new_keys, new_values),
            KVCacheMode::Int8 => {
                // INT8 support for RotatingKVCache is not part of B9 / issue
                // #481. Fall back to FP16 storage so the path is correct, even
                // if mis-configured. A future sub-issue can wire INT8 in.
                self.update_and_fetch_fp16(new_keys, new_values)
            }
            KVCacheMode::Turbo4Asym => self.update_and_fetch_turbo4_asym(new_keys, new_values),
            KVCacheMode::Turbo4 => {
                // Symmetric Turbo4 (issue #476) is not wired into RotatingKVCache
                // by B9 / issue #481 (RotatingKVCache currently supports only
                // Fp16/Int8/Turbo4Asym). Fall back to FP16 so a mis-configured
                // sliding-window model does not panic; a future sub-issue can
                // wire symmetric K/V quantization in.
                self.update_and_fetch_fp16(new_keys, new_values)
            }
            KVCacheMode::Turbo4Delegated => {
                // Delegated hot/cold (issue #479) is not wired into RotatingKVCache
                // by B7 / issue #479 (the delegated path targets dense caches).
                // Fall back to FP16 so a mis-configured sliding-window model does
                // not panic.
                self.update_and_fetch_fp16(new_keys, new_values)
            }
            KVCacheMode::Turbo3Asym => {
                // Turbo3Asym (issue #477) is not wired into RotatingKVCache in
                // this PR — wraparound + 3-bit re-pack alignment requires its
                // own analysis (mirrors B9 / issue #481 for Turbo4Asym). Fall
                // back to FP16 so sliding-window models with `--kv-cache-mode
                // fp16+turbo3` do not panic; a follow-up sub-issue can wire
                // Turbo3 into the rotating path once dense Turbo3 is validated.
                self.update_and_fetch_fp16(new_keys, new_values)
            }
        }
    }

    fn update_and_fetch_fp16(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let new_seq_len = {
            let shape = ffi::array_shape(&new_keys);
            shape[2]
        };

        if new_seq_len > 1 {
            return self.update_concat(new_keys, new_values, new_seq_len);
        }

        self.update_in_place(new_keys, new_values)
    }

    fn update_concat(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        new_seq_len: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        if self.keys.is_none() {
            self.offset += new_seq_len;
            self.idx = new_seq_len;
            self.keys = Some(ffi::contiguous(&new_keys, false));
            self.values = Some(ffi::contiguous(&new_values, false));
            return (new_keys, new_values);
        }

        let current_seq_len = {
            let shape = ffi::array_shape(self.keys.as_ref().unwrap());
            shape[2]
        };

        let concat_k = concatenate(self.keys.as_ref().unwrap(), &new_keys, 2);
        let concat_v = concatenate(self.values.as_ref().unwrap(), &new_values, 2);

        let total_len = current_seq_len + new_seq_len;
        self.offset += new_seq_len;

        if total_len > self.max_size {
            let start = total_len - self.max_size;
            let k = ffi::slice(
                &concat_k,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            let v = ffi::slice(
                &concat_v,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            self.idx = self.max_size;
            self.keys = Some(ffi::contiguous(&k, false));
            self.values = Some(ffi::contiguous(&v, false));
            (k, v)
        } else {
            self.idx = total_len;
            self.keys = Some(ffi::contiguous(&concat_k, false));
            self.values = Some(ffi::contiguous(&concat_v, false));
            (concat_k, concat_v)
        }
    }

    fn update_in_place(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            let buffer_size = shape[2];
            if buffer_size > self.max_size {
                let start = buffer_size - self.max_size;
                let ks = ffi::array_shape(self.keys.as_ref().unwrap());
                let vs = ffi::array_shape(self.values.as_ref().unwrap());
                self.keys = Some(ffi::contiguous(
                    &ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, start, 0],
                        &[ks[0], ks[1], buffer_size, ks[3]],
                    ),
                    false,
                ));
                self.values = Some(ffi::contiguous(
                    &ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, start, 0],
                        &[vs[0], vs[1], buffer_size, vs[3]],
                    ),
                    false,
                ));
                self.idx = self.max_size;
            }
        }

        if self.keys.is_none() {
            let shape = ffi::array_shape(&new_keys);
            let batch = shape[0];
            let heads = shape[1];
            let head_dim = shape[3];
            let value_shape = ffi::array_shape(&new_values);
            let value_head_dim = value_shape[3];

            let k_zeros = ffi::zeros(
                &[batch, heads, self.max_size, head_dim],
                ffi::array_dtype(&new_keys),
            );
            let v_zeros = ffi::zeros(
                &[batch, heads, self.max_size, value_head_dim],
                ffi::array_dtype(&new_values),
            );

            let k = ffi::slice_update(
                &k_zeros,
                &new_keys,
                &[0, 0, 0, 0],
                &[batch, heads, 1, head_dim],
            );
            let v = ffi::slice_update(
                &v_zeros,
                &new_values,
                &[0, 0, 0, 0],
                &[batch, heads, 1, value_head_dim],
            );

            self.offset = 1;
            self.idx = 1;
            self.keys = Some(ffi::contiguous(&k, false));
            self.values = Some(ffi::contiguous(&v, false));

            let k_out = ffi::slice(&k, &[0, 0, 0, 0], &[batch, heads, 1, head_dim]);
            let v_out = ffi::slice(&v, &[0, 0, 0, 0], &[batch, heads, 1, value_head_dim]);
            return (k_out, v_out);
        }

        let mut k_buffer = self.keys.take().unwrap();
        let mut v_buffer = self.values.take().unwrap();

        let shape = ffi::array_shape(&k_buffer);
        let batch = shape[0];
        let heads = shape[1];
        let buffer_size = shape[2];
        let head_dim = shape[3];
        let value_shape = ffi::array_shape(&v_buffer);
        let value_head_dim = value_shape[3];

        if self.idx >= buffer_size && buffer_size < self.max_size {
            let grow_by = self.step.min(self.max_size - buffer_size).max(0);
            if grow_by > 0 {
                let k_zeros = ffi::zeros(
                    &[batch, heads, grow_by, head_dim],
                    ffi::array_dtype(&new_keys),
                );
                let v_zeros = ffi::zeros(
                    &[batch, heads, grow_by, value_head_dim],
                    ffi::array_dtype(&new_values),
                );
                k_buffer = concatenate(&k_buffer, &k_zeros, 2);
                v_buffer = concatenate(&v_buffer, &v_zeros, 2);
            }
        }

        if self.idx >= self.max_size {
            self.idx = 0;
        }

        let pos = self.idx;
        let k_buffer = ffi::slice_update(
            &k_buffer,
            &new_keys,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, head_dim],
        );
        let v_buffer = ffi::slice_update(
            &v_buffer,
            &new_values,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, value_head_dim],
        );

        self.offset += 1;
        self.idx += 1;

        if self.offset < self.max_size {
            let k_out = ffi::slice(
                &k_buffer,
                &[0, 0, 0, 0],
                &[batch, heads, self.offset, head_dim],
            );
            let v_out = ffi::slice(
                &v_buffer,
                &[0, 0, 0, 0],
                &[batch, heads, self.offset, value_head_dim],
            );
            self.keys = Some(k_buffer);
            self.values = Some(v_buffer);
            (k_out, v_out)
        } else {
            let k_out = ffi::contiguous(&k_buffer, false);
            let v_out = ffi::contiguous(&v_buffer, false);
            self.keys = Some(k_buffer);
            self.values = Some(v_buffer);
            (k_out, v_out)
        }
    }

    // -----------------------------------------------------------------------
    // Turbo4Asym update path (B9, issue #481, epic #458)
    //
    // Storage layout when `mode == KVCacheMode::Turbo4Asym`:
    //   - keys     : [B, H, max_size, K_dim]    FP16 — same as Fp16 path
    //   - v_packed : [B, H, max_size, V_dim/2]  UINT8 — nibble-packed indices
    //   - v_norms  : [B, H, max_size, 1]        FP16 — per-token V L2 norms
    //   - values   : None — Turbo4Asym never stores fp16 V
    //
    // Block alignment invariant: `max_size % BLOCK_SIZE == 0` (asserted by
    // `new_with_mode`). Quantization is per-token along `head_dim`, so each
    // ring-buffer slot is independent — wraparound at any 32-aligned `idx`
    // lands on a fresh slot whose packed bytes can be overwritten without
    // disturbing neighbours.
    // -----------------------------------------------------------------------

    fn update_and_fetch_turbo4_asym(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let new_seq_len = {
            let shape = ffi::array_shape(&new_keys);
            shape[2]
        };

        // Cast incoming K/V to FP16 to match the cache contract, then quantize
        // V before any cache mutation.
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        // Lazy-init TurboQuantParams once V head_dim is known.
        if self.turbo_params.is_none() {
            let v_shape = ffi::array_shape(&new_values_f16);
            let v_head_dim = v_shape[3] as u32;
            self.turbo_params = Some(turbo::TurboQuantParams::new(v_head_dim, self.turbo_seed));
        }
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params just initialised");
        let (v_packed_new, v_norms_new, v_rescale_new) =
            turbo::quant::quantize_v_turbo4(&new_values_f16, params);

        if new_seq_len > 1 {
            self.update_turbo4_concat(
                new_keys_f16,
                v_packed_new,
                v_norms_new,
                v_rescale_new,
                new_seq_len,
            )
        } else {
            self.update_turbo4_in_place(new_keys_f16, v_packed_new, v_norms_new, v_rescale_new)
        }
    }

    /// Multi-token (prefill) path for Turbo4Asym. Mirrors
    /// `update_concat` but operates on the V sidecars instead of an FP16 V
    /// buffer. K side is identical to the FP16 path.
    fn update_turbo4_concat(
        &mut self,
        new_keys_f16: UniquePtr<MlxArray>,
        v_packed_new: UniquePtr<MlxArray>,
        v_norms_new: UniquePtr<MlxArray>,
        v_rescale_new: UniquePtr<MlxArray>,
        new_seq_len: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // First update: just install (Same shape semantics as Fp16 path.)
        if self.keys.is_none() {
            self.offset += new_seq_len;
            self.idx = new_seq_len;
            self.keys = Some(ffi::contiguous(&new_keys_f16, false));
            self.v_packed = Some(ffi::contiguous(&v_packed_new, false));
            self.v_norms = Some(ffi::contiguous(&v_norms_new, false));
            self.v_rescale = Some(ffi::contiguous(&v_rescale_new, false));
            // Build fp16 V output by dequantizing the packed slice we just stored.
            let params = self
                .turbo_params
                .as_ref()
                .expect("turbo_params populated by caller");
            let v_out = turbo::quant::dequantize_v_turbo4(&v_packed_new, &v_norms_new, params);
            return (new_keys_f16, v_out);
        }

        let current_seq_len = {
            let shape = ffi::array_shape(self.keys.as_ref().unwrap());
            shape[2]
        };

        let concat_k = concatenate(self.keys.as_ref().unwrap(), &new_keys_f16, 2);
        let concat_vp = concatenate(self.v_packed.as_ref().unwrap(), &v_packed_new, 2);
        let concat_vn = concatenate(self.v_norms.as_ref().unwrap(), &v_norms_new, 2);
        let concat_vr = match self.v_rescale.as_ref() {
            Some(vr) => concatenate(vr, &v_rescale_new, 2),
            None => ffi::contiguous(&v_rescale_new, false),
        };

        let total_len = current_seq_len + new_seq_len;
        self.offset += new_seq_len;

        // Block alignment is preserved: per-token quantization means slicing
        // a packed buffer along axis=2 always yields valid packed tokens. No
        // partial-block re-quantization is needed regardless of `start`.
        if total_len > self.max_size {
            let start = total_len - self.max_size;
            let k = ffi::slice(
                &concat_k,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            let vp = ffi::slice(
                &concat_vp,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            let vn = ffi::slice(
                &concat_vn,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            let vr = ffi::slice(
                &concat_vr,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            self.idx = self.max_size;
            self.keys = Some(ffi::contiguous(&k, false));
            self.v_packed = Some(ffi::contiguous(&vp, false));
            self.v_norms = Some(ffi::contiguous(&vn, false));
            self.v_rescale = Some(ffi::contiguous(&vr, false));
            let params = self
                .turbo_params
                .as_ref()
                .expect("turbo_params populated by caller");
            let v_out = turbo::quant::dequantize_v_turbo4(&vp, &vn, params);
            (k, v_out)
        } else {
            self.idx = total_len;
            self.keys = Some(ffi::contiguous(&concat_k, false));
            self.v_packed = Some(ffi::contiguous(&concat_vp, false));
            self.v_norms = Some(ffi::contiguous(&concat_vn, false));
            self.v_rescale = Some(ffi::contiguous(&concat_vr, false));
            let params = self
                .turbo_params
                .as_ref()
                .expect("turbo_params populated by caller");
            let v_out = turbo::quant::dequantize_v_turbo4(&concat_vp, &concat_vn, params);
            (concat_k, v_out)
        }
    }

    /// Single-token (decode) path for Turbo4Asym. Mirrors `update_in_place`,
    /// writing the new packed token + norm into the ring buffer at `self.idx`.
    /// Wraparound at `idx == max_size` resets `idx` to 0 — because each token
    /// is independently quantized, the overwrite is byte-correct without any
    /// re-quantize work.
    fn update_turbo4_in_place(
        &mut self,
        new_keys_f16: UniquePtr<MlxArray>,
        v_packed_new: UniquePtr<MlxArray>,
        v_norms_new: UniquePtr<MlxArray>,
        v_rescale_new: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // First-call init mirrors the FP16 path: pre-allocate the ring buffer.
        if self.keys.is_none() {
            let shape = ffi::array_shape(&new_keys_f16);
            let batch = shape[0];
            let heads = shape[1];
            let head_dim = shape[3];
            let vp_shape = ffi::array_shape(&v_packed_new);
            let v_packed_dim = vp_shape[3];

            let k_zeros = ffi::zeros(
                &[batch, heads, self.max_size, head_dim],
                ffi::array_dtype(&new_keys_f16),
            );
            let vp_zeros = ffi::zeros(&[batch, heads, self.max_size, v_packed_dim], dtype::UINT8);
            let vn_zeros = ffi::zeros(&[batch, heads, self.max_size, 1], dtype::FLOAT16);
            let vr_zeros = ffi::zeros(&[batch, heads, self.max_size, 1], dtype::FLOAT16);

            let k = ffi::slice_update(
                &k_zeros,
                &new_keys_f16,
                &[0, 0, 0, 0],
                &[batch, heads, 1, head_dim],
            );
            let vp = ffi::slice_update(
                &vp_zeros,
                &v_packed_new,
                &[0, 0, 0, 0],
                &[batch, heads, 1, v_packed_dim],
            );
            let vn = ffi::slice_update(
                &vn_zeros,
                &v_norms_new,
                &[0, 0, 0, 0],
                &[batch, heads, 1, 1],
            );
            let vr = ffi::slice_update(
                &vr_zeros,
                &v_rescale_new,
                &[0, 0, 0, 0],
                &[batch, heads, 1, 1],
            );

            self.offset = 1;
            self.idx = 1;
            self.keys = Some(ffi::contiguous(&k, false));
            self.v_packed = Some(ffi::contiguous(&vp, false));
            self.v_norms = Some(ffi::contiguous(&vn, false));
            self.v_rescale = Some(ffi::contiguous(&vr, false));

            // Build the FP16 V view from the single packed token.
            let params = self
                .turbo_params
                .as_ref()
                .expect("turbo_params populated by caller");
            let vp_slice = ffi::slice(&vp, &[0, 0, 0, 0], &[batch, heads, 1, v_packed_dim]);
            let vn_slice = ffi::slice(&vn, &[0, 0, 0, 0], &[batch, heads, 1, 1]);
            let v_out = turbo::quant::dequantize_v_turbo4(&vp_slice, &vn_slice, params);
            let k_out = ffi::slice(&k, &[0, 0, 0, 0], &[batch, heads, 1, head_dim]);
            return (k_out, v_out);
        }

        let mut k_buffer = self.keys.take().unwrap();
        let mut vp_buffer = self.v_packed.take().unwrap();
        let mut vn_buffer = self.v_norms.take().unwrap();
        let mut vr_buffer = self
            .v_rescale
            .take()
            .expect("v_rescale lockstep with v_norms in rotating cache");

        let shape = ffi::array_shape(&k_buffer);
        let batch = shape[0];
        let heads = shape[1];
        let buffer_size = shape[2];
        let head_dim = shape[3];
        let vp_shape = ffi::array_shape(&vp_buffer);
        let v_packed_dim = vp_shape[3];

        // Lazy buffer growth (mirrors fp16 path). For Turbo4Asym this only
        // fires before the first wraparound; once `buffer_size == max_size`
        // we stay there forever.
        if self.idx >= buffer_size && buffer_size < self.max_size {
            let grow_by = self.step.min(self.max_size - buffer_size).max(0);
            if grow_by > 0 {
                let k_zeros = ffi::zeros(
                    &[batch, heads, grow_by, head_dim],
                    ffi::array_dtype(&new_keys_f16),
                );
                let vp_zeros = ffi::zeros(&[batch, heads, grow_by, v_packed_dim], dtype::UINT8);
                let vn_zeros = ffi::zeros(&[batch, heads, grow_by, 1], dtype::FLOAT16);
                let vr_zeros = ffi::zeros(&[batch, heads, grow_by, 1], dtype::FLOAT16);
                k_buffer = concatenate(&k_buffer, &k_zeros, 2);
                vp_buffer = concatenate(&vp_buffer, &vp_zeros, 2);
                vn_buffer = concatenate(&vn_buffer, &vn_zeros, 2);
                vr_buffer = concatenate(&vr_buffer, &vr_zeros, 2);
            }
        }

        // Wraparound. Per-token independence makes this a simple in-place
        // overwrite; no block-edge re-quantization is required.
        if self.idx >= self.max_size {
            self.idx = 0;
        }

        let pos = self.idx;
        let k_buffer = ffi::slice_update(
            &k_buffer,
            &new_keys_f16,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, head_dim],
        );
        let vp_buffer = ffi::slice_update(
            &vp_buffer,
            &v_packed_new,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, v_packed_dim],
        );
        let vn_buffer = ffi::slice_update(
            &vn_buffer,
            &v_norms_new,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, 1],
        );
        let vr_buffer = ffi::slice_update(
            &vr_buffer,
            &v_rescale_new,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, 1],
        );

        self.offset += 1;
        self.idx += 1;

        // Build the FP16 V view from the visible window and return.
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params populated by caller");
        let result = if self.offset < self.max_size {
            let visible = self.offset;
            let k_out = ffi::slice(&k_buffer, &[0, 0, 0, 0], &[batch, heads, visible, head_dim]);
            let vp_view = ffi::slice(
                &vp_buffer,
                &[0, 0, 0, 0],
                &[batch, heads, visible, v_packed_dim],
            );
            let vn_view = ffi::slice(&vn_buffer, &[0, 0, 0, 0], &[batch, heads, visible, 1]);
            let v_out = turbo::quant::dequantize_v_turbo4(&vp_view, &vn_view, params);
            (k_out, v_out)
        } else {
            // Ring is full — return the contiguous full buffer view (matches
            // fp16 semantics: callers see the complete sliding window).
            let k_out = ffi::contiguous(&k_buffer, false);
            let v_out = turbo::quant::dequantize_v_turbo4(&vp_buffer, &vn_buffer, params);
            (k_out, v_out)
        };
        self.keys = Some(k_buffer);
        self.v_packed = Some(vp_buffer);
        self.v_norms = Some(vn_buffer);
        self.v_rescale = Some(vr_buffer);
        result
    }

    /// Get the current offset
    pub fn get_offset(&self) -> i32 {
        self.offset
    }

    /// Visible length exposed to decode attention.
    pub fn visible_len(&self) -> i32 {
        self.seq_len().min(self.offset).max(0)
    }

    /// Physical start index of the logical oldest token in the ring buffer.
    ///
    /// Before the cache wraps, the visible region starts at index 0. After
    /// wrapping, `idx` tracks the next write position, which is also the
    /// oldest logical token in the ring.
    pub fn logical_start(&self) -> i32 {
        let visible_len = self.visible_len();
        if visible_len == 0 || self.offset <= visible_len {
            0
        } else {
            self.idx.rem_euclid(visible_len)
        }
    }
}

impl Default for RotatingKVCache {
    fn default() -> Self {
        Self::new(4096)
    }
}

/// Chunked KV Cache for Llama 4's iGQA (Interleaved GQA) attention.
///
/// Maintains a sliding window cache that trims from the front when exceeding
/// `chunk_size`, while still tracking the global start position for mask logic.
pub struct ChunkedKVCache {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    pub chunk_size: i32,
    pub offset: i32,
    pub start_position: i32,
    step: i32,
}

impl ChunkedKVCache {
    /// Create a new chunked KV cache with specified chunk size
    pub fn new(chunk_size: i32) -> Self {
        Self {
            keys: None,
            values: None,
            chunk_size,
            offset: 0,
            start_position: 0,
            step: 256,
        }
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }

    /// Get the global offset (total tokens processed)
    pub fn get_offset(&self) -> i32 {
        self.offset
    }

    /// Get the start position (where visible window begins)
    pub fn get_start_position(&self) -> i32 {
        self.start_position
    }

    /// Trim the front of the cache if it exceeds chunk_size.
    ///
    /// This should be called before processing each layer.
    pub fn maybe_trim_front(&mut self) {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            let seq_len = (self.offset - self.start_position).min(shape[2]);

            if seq_len > self.chunk_size {
                let trim_amount = seq_len - self.chunk_size;
                self.start_position += trim_amount;

                let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
                let v_shape = ffi::array_shape(self.values.as_ref().unwrap());

                self.keys = Some(ffi::slice(
                    self.keys.as_ref().unwrap(),
                    &[0, 0, trim_amount, 0],
                    &[k_shape[0], k_shape[1], seq_len, k_shape[3]],
                ));
                self.values = Some(ffi::slice(
                    self.values.as_ref().unwrap(),
                    &[0, 0, trim_amount, 0],
                    &[v_shape[0], v_shape[1], seq_len, v_shape[3]],
                ));
            }
        }
    }

    /// Update cache with new key/value and return the visible portion
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let new_shape = ffi::array_shape(&new_keys);
        let new_seq_len = new_shape[2];
        let prev = self.offset - self.start_position;

        if self.keys.is_none() || (prev + new_seq_len) > self.get_buffer_size() {
            let b = new_shape[0];
            let n_kv_heads = new_shape[1];
            let k_head_dim = new_shape[3];
            let v_shape = ffi::array_shape(&new_values);
            let v_head_dim = v_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let new_buffer_size = n_steps * self.step;

            let new_k = ffi::zeros(
                &[b, n_kv_heads, new_buffer_size, k_head_dim],
                ffi::array_dtype(&new_keys),
            );
            let new_v = ffi::zeros(
                &[b, n_kv_heads, new_buffer_size, v_head_dim],
                ffi::array_dtype(&new_values),
            );

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
                    let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[k_shape[0], k_shape[1], prev, k_shape[3]],
                    ));
                    self.values = Some(ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[v_shape[0], v_shape[1], prev, v_shape[3]],
                    ));
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k, 2));
                self.values = Some(concatenate(self.values.as_ref().unwrap(), &new_v, 2));
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        self.offset += new_seq_len;
        let end = self.offset - self.start_position;

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());

        if prev > 0 {
            let k_before = ffi::slice(
                self.keys.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[k_shape[0], k_shape[1], prev, k_shape[3]],
            );
            let v_before = ffi::slice(
                self.values.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[v_shape[0], v_shape[1], prev, v_shape[3]],
            );

            if end < k_shape[2] {
                let k_after = ffi::slice(
                    self.keys.as_ref().unwrap(),
                    &[0, 0, end, 0],
                    &[k_shape[0], k_shape[1], k_shape[2], k_shape[3]],
                );
                let v_after = ffi::slice(
                    self.values.as_ref().unwrap(),
                    &[0, 0, end, 0],
                    &[v_shape[0], v_shape[1], v_shape[2], v_shape[3]],
                );
                self.keys = Some(concatenate(
                    &concatenate(&k_before, &new_keys, 2),
                    &k_after,
                    2,
                ));
                self.values = Some(concatenate(
                    &concatenate(&v_before, &new_values, 2),
                    &v_after,
                    2,
                ));
            } else {
                self.keys = Some(concatenate(&k_before, &new_keys, 2));
                self.values = Some(concatenate(&v_before, &new_values, 2));
            }
        } else if end < k_shape[2] {
            let k_after = ffi::slice(
                self.keys.as_ref().unwrap(),
                &[0, 0, end, 0],
                &[k_shape[0], k_shape[1], k_shape[2], k_shape[3]],
            );
            let v_after = ffi::slice(
                self.values.as_ref().unwrap(),
                &[0, 0, end, 0],
                &[v_shape[0], v_shape[1], v_shape[2], v_shape[3]],
            );
            self.keys = Some(concatenate(&new_keys, &k_after, 2));
            self.values = Some(concatenate(&new_values, &v_after, 2));
        } else {
            self.keys = Some(ffi::contiguous(&new_keys, false));
            self.values = Some(ffi::contiguous(&new_values, false));
        }

        (
            ffi::slice(
                self.keys.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[k_shape[0], k_shape[1], end, k_shape[3]],
            ),
            ffi::slice(
                self.values.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[v_shape[0], v_shape[1], end, v_shape[3]],
            ),
        )
    }

    fn get_buffer_size(&self) -> i32 {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            shape[2]
        } else {
            0
        }
    }
}

/// Structure-of-arrays metadata for batched decode position handling.
///
/// This keeps per-sequence offsets, query lengths, visible KV lengths, and
/// window sizes in a kernel-friendly representation instead of relying on
/// scalar `cache.offset` assumptions inside batched model code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchedAttentionMetadata {
    pub rope_offsets: Vec<i32>,
    pub query_lens: Vec<i32>,
    pub kv_lens: Vec<i32>,
    pub window_sizes: Vec<i32>,
}

impl BatchedAttentionMetadata {
    /// Build heterogeneous per-sequence metadata from standard KV caches.
    pub fn from_kv_caches(
        caches: &[&mut KVCache],
        query_lens: &[i32],
        window_sizes: &[i32],
    ) -> Result<Self, String> {
        let batch = caches.len();
        if query_lens.len() != batch {
            return Err(format!(
                "expected {} query lengths for batched attention metadata, got {}",
                batch,
                query_lens.len()
            ));
        }
        if window_sizes.len() != batch {
            return Err(format!(
                "expected {} window sizes for batched attention metadata, got {}",
                batch,
                window_sizes.len()
            ));
        }

        let mut rope_offsets = Vec::with_capacity(batch);
        let mut kv_lens = Vec::with_capacity(batch);
        for (cache, &query_len) in caches.iter().zip(query_lens.iter()) {
            if query_len < 0 {
                return Err(format!(
                    "query length must be non-negative for batched attention metadata, got {query_len}"
                ));
            }
            let offset = cache.offset;
            rope_offsets.push(offset);
            kv_lens.push(offset + query_len);
        }

        Ok(Self {
            rope_offsets,
            query_lens: query_lens.to_vec(),
            kv_lens,
            window_sizes: window_sizes.to_vec(),
        })
    }

    /// Build uniform metadata for full-attention batched decode/prefill paths.
    pub fn uniform_kv_caches(
        caches: &[&mut KVCache],
        query_len: i32,
        window_size: i32,
    ) -> Result<Self, String> {
        let query_lens = vec![query_len; caches.len()];
        let window_sizes = vec![window_size; caches.len()];
        Self::from_kv_caches(caches, &query_lens, &window_sizes)
    }

    pub fn len(&self) -> usize {
        self.rope_offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rope_offsets.is_empty()
    }
}

/// Decode-only paged attention metadata derived from per-sequence KV lengths.
///
/// The current dense-compat kernel treats `block_tables` as logical block
/// indices (`0..num_blocks`) for each sequence. A future physical paged-KV
/// backend can reuse the same shape while replacing these entries with actual
/// physical block identifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedDecodeMetadata {
    pub block_size: i32,
    pub kv_lens: Vec<i32>,
    pub block_table_offsets: Vec<i32>,
    pub block_tables: Vec<i32>,
}

/// Decode-only paged attention metadata for ring-buffer-backed rotating caches.
///
/// `logical_starts[i]` identifies the physical buffer index of the oldest
/// visible token for sequence `i`, allowing native paged decode kernels to
/// gather wrapped sliding-window buffers without first materializing a dense
/// linearized copy in Rust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotatingPagedDecodeMetadata {
    pub block_size: i32,
    pub kv_lens: Vec<i32>,
    pub logical_starts: Vec<i32>,
}

impl RotatingPagedDecodeMetadata {
    pub fn from_parts(
        kv_lens: &[i32],
        logical_starts: &[i32],
        block_size: i32,
    ) -> Result<Self, String> {
        if block_size <= 0 {
            return Err(format!(
                "rotating paged decode metadata requires block_size > 0, got {block_size}"
            ));
        }
        if kv_lens.len() != logical_starts.len() {
            return Err(format!(
                "rotating paged decode metadata length mismatch: {} kv_lens vs {} logical_starts",
                kv_lens.len(),
                logical_starts.len()
            ));
        }
        for (&kv_len, &logical_start) in kv_lens.iter().zip(logical_starts.iter()) {
            if kv_len < 0 {
                return Err(format!(
                    "rotating paged decode metadata requires non-negative kv lengths, got {kv_len}"
                ));
            }
            if logical_start < 0 {
                return Err(format!(
                    "rotating paged decode metadata requires non-negative logical starts, got {logical_start}"
                ));
            }
            if kv_len > 0 && logical_start >= kv_len {
                return Err(format!(
                    "rotating paged decode metadata requires logical_start < kv_len when kv_len > 0, got logical_start={logical_start}, kv_len={kv_len}"
                ));
            }
        }

        Ok(Self {
            block_size,
            kv_lens: kv_lens.to_vec(),
            logical_starts: logical_starts.to_vec(),
        })
    }

    pub fn len(&self) -> usize {
        self.kv_lens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kv_lens.is_empty()
    }
}

impl PagedDecodeMetadata {
    pub fn from_attention_metadata(
        metadata: &BatchedAttentionMetadata,
        block_size: i32,
    ) -> Result<Self, String> {
        Self::from_visible_lengths(&metadata.kv_lens, block_size)
    }

    pub fn from_visible_lengths(kv_lens: &[i32], block_size: i32) -> Result<Self, String> {
        if block_size <= 0 {
            return Err(format!(
                "paged decode metadata requires block_size > 0, got {block_size}"
            ));
        }

        let mut block_table_offsets = Vec::with_capacity(kv_lens.len() + 1);
        let mut block_tables = Vec::new();
        block_table_offsets.push(0);

        for &kv_len in kv_lens {
            if kv_len < 0 {
                return Err(format!(
                    "paged decode metadata requires non-negative kv lengths, got {kv_len}"
                ));
            }

            let block_count = if kv_len == 0 {
                0
            } else {
                (kv_len + block_size - 1) / block_size
            };
            for logical_block in 0..block_count {
                block_tables.push(logical_block);
            }
            block_table_offsets.push(block_tables.len() as i32);
        }

        Ok(Self {
            block_size,
            kv_lens: kv_lens.to_vec(),
            block_table_offsets,
            block_tables,
        })
    }

    pub fn len(&self) -> usize {
        self.kv_lens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kv_lens.is_empty()
    }
}

impl Default for ChunkedKVCache {
    fn default() -> Self {
        Self::new(8192)
    }
}

// --- Per-sequence cache isolation for continuous batching ---

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Unique identifier for a sequence in the batch.
///
/// Each active generation sequence receives a unique monotonically increasing
/// ID from the owning `CachePool`. The inner `u64` never wraps within any
/// reasonable server lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(u64);

impl SequenceId {
    /// Construct a `SequenceId` from a raw `u64` value.
    ///
    /// In production code, IDs are assigned by `CachePool::allocate`. This
    /// constructor is provided for tests, builders, and deserialization.
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Return the raw numeric identifier.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SequenceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "seq-{}", self.0)
    }
}

/// Logical owner/backend for one sequence's runtime state.
///
/// Phase 0 keeps all backends represented through the existing `Vec<KVCache>`
/// surface so behavior stays unchanged while the control plane gains an
/// explicit seam for future paged and model-owned state backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStateBackend {
    /// Standard per-layer external KV caches stored directly in the sequence.
    DenseKvCache,
    /// Paged block tables plus logical sequence metadata.
    PagedKvCache,
    /// Model-owned/internal state. The exposed `Vec<KVCache>` acts only as a
    /// compatibility placeholder for existing generation and scheduler paths.
    ModelOwned,
}

/// Backend/layout descriptor for allocating one sequence's runtime state.
///
/// Used by: `LanguageModel::sequence_state_layout()`, `CachePool::allocate()`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceStateLayout {
    pub backend: SequenceStateBackend,
    pub num_layers: usize,
    pub paged_layout: Option<PagedKvLayout>,
}

impl SequenceStateLayout {
    /// Allocate per-layer dense external KV caches for this sequence.
    pub const fn dense_kv_cache(num_layers: usize) -> Self {
        Self {
            backend: SequenceStateBackend::DenseKvCache,
            num_layers,
            paged_layout: None,
        }
    }

    /// Allocate per-layer paged KV state for this sequence.
    pub fn paged_kv_cache(paged_layout: PagedKvLayout) -> Self {
        Self {
            backend: SequenceStateBackend::PagedKvCache,
            num_layers: paged_layout.num_layers,
            paged_layout: Some(paged_layout),
        }
    }

    /// Allocate model-owned/internal sequence state with placeholder KV slots.
    pub const fn model_owned(num_layers: usize) -> Self {
        Self {
            backend: SequenceStateBackend::ModelOwned,
            num_layers,
            paged_layout: None,
        }
    }
}

/// One sequence's full set of layer caches.
///
/// Created by `CachePool::allocate` and tied to a single generation request.
/// The caller owns mutable access while the sequence is active and must call
/// `CachePool::release` when generation finishes.
pub struct SequenceCacheSet {
    /// Logical owner/backend for this sequence's state.
    pub backend: SequenceStateBackend,
    /// Per-layer KV caches (one entry per model layer).
    pub caches: Vec<KVCache>,
    /// Paged block-table state when `backend == PagedKvCache`.
    pub paged: Option<PagedSequenceState>,
    /// Unique identifier assigned by the pool.
    pub seq_id: SequenceId,
    /// Number of prompt tokens originally prefilled.
    pub prompt_len: usize,
    /// Current generation position (incremented during decode).
    pub current_offset: i32,
    /// Wall-clock time when this cache set was allocated.
    pub created_at: Instant,
    paged_layout: Option<PagedKvLayout>,
}

impl SequenceCacheSet {
    fn with_backend(
        seq_id: SequenceId,
        backend: SequenceStateBackend,
        caches: Vec<KVCache>,
        paged: Option<PagedSequenceState>,
        paged_layout: Option<PagedKvLayout>,
    ) -> Self {
        Self {
            backend,
            caches,
            paged,
            seq_id,
            prompt_len: 0,
            current_offset: 0,
            created_at: Instant::now(),
            paged_layout,
        }
    }

    /// Allocate a sequence state backed by standard external KV caches.
    pub fn dense_external(seq_id: SequenceId, caches: Vec<KVCache>) -> Self {
        Self::with_backend(
            seq_id,
            SequenceStateBackend::DenseKvCache,
            caches,
            None,
            None,
        )
    }

    /// Allocate a sequence state backed by paged block tables.
    pub fn paged(seq_id: SequenceId, paged_layout: PagedKvLayout) -> Self {
        let paged = PagedSequenceState::new(&paged_layout);
        Self::with_backend(
            seq_id,
            SequenceStateBackend::PagedKvCache,
            Vec::new(),
            Some(paged),
            Some(paged_layout),
        )
    }

    /// Allocate a sequence state for model-owned/internal caches.
    pub fn model_owned(seq_id: SequenceId) -> Self {
        Self::with_backend(
            seq_id,
            SequenceStateBackend::ModelOwned,
            Vec::new(),
            None,
            None,
        )
    }

    /// Total memory footprint of all layer caches in bytes.
    pub fn nbytes(&self) -> usize {
        let dense_bytes: usize = self.caches.iter().map(|c| c.nbytes()).sum();
        let paged_bytes = self
            .paged
            .as_ref()
            .zip(self.paged_layout.as_ref())
            .map_or(0, |(state, layout)| state.used_bytes(layout));
        dense_bytes + paged_bytes
    }

    pub fn paged_state(&self) -> Option<&PagedSequenceState> {
        self.paged.as_ref()
    }

    pub fn paged_state_mut(&mut self) -> Option<&mut PagedSequenceState> {
        self.paged.as_mut()
    }

    pub fn paged_stats(&self) -> Option<PagedCacheStats> {
        self.paged
            .as_ref()
            .zip(self.paged_layout.as_ref())
            .map(|(state, layout)| PagedCacheStats {
                allocated_blocks: state.reserved_blocks(),
                live_blocks: state.reserved_blocks(),
                free_blocks: 0,
                bytes_reserved: state.reserved_bytes(layout),
                bytes_in_use: state.used_bytes(layout),
            })
    }

    pub fn paged_layout(&self) -> Option<&PagedKvLayout> {
        self.paged_layout.as_ref()
    }
}

/// Pool that allocates and recycles per-sequence cache sets.
///
/// Designed for use by a continuous-batching scheduler. The pool assigns
/// monotonically increasing `SequenceId` values and enforces a hard upper
/// bound on concurrent active sequences.
///
/// Thread safety: `CachePool` itself is **not** `Sync`; callers in async
/// server code should wrap it in an appropriate lock (`Mutex` or `RwLock`).
pub struct CachePool {
    next_id: AtomicU64,
    active: HashMap<SequenceId, SequenceCacheSet>,
    max_sequences: usize,
    paged_pool: Option<PagedBlockPool>,
    /// Detached cache sets parked inside the pool during cross-request
    /// handoffs. See [`detach`] for the full design.
    detached: detach::DetachedMap,
}

impl CachePool {
    /// Create a new pool allowing up to `max_sequences` concurrent cache sets.
    pub fn new(max_sequences: usize) -> Self {
        Self {
            next_id: AtomicU64::new(0),
            active: HashMap::new(),
            max_sequences,
            paged_pool: None,
            detached: HashMap::new(),
        }
    }

    /// Allocate a fresh cache set for a new sequence.
    ///
    /// For batching models, calls `model.make_caches()` to build per-layer
    /// caches and enforces the `max_sequences` capacity limit.
    ///
    /// For non-batching models (internal RefCell/SSM caches), allocates a
    /// lightweight placeholder entry with dummy caches — without calling
    /// `make_caches()` — so that requests can be queued while another
    /// sequence is still generating.  The scheduler resets the model's
    /// internal caches at prefill time, not at enqueue time.
    pub fn allocate(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
    ) -> Result<SequenceId, String> {
        self.allocate_with_layout(model, None)
    }

    /// Allocate a fresh cache set using either the model default layout or an
    /// explicit server-side override.
    pub fn allocate_with_layout(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        layout_override: Option<SequenceStateLayout>,
    ) -> Result<SequenceId, String> {
        let layout = layout_override.unwrap_or_else(|| model.sequence_state_layout());
        if layout.backend == SequenceStateBackend::DenseKvCache
            && self.active.len() >= self.max_sequences
        {
            return Err(format!(
                "CachePool: max capacity ({}) reached, cannot allocate new sequence",
                self.max_sequences
            ));
        }

        let id = SequenceId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let entry = match layout.backend {
            SequenceStateBackend::DenseKvCache => {
                SequenceCacheSet::dense_external(id, model.make_caches())
            }
            SequenceStateBackend::PagedKvCache => {
                let paged_layout = layout.paged_layout.ok_or_else(|| {
                    "CachePool: paged backend requires a paged layout".to_string()
                })?;
                self.ensure_paged_pool(&paged_layout)?;
                SequenceCacheSet::with_backend(
                    id,
                    SequenceStateBackend::PagedKvCache,
                    model.make_caches(),
                    Some(PagedSequenceState::new(&paged_layout)),
                    Some(paged_layout),
                )
            }
            SequenceStateBackend::ModelOwned => SequenceCacheSet::model_owned(id),
        };
        self.active.insert(id, entry);
        Ok(id)
    }

    /// Return a mutable reference to the full `SequenceCacheSet` for the
    /// given sequence, or `None` if the ID is not active.
    pub fn get_mut(&mut self, id: SequenceId) -> Option<&mut SequenceCacheSet> {
        self.active.get_mut(&id)
    }

    pub fn get_paged_state(&self, id: SequenceId) -> Option<&PagedSequenceState> {
        self.active.get(&id)?.paged_state()
    }

    pub fn get_paged_state_mut(&mut self, id: SequenceId) -> Option<&mut PagedSequenceState> {
        self.active.get_mut(&id)?.paged_state_mut()
    }

    /// Return a mutable slice of the per-layer KV caches for direct use
    /// in `model.forward()`, or `None` if the ID is not active.
    pub fn get_caches_mut(&mut self, id: SequenceId) -> Option<&mut [KVCache]> {
        self.active.get_mut(&id).map(|s| s.caches.as_mut_slice())
    }

    /// Return cache slices for multiple active sequences in one call.
    ///
    /// This centralizes the aliasing/unsafe boundary so scheduler code does
    /// not need to reconstruct `&mut [KVCache]` slices from raw pointers.
    pub fn get_batch_caches_mut<'a>(
        &'a mut self,
        ids: &[SequenceId],
    ) -> Result<Vec<&'a mut [KVCache]>, String> {
        let mut cache_ptrs: Vec<(*mut KVCache, usize)> = Vec::with_capacity(ids.len());
        for &id in ids {
            let (ptr, len) = {
                let caches = self
                    .get_caches_mut(id)
                    .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
                (caches.as_mut_ptr(), caches.len())
            };
            cache_ptrs.push((ptr, len));
        }

        // SAFETY: each `SequenceId` maps to a distinct `SequenceCacheSet`
        // allocation inside the HashMap, and callers ensure the same id is not
        // requested twice in one batch. The returned slices are tied to the
        // lifetime of `&mut self` and no mutation of `self.active` occurs
        // between pointer extraction and slice reconstruction.
        Ok(cache_ptrs
            .iter()
            .map(|&(ptr, len)| unsafe { std::slice::from_raw_parts_mut(ptr, len) })
            .collect())
    }

    /// Release a sequence's caches, reclaiming the pool slot.
    ///
    /// This is a no-op if `id` is not currently active.
    pub fn release(&mut self, id: SequenceId) {
        if let Some(mut sequence) = self.active.remove(&id) {
            if let Some(pool) = self.paged_pool.as_mut() {
                if let Some(state) = sequence.paged_state_mut() {
                    let _ = pool.release_sequence(state);
                }
            }
        }
    }

    /// Number of sequences currently holding active cache sets.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Sum of `nbytes()` across all active cache sets, plus any detached
    /// cache sets currently parked inside the pool.
    ///
    /// Parked sets are tensors the pool still physically holds in-flight
    /// during cross-request handoffs (see `detach` / `adopt`), so including
    /// them here keeps memory accounting consistent for schedulers that
    /// admission-control on `memory_usage_bytes()`.
    ///
    /// Used by: `BatchScheduler::update_gauges` (server observability), any
    /// admission-control path that caps KV memory before accepting new work.
    pub fn memory_usage_bytes(&self) -> usize {
        let active_bytes: usize = self.active.values().map(|s| s.nbytes()).sum();
        let parked_bytes: usize = self.detached.values().map(|d| d.nbytes()).sum();
        // Per-page Turbo4 sidecars in the paged pool are owned by the pool
        // itself rather than by any individual `SequenceCacheSet`, so they
        // are not visible to the per-sequence `nbytes()` walk above. Add
        // them explicitly so admission-control sees the true KV footprint
        // for paged Turbo4 deployments (#482).
        let pool_sidecar_bytes: usize = self
            .paged_pool
            .as_ref()
            .map(|pool| pool.turbo_sidecar_bytes())
            .unwrap_or(0);
        active_bytes + parked_bytes + pool_sidecar_bytes
    }

    pub fn append_paged_tokens(
        &mut self,
        id: SequenceId,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<(), String> {
        let pool = self
            .paged_pool
            .as_mut()
            .ok_or_else(|| "CachePool: paged backend is not initialized".to_string())?;
        let state = self
            .active
            .get_mut(&id)
            .ok_or_else(|| format!("CachePool: sequence {id} not found"))?
            .paged_state_mut()
            .ok_or_else(|| format!("CachePool: sequence {id} is not paged"))?;
        pool.append_tokens(state, layer_idx, token_count)
    }

    pub fn trim_paged_tokens(
        &mut self,
        id: SequenceId,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<usize, String> {
        let pool = self
            .paged_pool
            .as_mut()
            .ok_or_else(|| "CachePool: paged backend is not initialized".to_string())?;
        let state = self
            .active
            .get_mut(&id)
            .ok_or_else(|| format!("CachePool: sequence {id} not found"))?
            .paged_state_mut()
            .ok_or_else(|| format!("CachePool: sequence {id} is not paged"))?;
        pool.trim_tokens(state, layer_idx, token_count)
    }

    pub fn rewind_paged_tokens(
        &mut self,
        id: SequenceId,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<usize, String> {
        let pool = self
            .paged_pool
            .as_mut()
            .ok_or_else(|| "CachePool: paged backend is not initialized".to_string())?;
        let state = self
            .active
            .get_mut(&id)
            .ok_or_else(|| format!("CachePool: sequence {id} not found"))?
            .paged_state_mut()
            .ok_or_else(|| format!("CachePool: sequence {id} is not paged"))?;
        pool.rewind_tokens(state, layer_idx, token_count)
    }

    pub fn paged_stats(&self) -> Option<PagedCacheStats> {
        let pool = self.paged_pool.as_ref()?;
        Some(
            pool.stats_for_sequences(
                self.active
                    .values()
                    .filter_map(|sequence| sequence.paged_state()),
            ),
        )
    }

    pub fn paged_block_size(&self) -> Option<usize> {
        self.paged_pool
            .as_ref()
            .map(|pool| pool.layout().block_size)
    }

    /// Read-only access to the underlying [`PagedBlockPool`].
    ///
    /// Used by: tests and read-only diagnostics that need to peek at per-page
    /// Turbo4 sidecar state without going through the higher-level
    /// `CachePool` API.
    pub fn paged_pool_ref(&self) -> Option<&PagedBlockPool> {
        self.paged_pool.as_ref()
    }

    /// Mutable access to the underlying [`PagedBlockPool`].
    ///
    /// Used by: scheduler / model code that installs Turbo4 sidecar pages
    /// directly on the pool, and by unit tests for the same purpose.
    pub fn paged_pool_mut(&mut self) -> Option<&mut PagedBlockPool> {
        self.paged_pool.as_mut()
    }

    /// Mirror the visible dense-cache offsets into the paged backend state for
    /// one sequence.
    ///
    /// This keeps server decode/pre-fill lifecycle bookkeeping aligned while
    /// the actual model execution still runs on dense compatibility caches.
    pub fn sync_paged_state_with_dense(&mut self, id: SequenceId) -> Result<(), String> {
        let sequence = self
            .active
            .get_mut(&id)
            .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
        let target_lens: Vec<usize> = sequence
            .caches
            .iter()
            .map(|cache| cache.seq_len().max(0) as usize)
            .collect();
        self.sync_paged_state_with_lengths(id, &target_lens)
    }

    /// Mirror explicit visible lengths into the paged backend state for one sequence.
    pub fn sync_paged_state_with_lengths(
        &mut self,
        id: SequenceId,
        target_lens: &[usize],
    ) -> Result<(), String> {
        let pool = match self.paged_pool.as_mut() {
            Some(pool) => pool,
            None => return Ok(()),
        };
        let sequence = self
            .active
            .get_mut(&id)
            .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
        let state = match sequence.paged_state_mut() {
            Some(state) => state,
            None => return Ok(()),
        };

        if target_lens.len() != state.layers.len() {
            return Err(format!(
                "CachePool: expected {} paged layer lengths for {id}, got {}",
                state.layers.len(),
                target_lens.len()
            ));
        }

        for (layer_idx, target_len) in target_lens.iter().copied().enumerate() {
            let current_len = state.layers[layer_idx].len;
            if target_len > current_len {
                pool.append_tokens(state, layer_idx, target_len - current_len)?;
            } else if target_len < current_len {
                pool.trim_tokens(state, layer_idx, current_len - target_len)?;
            }
        }
        Ok(())
    }

    /// Restore externally serialized paged state into an active sequence and
    /// register its blocks with the shared allocator.
    pub fn restore_paged_state(
        &mut self,
        id: SequenceId,
        restored: PagedSequenceState,
    ) -> Result<(), String> {
        let layout = {
            let sequence = self
                .active
                .get(&id)
                .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
            if sequence.backend != SequenceStateBackend::PagedKvCache {
                return Err(format!(
                    "CachePool: sequence {id} is not using the paged backend"
                ));
            }

            sequence
                .paged_layout
                .clone()
                .ok_or_else(|| format!("CachePool: sequence {id} is missing paged layout"))?
        };
        if restored.block_size != layout.block_size {
            return Err(format!(
                "CachePool: restored block size {} does not match layout block size {}",
                restored.block_size, layout.block_size
            ));
        }
        if restored.layers.len() != layout.num_layers {
            return Err(format!(
                "CachePool: restored layer count {} does not match layout layer count {}",
                restored.layers.len(),
                layout.num_layers
            ));
        }

        let mut existing = {
            let sequence = self
                .active
                .get_mut(&id)
                .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
            sequence.paged.take()
        };

        self.ensure_paged_pool(&layout)?;
        let pool = self
            .paged_pool
            .as_mut()
            .expect("paged pool must exist after ensure_paged_pool");
        if let Some(existing) = existing.as_mut() {
            pool.release_sequence(existing)?;
        }
        pool.restore_sequence(&restored)?;

        let sequence = self
            .active
            .get_mut(&id)
            .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
        sequence.paged = Some(restored);
        Ok(())
    }

    /// Maximum number of concurrent sequences this pool allows.
    pub fn max_sequences(&self) -> usize {
        self.max_sequences
    }

    fn ensure_paged_pool(&mut self, layout: &PagedKvLayout) -> Result<(), String> {
        if let Some(pool) = self.paged_pool.as_ref() {
            if pool.layout() != layout {
                return Err("CachePool: paged layout mismatch for active paged backend".to_string());
            }
            return Ok(());
        }
        self.paged_pool = Some(PagedBlockPool::new(layout.clone()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_trim_clears_storage_when_fully_rewound() {
        let mut cache = KVCache::new();
        let keys = ffi::from_slice_f32(&[1.0, 2.0], &[1, 1, 2, 1]);
        let values = ffi::from_slice_f32(&[3.0, 4.0], &[1, 1, 2, 1]);
        cache.update(keys, values);

        assert_eq!(cache.seq_len(), 2);
        assert!(!cache.is_empty());
        assert_eq!(cache.trim(5), 2);
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn rotating_cache_wraps_single_token_updates_to_window_size() {
        let mut cache = RotatingKVCache::new(2);
        let first = ffi::from_slice_f32(&[1.0], &[1, 1, 1, 1]);
        let second = ffi::from_slice_f32(&[2.0], &[1, 1, 1, 1]);
        let third = ffi::from_slice_f32(&[3.0], &[1, 1, 1, 1]);
        let values = |x| ffi::from_slice_f32(&[x], &[1, 1, 1, 1]);

        cache.update_and_fetch(first, values(1.0));
        cache.update_and_fetch(second, values(2.0));
        let (keys, _values) = cache.update_and_fetch(third, values(3.0));

        assert_eq!(cache.get_offset(), 3);
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(ffi::array_shape(&keys), vec![1, 1, 2, 1]);
    }

    #[test]
    fn chunked_cache_trim_front_advances_visible_window() {
        let mut cache = ChunkedKVCache::new(2);
        let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 1, 3, 1]);
        let values = ffi::from_slice_f32(&[4.0, 5.0, 6.0], &[1, 1, 3, 1]);
        cache.update_and_fetch(keys, values);
        cache.maybe_trim_front();

        assert_eq!(cache.get_offset(), 3);
        assert_eq!(cache.get_start_position(), 1);
        assert_eq!(
            ffi::array_shape(cache.keys.as_ref().unwrap()),
            vec![1, 1, 2, 1]
        );
    }

    // --- CachePool tests ---

    /// Minimal model stub for CachePool tests. Produces N empty KVCaches.
    struct StubModel {
        num_layers: usize,
    }

    impl crate::generate::LanguageModel for StubModel {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::zeros(&[1], 0)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            (0..self.num_layers).map(|_| KVCache::new()).collect()
        }

        fn num_layers(&self) -> usize {
            self.num_layers
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }
    }

    struct PagedModel {
        layout: PagedKvLayout,
    }

    impl crate::generate::LanguageModel for PagedModel {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::zeros(&[1], 0)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            Vec::new()
        }

        fn num_layers(&self) -> usize {
            self.layout.num_layers
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }

        fn sequence_state_layout(&self) -> SequenceStateLayout {
            SequenceStateLayout::paged_kv_cache(self.layout.clone())
        }
    }

    #[test]
    fn cache_pool_allocate_and_release() {
        let model = StubModel { num_layers: 4 };
        let mut pool = CachePool::new(8);

        let id1 = pool.allocate(&model).expect("should allocate");
        let id2 = pool.allocate(&model).expect("should allocate");

        assert_ne!(id1, id2);
        assert_eq!(pool.active_count(), 2);
        assert_eq!(
            pool.get_mut(id1).unwrap().backend,
            SequenceStateBackend::DenseKvCache
        );
        assert_eq!(
            pool.get_mut(id2).unwrap().backend,
            SequenceStateBackend::DenseKvCache
        );

        // Each sequence should have 4 layer caches
        assert_eq!(pool.get_caches_mut(id1).unwrap().len(), 4);
        assert_eq!(pool.get_caches_mut(id2).unwrap().len(), 4);

        pool.release(id1);
        assert_eq!(pool.active_count(), 1);
        assert!(pool.get_mut(id1).is_none());
        assert!(pool.get_mut(id2).is_some());

        pool.release(id2);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn cache_pool_refuses_allocation_when_full() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(2);

        pool.allocate(&model).expect("first");
        pool.allocate(&model).expect("second");

        let result = pool.allocate(&model);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("max capacity"));
    }

    #[test]
    fn cache_pool_release_reopens_slot() {
        let model = StubModel { num_layers: 1 };
        let mut pool = CachePool::new(1);

        let id = pool.allocate(&model).expect("first");
        assert!(pool.allocate(&model).is_err());

        pool.release(id);
        assert_eq!(pool.active_count(), 0);

        // Slot should be available again
        let id2 = pool.allocate(&model).expect("after release");
        assert_ne!(id, id2); // IDs are monotonic, never reused
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn cache_pool_independent_state() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(4);

        let id1 = pool.allocate(&model).unwrap();
        let id2 = pool.allocate(&model).unwrap();

        // Mutate caches for sequence 1 only
        {
            let caches = pool.get_caches_mut(id1).unwrap();
            let keys = ffi::from_slice_f32(&[1.0, 2.0], &[1, 1, 2, 1]);
            let values = ffi::from_slice_f32(&[3.0, 4.0], &[1, 1, 2, 1]);
            caches[0].update(keys, values);
        }

        // Sequence 2 caches should still be empty
        {
            let caches = pool.get_caches_mut(id2).unwrap();
            assert!(caches[0].is_empty());
            assert!(caches[1].is_empty());
        }

        // Sequence 1 cache should have data
        {
            let caches = pool.get_caches_mut(id1).unwrap();
            assert!(!caches[0].is_empty());
            assert_eq!(caches[0].seq_len(), 2);
            // Second layer still empty
            assert!(caches[1].is_empty());
        }
    }

    #[test]
    fn cache_pool_collects_batch_cache_slices() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(4);

        let id1 = pool.allocate(&model).unwrap();
        let id2 = pool.allocate(&model).unwrap();

        let mut batch = pool.get_batch_caches_mut(&[id1, id2]).unwrap();
        assert_eq!(batch.len(), 2);
        batch[0][0].offset = 3;
        batch[1][1].offset = 5;
        drop(batch);

        assert_eq!(pool.get_caches_mut(id1).unwrap()[0].offset, 3);
        assert_eq!(pool.get_caches_mut(id2).unwrap()[1].offset, 5);
    }

    #[test]
    fn cache_pool_memory_usage() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(4);

        // Empty pool
        assert_eq!(pool.memory_usage_bytes(), 0);

        let id1 = pool.allocate(&model).unwrap();

        // Freshly allocated caches have no data
        assert_eq!(pool.memory_usage_bytes(), 0);

        // Add some data to one cache
        {
            let caches = pool.get_caches_mut(id1).unwrap();
            let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
            let values = ffi::from_slice_f32(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
            caches[0].update(keys, values);
        }

        let mem_after = pool.memory_usage_bytes();
        assert!(mem_after > 0);

        // Release should bring memory tracking back to zero
        pool.release(id1);
        assert_eq!(pool.memory_usage_bytes(), 0);
    }

    #[test]
    fn cache_pool_sequence_metadata() {
        let model = StubModel { num_layers: 1 };
        let mut pool = CachePool::new(4);

        let id = pool.allocate(&model).unwrap();
        let entry = pool.get_mut(id).unwrap();

        assert_eq!(entry.seq_id, id);
        assert_eq!(entry.prompt_len, 0);
        assert_eq!(entry.current_offset, 0);

        // Simulate prefill state update
        entry.prompt_len = 42;
        entry.current_offset = 42;

        let entry = pool.get_mut(id).unwrap();
        assert_eq!(entry.prompt_len, 42);
        assert_eq!(entry.current_offset, 42);
    }

    #[test]
    fn cache_pool_release_nonexistent_is_noop() {
        let mut pool = CachePool::new(4);
        let fake_id = SequenceId(9999);
        pool.release(fake_id); // should not panic
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn sequence_id_display() {
        let id = SequenceId(42);
        assert_eq!(format!("{id}"), "seq-42");
        assert_eq!(id.as_u64(), 42);
    }

    #[test]
    fn cache_pool_rejects_non_batching_model() {
        struct NonBatchModel;

        impl crate::generate::LanguageModel for NonBatchModel {
            fn forward(
                &self,
                _input_ids: &MlxArray,
                _caches: &mut [KVCache],
                _mask: Option<&MlxArray>,
            ) -> UniquePtr<MlxArray> {
                ffi::zeros(&[1], 0)
            }

            fn make_caches(&self) -> Vec<KVCache> {
                vec![KVCache::new()]
            }

            fn num_layers(&self) -> usize {
                1
            }

            fn eos_token_ids(&self) -> Vec<i32> {
                vec![0]
            }

            fn supports_batching(&self) -> bool {
                false
            }
        }

        let model = NonBatchModel;
        let mut pool = CachePool::new(8);

        // Non-batching models use lightweight placeholders — multiple
        // allocations are allowed so requests can be queued while another
        // sequence is generating.
        let first = pool.allocate(&model);
        assert!(first.is_ok());
        let first = first.unwrap();
        assert_eq!(pool.active_count(), 1);
        assert_eq!(
            pool.get_mut(first).unwrap().backend,
            SequenceStateBackend::ModelOwned
        );

        let second = pool.allocate(&model);
        assert!(second.is_ok());
        let second = second.unwrap();
        assert_eq!(pool.active_count(), 2);

        // Release both
        pool.release(first);
        pool.release(second);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn paged_layout_validates_block_geometry() {
        assert!(PagedKvLayout::uniform(2, 0, 128).is_err());
        assert!(PagedKvLayout::uniform(2, 4, 130).is_err());
        assert!(PagedKvLayout::new(4, Vec::new()).is_err());
    }

    #[test]
    fn cache_pool_allocates_paged_sequence_state() {
        let layout = PagedKvLayout::uniform(2, 4, 128).unwrap();
        let model = PagedModel {
            layout: layout.clone(),
        };
        let mut pool = CachePool::new(4);

        let id = pool.allocate(&model).unwrap();
        let entry = pool.get_mut(id).unwrap();
        assert_eq!(entry.backend, SequenceStateBackend::PagedKvCache);
        assert!(entry.caches.is_empty());
        assert_eq!(entry.paged_state().unwrap().layers.len(), layout.num_layers);
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 0,
                live_blocks: 0,
                free_blocks: 0,
                bytes_reserved: 0,
                bytes_in_use: 0,
            }
        );
    }

    #[test]
    fn cache_pool_restores_paged_sequence_state_into_allocator() {
        let layout = PagedKvLayout::uniform(2, 4, 128).unwrap();
        let model = PagedModel {
            layout: layout.clone(),
        };
        let mut pool = CachePool::new(4);
        let id = pool.allocate(&model).unwrap();

        let restored = PagedSequenceState {
            block_size: layout.block_size,
            layers: vec![
                PagedLayerState {
                    block_ids: vec![PagedBlockId::from_raw(7), PagedBlockId::from_raw(8)],
                    len: 6,
                    logical_start: 0,
                },
                PagedLayerState {
                    block_ids: vec![PagedBlockId::from_raw(42)],
                    len: 3,
                    logical_start: 0,
                },
            ],
        };

        pool.restore_paged_state(id, restored).unwrap();

        assert_eq!(
            pool.paged_stats(),
            Some(PagedCacheStats {
                allocated_blocks: 3,
                live_blocks: 3,
                free_blocks: 0,
                bytes_reserved: 384,
                bytes_in_use: 288,
            })
        );

        pool.append_paged_tokens(id, 1, 2).unwrap();
        assert_eq!(
            pool.paged_stats(),
            Some(PagedCacheStats {
                allocated_blocks: 4,
                live_blocks: 4,
                free_blocks: 0,
                bytes_reserved: 512,
                bytes_in_use: 352,
            })
        );
    }

    #[test]
    fn cache_pool_paged_append_trim_release_and_reuse() {
        let layout = PagedKvLayout::uniform(2, 4, 128).unwrap();
        let model = PagedModel {
            layout: layout.clone(),
        };
        let mut pool = CachePool::new(4);

        let id1 = pool.allocate(&model).unwrap();
        pool.append_paged_tokens(id1, 0, 6).unwrap();

        let (first_block, second_block) = {
            let layer = pool.get_paged_state(id1).unwrap().layer(0).unwrap();
            assert_eq!(layer.len, 6);
            assert_eq!(layer.visible_len(), 6);
            assert_eq!(layer.reserved_blocks(), 2);
            (layer.block_ids[0], layer.block_ids[1])
        };
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 2,
                live_blocks: 2,
                free_blocks: 0,
                bytes_reserved: 256,
                bytes_in_use: 192,
            }
        );
        assert_eq!(pool.memory_usage_bytes(), 192);

        assert_eq!(pool.trim_paged_tokens(id1, 0, 1).unwrap(), 1);
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 2,
                live_blocks: 2,
                free_blocks: 0,
                bytes_reserved: 256,
                bytes_in_use: 160,
            }
        );

        assert_eq!(pool.rewind_paged_tokens(id1, 0, 2).unwrap(), 2);
        {
            let layer = pool.get_paged_state(id1).unwrap().layer(0).unwrap();
            assert_eq!(layer.len, 3);
            assert_eq!(layer.block_ids, vec![first_block]);
        }
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 2,
                live_blocks: 1,
                free_blocks: 1,
                bytes_reserved: 128,
                bytes_in_use: 96,
            }
        );

        pool.append_paged_tokens(id1, 1, 4).unwrap();
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 3,
                live_blocks: 2,
                free_blocks: 1,
                bytes_reserved: 256,
                bytes_in_use: 224,
            }
        );

        pool.release(id1);
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 3,
                live_blocks: 0,
                free_blocks: 3,
                bytes_reserved: 0,
                bytes_in_use: 0,
            }
        );

        let id2 = pool.allocate(&model).unwrap();
        pool.append_paged_tokens(id2, 0, 4).unwrap();
        let reused_block = pool
            .get_paged_state(id2)
            .unwrap()
            .layer(0)
            .unwrap()
            .block_ids[0];
        assert_eq!(reused_block, first_block);
        assert_ne!(reused_block, second_block);
    }

    #[test]
    fn cache_pool_can_override_dense_model_with_paged_sequence_state() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(4);
        let layout = SequenceStateLayout::paged_kv_cache(PagedKvLayout::uniform(2, 4, 4).unwrap());

        let id = pool.allocate_with_layout(&model, Some(layout)).unwrap();
        let entry = pool.get_mut(id).unwrap();

        assert_eq!(entry.backend, SequenceStateBackend::PagedKvCache);
        assert_eq!(entry.caches.len(), 2);
        assert!(entry.paged_state().is_some());
    }

    #[test]
    fn sync_paged_state_with_dense_cache_offsets_tracks_rewinds() {
        let model = StubModel { num_layers: 1 };
        let mut pool = CachePool::new(4);
        let layout = SequenceStateLayout::paged_kv_cache(PagedKvLayout::uniform(1, 4, 4).unwrap());
        let id = pool.allocate_with_layout(&model, Some(layout)).unwrap();

        {
            let caches = pool.get_caches_mut(id).unwrap();
            let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
            let values = ffi::from_slice_f32(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
            caches[0].update(keys, values);
        }
        pool.sync_paged_state_with_dense(id).unwrap();
        assert_eq!(pool.get_paged_state(id).unwrap().layer(0).unwrap().len, 4);

        {
            let caches = pool.get_caches_mut(id).unwrap();
            assert_eq!(caches[0].trim(2), 2);
        }
        pool.sync_paged_state_with_dense(id).unwrap();
        assert_eq!(pool.get_paged_state(id).unwrap().layer(0).unwrap().len, 2);
    }

    #[test]
    fn batched_decode_metadata_tracks_heterogeneous_lengths() {
        let mut cache_a = KVCache::new();
        cache_a.offset = 3;
        let mut cache_b = KVCache::new();
        cache_b.offset = 9;
        let caches = vec![&mut cache_a, &mut cache_b];

        let metadata =
            BatchedAttentionMetadata::from_kv_caches(&caches, &[1, 4], &[0, 32]).unwrap();

        assert_eq!(metadata.rope_offsets, vec![3, 9]);
        assert_eq!(metadata.query_lens, vec![1, 4]);
        assert_eq!(metadata.kv_lens, vec![4, 13]);
        assert_eq!(metadata.window_sizes, vec![0, 32]);
    }

    #[test]
    fn batched_attention_metadata_rejects_mismatched_lengths() {
        let mut cache = KVCache::new();
        let caches = vec![&mut cache];

        assert!(BatchedAttentionMetadata::from_kv_caches(&caches, &[1, 2], &[0]).is_err());
        assert!(BatchedAttentionMetadata::from_kv_caches(&caches, &[1], &[0, 1]).is_err());
    }

    #[test]
    fn paged_decode_metadata_builds_logical_block_tables() {
        let metadata = BatchedAttentionMetadata {
            rope_offsets: vec![0, 4],
            query_lens: vec![1, 1],
            kv_lens: vec![3, 5],
            window_sizes: vec![0, 0],
        };

        let paged = PagedDecodeMetadata::from_attention_metadata(&metadata, 2).unwrap();

        assert_eq!(paged.block_size, 2);
        assert_eq!(paged.kv_lens, vec![3, 5]);
        assert_eq!(paged.block_table_offsets, vec![0, 2, 5]);
        assert_eq!(paged.block_tables, vec![0, 1, 0, 1, 2]);
    }

    #[test]
    fn paged_decode_metadata_rejects_invalid_block_size() {
        let metadata = BatchedAttentionMetadata {
            rope_offsets: vec![0],
            query_lens: vec![1],
            kv_lens: vec![1],
            window_sizes: vec![0],
        };

        assert!(PagedDecodeMetadata::from_attention_metadata(&metadata, 0).is_err());
    }
}

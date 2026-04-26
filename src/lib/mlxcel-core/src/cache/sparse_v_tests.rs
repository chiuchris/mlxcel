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

//! MLX-using unit tests for the sparse-V dequant scaffold (issue #480).
//!
//! Pure-Rust env-var tests live inline in `cache/turbo/sparse_v.rs` so they
//! still run when the Metal/CUDA features are off. The tests in this file
//! all touch the MLX FFI and must run on a host with a working
//! `mlxcel-core` build.

use cxx::UniquePtr;

use super::turbo::{self, sparse_v, TurboQuantParams};
use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;

// ---------------------------------------------------------------------------
// Helpers (mirrors `turbo_tests.rs` style)
// ---------------------------------------------------------------------------

/// Deterministic LCG-driven f32 tensor on `[shape]`.
fn synth_tensor(shape: &[i32], seed: u32) -> UniquePtr<MlxArray> {
    let total: usize = shape.iter().map(|&d| d as usize).product();
    let mut state = if seed == 0 { 0xCAFE_BABE } else { seed };
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let x = (state >> 1) as f32 / (i32::MAX as f32);
        data.push(x);
    }
    ffi::from_slice_f32(&data, shape)
}

fn flatten_fp32(arr: &MlxArray) -> Vec<f32> {
    let a = ffi::astype(arr, dtype::FLOAT32);
    ffi::eval(&a);
    let bytes = ffi::array_to_raw_bytes(&a);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn rms_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "rms_diff: length mismatch");
    let s: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum();
    ((s / a.len() as f64).sqrt()) as f32
}

// ---------------------------------------------------------------------------
// compute_alive_mask
// ---------------------------------------------------------------------------

#[test]
fn alive_mask_shape_matches_kv_grid_non_grouped() {
    // Hq == Hkv: trivial pass-through aggregation.
    let attn = synth_tensor(&[2, 4, 3, 8], 1);
    let mask = sparse_v::compute_alive_mask(&attn, /*kv_heads=*/ 4, 0.5);
    let s = ffi::array_shape(&mask);
    assert_eq!(s, vec![2_i32, 4, 1, 8]);
}

#[test]
fn alive_mask_shape_matches_kv_grid_grouped() {
    // Hq=8, Hkv=2 → n_rep=4. Output should still be [B, Hkv=2, 1, Tk=8].
    let attn = synth_tensor(&[1, 8, 5, 8], 2);
    let mask = sparse_v::compute_alive_mask(&attn, /*kv_heads=*/ 2, 0.5);
    let s = ffi::array_shape(&mask);
    assert_eq!(s, vec![1_i32, 2, 1, 8]);
}

#[test]
fn alive_mask_threshold_zero_keeps_everything_alive() {
    // With threshold=0.0, every position whose attention weight is > 0
    // stays alive. Synth tensor is all positive (LCG returns [-1, 1] but
    // we pass abs values via softmax-like distribution below).
    //
    // Build a deterministic positive attention tensor: softmax(synth) is
    // strictly positive and sums to 1 along the last axis.
    let synth = synth_tensor(&[1, 1, 2, 4], 7);
    let attn = ffi::softmax_precise(&synth, -1);
    let mask = sparse_v::compute_alive_mask(&attn, 1, 0.0);
    let v = flatten_fp32(&mask);
    // All entries should be 1.0 (alive) since every softmax output is > 0.
    for (i, &x) in v.iter().enumerate() {
        assert!(
            (x - 1.0).abs() < 1e-6,
            "alive_mask[{i}] = {x}, expected 1.0 (everything alive at threshold=0)"
        );
    }
}

#[test]
fn alive_mask_threshold_one_kills_everything() {
    // With threshold=1.0, no softmax output exceeds 1.0 strictly (max is
    // exactly 1.0 only in the degenerate one-hot case; here the synth
    // tensor is non-degenerate so max is < 1).
    let synth = synth_tensor(&[1, 1, 2, 4], 11);
    let attn = ffi::softmax_precise(&synth, -1);
    let mask = sparse_v::compute_alive_mask(&attn, 1, 1.0);
    let v = flatten_fp32(&mask);
    for (i, &x) in v.iter().enumerate() {
        assert!(
            x.abs() < 1e-6,
            "alive_mask[{i}] = {x}, expected 0.0 (everything dead at threshold=1.0)"
        );
    }
}

#[test]
fn alive_mask_aggregates_q_axis_via_max() {
    // Construct a [1, 1, 2, 4] attention tensor with one Q row above the
    // threshold and one below. The output mask should keep all KV slots
    // alive that any Q position attends to above threshold.
    //
    // Layout: Q0 = [0.9, 0.05, 0.04, 0.01] (above 0.5 only at slot 0)
    //         Q1 = [0.02, 0.03, 0.85, 0.10] (above 0.5 only at slot 2)
    // After max over Q axis: [0.9, 0.05, 0.85, 0.10]
    // With threshold = 0.5: alive = [1, 0, 1, 0]
    let data: Vec<f32> = vec![0.9, 0.05, 0.04, 0.01, 0.02, 0.03, 0.85, 0.10];
    let attn = ffi::from_slice_f32(&data, &[1, 1, 2, 4]);
    let mask = sparse_v::compute_alive_mask(&attn, 1, 0.5);
    let v = flatten_fp32(&mask);
    assert_eq!(v.len(), 4);
    let expected = [1.0_f32, 0.0, 1.0, 0.0];
    for (i, (&got, &want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "alive_mask[{i}] = {got}, expected {want}"
        );
    }
}

#[test]
fn alive_mask_aggregates_grouped_heads_via_max() {
    // Hq=4, Hkv=2 → n_rep=2. Different Q-heads inside a group should OR
    // together (via max). Build:
    //   Hq=0 (group 0): Tq=1, Tk=2 = [0.8, 0.1]   alive=[1,0]
    //   Hq=1 (group 0): Tq=1, Tk=2 = [0.05, 0.9]  alive=[0,1]
    //   Hq=2 (group 1): Tq=1, Tk=2 = [0.05, 0.05] alive=[0,0]
    //   Hq=3 (group 1): Tq=1, Tk=2 = [0.7, 0.05]  alive=[1,0]
    // After grouped aggregation: Hkv=0 max-OR = [1,1], Hkv=1 = [1,0]
    let data: Vec<f32> = vec![
        // Hq=0
        0.8, 0.1, // Hq=1
        0.05, 0.9, // Hq=2
        0.05, 0.05, // Hq=3
        0.7, 0.05,
    ];
    let attn = ffi::from_slice_f32(&data, &[1, 4, 1, 2]);
    let mask = sparse_v::compute_alive_mask(&attn, /*kv_heads=*/ 2, 0.5);
    let s = ffi::array_shape(&mask);
    assert_eq!(s, vec![1_i32, 2, 1, 2]);
    let v = flatten_fp32(&mask);
    let expected = [1.0_f32, 1.0, 1.0, 0.0];
    for (i, (&got, &want)) in v.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "grouped alive_mask[{i}] = {got}, expected {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// attention_sparse_v_turbo4 — correctness equivalence at threshold=0
// ---------------------------------------------------------------------------

/// Helper: full-dequant reference attention path. Uses the same MLX
/// primitives as `attention_sparse_v_turbo4` but skips the alive-mask
/// branch entirely.
fn full_dequant_attention(
    q: &MlxArray,
    k: &MlxArray,
    v_packed: &MlxArray,
    v_norms: &MlxArray,
    params: &TurboQuantParams,
    scale: f32,
) -> UniquePtr<MlxArray> {
    let v_dq = turbo::quant::dequantize_v_turbo4(v_packed, v_norms, params);
    let q_shape = ffi::array_shape(q);
    let k_shape = ffi::array_shape(k);
    let b = q_shape[0];
    let hq = q_shape[1];
    let kv_heads = k_shape[1];
    let n_rep = hq / kv_heads;

    let (k_for_q, v_for_q) = if n_rep == 1 {
        (ffi::contiguous(k, false), ffi::contiguous(&v_dq, false))
    } else {
        let kt = k_shape[2];
        let kd = k_shape[3];
        let k_exp = ffi::expand_dims(k, 2);
        let k_tiled = ffi::broadcast_to(&k_exp, &[b, kv_heads, n_rep, kt, kd]);
        let k_full = ffi::reshape(&k_tiled, &[b, hq, kt, kd]);
        let vs = ffi::array_shape(&v_dq);
        let vt = vs[2];
        let vd = vs[3];
        let v_exp = ffi::expand_dims(&v_dq, 2);
        let v_tiled = ffi::broadcast_to(&v_exp, &[b, kv_heads, n_rep, vt, vd]);
        let v_full = ffi::reshape(&v_tiled, &[b, hq, vt, vd]);
        (k_full, v_full)
    };

    let k_t = ffi::transpose_axes(&k_for_q, &[0, 1, 3, 2]);
    let q_f32 = ffi::astype(q, dtype::FLOAT32);
    let k_t_f32 = ffi::astype(&k_t, dtype::FLOAT32);
    let v_f32 = ffi::astype(&v_for_q, dtype::FLOAT32);
    let qk = ffi::matmul(&q_f32, &k_t_f32);
    let scale_arr = ffi::full_f32(&[1], scale, dtype::FLOAT32);
    let scores = ffi::multiply(&qk, &scale_arr);
    let attn = ffi::softmax_precise(&scores, -1);
    let out = ffi::matmul(&attn, &v_f32);
    ffi::astype(&out, dtype::FLOAT16)
}

#[test]
fn sparse_v_attention_threshold_zero_matches_full_dequant() {
    // With threshold = 0.0 the alive mask short-circuits and the sparse-V
    // path collapses to the full-dequant reference within FP16 round-off.
    let head_dim: i32 = 64;
    let b: i32 = 1;
    let hq: i32 = 2;
    let hkv: i32 = 2;
    let tq: i32 = 4;
    let tk: i32 = 8;

    // Build Q, K (FP16), and packed V.
    let q_f32 = synth_tensor(&[b, hq, tq, head_dim], 100);
    let k_f32 = synth_tensor(&[b, hkv, tk, head_dim], 200);
    let v_f32 = synth_tensor(&[b, hkv, tk, head_dim], 300);
    let q = ffi::astype(&q_f32, dtype::FLOAT16);
    let k = ffi::astype(&k_f32, dtype::FLOAT16);
    let v_f16 = ffi::astype(&v_f32, dtype::FLOAT16);

    let params = TurboQuantParams::new(head_dim as u32, 0xC0FFEE);
    let (v_packed, v_norms) = turbo::quant::quantize_v_turbo4(&v_f16, &params);

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let sparse_out = sparse_v::attention_sparse_v_turbo4(
        &q, &k, &v_packed, &v_norms, &params, scale, None, /*threshold=*/ 0.0,
    );
    let full_out = full_dequant_attention(&q, &k, &v_packed, &v_norms, &params, scale);

    let s_vec = flatten_fp32(&sparse_out);
    let f_vec = flatten_fp32(&full_out);
    let rms = rms_diff(&s_vec, &f_vec);
    // FP16 has ~1024 representable values per power-of-two; the matmul →
    // softmax → matmul chain sees several FP16/FP32 round-trips. Anything
    // below 5e-3 RMS is well within FP16 noise.
    assert!(
        rms < 5e-3,
        "sparse-V (threshold=0) vs full-dequant RMS = {rms}; expected < 5e-3"
    );
}

#[test]
fn sparse_v_attention_threshold_one_zeros_output() {
    // With threshold = 1.0 every alive bit is 0 (no softmax output is > 1
    // for a non-degenerate distribution). The output should be ~zero.
    let head_dim: i32 = 64;
    let b: i32 = 1;
    let hq: i32 = 2;
    let hkv: i32 = 2;
    let tq: i32 = 4;
    let tk: i32 = 8;

    let q_f32 = synth_tensor(&[b, hq, tq, head_dim], 400);
    let k_f32 = synth_tensor(&[b, hkv, tk, head_dim], 500);
    let v_f32 = synth_tensor(&[b, hkv, tk, head_dim], 600);
    let q = ffi::astype(&q_f32, dtype::FLOAT16);
    let k = ffi::astype(&k_f32, dtype::FLOAT16);
    let v_f16 = ffi::astype(&v_f32, dtype::FLOAT16);

    let params = TurboQuantParams::new(head_dim as u32, 0xDECAFBAD);
    let (v_packed, v_norms) = turbo::quant::quantize_v_turbo4(&v_f16, &params);

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let out = sparse_v::attention_sparse_v_turbo4(
        &q, &k, &v_packed, &v_norms, &params, scale, None, /*threshold=*/ 1.0,
    );
    let v = flatten_fp32(&out);
    let max_abs = v.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
    assert!(
        max_abs < 1e-3,
        "sparse-V (threshold=1.0) max |out| = {max_abs}; expected ~0"
    );
}

#[test]
fn sparse_v_attention_with_grouped_heads() {
    // Hq=4, Hkv=2 (n_rep=2). At threshold=0 the grouped path should match
    // the full-dequant grouped reference within FP16 noise.
    let head_dim: i32 = 64;
    let b: i32 = 1;
    let hq: i32 = 4;
    let hkv: i32 = 2;
    let tq: i32 = 2;
    let tk: i32 = 8;

    let q = ffi::astype(&synth_tensor(&[b, hq, tq, head_dim], 700), dtype::FLOAT16);
    let k = ffi::astype(&synth_tensor(&[b, hkv, tk, head_dim], 800), dtype::FLOAT16);
    let v_f16 = ffi::astype(&synth_tensor(&[b, hkv, tk, head_dim], 900), dtype::FLOAT16);

    let params = TurboQuantParams::new(head_dim as u32, 0xBEEFFACE);
    let (v_packed, v_norms) = turbo::quant::quantize_v_turbo4(&v_f16, &params);

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let sparse_out = sparse_v::attention_sparse_v_turbo4(
        &q, &k, &v_packed, &v_norms, &params, scale, None, 0.0,
    );
    let full_out = full_dequant_attention(&q, &k, &v_packed, &v_norms, &params, scale);
    let rms = rms_diff(&flatten_fp32(&sparse_out), &flatten_fp32(&full_out));
    assert!(
        rms < 5e-3,
        "sparse-V grouped (threshold=0) vs full-dequant RMS = {rms}; expected < 5e-3"
    );
}

// ---------------------------------------------------------------------------
// KVCache::sparse_v_* accessors
// ---------------------------------------------------------------------------

#[test]
fn sparse_v_available_returns_false_for_fp16_mode() {
    // Fp16 cache must always report false regardless of env var. This is
    // the "no-op on non-Turbo4 modes" requirement from the issue.
    let _guard = SparseVThresholdGuard::set("1.0");
    let cache = super::KVCache::new_with_mode(super::KVCacheMode::Fp16);
    assert!(!cache.sparse_v_available());
    assert!(cache.v_packed().is_none());
}

#[test]
fn sparse_v_available_returns_false_for_int8_mode() {
    // Int8 cache also must report false. (No packed V in the Turbo sense.)
    let _guard = SparseVThresholdGuard::set("1e-6");
    let cache = super::KVCache::new_with_mode(super::KVCacheMode::Int8);
    assert!(!cache.sparse_v_available());
}

#[test]
fn sparse_v_attention_via_cache_matches_full_dequant_at_threshold_zero() {
    // End-to-end: build a Turbo4Asym cache, push K/V through, then call
    // `cache.sparse_v_attention` and compare with the full-dequant
    // reference. With threshold=0 the two paths must agree to FP16 noise.
    let _guard = SparseVThresholdGuard::set("0.0001");
    // Threshold 1e-4 is set in the env so the cache reports
    // `sparse_v_available == true` (it just needs > 0). The actual
    // numerical equivalence test below uses threshold = 0 directly.

    let head_dim: i32 = 64;
    let b: i32 = 1;
    let hq: i32 = 2;
    let hkv: i32 = 2;
    let tk: i32 = 4;

    let k_in = ffi::astype(&synth_tensor(&[b, hkv, tk, head_dim], 33), dtype::FLOAT16);
    let v_in = ffi::astype(&synth_tensor(&[b, hkv, tk, head_dim], 44), dtype::FLOAT16);

    let mut cache = super::KVCache::new_with_mode(super::KVCacheMode::Turbo4Asym);
    let (k_cached, _v_cached) = cache.update_and_fetch(k_in, v_in);

    assert!(cache.sparse_v_available(), "cache should report sparse_v_available");
    assert!(cache.v_packed().is_some(), "v_packed must be populated");
    assert!(cache.v_norms().is_some(), "v_norms must be populated");
    assert!(cache.turbo_params().is_some(), "turbo_params must be populated");

    let q = ffi::astype(&synth_tensor(&[b, hq, 1, head_dim], 55), dtype::FLOAT16);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Use the explicit-zero entry point so the test does not depend on the
    // env-var-cached threshold (which the OnceLock may have fixed earlier
    // in the test binary's life).
    let sparse_zero = sparse_v::attention_sparse_v_turbo4(
        &q,
        &k_cached,
        cache.v_packed().unwrap(),
        cache.v_norms().unwrap(),
        cache.turbo_params().unwrap(),
        scale,
        None,
        0.0,
    );
    let full = full_dequant_attention(
        &q,
        &k_cached,
        cache.v_packed().unwrap(),
        cache.v_norms().unwrap(),
        cache.turbo_params().unwrap(),
        scale,
    );
    let rms = rms_diff(&flatten_fp32(&sparse_zero), &flatten_fp32(&full));
    assert!(
        rms < 5e-3,
        "via-cache sparse-V (threshold=0) vs full-dequant RMS = {rms}; expected < 5e-3"
    );
}

#[test]
fn sparse_v_attention_returns_none_when_not_available() {
    // On a Fp16 cache, sparse_v_attention must return None so callers can
    // fall back to the standard attention path.
    let cache = super::KVCache::new_with_mode(super::KVCacheMode::Fp16);
    let q = ffi::astype(&synth_tensor(&[1, 1, 1, 64], 1), dtype::FLOAT16);
    let k = ffi::astype(&synth_tensor(&[1, 1, 1, 64], 2), dtype::FLOAT16);
    let result = cache.sparse_v_attention(&q, &k, 1.0, None);
    assert!(
        result.is_none(),
        "sparse_v_attention must return None for non-Turbo4 caches"
    );
}

/// Test-only RAII guard that mutates `MLXCEL_SPARSE_V_THRESHOLD` for the
/// lifetime of the guard. Note the global `OnceLock` cache in
/// `sparse_v::threshold()` means tests should not assume mid-test
/// reconfiguration takes effect; the guard is here to set the env *before*
/// any other test in this binary calls `threshold()`.
struct SparseVThresholdGuard {
    prev: Option<String>,
}

impl SparseVThresholdGuard {
    #[allow(dead_code)] // used only by the cache-mode unit tests below
    fn set(value: &str) -> Self {
        let prev = std::env::var(super::turbo::sparse_v::ENV_VAR).ok();
        // SAFETY: env mutation is single-threaded in cargo test by default
        // for the affected scope; we only use this in `#[test]` paths.
        unsafe { std::env::set_var(super::turbo::sparse_v::ENV_VAR, value) };
        Self { prev }
    }
}

impl Drop for SparseVThresholdGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => unsafe { std::env::set_var(super::turbo::sparse_v::ENV_VAR, v) },
            None => unsafe { std::env::remove_var(super::turbo::sparse_v::ENV_VAR) },
        }
    }
}

#[test]
fn sparse_v_attention_threshold_default_keeps_quality() {
    // At the default threshold (1e-6) the sparse-V output should be within
    // FP16 round-off of the full-dequant baseline. Any KV slot with
    // attention weight < 1e-6 contributes negligibly to the output, so
    // zeroing it changes nothing measurable.
    let head_dim: i32 = 64;
    let b: i32 = 1;
    let hq: i32 = 2;
    let hkv: i32 = 2;
    let tq: i32 = 4;
    let tk: i32 = 32; // longer K to guarantee the threshold kicks in

    let q = ffi::astype(&synth_tensor(&[b, hq, tq, head_dim], 1100), dtype::FLOAT16);
    let k = ffi::astype(&synth_tensor(&[b, hkv, tk, head_dim], 1200), dtype::FLOAT16);
    let v_f16 = ffi::astype(&synth_tensor(&[b, hkv, tk, head_dim], 1300), dtype::FLOAT16);

    let params = TurboQuantParams::new(head_dim as u32, 0xFEED_FACE);
    let (v_packed, v_norms) = turbo::quant::quantize_v_turbo4(&v_f16, &params);

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let sparse_out = sparse_v::attention_sparse_v_turbo4(
        &q,
        &k,
        &v_packed,
        &v_norms,
        &params,
        scale,
        None,
        sparse_v::DEFAULT_THRESHOLD,
    );
    let full_out = full_dequant_attention(&q, &k, &v_packed, &v_norms, &params, scale);
    let rms = rms_diff(&flatten_fp32(&sparse_out), &flatten_fp32(&full_out));
    // 1e-6 threshold should produce identical (within FP16 noise) output.
    assert!(
        rms < 5e-3,
        "sparse-V (threshold=1e-6) vs full-dequant RMS = {rms}; expected < 5e-3"
    );
}

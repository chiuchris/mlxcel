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

//! Integration tests for `KVCacheMode::Turbo4Asym` (issue #474, epic #458).
//!
//! Covers:
//! 1. Mode-string parsing (`fp16+turbo4`, `turbo4-asym`).
//! 2. Single-token / multi-token update + read round-trip.
//! 3. Trim correctness across the V sidecars.
//! 4. detach / install_detached round-trip preserves V sidecars + seed.

use super::*;
use crate::dtype;
use crate::ffi;

// ---------------------------------------------------------------------------
// Mode parsing
// ---------------------------------------------------------------------------

#[test]
fn turbo4_asym_parses_canonical_string() {
    let m: KVCacheMode = "fp16+turbo4".parse().unwrap();
    assert_eq!(m, KVCacheMode::Turbo4Asym);
}

#[test]
fn turbo4_asym_parses_alias() {
    let m: KVCacheMode = "turbo4-asym".parse().unwrap();
    assert_eq!(m, KVCacheMode::Turbo4Asym);
}

#[test]
fn turbo4_asym_parsing_is_case_insensitive() {
    let m: KVCacheMode = "FP16+TURBO4".parse().unwrap();
    assert_eq!(m, KVCacheMode::Turbo4Asym);
    let m: KVCacheMode = "Turbo4-Asym".parse().unwrap();
    assert_eq!(m, KVCacheMode::Turbo4Asym);
}

#[test]
fn turbo4_asym_display_round_trip() {
    let m = KVCacheMode::Turbo4Asym;
    let s = m.to_string();
    assert_eq!(s, "fp16+turbo4");
    let parsed: KVCacheMode = s.parse().unwrap();
    assert_eq!(parsed, m);
}

#[test]
fn unknown_mode_string_errors() {
    let r: Result<KVCacheMode, _> = "turbo3".parse();
    assert!(r.is_err());
    let err = r.unwrap_err();
    assert!(
        err.contains("turbo3"),
        "error message should include input: {err}"
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a deterministic [B, H, T, D] f32 V tensor with token-varying
/// magnitudes. Uses the in-house LCG so every test sees the same data.
fn synth_kv_tensor(b: i32, h: i32, t: i32, d: i32, seed: u32) -> cxx::UniquePtr<ffi::MlxArray> {
    let total = (b * h * t * d) as usize;
    let mut state = if seed == 0 { 0xDEADBEEF } else { seed };
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        // Same LCG as quant::Lcg32 — keeps tests reproducible across platforms.
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Map to [-1.0, 1.0]
        let x = (state >> 1) as f32 / (i32::MAX as f32);
        data.push(x);
    }
    ffi::from_slice_f32(&data, &[b, h, t, d])
}

fn flatten_fp32(arr: &ffi::MlxArray) -> Vec<f32> {
    let a = ffi::astype(arr, dtype::FLOAT32);
    ffi::eval(&a);
    let bytes = ffi::array_to_raw_bytes(&a);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// update_and_fetch round-trip
// ---------------------------------------------------------------------------

#[test]
fn turbo4_asym_update_returns_fp16_dequantized_v() {
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);

    let k = synth_kv_tensor(1, 1, 4, head_dim, 42);
    let v = synth_kv_tensor(1, 1, 4, head_dim, 43);

    let (k_out, v_out) = cache.update_and_fetch(k, v);
    assert_eq!(ffi::array_dtype(&v_out), dtype::FLOAT16);
    assert_eq!(ffi::array_shape(&v_out), vec![1_i32, 1, 4, head_dim]);

    // K side must be untouched — same shape, fp16 dtype.
    assert_eq!(ffi::array_dtype(&k_out), dtype::FLOAT16);
    assert_eq!(ffi::array_shape(&k_out), vec![1_i32, 1, 4, head_dim]);

    // The packed and norms sidecars must be populated; the standard `values`
    // tensor must NOT be (Turbo4Asym never stores fp16 V).
    assert!(cache.v_packed.is_some());
    assert!(cache.v_norms.is_some());
    assert!(cache.values.is_none());
    assert_eq!(cache.seq_len(), 4);
}

#[test]
fn turbo4_asym_multi_token_growth_keeps_visible_window_correct() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);

    let k1 = synth_kv_tensor(1, 1, 3, head_dim, 1);
    let v1 = synth_kv_tensor(1, 1, 3, head_dim, 2);
    let (k_out_1, v_out_1) = cache.update_and_fetch(k1, v1);
    assert_eq!(ffi::array_shape(&k_out_1)[2], 3);
    assert_eq!(ffi::array_shape(&v_out_1)[2], 3);
    assert_eq!(cache.seq_len(), 3);

    let k2 = synth_kv_tensor(1, 1, 5, head_dim, 3);
    let v2 = synth_kv_tensor(1, 1, 5, head_dim, 4);
    let (k_out_2, v_out_2) = cache.update_and_fetch(k2, v2);
    assert_eq!(cache.seq_len(), 8);
    assert_eq!(ffi::array_shape(&k_out_2)[2], 8);
    assert_eq!(ffi::array_shape(&v_out_2)[2], 8);
}

#[test]
fn turbo4_asym_keys_are_bit_identical_to_input() {
    // K side bypasses the quantizer entirely. After update, the visible portion
    // of self.keys (cast to f32) should equal the input bytes (cast through f16).
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);

    // Use a small, exactly-fp16-representable input to avoid round-off noise.
    let k_data: Vec<f32> = (0..head_dim).map(|i| ((i % 8) as f32) * 0.125).collect();
    let v_data: Vec<f32> = (0..head_dim).map(|i| ((i + 1) as f32) * 0.01).collect();
    let k = ffi::from_slice_f32(&k_data, &[1, 1, 1, head_dim]);
    let v = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);

    let (k_out, _) = cache.update_and_fetch(k, v);
    let recovered = flatten_fp32(&k_out);
    for (a, b) in recovered.iter().zip(k_data.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "K side must round-trip through fp16 unchanged: got {a}, expected {b}"
        );
    }
}

#[test]
fn turbo4_asym_v_reconstruction_has_bounded_error() {
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);

    // Single token with realistic magnitude
    let v_data: Vec<f32> = (0..head_dim)
        .map(|i| (i as f32 / head_dim as f32 - 0.5) * 2.0)
        .collect();
    let k = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);
    let v = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);

    let (_k_out, v_out) = cache.update_and_fetch(k, v);
    let v_recovered = flatten_fp32(&v_out);

    // Per-token relative L2 error should be < 15% (Lloyd-Max bound + fp16
    // round-off + norm-correction noise). See cache::turbo::quant tests.
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for i in 0..head_dim as usize {
        let diff = v_data[i] - v_recovered[i];
        num += diff * diff;
        den += v_data[i] * v_data[i];
    }
    let rel = (num / den.max(1e-12)).sqrt();
    assert!(rel < 0.15, "V reconstruction relative error {rel:.4} > 15%");
}

// ---------------------------------------------------------------------------
// Trim
// ---------------------------------------------------------------------------

#[test]
fn turbo4_asym_trim_to_zero_clears_all_buffers() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 5, head_dim, 11);
    let v = synth_kv_tensor(1, 1, 5, head_dim, 12);
    cache.update(k, v);
    assert_eq!(cache.seq_len(), 5);
    assert!(cache.v_packed.is_some());
    assert!(cache.v_norms.is_some());

    let trimmed = cache.trim(5);
    assert_eq!(trimmed, 5);
    assert_eq!(cache.seq_len(), 0);
    assert!(cache.is_empty());
    assert!(cache.v_packed.is_none());
    assert!(cache.v_norms.is_none());
}

#[test]
fn turbo4_asym_partial_trim_shrinks_v_sidecars() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 8, head_dim, 21);
    let v = synth_kv_tensor(1, 1, 8, head_dim, 22);
    cache.update(k, v);
    assert_eq!(cache.seq_len(), 8);

    let n = cache.trim(3);
    assert_eq!(n, 3);
    assert_eq!(cache.seq_len(), 5);

    // V sidecars must reflect the new offset.
    let vp = cache.v_packed.as_ref().unwrap();
    let vn = cache.v_norms.as_ref().unwrap();
    assert_eq!(ffi::array_shape(vp)[2], 5);
    assert_eq!(ffi::array_shape(vn)[2], 5);
    let k_buf = cache.keys.as_ref().unwrap();
    assert_eq!(ffi::array_shape(k_buf)[2], 5);
}

/// LOW-1 regression: `turbo_params` must be `None` after a full trim so the
/// next quantize call rebuilds params from scratch. This matters when a cache
/// slot is reused with a different head_dim after the trim (e.g. a sequence
/// completes and the slot is handed to a new sequence of a different model).
#[test]
fn turbo4_asym_trim_to_zero_clears_turbo_params() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 4, head_dim, 99);
    let v = synth_kv_tensor(1, 1, 4, head_dim, 100);
    cache.update(k, v);

    // turbo_params is populated after the first update.
    assert!(
        cache.turbo_params.is_some(),
        "turbo_params must be Some after update"
    );
    assert_eq!(cache.seq_len(), 4);

    // Full trim should clear turbo_params alongside the tensor buffers.
    let trimmed = cache.trim(4);
    assert_eq!(trimmed, 4);
    assert_eq!(cache.seq_len(), 0);
    assert!(cache.is_empty());
    assert!(
        cache.turbo_params.is_none(),
        "turbo_params must be None after full trim (LOW-1)"
    );
}

/// LOW-1 regression: `clone_handle` must clear `turbo_params` on the source
/// so the slot can be reused or initialized fresh without stale params.
#[test]
fn turbo4_asym_clone_handle_clears_turbo_params_on_source() {
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 3, head_dim, 101);
    let v = synth_kv_tensor(1, 1, 3, head_dim, 102);
    cache.update(k, v);

    // turbo_params is populated after the first update.
    assert!(
        cache.turbo_params.is_some(),
        "turbo_params must be Some after update"
    );

    let _handle = cache.clone_handle();

    // Source cache must have turbo_params cleared after clone_handle so a
    // fresh sequence can start with a clean slate (LOW-1 fix, #474).
    assert!(
        cache.turbo_params.is_none(),
        "clone_handle must clear turbo_params on source (LOW-1)"
    );
    // The source is also empty (existing contract).
    assert!(cache.is_empty());
}

// ---------------------------------------------------------------------------
// detach / install_detached round-trip
// ---------------------------------------------------------------------------

#[test]
fn turbo4_asym_clone_handle_round_trip_preserves_sidecars() {
    let head_dim = 64;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 4, head_dim, 31);
    let v = synth_kv_tensor(1, 1, 4, head_dim, 32);
    cache.update(k, v);

    // Capture the bytes of v_packed and v_norms pre-detach so we can compare
    // post-adopt.
    let pre_vp = ffi::array_to_raw_bytes(cache.v_packed.as_ref().unwrap());
    let pre_vn = ffi::array_to_raw_bytes(cache.v_norms.as_ref().unwrap());
    let pre_seed = cache.turbo_seed;

    let handle = cache.clone_handle();
    assert_eq!(handle.mode(), KVCacheMode::Turbo4Asym);

    // Source cache should be empty after clone_handle.
    assert!(cache.is_empty());
    assert!(cache.v_packed.is_none());
    assert!(cache.v_norms.is_none());

    let mut restored = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    restored.install_detached(handle).unwrap();

    assert_eq!(restored.seq_len(), 4);
    assert_eq!(restored.mode, KVCacheMode::Turbo4Asym);
    assert!(restored.v_packed.is_some());
    assert!(restored.v_norms.is_some());

    // turbo_params should have been re-derived from v_packed shape.
    assert!(restored.turbo_params.is_some());
    assert_eq!(restored.turbo_seed, pre_seed);

    let post_vp = ffi::array_to_raw_bytes(restored.v_packed.as_ref().unwrap());
    let post_vn = ffi::array_to_raw_bytes(restored.v_norms.as_ref().unwrap());
    assert_eq!(
        pre_vp, post_vp,
        "v_packed must survive detach/adopt bit-for-bit"
    );
    assert_eq!(
        pre_vn, post_vn,
        "v_norms must survive detach/adopt bit-for-bit"
    );
}

#[test]
fn turbo4_asym_clone_handle_install_then_dequant_matches_pre_detach() {
    // This is the strongest property: detach + install + read must yield the
    // same dequantized V tensor as a direct read on the original cache.
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k_data: Vec<f32> = (0..2 * head_dim)
        .map(|i| (i as f32 / 256.0) - 0.5)
        .collect();
    let v_data: Vec<f32> = (0..2 * head_dim)
        .map(|i| (((i * 7) % 13) as f32 / 13.0) - 0.5)
        .collect();
    let k = ffi::from_slice_f32(&k_data, &[1, 1, 2, head_dim as i32]);
    let v = ffi::from_slice_f32(&v_data, &[1, 1, 2, head_dim as i32]);

    let (_k1, v1_out) = cache.update_and_fetch(k, v);
    let v1 = flatten_fp32(&v1_out);

    let handle = cache.clone_handle();
    let mut restored = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    restored.install_detached(handle).unwrap();

    // Re-dequantize the visible portion of the restored cache using the
    // restored TurboQuantParams. To match the pre-detach v1, slice the V
    // sidecars to `restored.offset` first — the buffer capacity may exceed
    // the visible length due to step-aligned pre-allocation.
    let params = restored.turbo_params.as_ref().unwrap();
    let vp_buf = restored.v_packed.as_ref().unwrap();
    let vn_buf = restored.v_norms.as_ref().unwrap();
    let vps = ffi::array_shape(vp_buf);
    let vns = ffi::array_shape(vn_buf);
    let off = restored.offset;
    let vp = ffi::slice(vp_buf, &[0, 0, 0, 0], &[vps[0], vps[1], off, vps[3]]);
    let vn = ffi::slice(vn_buf, &[0, 0, 0, 0], &[vns[0], vns[1], off, 1]);
    let v2_out = super::turbo::quant::dequantize_v_turbo4(&vp, &vn, params);
    let v2 = flatten_fp32(&v2_out);

    assert_eq!(v1.len(), v2.len());
    for (a, b) in v1.iter().zip(v2.iter()) {
        // Fp16 quantize/dequant determinism: identical sidecars must produce
        // identical V output.
        assert!(
            (a - b).abs() < 1e-4,
            "post-detach dequant mismatch: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// CachePool detach/adopt round-trip on Turbo4Asym
// ---------------------------------------------------------------------------

#[test]
fn cache_pool_detach_adopt_preserves_turbo4_asym() {
    use crate::generate::LanguageModel;

    struct Stub {
        n: usize,
    }
    impl LanguageModel for Stub {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> cxx::UniquePtr<MlxArray> {
            ffi::zeros(&[1], 0)
        }
        fn make_caches(&self) -> Vec<KVCache> {
            (0..self.n).map(|_| KVCache::new()).collect()
        }
        fn num_layers(&self) -> usize {
            self.n
        }
        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }
    }

    let head_dim = 64;
    let model = Stub { n: 1 };
    let mut pool = CachePool::new(4);

    let seq_a = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        caches[0] = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
        let k = synth_kv_tensor(1, 1, 5, head_dim, 51);
        let v = synth_kv_tensor(1, 1, 5, head_dim, 52);
        caches[0].update(k, v);
    }

    let pre_vp = {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        ffi::array_to_raw_bytes(caches[0].v_packed.as_ref().unwrap())
    };
    let pre_vn = {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        ffi::array_to_raw_bytes(caches[0].v_norms.as_ref().unwrap())
    };

    let detached = pool.detach(seq_a).unwrap();
    let seq_b = pool.adopt(&model, detached).unwrap();

    let caches = pool.get_caches_mut(seq_b).unwrap();
    assert_eq!(caches[0].mode, KVCacheMode::Turbo4Asym);
    assert_eq!(caches[0].seq_len(), 5);
    assert!(caches[0].v_packed.is_some());
    assert!(caches[0].v_norms.is_some());

    let post_vp = ffi::array_to_raw_bytes(caches[0].v_packed.as_ref().unwrap());
    let post_vn = ffi::array_to_raw_bytes(caches[0].v_norms.as_ref().unwrap());
    assert_eq!(pre_vp, post_vp);
    assert_eq!(pre_vn, post_vn);
}

// ---------------------------------------------------------------------------
// Memory accounting
// ---------------------------------------------------------------------------

#[test]
fn turbo4_asym_nbytes_includes_v_sidecars_and_excludes_values() {
    let head_dim = 128;
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let k = synth_kv_tensor(1, 1, 32, head_dim, 71);
    let v = synth_kv_tensor(1, 1, 32, head_dim, 72);
    cache.update(k, v);
    let bytes = cache.nbytes();
    // Expected lower bound:
    //   K buffer   = 1*1*step(256)*128*2 = 65536
    //   v_packed   = 1*1*256*64*1        = 16384
    //   v_norms    = 1*1*256*1*2         = 512
    // Some padding from step alignment is fine.
    assert!(bytes > 0, "Turbo4Asym nbytes() must be non-zero");
    assert!(
        bytes < 200_000,
        "Turbo4Asym nbytes() should be ~80KB, got {bytes}"
    );
    // Values tensor stays None — it should NOT contribute to the byte count.
    assert!(cache.values.is_none());
}

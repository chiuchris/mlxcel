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

//! Proportional RoPE helper for Gemma 4 full-attention layers.
//!
//! The Gemma 4 reference (`mlx_vlm/models/gemma4/rope_utils.py`) defines a
//! `ProportionalRoPE` variant whose frequency exponents are normalized by the
//! FULL head dimension (`dims`) rather than the rotated-only slice
//! (`rotated_dims = int(dims * partial_rotary_factor // 2) * 2`). Only the
//! first `rotated_dims / 2` HF-style pairs actually rotate; the remaining
//! entries pass through unchanged.
//!
//! This module mirrors upstream's slice / concat / `mx.fast.rope` / re-splice
//! layout. The zero-padded `freqs` shortcut (fill the tail of `freqs` with
//! zeros and pass the full `head_dim` to `fast_rope_with_freqs`) is **NOT**
//! equivalent, because MLX's `fast::rope` computes `inv_freqs = 1 / freqs`
//! internally — a zero entry would therefore produce `inf` angles and `NaN`
//! output. We must operate on the truly-rotated slice.
//!
//! Used by: Gemma4

use crate::{
    array_shape, concatenate, copy, fast_rope_with_freqs, from_slice_f32, slice, MlxArray,
    UniquePtr,
};

/// Compute the frequency table for proportional RoPE.
///
/// Returns a length-`rotated_dims / 2` tensor where
/// `freqs[i] = factor * base^(2 * i / head_dim)`.
///
/// Note the denominator is the FULL `head_dim`, not `rotated_dims` — this is
/// the defining feature of "proportional" RoPE.
///
/// `rotated_dims = 2 * floor(partial_rotary_factor * head_dim / 2)`.
/// Returns `None` when `rotated_dims <= 0`, in which case
/// [`apply_proportional_rope`] treats the input as identity.
///
/// # Panics
///
/// Panics if `head_dim` is not a positive even integer, if `base <= 0`, or
/// if `factor <= 0`.
pub fn compute_proportional_rope_freqs(
    head_dim: i32,
    partial_rotary_factor: f32,
    base: f32,
    factor: f32,
) -> Option<UniquePtr<MlxArray>> {
    assert!(
        head_dim > 0 && head_dim % 2 == 0,
        "compute_proportional_rope_freqs: head_dim must be positive and even, got {head_dim}"
    );
    assert!(
        partial_rotary_factor.is_finite() && partial_rotary_factor >= 0.0,
        "compute_proportional_rope_freqs: partial_rotary_factor must be finite and \
         non-negative, got {partial_rotary_factor}"
    );
    assert!(
        base.is_finite() && base > 0.0,
        "compute_proportional_rope_freqs: base must be finite and positive, got {base}"
    );
    assert!(
        factor.is_finite() && factor > 0.0,
        "compute_proportional_rope_freqs: factor must be finite and positive, got {factor}"
    );

    let rope_angles = ((partial_rotary_factor as f64) * (head_dim as f64) / 2.0).floor() as i32;
    let rope_angles = rope_angles.clamp(0, head_dim / 2);
    if rope_angles <= 0 {
        return None;
    }

    let mut freqs = Vec::with_capacity(rope_angles as usize);
    for i in 0..rope_angles {
        // exponent = (2 * i) / head_dim — denominator is FULL head_dim,
        // matching upstream's `arange(0, rotated_dims, 2) / dims` expression.
        let exponent = (2 * i) as f32 / head_dim as f32;
        freqs.push(factor * base.powf(exponent));
    }
    Some(from_slice_f32(&freqs, &[rope_angles]))
}

/// Apply proportional RoPE to `x` along the last dimension.
///
/// The last dimension of `x` must equal `head_dim`. Only the first
/// `rotated_dims` entries are rotated — the remainder passes through
/// unchanged — mirroring the upstream Python reference
/// (`mlx_vlm.models.gemma4.rope_utils.ProportionalRoPE.__call__`).
///
/// `freqs` must have been produced by [`compute_proportional_rope_freqs`]
/// with the same `head_dim` and `partial_rotary_factor`. `None` means no
/// rotation (identity), matching the reference behaviour when
/// `rotated_dims <= 0`.
pub fn apply_proportional_rope(
    x: &MlxArray,
    head_dim: i32,
    partial_rotary_factor: f32,
    offset: i32,
    freqs: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let rope_angles = ((partial_rotary_factor as f64) * (head_dim as f64) / 2.0).floor() as i32;
    let rotated_dims = 2 * rope_angles.max(0);

    if rotated_dims == 0 || freqs.is_none() {
        return copy(x);
    }

    let freqs = freqs.expect("freqs must be Some when rotated_dims > 0");
    let shape = array_shape(x);
    let rank = shape.len() as i32;
    assert!(rank >= 1, "apply_proportional_rope: x must have rank >= 1");
    let last_axis = rank - 1;
    let half = head_dim / 2;
    let last_dim = shape[last_axis as usize];
    assert!(
        last_dim >= head_dim,
        "apply_proportional_rope: last dim ({last_dim}) must be >= head_dim ({head_dim})"
    );

    // head = x[..., :head_dim]; tail = x[..., head_dim:]
    let start_full = vec![0_i32; rank as usize];
    let mut stop_full = vec![i32::MAX; rank as usize];
    stop_full[last_axis as usize] = head_dim;
    let head = slice(x, &start_full, &stop_full);

    // left = head[..., :half]; right = head[..., half:]
    let mut stop_half = stop_full.clone();
    stop_half[last_axis as usize] = half;
    let left = slice(&head, &start_full, &stop_half);

    let mut start_half = start_full.clone();
    start_half[last_axis as usize] = half;
    let mut stop_head = stop_full.clone();
    stop_head[last_axis as usize] = head_dim;
    let right = slice(&head, &start_half, &stop_head);

    // rotated = concat([left[..., :rotated_dims/2], right[..., :rotated_dims/2]], -1)
    let rot_half = rotated_dims / 2;
    let mut stop_rot = stop_full.clone();
    stop_rot[last_axis as usize] = rot_half;
    let left_first = slice(&left, &start_full, &stop_rot);
    let right_first = slice(&right, &start_full, &stop_rot);
    let rotated = concatenate(&left_first, &right_first, last_axis);

    // Apply mx.fast.rope on the packed [..., rotated_dims] tensor.
    let rotated_out = fast_rope_with_freqs(&rotated, rotated_dims, false, 1.0, offset, freqs);

    // Split rotated_out back into its two halves.
    let mut start_rot = start_full.clone();
    let mut stop_rot_first = stop_full.clone();
    stop_rot_first[last_axis as usize] = rot_half;
    let rot_a = slice(&rotated_out, &start_rot, &stop_rot_first);
    start_rot[last_axis as usize] = rot_half;
    let mut stop_rot_full = stop_full.clone();
    stop_rot_full[last_axis as usize] = rotated_dims;
    let rot_b = slice(&rotated_out, &start_rot, &stop_rot_full);

    // Rebuild the two halves: replace the first rot_half entries in each.
    // left_new  = concat([rot_a, left [..., rot_half:half]], -1)
    // right_new = concat([rot_b, right[..., rot_half:half]], -1)
    let mut start_rest = start_full.clone();
    start_rest[last_axis as usize] = rot_half;
    let mut stop_rest = stop_full.clone();
    stop_rest[last_axis as usize] = half;
    let left_rest = slice(&left, &start_rest, &stop_rest);
    let right_rest = slice(&right, &start_rest, &stop_rest);
    let left_new = concatenate(&rot_a, &left_rest, last_axis);
    let right_new = concatenate(&rot_b, &right_rest, last_axis);

    let head_new = concatenate(&left_new, &right_new, last_axis);

    if last_dim == head_dim {
        return head_new;
    }

    // Preserve any elements past head_dim (e.g., deepseek-style padded
    // head_dims). In practice Gemma 4 never hits this branch.
    let mut start_tail = start_full.clone();
    start_tail[last_axis as usize] = head_dim;
    let stop_tail = stop_full.clone();
    let tail = slice(x, &start_tail, &stop_tail);
    concatenate(&head_new, &tail, last_axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{allclose, array_to_raw_bytes, astype, dtype, eval, item_bool};

    #[test]
    fn proportional_rope_freqs_match_python_formula() {
        // Gemma 4 full-attention layer: head_dim=256, partial_rotary_factor=0.25,
        // rope_theta=1_000_000 → rope_angles = 32.
        let head_dim = 256_i32;
        let prf = 0.25_f32;
        let base = 1_000_000.0_f32;
        let factor = 1.0_f32;
        let freqs = compute_proportional_rope_freqs(head_dim, prf, base, factor)
            .expect("proportional freqs must exist for prf=0.25");
        eval(&freqs);

        assert_eq!(
            array_shape(&freqs),
            vec![32],
            "freqs length must equal rope_angles (32 for head_dim=256, prf=0.25)"
        );

        let freqs_f32 = astype(&freqs, dtype::FLOAT32);
        eval(&freqs_f32);
        let bytes = array_to_raw_bytes(&freqs_f32);
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(values.len(), 32);

        for (i, &got) in values.iter().enumerate() {
            // Expected: base^(2*i / head_dim)
            let expected = base.powf((2 * i) as f32 / head_dim as f32);
            let rel = (got - expected).abs() / expected.max(1.0);
            assert!(
                rel < 1e-4,
                "freqs[{i}] expected {expected}, got {got} (rel {rel})"
            );
        }
    }

    #[test]
    fn proportional_rope_zero_partial_factor_is_identity() {
        // partial_rotary_factor = 0 ⇒ rotated_dims = 0 ⇒ no-op.
        let head_dim = 128_i32;
        let freqs = compute_proportional_rope_freqs(head_dim, 0.0, 10_000.0, 1.0);
        assert!(freqs.is_none(), "freqs must be None when rope_angles = 0");

        let x = crate::ones(&[1, 2, 3, head_dim], dtype::FLOAT32);
        let out = apply_proportional_rope(&x, head_dim, 0.0, 0, None);
        eval(&out);
        let close = allclose(&out, &x, 1e-6, 1e-6);
        eval(&close);
        assert!(item_bool(&close), "zero-rotation path must be identity");
    }

    #[test]
    fn proportional_rope_nontrivial_offset_produces_nonidentity_rotation() {
        // Smoke test: for a non-zero offset and a non-trivial input, the
        // rotated output must differ from the input (at least on the rotated
        // portion). This catches accidental no-op regressions.
        let head_dim = 64_i32;
        let prf = 0.5_f32;
        let base = 10_000.0_f32;
        let total = (2 * 3) * head_dim as usize;
        let data: Vec<f32> = (0..total).map(|i| (i as f32 * 0.03).cos()).collect();
        let x = from_slice_f32(&data, &[1, 2, 3, head_dim]);

        let freqs =
            compute_proportional_rope_freqs(head_dim, prf, base, 1.0).expect("freqs must exist");
        let rotated = apply_proportional_rope(&x, head_dim, prf, 5, freqs.as_ref());
        eval(&rotated);
        let close = allclose(&rotated, &x, 1e-5, 1e-5);
        eval(&close);
        assert!(
            !item_bool(&close),
            "non-zero offset must rotate the first rotated_dims slots"
        );
    }

    #[test]
    fn proportional_rope_preserves_non_rotated_tail() {
        // Values in the non-rotated tail of each half (indices
        // [rotated_dims/2, head_dim/2) of `left` and `right`) must pass
        // through unchanged.
        let head_dim = 64_i32;
        let prf = 0.5_f32; // rope_angles = 16, rotated_dims = 32, rot_half = 16.
        let base = 10_000.0_f32;
        let half = (head_dim / 2) as usize;
        let rot_half = 16_usize;

        let total = head_dim as usize;
        let data: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let x = from_slice_f32(&data, &[1, 1, 1, head_dim]);
        let freqs = compute_proportional_rope_freqs(head_dim, prf, base, 1.0).unwrap();
        let out = apply_proportional_rope(&x, head_dim, prf, 7, freqs.as_ref());
        eval(&out);

        let out_bytes = array_to_raw_bytes(&astype(&out, dtype::FLOAT32));
        let out_values: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // left tail: indices [rot_half, half) in the head — in flat layout
        // those map to positions [rot_half, half).
        for i in rot_half..half {
            let expected = data[i];
            let got = out_values[i];
            assert!(
                (got - expected).abs() < 1e-6,
                "left tail position {i}: expected {expected}, got {got}"
            );
        }
        // right tail: same sub-range but offset by `half` (start of right
        // half) → positions [half + rot_half, half + half) = [48, 64).
        for i in (half + rot_half)..(2 * half) {
            let expected = data[i];
            let got = out_values[i];
            assert!(
                (got - expected).abs() < 1e-6,
                "right tail position {i}: expected {expected}, got {got}"
            );
        }
    }
}

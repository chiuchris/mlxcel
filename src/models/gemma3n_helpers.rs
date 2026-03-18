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

//! Shared Gemma3n multimodal math helpers.
//!
//! These helpers are pure tensor-preparation utilities used by the Gemma3n
//! multimodal path. Keeping them outside `gemma3n.rs` localizes a hotspot that
//! changes independently from the decoder and layer definitions.

use mlxcel_core::{MlxArray, UniquePtr};

/// Stack arrays along a new axis.
pub(crate) fn stack_arrays(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    let ptrs: Vec<*const MlxArray> = arrays
        .iter()
        .map(|array| array.as_ref().unwrap() as *const _)
        .collect();
    mlxcel_core::stack(&ptrs, axis)
}

/// Compute magnitude (RMS) of an array along the last axis.
pub(crate) fn compute_magnitude(x: &MlxArray) -> UniquePtr<MlxArray> {
    let sq = mlxcel_core::square(x);
    let mean = mlxcel_core::mean_axis(&sq, -1, true);
    mlxcel_core::sqrt(&mean)
}

/// Normalize magnitudes of arrays starting from index 1.
pub(crate) fn normalize_magnitudes(
    arrays: &mut [UniquePtr<MlxArray>],
    target_magnitude: &MlxArray,
) {
    normalize_magnitudes_from_idx(arrays, 1, target_magnitude);
}

/// Normalize magnitudes of arrays starting from the specified index.
pub(crate) fn normalize_magnitudes_from_idx(
    arrays: &mut [UniquePtr<MlxArray>],
    start_idx: usize,
    target_magnitude: &MlxArray,
) {
    let eps = mlxcel_core::full_f32(&[1], 1e-6, mlxcel_core::array_dtype(target_magnitude));
    for item in arrays.iter_mut().skip(start_idx) {
        let mag = compute_magnitude(item);
        let mag_safe = mlxcel_core::maximum(&mag, &eps);
        let scale = mlxcel_core::divide(target_magnitude, &mag_safe);
        *item = mlxcel_core::multiply(item, &scale);
    }
}

/// Compute the mean of multiple arrays.
pub(crate) fn mean_arrays(arrays: &[UniquePtr<MlxArray>]) -> UniquePtr<MlxArray> {
    let stacked = stack_arrays(arrays, 0);
    mlxcel_core::mean_axis(&stacked, 0, false)
}

/// Apply softcap to logits: `cap * tanh(logits / cap)`.
pub(crate) fn apply_softcap(logits: &MlxArray, cap: f32) -> UniquePtr<MlxArray> {
    let cap_arr = mlxcel_core::full_f32(&[1], cap, mlxcel_core::array_dtype(logits));
    let scaled = mlxcel_core::divide(logits, &cap_arr);
    let tanh_out = mlxcel_core::tanh(&scaled);
    mlxcel_core::multiply(&tanh_out, &cap_arr)
}

/// Slice per-layer input for a specific layer.
pub(crate) fn slice_layer_input(
    per_layer_inputs: &MlxArray,
    layer_idx: i32,
    batch: i32,
    seq_len: i32,
    hidden_size: i32,
) -> UniquePtr<MlxArray> {
    let start = vec![0, 0, layer_idx, 0];
    let stop = vec![batch, seq_len, layer_idx + 1, hidden_size];
    let sliced = mlxcel_core::slice(per_layer_inputs, &start, &stop);
    mlxcel_core::squeeze_axis(&sliced, 2)
}

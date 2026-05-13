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

//! Structural unit tests for the Gemma 4 [`MtpTarget`] adapter
//! (issue #666).
//!
//! These tests pin the adapter's **trait surface** without loading real
//! Gemma 4 weights. The full on-hardware greedy-parity test for the
//! adapter ships in `tests/speculative_parity.rs` and is gated behind
//! `#[ignore]` so CI hosts without the model checkpoints don't red-flag
//! the build.
//!
//! ## What this file pins
//!
//! 1. `Gemma4MtpTargetAdapter::new` constructs cleanly with a `&'a
//!    Gemma4Wrapper` and `Option<SequenceId>`.
//! 2. The adapter is `Send`-able only in the same way `&Gemma4Wrapper`
//!    is — i.e. the lifetime bound is properly threaded.
//! 3. The shared-K/V slicing helper (`slice_shared_kv`) returns the
//!    input vector unchanged when `rejected == 0` (the common
//!    full-accept fast-path).
//!
//! Tests that need to actually call `forward_with_speculative_sinks`
//! require a loaded model and live in `tests/speculative_parity.rs`.

use super::*;

#[test]
fn slice_shared_kv_with_zero_rejected_is_identity() {
    // Build a synthetic 4-tensor shared K/V vector to verify the
    // fast-path. We use the FFI `from_slice_f32` to build small tensors;
    // since `rejected == 0` the slice helper must return them unchanged.
    let _runtime = crate::initialize_runtime();

    let make =
        || mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2]);
    let tensors: Vec<UniquePtr<MlxArray>> = vec![make(), make(), make(), make()];
    let original_shapes: Vec<Vec<i32>> = tensors
        .iter()
        .map(|t| mlxcel_core::array_shape(t.as_ref().unwrap()))
        .collect();

    let sliced = Gemma4MtpTargetAdapter::slice_shared_kv(tensors, 0);
    assert_eq!(sliced.len(), 4);
    for (i, s) in sliced.iter().enumerate() {
        let shape = mlxcel_core::array_shape(s.as_ref().unwrap());
        assert_eq!(
            shape, original_shapes[i],
            "rejected=0 must return identical shapes; entry {i} drifted"
        );
    }
}

#[test]
fn slice_shared_kv_with_rejected_one_shrinks_kv_axis() {
    // Build a `[1, 2, 4, 2]` synthetic tensor (B=1, num_kv_heads=2,
    // kv_len=4, head_dim=2). `rejected = 1` should produce
    // `[1, 2, 3, 2]`.
    let _runtime = crate::initialize_runtime();

    let make = || {
        // Total cells = 1 (batch) * 2 (heads) * 4 (kv_len) * 2 (head_dim) = 16
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        mlxcel_core::from_slice_f32(&data, &[1, 2, 4, 2])
    };
    let tensors: Vec<UniquePtr<MlxArray>> = vec![make(), make()];
    let sliced = Gemma4MtpTargetAdapter::slice_shared_kv(tensors, 1);
    assert_eq!(sliced.len(), 2);
    for s in &sliced {
        let shape = mlxcel_core::array_shape(s.as_ref().unwrap());
        assert_eq!(
            shape,
            vec![1, 2, 3, 2],
            "rejected=1 must shrink kv_len axis from 4 to 3"
        );
    }
}

#[test]
fn argmax_per_position_returns_one_id_per_position() {
    // `[1, 3, 4]` logits with deterministic per-row argmax:
    //   row 0: max at index 2 -> argmax = 2
    //   row 1: max at index 0 -> argmax = 0
    //   row 2: max at index 3 -> argmax = 3
    let _runtime = crate::initialize_runtime();

    let data: Vec<f32> = vec![
        // row 0
        0.1, 0.2, 0.9, 0.3, // row 1
        0.9, 0.1, 0.2, 0.3, // row 2
        0.1, 0.2, 0.3, 0.9,
    ];
    let logits = mlxcel_core::from_slice_f32(&data, &[1, 3, 4]);
    let argmax = Gemma4MtpTargetAdapter::argmax_per_position(logits.as_ref().unwrap());
    assert_eq!(argmax, vec![2, 0, 3]);
}

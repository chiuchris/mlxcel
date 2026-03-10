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

use super::{RMSNormGated, precise_swiglu_gate, restore_dtype};
use mlxcel_core::dtype;

fn assert_allclose(actual: &mlxcel_core::MlxArray, expected: &mlxcel_core::MlxArray) {
    let close = mlxcel_core::allclose(actual, expected, 1e-3, 1e-3);
    mlxcel_core::eval(&close);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
#[ignore = "requires serial MLX execution"]
fn precise_swiglu_gate_matches_float32_reference_and_restores_dtype() {
    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[1.5, -0.5, 0.25, -1.0], &[2, 2]),
        dtype::FLOAT16,
    );
    let gate = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[5.0, -3.0, 0.75, -0.25], &[2, 2]),
        dtype::FLOAT16,
    );

    let actual = precise_swiglu_gate(&x, &gate, dtype::FLOAT16);

    let gate_f32 = mlxcel_core::astype(&gate, dtype::FLOAT32);
    let gate_silu = mlxcel_core::silu(&gate_f32);
    let x_f32 = mlxcel_core::astype(&x, dtype::FLOAT32);
    let expected_f32 = mlxcel_core::multiply(&gate_silu, &x_f32);
    let expected = mlxcel_core::astype(&expected_f32, dtype::FLOAT16);

    assert_eq!(mlxcel_core::array_dtype(&actual), dtype::FLOAT16);
    assert_allclose(&actual, &expected);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn restore_dtype_casts_promoted_values_back_to_input_dtype() {
    let value = mlxcel_core::from_slice_f32(&[1.0, 2.0], &[1, 2]);
    let restored = restore_dtype(value, dtype::FLOAT16);

    assert_eq!(mlxcel_core::array_dtype(&restored), dtype::FLOAT16);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn rms_norm_gated_forward_restores_hidden_state_dtype_without_gate() {
    let weight = mlxcel_core::astype(&mlxcel_core::ones(&[2], dtype::FLOAT32), dtype::FLOAT16);
    let norm = RMSNormGated::new(weight, 1e-6);
    let hidden = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[1.0, -1.0], &[1, 2]),
        dtype::FLOAT16,
    );

    let output = norm.forward(&hidden, None);

    assert_eq!(mlxcel_core::array_dtype(&output), dtype::FLOAT16);
}

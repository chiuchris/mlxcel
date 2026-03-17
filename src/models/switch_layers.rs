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

//! Shared SwitchLinear / SwitchGLU for MoE models
//!
//! Used by: KimiLinear, LongcatFlashNgram, DeepSeekV3, DeepSeekV32, GLM4Moe,
//!          GLM4MoeLite, ExaOneMoe, Mixtral, Qwen3Moe, PhiMoE, OLMoE, etc.
//!
//! SwitchLinear: per-expert 3D matmul (quantized via gather_qmm, regular via gather_mm)
//! SwitchGLU: SwiGLU MLP routing through SwitchLinear
//! group_mask_scores: group-based expert masking for MoE gates with n_group > 1

use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Per-expert 3D linear layer (falls back to gather_mm for non-quantized models)
/// Supports affine, mxfp4, nvfp4, and mxfp8 quantization modes.
pub enum SwitchLinear {
    /// Quantized path: uses gather_qmm
    Quantized {
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: Option<UniquePtr<MlxArray>>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
    /// Non-quantized path: uses gather_mm
    Regular { weight: UniquePtr<MlxArray> },
}

impl SwitchLinear {
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray, sorted: bool) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => {
                let biases_ptr: *const MlxArray = match biases {
                    Some(b) => b.as_ref().unwrap() as *const MlxArray,
                    None => std::ptr::null(),
                };
                unsafe {
                    mlxcel_core::gather_qmm(
                        x,
                        weight,
                        scales,
                        biases_ptr,
                        std::ptr::null(),
                        indices as *const _,
                        true,
                        *group_size,
                        *bits,
                        sorted,
                        mode,
                    )
                }
            }
            Self::Regular { weight } => {
                // Python: gather_mm(x, weight.swapaxes(-1, -2), rhs_indices=indices)
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
                unsafe {
                    mlxcel_core::gather_mm(x, &wt, std::ptr::null(), indices as *const _, sorted)
                }
            }
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Self::from_weights_with_mode(weights, prefix, group_size, bits, "affine")
    }

    pub fn from_weights_with_mode(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<Self, String> {
        let weight = weights
            .get(&format!("{}.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing weight: {}", prefix))?;

        let scales_key = format!("{}.scales", prefix);
        if weights.contains_key(&scales_key) {
            // Quantized path
            let scales = weights
                .get(&scales_key)
                .map(|w| mlxcel_core::copy(w))
                .unwrap();
            // biases may not exist for mxfp4/nvfp4/mxfp8 modes
            let biases = weights
                .get(&format!("{}.biases", prefix))
                .map(|w| mlxcel_core::copy(w));

            Ok(Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode: mode.to_string(),
            })
        } else {
            // Non-quantized fallback
            Ok(Self::Regular { weight })
        }
    }
}

pub struct SwitchGLU {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGLU {
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let indices_shape = mlxcel_core::array_shape(indices);
        let n_tokens = indices_shape[0];
        let top_k = indices_shape[1];
        let total = n_tokens * top_k;
        let do_sort = total >= 64;

        let x_exp = mlxcel_core::expand_dims(x, -2);
        let x_exp = mlxcel_core::expand_dims(&x_exp, -3);

        if do_sort {
            let (sorted_x, sorted_idx, inv_order) = gather_sort(&x_exp, indices);
            let x_gate = self.gate_proj.forward(&sorted_x, &sorted_idx, true);
            let x_up = self.up_proj.forward(&sorted_x, &sorted_idx, true);
            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);
            let output = self.down_proj.forward(&activated, &sorted_idx, true);
            scatter_unsort(&output, &inv_order, &indices_shape)
        } else {
            let x_gate = self.gate_proj.forward(&x_exp, indices, false);
            let x_up = self.up_proj.forward(&x_exp, indices, false);
            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);
            let output = self.down_proj.forward(&activated, indices, false);
            mlxcel_core::squeeze_axis(&output, -2)
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                group_size,
                bits,
            )?,
            up_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                group_size,
                bits,
            )?,
            down_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                group_size,
                bits,
            )?,
        })
    }
}

/// Sort tokens by expert index for efficient gather_qmm/gather_mm
/// Used by: SwitchGLU, GptOss
pub fn gather_sort(
    x: &MlxArray,
    indices: &MlxArray,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let indices_shape = mlxcel_core::array_shape(indices);
    let top_k = indices_shape[indices_shape.len() - 1];
    let flat_indices = mlxcel_core::reshape(indices, &[-1]);
    let order = mlxcel_core::argsort(&flat_indices, -1);
    let inv_order = mlxcel_core::argsort(&order, -1);
    let x_shape = mlxcel_core::array_shape(x);
    let x_flat = mlxcel_core::reshape(x, &[x_shape[0], 1, x_shape[3]]);
    let top_k_arr = mlxcel_core::from_slice_i32(&[top_k], &[1]);
    let token_indices = mlxcel_core::divide(&order, &top_k_arr);
    let token_indices = mlxcel_core::astype(&token_indices, dtype::INT32);
    let sorted_x = mlxcel_core::take(&x_flat, &token_indices, 0);
    let sorted_indices = mlxcel_core::take(&flat_indices, &order, 0);
    (sorted_x, sorted_indices, inv_order)
}

/// Unsort tokens back to original order
fn scatter_unsort(x: &MlxArray, inv_order: &MlxArray, orig_shape: &[i32]) -> UniquePtr<MlxArray> {
    let unsorted = mlxcel_core::take(x, inv_order, 0);
    let x_shape = mlxcel_core::array_shape(&unsorted);
    let n_tokens = orig_shape[0];
    let top_k = orig_shape[1];
    let reshaped = mlxcel_core::reshape(&unsorted, &[n_tokens, top_k, x_shape[1], x_shape[2]]);
    mlxcel_core::squeeze_axis(&reshaped, 2)
}

/// Group-based expert masking for MoE gates with n_group > 1.
///
/// Selects the top `topk_group` expert groups (by sum of top-2 scores per group)
/// and zeros out scores for experts in non-selected groups.
///
/// Used by: DeepSeekV3, DeepSeekV32, GLM4Moe, GLM4MoeLite, ExaOneMoe
///
/// Reference: mlx-lm deepseek_v3.py group_expert_select()
pub fn group_mask_scores(scores: &MlxArray, n_group: i32, topk_group: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(scores);
    let n = shape[0];
    let n_experts = shape[1];
    let experts_per_group = n_experts / n_group;

    // Unflatten: [n, n_experts] -> [n, n_group, experts_per_group]
    let grouped = mlxcel_core::reshape(scores, &[n, n_group, experts_per_group]);

    // Compute group_scores = sum of top-2 expert scores per group
    let neg_grouped = mlxcel_core::negative(&grouped);
    let part_idx = mlxcel_core::argpartition(&neg_grouped, 1, -1); // kth=1 for top-2
    let top2_idx = slice_axis(&part_idx, -1, 0, 2); // [n, n_group, 2]
    let top2_vals = mlxcel_core::take_along_axis(&grouped, &top2_idx, -1); // [n, n_group, 2]
    let group_scores = mlxcel_core::sum_axis(&top2_vals, -1, true); // [n, n_group, 1]

    // Find bottom-k groups to zero out (k = n_group - topk_group)
    let k = n_group - topk_group;
    let group_idx = mlxcel_core::argpartition(&group_scores, k - 1, -2); // [n, n_group, 1]
    let group_idx = slice_axis(&group_idx, -2, 0, k); // [n, k, 1]

    // Zero out experts in non-selected groups
    let zero = mlxcel_core::full_f32(&[1], 0.0, dtype::FLOAT32);
    let grouped = mlxcel_core::put_along_axis(&grouped, &group_idx, &zero, -2);

    // Flatten back: [n, n_group, experts_per_group] -> [n, n_experts]
    mlxcel_core::reshape(&grouped, &[n, n_experts])
}

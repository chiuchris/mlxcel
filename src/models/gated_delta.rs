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

//! Shared Gated Delta Rule implementation for linear attention layers.
//!
//! This module provides the core gated delta net primitives used by models
//! that employ hybrid transformer + linear attention architectures.
//!
//! Used by: Qwen3Next, Qwen3.5, KimiLinear
//!
//! Reference: mlx-lm/mlx_lm/models/gated_delta.py

use mlxcel_core::utils::{silu, softplus, stack_arrays};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

// Cache.
/// Cache for GatedDeltaNet (linear attention) layers.
/// Stores conv1d state and recurrent SSM state.
///
/// Used by: Qwen3Next, KimiLinear
pub struct GatedDeltaCache {
    pub conv_state: Option<UniquePtr<MlxArray>>, // [batch, kernel-1, conv_dim]
    pub state_cache: Option<UniquePtr<MlxArray>>, // [batch, num_v_heads, head_v_dim, head_k_dim]
    pub offset: i32,
}

impl GatedDeltaCache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            state_cache: None,
            offset: 0,
        }
    }

    pub fn advance(&mut self, step: i32) {
        self.offset += step;
    }
}

impl Default for GatedDeltaCache {
    fn default() -> Self {
        Self::new()
    }
}

// Core Functions.
/// Compute gating values from A_log, a, and dt_bias.
/// g = exp(-exp(A_log) * softplus(a + dt_bias))
///
/// Used by: Qwen3Next, KimiLinear
pub fn compute_g(a_log: &MlxArray, a: &MlxArray, dt_bias: &MlxArray) -> UniquePtr<MlxArray> {
    let a_plus_dt = mlxcel_core::add(a, dt_bias);
    let sp = softplus(&a_plus_dt);
    let exp_a_log = mlxcel_core::exp(a_log);
    let neg_product = mlxcel_core::negative(&mlxcel_core::multiply(&exp_a_log, &sp));
    mlxcel_core::exp(&neg_product)
}

/// Single recurrent step of the gated delta rule.
///
/// Shapes:
///   - q, k: [B, H, Dk]
///   - v: [B, H, Dv]
///   - g: [B, H] or [B, H, Dk]
///   - beta: [B, H]
///   - state: [B, H, Dv, Dk]
///
/// Returns: (y: [B, H, Dv], new_state: [B, H, Dv, Dk])
///
/// Used by: Qwen3Next, KimiLinear
pub fn gated_delta_step(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    g: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
    mask: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let old_state = mlxcel_core::copy(state);

    // Decay: state = state * g
    let g_ndim = mlxcel_core::array_ndim(g);
    let decay = if g_ndim == 2 {
        // g: [B, H] -> [B, H, 1, 1]
        let g1 = mlxcel_core::expand_dims(g, -1);
        mlxcel_core::expand_dims(&g1, -1)
    } else if g_ndim == 3 {
        // g: [B, H, Dk] -> [B, H, 1, Dk]
        mlxcel_core::expand_dims(g, -2)
    } else {
        panic!("Unsupported gating shape");
    };

    let mut new_state = mlxcel_core::multiply(state, &decay);

    // kv_mem = (state * k[..., None, :]).sum(axis=-1) -> [B, H, Dv]
    let k_exp = mlxcel_core::expand_dims(k, -2);
    let kv_mem = mlxcel_core::sum_axis(&mlxcel_core::multiply(&new_state, &k_exp), -1, false);

    // delta = (v - kv_mem) * beta[..., None]
    let beta_exp = mlxcel_core::expand_dims(beta, -1);
    let delta = mlxcel_core::multiply(&mlxcel_core::subtract(v, &kv_mem), &beta_exp);

    // state = state + k[..., None, :] * delta[..., None]
    let delta_exp = mlxcel_core::expand_dims(&delta, -1);
    new_state = mlxcel_core::add(&new_state, &mlxcel_core::multiply(&k_exp, &delta_exp));

    // Output projection: y = (state * q[..., None, :]).sum(axis=-1)
    let q_exp = mlxcel_core::expand_dims(q, -2);
    let y = mlxcel_core::sum_axis(&mlxcel_core::multiply(&new_state, &q_exp), -1, false);

    // Apply mask if provided
    let new_state = if let Some(m) = mask {
        // m: [B] -> [B, 1, 1, 1]
        let m1 = mlxcel_core::expand_dims(m, 1);
        let m2 = mlxcel_core::expand_dims(&m1, 2);
        let m3 = mlxcel_core::expand_dims(&m2, 3);
        mlxcel_core::where_cond(&m3, &new_state, &old_state)
    } else {
        new_state
    };

    (y, new_state)
}

/// Ops-based implementation for prompt prefill (sequential loop).
///
/// Shapes:
///   - q, k: [B, T, Hk, Dk]
///   - v: [B, T, Hv, Dv]
///   - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
///   - beta: [B, T, Hv]
///   - state: [B, Hv, Dv, Dk]
///
/// Returns: (y: [B, T, Hv, Dv], state: [B, Hv, Dv, Dk])
///
/// Used by: Qwen3Next, KimiLinear
pub fn gated_delta_ops(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    g: &MlxArray,
    beta: &MlxArray,
    state: Option<&MlxArray>,
    mask: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let shape = mlxcel_core::array_shape(q);
    let b = shape[0];
    let t = shape[1] as usize;
    let hk = shape[2];
    let dk = shape[3];
    let v_shape = mlxcel_core::array_shape(v);
    let hv = v_shape[2];
    let dv = v_shape[3];

    // Initialize state if not provided
    let mut current_state = if let Some(s) = state {
        mlxcel_core::copy(s)
    } else {
        mlxcel_core::zeros(&[b, hv, dv, dk], dtype::FLOAT32)
    };

    // Compute repeat factor for GQA
    let repeat_factor = hv / hk;

    // Repeat q and k if needed
    let q = if repeat_factor > 1 {
        mlxcel_core::repeat(q, repeat_factor as i32, -2)
    } else {
        mlxcel_core::copy(q)
    };
    let k = if repeat_factor > 1 {
        mlxcel_core::repeat(k, repeat_factor as i32, -2)
    } else {
        mlxcel_core::copy(k)
    };

    // Process each timestep
    let mut ys = Vec::with_capacity(t);
    for t_idx in 0..t {
        // Extract timestep t using slicing
        let q_t = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(
                &q,
                &[0, t_idx as i32, 0, 0],
                &[b, (t_idx + 1) as i32, hv, dk],
            ),
            1,
        );
        let k_t = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(
                &k,
                &[0, t_idx as i32, 0, 0],
                &[b, (t_idx + 1) as i32, hv, dk],
            ),
            1,
        );
        let v_t = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(
                v,
                &[0, t_idx as i32, 0, 0],
                &[b, (t_idx + 1) as i32, hv, dv],
            ),
            1,
        );

        let g_ndim = mlxcel_core::array_ndim(g);
        let g_t = if g_ndim == 4 {
            mlxcel_core::squeeze_axis(
                &mlxcel_core::slice(
                    g,
                    &[0, t_idx as i32, 0, 0],
                    &[b, (t_idx + 1) as i32, hv, dk],
                ),
                1,
            )
        } else {
            mlxcel_core::squeeze_axis(
                &mlxcel_core::slice(g, &[0, t_idx as i32, 0], &[b, (t_idx + 1) as i32, hv]),
                1,
            )
        };

        let beta_t = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(beta, &[0, t_idx as i32, 0], &[b, (t_idx + 1) as i32, hv]),
            1,
        );

        let mask_t = mask.map(|m| {
            mlxcel_core::squeeze_axis(
                &mlxcel_core::slice(m, &[0, t_idx as i32], &[b, (t_idx + 1) as i32]),
                1,
            )
        });

        let (y, new_state) = gated_delta_step(
            &q_t,
            &k_t,
            &v_t,
            &g_t,
            &beta_t,
            &current_state,
            mask_t.as_deref(),
        );

        ys.push(y);
        current_state = new_state;
    }

    // Stack outputs: [B, T, Hv, Dv]
    let y = stack_arrays(&ys, 1);

    (y, current_state)
}

/// Main gated delta update function.
///
/// Computes beta = sigmoid(b) and g = exp(-exp(A_log) * softplus(a + dt_bias)),
/// then runs the ops-based implementation.
///
/// Used by: Qwen3Next, KimiLinear
pub fn gated_delta_update(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a: &MlxArray,
    b: &MlxArray,
    a_log: &MlxArray,
    dt_bias: &MlxArray,
    state: Option<&MlxArray>,
    mask: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Compute beta = sigmoid(b)
    let beta = mlxcel_core::sigmoid(b);

    // Compute gating g = exp(-exp(A_log) * softplus(a + dt_bias))
    let g = compute_g(a_log, a, dt_bias);

    // Run the ops-based implementation
    gated_delta_ops(q, k, v, &g, &beta, state, mask)
}

// RMSNorm with optional gating.
/// RMSNorm with optional SwiGLU gating (for GatedDeltaNet output).
/// Gating: silu(gate) * rms_norm(x)
///
/// Used by: Qwen3Next, Qwen3.5, KimiLinear
pub struct RMSNormGated {
    weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl RMSNormGated {
    pub fn new(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &MlxArray, gate: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let target_dtype = mlxcel_core::array_dtype(x);

        // RMS normalization
        let x_sq = mlxcel_core::square(x);
        let mean_sq = mlxcel_core::mean_axis(&x_sq, -1, true);
        let eps_arr = mlxcel_core::full_f32(&[1], self.eps, dtype::FLOAT32);
        let rms = mlxcel_core::sqrt(&mlxcel_core::add(&mean_sq, &eps_arr));
        let normed = mlxcel_core::divide(x, &rms);
        let scaled = mlxcel_core::multiply(&normed, &self.weight);

        // Apply SwiGLU gating: silu(gate) * x
        // Python mlx-lm promotes the gated path to float32 before restoring the
        // hidden-state dtype so Qwen3Next/Qwen3.5 keep the expected precision.
        if let Some(g) = gate {
            precise_swiglu_gate(&scaled, g, target_dtype)
        } else {
            restore_dtype(scaled, target_dtype)
        }
    }
}

fn restore_dtype(value: UniquePtr<MlxArray>, target_dtype: i32) -> UniquePtr<MlxArray> {
    if mlxcel_core::array_dtype(&value) == target_dtype {
        value
    } else {
        mlxcel_core::astype(&value, target_dtype)
    }
}

fn precise_swiglu_gate(x: &MlxArray, gate: &MlxArray, target_dtype: i32) -> UniquePtr<MlxArray> {
    let gate_f32 = mlxcel_core::astype(gate, dtype::FLOAT32);
    let gate_silu = silu(&gate_f32);
    let x_f32 = mlxcel_core::astype(x, dtype::FLOAT32);
    let product = mlxcel_core::multiply(&gate_silu, &x_f32);
    restore_dtype(product, target_dtype)
}

#[cfg(test)]
#[path = "gated_delta_tests.rs"]
mod tests;

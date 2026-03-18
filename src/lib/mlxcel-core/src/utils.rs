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

//! Common utility functions for mlxcel-core
//!
//! This module provides shared utility functions used across multiple models,
//! reducing code duplication and ensuring consistency.

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;

// Array Slicing Utilities.
/// Slice an array along a specified axis.
///
/// # Arguments
/// * `x` - Input array
/// * `axis` - Axis to slice along (supports negative indexing)
/// * `start` - Start index (supports negative indexing)
/// * `end` - End index. Use -1 to mean "to the end of axis" (Python slice semantics)
///
/// # Example
/// ```ignore
/// // Slice x[:, 0:10, :] along axis 1
/// let sliced = slice_axis(&x, 1, 0, 10);
///
/// // Slice x[:, 5:, :] along axis 1 (5 to end)
/// let sliced = slice_axis(&x, 1, 5, -1);
/// ```
pub fn slice_axis(x: &MlxArray, axis: i32, start: i32, end: i32) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(x);
    let ndim = shape.len();

    // Handle negative axis
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;

    let dim_size = shape[axis];

    // Handle end index:
    // - end = -1 means "to the end of axis" (Python slice semantics)
    // - other negative values are relative to end
    let end = if end == -1 {
        dim_size
    } else if end < 0 {
        dim_size + end
    } else {
        end.min(dim_size)
    };

    // Handle negative start
    let start = if start < 0 {
        (dim_size + start).max(0)
    } else {
        start.min(dim_size)
    };

    // Build starts and stops vectors
    let mut starts = vec![0i32; ndim];
    let mut stops: Vec<i32> = shape.clone();
    starts[axis] = start;
    stops[axis] = end;

    ffi::slice(x, &starts, &stops)
}

// Attention Mask Utilities.
/// Create a causal attention mask.
///
/// Creates a lower triangular mask of shape [size, size + offset] where:
/// - 1.0 indicates positions that can be attended to
/// - -inf indicates positions that should be masked
///
/// # Arguments
/// * `size` - Size of the query sequence
/// * `offset` - Offset for KV cache (number of previously cached tokens)
///
/// # Returns
/// Mask of shape [size, size + offset] with -inf in upper triangular region
pub fn create_causal_mask(size: i32, offset: i32) -> UniquePtr<MlxArray> {
    let total_len = size + offset;

    // Create lower triangular mask (1 = attend, 0 = mask)
    let ones = ffi::ones(&[size, total_len], dtype::FLOAT32);
    let mask = ffi::tril(&ones, offset);

    // Convert to attention mask format: where mask=1 -> 0, where mask=0 -> -inf
    // Use where_cond to avoid NaN from 0 * -inf
    let zeros = ffi::zeros(&[size, total_len], dtype::FLOAT32);
    let neg_inf = ffi::full_f32(&[size, total_len], f32::NEG_INFINITY, dtype::FLOAT32);
    let bool_mask = ffi::greater(&mask, &zeros); // mask > 0 gives bool mask

    ffi::where_cond(&bool_mask, &zeros, &neg_inf)
}

/// Create a causal attention mask with sliding window.
///
/// # Arguments
/// * `size` - Size of the query sequence
/// * `offset` - Offset for KV cache
/// * `window` - Sliding window size (None for full attention)
///
/// # Returns
/// Mask with sliding window constraint applied
pub fn create_causal_mask_with_window(
    size: i32,
    offset: i32,
    window: Option<i32>,
) -> UniquePtr<MlxArray> {
    let total_len = size + offset;

    // Create lower triangular mask (1 = attend, 0 = mask)
    let ones = ffi::ones(&[size, total_len], dtype::FLOAT32);
    let mut mask = ffi::tril(&ones, offset);

    // Apply sliding window if specified
    if let Some(w) = window {
        // Create upper bound mask (window positions before diagonal)
        let upper_mask = ffi::triu(&ones, offset - w + 1);
        mask = ffi::multiply(&mask, &upper_mask);
    }

    // Convert to attention mask format: where mask=1 -> 0, where mask=0 -> -inf
    // Use where_cond to avoid NaN from 0 * -inf
    let zeros = ffi::zeros(&[size, total_len], dtype::FLOAT32);
    let neg_inf = ffi::full_f32(&[size, total_len], f32::NEG_INFINITY, dtype::FLOAT32);
    let bool_mask = ffi::greater(&mask, &zeros); // mask > 0 gives bool mask

    ffi::where_cond(&bool_mask, &zeros, &neg_inf)
}

// KV Cache Utilities.
/// Repeat key/value tensors for grouped-query attention.
///
/// When n_kv_heads < n_heads, we need to repeat K and V to match Q dimensions.
///
/// # Arguments
/// * `x` - Input tensor of shape [batch, n_kv_heads, seq_len, head_dim]
/// * `n_rep` - Number of times to repeat (n_heads / n_kv_heads)
///
/// # Returns
/// Tensor of shape [batch, n_heads, seq_len, head_dim]
pub fn repeat_kv(x: &MlxArray, n_rep: i32) -> UniquePtr<MlxArray> {
    if n_rep == 1 {
        return ffi::copy(x);
    }

    let shape = ffi::array_shape(x);
    let batch = shape[0];
    let n_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    // Reshape to [batch, n_kv_heads, 1, seq_len, head_dim]
    let x_exp = ffi::reshape(x, &[batch, n_kv_heads, 1, seq_len, head_dim]);

    // Broadcast to [batch, n_kv_heads, n_rep, seq_len, head_dim]
    let x_broad = ffi::broadcast_to(&x_exp, &[batch, n_kv_heads, n_rep, seq_len, head_dim]);

    // Reshape to [batch, n_kv_heads * n_rep, seq_len, head_dim]
    ffi::reshape(&x_broad, &[batch, n_kv_heads * n_rep, seq_len, head_dim])
}

// Activation Functions.
/// SiLU (Swish) activation: x * sigmoid(x) — compiled kernel fusion
#[inline]
pub fn silu(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::compiled_silu(x)
}

/// GELU activation with sigmoid approximation: x * sigmoid(1.702 * x)
///
/// NOTE: This is NOT the same as exact GELU or tanh-approximate GELU.
/// For exact GELU, use `ffi::gelu()` (re-exported as `mlxcel_core::gelu`).
/// For tanh-approximate GELU, use `gelu_approx()`.
#[inline]
pub fn gelu_sigmoid(x: &MlxArray) -> UniquePtr<MlxArray> {
    let x_dtype = ffi::array_dtype(x);
    let coef = ffi::full_f32(&[1], 1.702, x_dtype);
    let scaled = ffi::multiply(&coef, x);
    let sigmoid_x = ffi::sigmoid(&scaled);
    ffi::multiply(x, &sigmoid_x)
}

/// ReLU squared activation: max(0, x)^2 — compiled kernel fusion
#[inline]
pub fn relu_squared(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::compiled_relu_squared(x)
}

/// Softplus activation: log(1 + exp(x))
#[inline]
pub fn softplus(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::softplus(x)
}

/// GELU approximate activation (tanh-based approximation)
/// This is the faster approximation used by many models like Phi
#[inline]
pub fn gelu_approx(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::gelu_approx(x)
}

/// GeGELU activation for Phi3Small
/// Splits input into gelu and linear parts (interleaved), applies gelu to first half,
/// then computes: gelu(x[::2]) * (x[1::2] + 1.0)
///
/// # Arguments
/// * `x` - Input array where last dim will be split into interleaved gelu/linear parts
/// * `limit` - Clipping limit for numerical stability
pub fn gegelu(x: &MlxArray, limit: f32) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(x);
    let ndim = shape.len();
    let last_dim = shape[ndim - 1];
    let half_dim = last_dim / 2;

    // Split into gelu part (even indices) and linear part (odd indices)
    // Reshape: [B, L, D] -> [B, L, D/2, 2]
    let mut new_shape = shape.clone();
    new_shape[ndim - 1] = half_dim;
    new_shape.push(2);

    let x_reshaped = ffi::reshape(x, &new_shape);

    // Select gelu_part (index 0) and linear_part (index 1) along last axis
    // Using slice: gelu_part = x_reshaped[..., :, 0], linear_part = x_reshaped[..., :, 1]
    let mut starts = vec![0i32; ndim + 1];
    let mut stops: Vec<i32> = new_shape.clone();

    // gelu_part: slice [..., :, 0:1] then squeeze
    starts[ndim] = 0;
    stops[ndim] = 1;
    let gelu_part = ffi::slice(&x_reshaped, &starts, &stops);
    let gelu_part = ffi::squeeze_axis(&gelu_part, ndim as i32);

    // linear_part: slice [..., :, 1:2] then squeeze
    starts[ndim] = 1;
    stops[ndim] = 2;
    let linear_part = ffi::slice(&x_reshaped, &starts, &stops);
    let linear_part = ffi::squeeze_axis(&linear_part, ndim as i32);

    // Clip both parts for numerical stability
    let neg_limit = ffi::full_f32(&[1], -limit, dtype::FLOAT32);
    let pos_limit = ffi::full_f32(&[1], limit, dtype::FLOAT32);

    let a_gelu = ffi::clip(&gelu_part, &neg_limit, &pos_limit);
    let a_linear = ffi::clip(&linear_part, &neg_limit, &pos_limit);

    // Apply GELU approximation: x * sigmoid(1.702 * x)
    let coef = ffi::full_f32(&[1], 1.702, dtype::FLOAT32);
    let scaled = ffi::multiply(&coef, &a_gelu);
    let sigmoid_x = ffi::sigmoid(&scaled);
    let out_gelu = ffi::multiply(&a_gelu, &sigmoid_x);

    // Compute: out_gelu * (a_linear + 1.0)
    let ones = ffi::full_f32(&[1], 1.0, dtype::FLOAT32);
    let linear_plus_one = ffi::add(&a_linear, &ones);
    ffi::multiply(&out_gelu, &linear_plus_one)
}

// Gemma-specific Functions.
/// Softcap function for Gemma 2/3 attention and logits.
///
/// Applies tanh(x / cap) * cap to prevent extreme values.
///
/// # Arguments
/// * `x` - Input array
/// * `cap` - Softcapping value
///
/// # Returns
/// Softcapped array
pub fn softcap(x: &MlxArray, cap: f32) -> UniquePtr<MlxArray> {
    let scaled = crate::divide_scalar(x, cap);
    let tanhed = ffi::tanh(&scaled);
    crate::multiply_scalar(&tanhed, cap)
}

/// Clip residual addition for float16 overflow prevention (Gemma 3).
///
/// When using float16, casts to float32, adds, clips to float16 range,
/// and casts back to float16. For other dtypes, performs normal addition.
///
/// # Arguments
/// * `x` - First input array
/// * `y` - Second input array (to be added to x)
///
/// # Returns
/// Clipped residual sum
pub fn clip_residual_f16(x: &MlxArray, y: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype_code = ffi::array_dtype(x);

    // Check if dtype is float16 (dtype code 9)
    if dtype_code != dtype::FLOAT16 {
        // Not float16, just add normally
        return ffi::add(x, y);
    }

    // float16 max is approximately 65504
    let bound = 65504.0f32;

    // Cast to f32
    let x_f32 = ffi::astype(x, dtype::FLOAT32);
    let y_f32 = ffi::astype(y, dtype::FLOAT32);

    // Add
    let sum = ffi::add(&x_f32, &y_f32);

    // Create bound arrays
    let min_bound = ffi::full_f32(&[1], -bound, dtype::FLOAT32);
    let max_bound = ffi::full_f32(&[1], bound, dtype::FLOAT32);

    // Clip
    let clipped = ffi::clip(&sum, &min_bound, &max_bound);

    // Cast back to f16
    ffi::astype(&clipped, dtype::FLOAT16)
}

// Shape Utilities.
/// Concatenate two arrays along the specified axis.
#[inline]
pub fn concatenate(a: &MlxArray, b: &MlxArray, axis: i32) -> UniquePtr<MlxArray> {
    crate::concatenate(a, b, axis)
}

/// Stack arrays along a new axis.
pub fn stack_arrays(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    let ptrs: Vec<*const MlxArray> = arrays
        .iter()
        .map(|a| a.as_ref().unwrap() as *const _)
        .collect();
    unsafe { ffi::stack(&ptrs, axis) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_axis_basic() {
        // Create a simple test array
        let x = ffi::ones(&[2, 10, 4], dtype::FLOAT32);

        // Slice middle portion
        let sliced = slice_axis(&x, 1, 2, 5);
        let shape = ffi::array_shape(&sliced);
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_slice_axis_end_minus_one() {
        let x = ffi::ones(&[2, 10, 4], dtype::FLOAT32);

        // Slice from index 5 to end using -1
        let sliced = slice_axis(&x, 1, 5, -1);
        let shape = ffi::array_shape(&sliced);
        assert_eq!(shape, vec![2, 5, 4]); // 10 - 5 = 5
    }

    #[test]
    fn test_repeat_kv() {
        let x = ffi::ones(&[1, 4, 10, 64], dtype::FLOAT32);

        // Repeat 2 times (4 heads -> 8 heads)
        let repeated = repeat_kv(&x, 2);
        let shape = ffi::array_shape(&repeated);
        assert_eq!(shape, vec![1, 8, 10, 64]);
    }

    #[test]
    fn test_repeat_kv_no_repeat() {
        let x = ffi::ones(&[1, 8, 10, 64], dtype::FLOAT32);

        // No repeat needed
        let repeated = repeat_kv(&x, 1);
        let shape = ffi::array_shape(&repeated);
        assert_eq!(shape, vec![1, 8, 10, 64]);
    }
}

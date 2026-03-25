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

//! Direct C++ bindings to MLX via cxx
//!
//! This crate provides direct bindings to MLX C++ API, bypassing the mlx-c wrapper
//! for improved performance.

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "mlx_cxx")]
mod ffi {
    // Opaque types - these are defined in C++ and we just hold pointers to them
    unsafe extern "C++" {
        include!("mlx_cxx_bridge.h");

        /// Opaque wrapper for mlx::core::array
        type MlxArray;

        /// Opaque wrapper for mlx::core::Stream
        type MlxStream;

        // Stream functions.
        /// Get the default stream
        fn default_stream() -> UniquePtr<MlxStream>;

        /// Create a new stream on the specified device
        fn new_stream_on_device(gpu: bool) -> UniquePtr<MlxStream>;

        /// Synchronize a stream
        fn synchronize_stream(stream: &MlxStream);

        // Array factory functions.
        /// Create array filled with zeros
        fn zeros(shape: &[i32], dtype: i32) -> UniquePtr<MlxArray>;

        /// Create array filled with zeros on specific stream
        fn zeros_stream(shape: &[i32], dtype: i32, stream: &MlxStream) -> UniquePtr<MlxArray>;

        /// Create array filled with ones
        fn ones(shape: &[i32], dtype: i32) -> UniquePtr<MlxArray>;

        /// Create array filled with ones on specific stream
        fn ones_stream(shape: &[i32], dtype: i32, stream: &MlxStream) -> UniquePtr<MlxArray>;

        /// Create array with specific f32 value
        fn full_f32(shape: &[i32], value: f32, dtype: i32) -> UniquePtr<MlxArray>;

        /// Create identity/eye matrix (n rows, m cols, k diagonal offset)
        fn eye(n: i32, m: i32, k: i32, dtype: i32) -> UniquePtr<MlxArray>;

        /// Create linearly spaced values
        fn linspace(start: f32, stop: f32, num: i32, dtype: i32) -> UniquePtr<MlxArray>;

        /// Create zeros with same shape as input
        fn zeros_like(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Create ones with same shape as input
        fn ones_like(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Create array filled with value, same shape as input
        fn full_like(a: &MlxArray, value: f32) -> UniquePtr<MlxArray>;

        /// Create array from f32 slice
        fn from_slice_f32(data: &[f32], shape: &[i32]) -> UniquePtr<MlxArray>;

        /// Create array from i32 slice
        fn from_slice_i32(data: &[i32], shape: &[i32]) -> UniquePtr<MlxArray>;

        /// Create array from u32 slice (for quantized weights)
        fn from_slice_u32(data: &[u32], shape: &[i32]) -> UniquePtr<MlxArray>;

        /// Create array from i64 slice
        fn from_slice_i64(data: &[i64], shape: &[i32]) -> UniquePtr<MlxArray>;

        /// Create array from raw bytes with specified dtype
        fn from_bytes(data: &[u8], shape: &[i32], dtype: i32) -> UniquePtr<MlxArray>;

        /// Create half-precision array from raw bytes
        fn from_bytes_f16(data: &[u8], shape: &[i32], bfloat16: bool) -> UniquePtr<MlxArray>;

        // Array property accessors.
        /// Get the shape of an array
        fn array_shape(arr: &MlxArray) -> Vec<i32>;

        /// Get the dtype of an array (as integer code)
        fn array_dtype(arr: &MlxArray) -> i32;

        /// Get the number of elements in an array
        fn array_size(arr: &MlxArray) -> usize;

        /// Get the number of dimensions of an array
        fn array_ndim(arr: &MlxArray) -> usize;

        /// Get the size of each element in bytes
        fn array_itemsize(arr: &MlxArray) -> usize;

        /// Get the total size in bytes
        fn array_nbytes(arr: &MlxArray) -> usize;

        // Scalar extraction.
        /// Extract f32 scalar value
        fn item_f32(arr: &MlxArray) -> f32;

        /// Extract i32 scalar value
        fn item_i32(arr: &MlxArray) -> i32;

        /// Extract i64 scalar value
        fn item_i64(arr: &MlxArray) -> i64;

        /// Extract bool scalar value
        fn item_bool(arr: &MlxArray) -> bool;

        /// Copy evaluated array data to a raw byte buffer.
        /// Used by: KV cache serialization for disaggregated inference
        fn array_to_raw_bytes(arr: &MlxArray) -> Vec<u8>;

        // Evaluation.
        /// Evaluate an array
        fn eval(arr: &MlxArray);

        /// Evaluate multiple arrays at once
        unsafe fn eval_all(arrays: &[*const MlxArray]);

        // Element-wise binary operations.
        /// Element-wise addition
        fn add(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise subtraction
        fn subtract(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise multiplication
        fn multiply(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise division
        fn divide(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise remainder (modulo)
        fn remainder(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise maximum
        fn maximum(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise minimum
        fn minimum(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        // Element-wise unary operations.
        /// Element-wise negation
        fn negative(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise absolute value
        fn abs(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise exponential
        fn exp(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise natural logarithm
        fn log(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise square root
        fn sqrt(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise reciprocal square root
        fn rsqrt(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise square
        fn square(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise sine
        fn sin(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise cosine
        fn cos(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic tangent
        fn tanh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise sigmoid
        fn sigmoid(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise floor
        fn floor(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise ceiling
        fn ceil(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise round
        fn round(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise sign
        fn sign(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise reciprocal (1/x)
        fn reciprocal(a: &MlxArray) -> UniquePtr<MlxArray>;

        // Trigonometric functions
        /// Element-wise tangent
        fn tan(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic sine
        fn sinh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic cosine
        fn cosh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise arc sine
        fn arcsin(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise arc cosine
        fn arccos(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise arc tangent
        fn arctan(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise arc tangent of a/b
        fn arctan2(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic arc sine
        fn arcsinh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic arc cosine
        fn arccosh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic arc tangent
        fn arctanh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Convert radians to degrees
        fn degrees(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Convert degrees to radians
        fn radians(a: &MlxArray) -> UniquePtr<MlxArray>;

        // Mathematical/Special functions
        /// Error function
        fn erf(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Inverse error function
        fn erfinv(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// exp(x) - 1, numerically stable for small x
        fn expm1(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Base-2 logarithm
        fn log2(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Base-10 logarithm
        fn log10(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// log(exp(a) + exp(b)), numerically stable
        fn logaddexp(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise power a^b
        fn power(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        // Checks
        /// Check if elements are NaN
        fn isnan(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Check if elements are infinite
        fn isinf(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Check if elements are finite
        fn isfinite(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Check if elements are negative infinity
        fn isneginf(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Check if elements are positive infinity
        fn isposinf(a: &MlxArray) -> UniquePtr<MlxArray>;

        // Reduction operations.
        /// Sum all elements
        fn sum_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Sum along axis
        fn sum_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Mean of all elements
        fn mean_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Mean along axis
        fn mean_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Max of all elements
        fn max_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Max along axis
        fn max_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Min of all elements
        fn min_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Min along axis
        fn min_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Product of all elements
        fn prod_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Product along axis
        fn prod_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Variance of all elements
        fn var_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Variance along axis (ddof = degrees of freedom correction)
        fn var_axis(a: &MlxArray, axis: i32, keepdims: bool, ddof: i32) -> UniquePtr<MlxArray>;

        /// Standard deviation of all elements
        fn std_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Standard deviation along axis
        fn std_axis(a: &MlxArray, axis: i32, keepdims: bool, ddof: i32) -> UniquePtr<MlxArray>;

        /// Logsumexp of all elements
        fn logsumexp_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Logsumexp along axis
        fn logsumexp_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// All elements are true
        fn all_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Any element is true
        fn any_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        // Matrix operations.
        /// Matrix multiplication
        fn matmul(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Transpose (swap last two dimensions)
        fn transpose(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Transpose with specified axes
        fn transpose_axes(a: &MlxArray, axes: &[i32]) -> UniquePtr<MlxArray>;

        /// Reshape array
        fn reshape(a: &MlxArray, shape: &[i32]) -> UniquePtr<MlxArray>;

        // Shape operations.
        /// Expand dimensions at axis
        fn expand_dims(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Remove all singleton dimensions
        fn squeeze(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Remove singleton dimension at axis
        fn squeeze_axis(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Broadcast to shape
        fn broadcast_to(a: &MlxArray, shape: &[i32]) -> UniquePtr<MlxArray>;

        /// Flatten array to 1D
        fn flatten(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Flatten array between start_axis and end_axis
        fn flatten_range(a: &MlxArray, start_axis: i32, end_axis: i32) -> UniquePtr<MlxArray>;

        /// Move axis from source to destination
        fn moveaxis(a: &MlxArray, source: i32, destination: i32) -> UniquePtr<MlxArray>;

        /// Pad array (pad_width is pairs of [before, after] for each dim)
        fn pad(a: &MlxArray, pad_width: &[i32], pad_value: f32) -> UniquePtr<MlxArray>;

        /// Extract or construct diagonal
        fn diag(a: &MlxArray, k: i32) -> UniquePtr<MlxArray>;

        /// Extract diagonal from matrix
        fn diagonal(a: &MlxArray, offset: i32, axis1: i32, axis2: i32) -> UniquePtr<MlxArray>;

        // Type conversion.
        /// Convert array to specified dtype
        fn astype(a: &MlxArray, dtype: i32) -> UniquePtr<MlxArray>;

        // Copy.
        /// Copy array
        fn copy(a: &MlxArray) -> UniquePtr<MlxArray>;

        // High-level operations for LLM inference.
        /// Softmax along axis
        fn softmax(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Log-softmax along axis (numerically stable)
        fn log_softmax(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// RMS normalization
        fn rms_norm(x: &MlxArray, weight: &MlxArray, eps: f32) -> UniquePtr<MlxArray>;

        /// Layer normalization
        fn layer_norm(
            x: &MlxArray,
            weight: &MlxArray,
            bias: &MlxArray,
            eps: f32,
        ) -> UniquePtr<MlxArray>;

        /// Slice array with start, stop
        fn slice(a: &MlxArray, starts: &[i32], stops: &[i32]) -> UniquePtr<MlxArray>;

        /// Slice update: src[starts:stops] = update
        fn slice_update(
            src: &MlxArray,
            update: &MlxArray,
            starts: &[i32],
            stops: &[i32],
        ) -> UniquePtr<MlxArray>;

        /// Argmax along axis
        fn argmax(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Where (conditional select)
        fn where_cond(condition: &MlxArray, x: &MlxArray, y: &MlxArray) -> UniquePtr<MlxArray>;

        /// Greater than comparison
        fn greater(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Less than comparison
        fn less(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Equal comparison
        fn equal(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Seed the global MLX random number generator
        fn random_seed(seed: u64);

        /// Random categorical sampling
        fn random_categorical(logits: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        // Transformer-specific high-level operations.
        /// Apply rotary embedding to input
        fn apply_rope(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray>;

        /// RoPE forward (compute cos, sin for position embedding)
        fn rope_forward(
            x: &MlxArray,
            head_dim: i32,
            theta: f32,
            offset: i32,
            traditional: bool,
        ) -> UniquePtr<MlxArray>;

        /// Scaled dot-product attention
        unsafe fn scaled_dot_product_attention(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            scale: f32,
            mask: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Linear layer forward (with optional bias)
        unsafe fn linear_forward(
            x: &MlxArray,
            weight: &MlxArray,
            bias: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Quantized linear layer forward
        /// biases: nullable for mxfp4/nvfp4/mxfp8 modes
        unsafe fn quantized_linear_forward(
            x: &MlxArray,
            weight: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            linear_bias: *const MlxArray,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        /// SwiGLU MLP forward
        fn swiglu_mlp_forward(
            x: &MlxArray,
            gate_proj: &MlxArray,
            up_proj: &MlxArray,
            down_proj: &MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Compiled SwiGLU activation with kernel fusion
        /// Uses mlx::core::compile(shapeless=true) like Python's @mx.compile
        /// output = silu(gate) * x
        fn compiled_swiglu_activation(gate: &MlxArray, x: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compiled relu_squared: square(maximum(x, 0)) — single fused kernel
        fn compiled_relu_squared(x: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compiled silu: x * sigmoid(x) — single fused kernel
        fn compiled_silu(x: &MlxArray) -> UniquePtr<MlxArray>;

        /// Full transformer layer forward (maximum FFI reduction)
        unsafe fn transformer_layer_forward(
            x: &MlxArray,
            attn_norm_weight: &MlxArray,
            q_proj: &MlxArray,
            k_proj: &MlxArray,
            v_proj: &MlxArray,
            o_proj: &MlxArray,
            ffn_norm_weight: &MlxArray,
            gate_proj: &MlxArray,
            up_proj: &MlxArray,
            down_proj: &MlxArray,
            kv_cache_k: *const MlxArray,
            kv_cache_v: *const MlxArray,
            n_heads: i32,
            n_kv_heads: i32,
            head_dim: i32,
            rope_theta: f32,
            rope_offset: i32,
            norm_eps: f32,
        ) -> UniquePtr<MlxArray>;

        // Advanced indexing operations.
        /// Take elements along axis using indices
        fn take(a: &MlxArray, indices: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Gather elements using multi-dimensional indexing
        unsafe fn gather(
            a: &MlxArray,
            indices: &[*const MlxArray],
            axes: &[i32],
            slice_sizes: &[i32],
        ) -> UniquePtr<MlxArray>;

        /// Take along axis (like numpy.take_along_axis)
        fn take_along_axis(a: &MlxArray, indices: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Put along axis (scatter update)
        fn put_along_axis(
            a: &MlxArray,
            indices: &MlxArray,
            values: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        /// Tile/repeat array
        fn tile(a: &MlxArray, reps: &[i32]) -> UniquePtr<MlxArray>;

        /// Repeat array along axis
        fn repeat(a: &MlxArray, repeats: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// Stack arrays along new axis
        unsafe fn stack(arrays: &[*const MlxArray], axis: i32) -> UniquePtr<MlxArray>;

        /// Concatenate arrays along existing axis
        unsafe fn concatenate(arrays: &[*const MlxArray], axis: i32) -> UniquePtr<MlxArray>;

        /// Create range of f32 values
        fn arange_f32(start: f32, stop: f32, step: f32) -> UniquePtr<MlxArray>;

        /// Create range of i32 values
        fn arange_i32(start: i32, stop: i32, step: i32) -> UniquePtr<MlxArray>;

        // Logical operations.
        /// Element-wise logical NOT
        fn logical_not(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise logical AND
        fn logical_and(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise logical OR
        fn logical_or(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// All along axis
        fn all_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Any along axis
        fn any_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Greater than or equal comparison
        fn greater_equal(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Less than or equal comparison
        fn less_equal(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Not equal comparison
        fn not_equal(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        // Activation functions.
        /// SiLU (Swish) activation: x * sigmoid(x)
        fn silu(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// GELU activation
        fn gelu(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Approximate GELU (using tanh)
        fn gelu_approx(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// ReLU activation
        fn relu(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Leaky ReLU activation
        fn leaky_relu(a: &MlxArray, negative_slope: f32) -> UniquePtr<MlxArray>;

        // Sorting and searching.
        /// Argsort along axis
        fn argsort(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Argpartition along axis
        fn argpartition(a: &MlxArray, kth: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// Argmin along axis
        fn argmin(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Top-k elements along axis
        fn topk(a: &MlxArray, k: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// Sort along axis
        fn sort(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Partition along axis
        fn partition(a: &MlxArray, kth: i32, axis: i32) -> UniquePtr<MlxArray>;

        // Cumulative operations
        /// Cumulative max along axis
        fn cummax(a: &MlxArray, axis: i32, reverse: bool, inclusive: bool) -> UniquePtr<MlxArray>;

        /// Cumulative min along axis
        fn cummin(a: &MlxArray, axis: i32, reverse: bool, inclusive: bool) -> UniquePtr<MlxArray>;

        /// Cumulative product along axis
        fn cumprod(a: &MlxArray, axis: i32, reverse: bool, inclusive: bool) -> UniquePtr<MlxArray>;

        // Scatter operations
        /// Scatter updates into array at indices
        fn scatter(
            a: &MlxArray,
            indices: &MlxArray,
            updates: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        /// Scatter add updates into array at indices
        fn scatter_add(
            a: &MlxArray,
            indices: &MlxArray,
            updates: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        /// Scatter max updates into array at indices
        fn scatter_max(
            a: &MlxArray,
            indices: &MlxArray,
            updates: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        /// Scatter min updates into array at indices
        fn scatter_min(
            a: &MlxArray,
            indices: &MlxArray,
            updates: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        /// Scatter multiply updates into array at indices
        fn scatter_prod(
            a: &MlxArray,
            indices: &MlxArray,
            updates: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        // Bitwise operations
        /// Bitwise AND
        fn bitwise_and(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Bitwise OR
        fn bitwise_or(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Bitwise XOR
        fn bitwise_xor(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Left shift
        fn left_shift(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Right shift
        fn right_shift(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        // Linear algebra
        /// Tensor dot product
        fn tensordot(a: &MlxArray, b: &MlxArray, axes: i32) -> UniquePtr<MlxArray>;

        /// Inner product
        fn inner(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Outer product
        fn outer(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Matrix trace
        fn trace(a: &MlxArray, offset: i32, axis1: i32, axis2: i32) -> UniquePtr<MlxArray>;

        /// Roll (circular shift) along axis
        fn roll(a: &MlxArray, shift: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// Replace NaN/Inf with specified values
        fn nan_to_num(
            a: &MlxArray,
            nan_val: f32,
            posinf_val: f32,
            neginf_val: f32,
        ) -> UniquePtr<MlxArray>;

        /// Stop gradient (for autograd)
        fn stop_gradient(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// 2D convolution
        fn conv2d(
            input: &MlxArray,
            weight: &MlxArray,
            stride_h: i32,
            stride_w: i32,
            padding_h: i32,
            padding_w: i32,
            dilation_h: i32,
            dilation_w: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        /// 2D average pooling
        /// Used by: VisionModule (Gemma3 AvgPool projector)
        fn avg_pool2d(
            input: &MlxArray,
            kernel_h: i32,
            kernel_w: i32,
            stride_h: i32,
            stride_w: i32,
            padding_h: i32,
            padding_w: i32,
        ) -> UniquePtr<MlxArray>;

        // MoE (Mixture of Experts) operations.
        /// Gather matrix multiply for MoE
        /// sorted_indices: if true, lhs_indices are pre-sorted for better memory access
        unsafe fn gather_mm(
            a: &MlxArray,
            b: &MlxArray,
            lhs_indices: *const MlxArray,
            rhs_indices: *const MlxArray,
            sorted_indices: bool,
        ) -> UniquePtr<MlxArray>;

        /// Gather quantized matrix multiply for MoE
        /// sorted_indices: if true, lhs_indices are pre-sorted for better memory access
        unsafe fn gather_qmm(
            x: &MlxArray,
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            lhs_indices: *const MlxArray,
            rhs_indices: *const MlxArray,
            transpose: bool,
            group_size: i32,
            bits: i32,
            sorted_indices: bool,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        /// Direct quantized matrix multiplication
        /// y = x @ dequantize(w, scales, biases).T if transpose else x @ dequantize(w, scales, biases)
        unsafe fn quantized_matmul(
            x: &MlxArray,
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            transpose: bool,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        /// Dequantize quantized weights to full precision
        /// biases: nullable for mxfp4/nvfp4/mxfp8 modes
        unsafe fn dequantize(
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        // Embedding.
        /// Embedding lookup (same as take along axis 0)
        fn embedding(weight: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray>;

        /// Quantized embedding lookup with dequantization
        /// biases: nullable for mxfp4/nvfp4/mxfp8 modes
        unsafe fn quantized_embedding(
            weight: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            indices: &MlxArray,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        // Fast operations (using MLX fast kernels).
        /// Fast RoPE using MLX fast kernel
        fn fast_rope(
            x: &MlxArray,
            dims: i32,
            traditional: bool,
            base: f32,
            scale: f32,
            offset: i32,
        ) -> UniquePtr<MlxArray>;

        /// Fast RoPE with custom frequencies (for Yarn RoPE)
        fn fast_rope_with_freqs(
            x: &MlxArray,
            dims: i32,
            traditional: bool,
            scale: f32,
            offset: i32,
            freqs: &MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Fast RMS norm using MLX fast kernel
        fn fast_rms_norm(x: &MlxArray, weight: &MlxArray, eps: f32) -> UniquePtr<MlxArray>;

        /// Fast layer norm using MLX fast kernel
        unsafe fn fast_layer_norm(
            x: &MlxArray,
            weight: *const MlxArray,
            bias: *const MlxArray,
            eps: f32,
        ) -> UniquePtr<MlxArray>;

        /// Fast scaled dot product attention using MLX fast kernel
        unsafe fn fast_scaled_dot_product_attention(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            scale: f32,
            mask: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Fast SDPA with optional sinks (per-head attention bias for first position)
        /// Used by: GptOss
        #[allow(clippy::too_many_arguments)]
        unsafe fn fast_scaled_dot_product_attention_with_sinks(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            scale: f32,
            mask: *const MlxArray,
            sinks: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// SDPA with explicit causal masking for prefill
        fn fast_scaled_dot_product_attention_causal(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            scale: f32,
        ) -> UniquePtr<MlxArray>;

        /// Fused QKV projection + reshape + transpose + RoPE
        /// Reduces ~5 FFI calls (projection, reshape, transpose, rope) to 1
        #[allow(clippy::too_many_arguments)]
        unsafe fn fused_qkv_project_and_rope(
            x: &MlxArray,
            weight: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            num_heads: i32,
            head_dim: i32,
            rope_dims: i32,
            rope_base: f32,
            cache_offset: i32,
            group_size: i32,
            bits: i32,
            apply_rope: bool,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        // Compiled operations (with kernel fusion).
        /// Compiled MoE expert forward with quantized weights
        /// Falls back to non-compiled for non-affine modes (mxfp4/nvfp4/mxfp8)
        unsafe fn compiled_moe_expert_forward(
            x: &MlxArray,
            gate_proj: &MlxArray,
            gate_scales: &MlxArray,
            gate_biases: *const MlxArray,
            up_proj: &MlxArray,
            up_scales: &MlxArray,
            up_biases: *const MlxArray,
            down_proj: &MlxArray,
            down_scales: &MlxArray,
            down_biases: *const MlxArray,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        // Compiled MoE gate (sigmoid + topk + normalize + scale in one graph)
        /// Matches Python @mx.compile group_expert_select()
        fn compiled_moe_gate(
            gates: &MlxArray,
            correction_bias: &MlxArray,
            top_k: i32,
            scaling_factor: f32,
            norm_topk_prob: bool,
            indices_out: &mut UniquePtr<MlxArray>,
            scores_out: &mut UniquePtr<MlxArray>,
        );

        // Fused MoE forward: gate + switch_mlp + score weighting + shared expert
        /// Combines ~25 FFI calls into a single C++ function
        #[allow(clippy::too_many_arguments)]
        unsafe fn fused_moe_forward(
            x: &MlxArray,
            gate_weight: &MlxArray,
            correction_bias: &MlxArray,
            fc1_weight: &MlxArray,
            fc1_scales: &MlxArray,
            fc1_biases: &MlxArray,
            fc2_weight: &MlxArray,
            fc2_scales: &MlxArray,
            fc2_biases: &MlxArray,
            shared_up_weight: *const MlxArray,
            shared_up_scales: *const MlxArray,
            shared_up_biases: *const MlxArray,
            shared_down_weight: *const MlxArray,
            shared_down_scales: *const MlxArray,
            shared_down_biases: *const MlxArray,
            top_k: i32,
            scaling_factor: f32,
            norm_topk_prob: bool,
            group_size: i32,
            bits: i32,
        ) -> UniquePtr<MlxArray>;

        // SSM (Mamba2) fused Metal kernel.
        /// Check if SSM Metal kernel is available
        fn ssm_kernel_available() -> bool;

        /// Fused SSM update kernel for single-token decode
        /// Replaces ~55 individual ops with a single Metal kernel call
        /// Used by: NemotronH, NemotronNAS, Mamba2
        fn ssm_update_kernel(
            hidden_states: &MlxArray,
            a_log: &MlxArray,
            b: &MlxArray,
            c: &MlxArray,
            d: &MlxArray,
            dt: &MlxArray,
            dt_bias: &MlxArray,
            state_in: &MlxArray,
            time_step_min: f32,
            time_step_max: f32,
            output: &mut UniquePtr<MlxArray>,
            next_state: &mut UniquePtr<MlxArray>,
        );

        // Fused Mamba2 mixer forward for single-token decode.
        /// Combines in_proj + conv1d + SSM kernel + MambaRMSNormGated + out_proj into one C++ call.
        /// Replaces ~23 FFI round-trips for the hot decode path.
        /// Used by: NemotronH
        #[allow(clippy::too_many_arguments)]
        unsafe fn fused_mamba2_forward(
            hidden_states: &MlxArray,
            // in_proj (quantized)
            in_proj_weight: &MlxArray,
            in_proj_scales: &MlxArray,
            in_proj_biases: *const MlxArray,     // nullable
            // conv1d
            conv_weight: &MlxArray,
            conv_bias: *const MlxArray,           // nullable
            // SSM parameters
            a_log: &MlxArray,
            d: &MlxArray,
            dt_bias: &MlxArray,
            // norm weight
            norm_weight: &MlxArray,
            // out_proj (quantized)
            out_proj_weight: &MlxArray,
            out_proj_scales: &MlxArray,
            out_proj_biases: *const MlxArray,    // nullable
            // cache state inputs
            conv_state_in: &MlxArray,
            ssm_state_in: &MlxArray,
            // mamba2 config
            intermediate_size: i32,
            conv_dim: i32,
            conv_kernel_size: i32,
            num_heads: i32,
            head_dim: i32,
            n_groups: i32,
            ssm_state_size: i32,
            time_step_min: f32,
            time_step_max: f32,
            norm_eps: f32,
            // quantization config
            group_size: i32,
            bits: i32,
            // outputs
            output: &mut UniquePtr<MlxArray>,
            conv_state_out: &mut UniquePtr<MlxArray>,
            ssm_state_out: &mut UniquePtr<MlxArray>,
        );

        // NemotronH full-forward decode (opaque handle pattern).
        #[allow(clippy::too_many_arguments)]
        unsafe fn nemotron_register_model(
            embed_w: &MlxArray, embed_s: &MlxArray, embed_b: &MlxArray,
            final_norm_w: &MlxArray,
            lm_head_w: &MlxArray, lm_head_s: &MlxArray, lm_head_b: *const MlxArray,
            norm_weights: &[*const MlxArray],
            block_types: &[i32],
            mamba_weights: &[*const MlxArray],
            moe_weights: &[*const MlxArray],
            attn_weights: &[*const MlxArray],
            norm_eps: f32, group_size: i32, bits: i32,
            m_inter: i32, m_cdim: i32, m_ck: i32,
            m_heads: i32, m_hdim: i32, m_groups: i32, m_state: i32,
            m_ts_min: f32, m_ts_max: f32, m_neps: f32,
            moe_tk: i32, moe_sc: f32, moe_norm: bool,
            a_heads: i32, a_kvh: i32, a_hdim: i32,
            a_rope: f32, a_scale: f32,
        ) -> u64;

        fn nemotron_free_model(handle: u64);

        #[allow(clippy::too_many_arguments)]
        unsafe fn nemotron_decode_step(
            handle: u64,
            input_ids: &MlxArray,
            mamba_conv_in: &[*const MlxArray],
            mamba_ssm_in: &[*const MlxArray],
            attn_kv_keys: &[*const MlxArray],
            attn_kv_values: &[*const MlxArray],
            attn_kv_offsets: &[i32],
            logits: &mut UniquePtr<MlxArray>,
            mamba_conv_out: &mut [UniquePtr<MlxArray>],
            mamba_ssm_out: &mut [UniquePtr<MlxArray>],
        );

        // Memory management.
        /// Clear memory cache
        fn clear_memory_cache();

        /// Async eval single array
        fn async_eval(arr: &MlxArray);

        /// Async eval multiple arrays at once
        unsafe fn async_eval_all(arrays: &[*const MlxArray]);

        /// Synchronize default stream
        fn synchronize_default();

        /// Set default device for subsequent operations
        fn set_default_device(gpu: bool);

        /// Set wired memory limit, returns previous limit
        fn set_wired_limit(limit: usize) -> usize;

        /// Get current wired memory limit
        fn get_wired_limit() -> usize;

        /// Get max GPU memory size (works across Metal and CUDA backends)
        fn gpu_max_memory_size() -> usize;

        /// Create new GPU stream
        fn new_gpu_stream() -> UniquePtr<MlxStream>;

        // Optimized generation functions.
        /// Extract last token logits: logits[:, -1, :] -> [batch, vocab]
        /// Optimized for sampling during generation
        fn slice_last_logits(logits: &MlxArray) -> UniquePtr<MlxArray>;

        /// Slice on the last dimension only: a[..., start:end]
        /// Useful for fused QKV/gate_up projections
        fn slice_last_dim(a: &MlxArray, start: i32, end: i32) -> UniquePtr<MlxArray>;

        /// Argmax on last axis for greedy sampling
        fn argmax_last_axis(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Reshape token for next forward pass: [] or [batch] -> [batch, 1]
        /// Allows passing token array directly without extracting scalar
        fn reshape_token_for_forward(token: &MlxArray) -> UniquePtr<MlxArray>;

        /// Async eval two arrays at once (for lookahead pipelining)
        fn async_eval_pair(a: &MlxArray, b: &MlxArray);

        /// Set default stream for subsequent operations
        fn set_default_stream(stream: &MlxStream);

        /// Check whether the current default device is GPU
        fn is_gpu_available() -> bool;

        /// Fused sampling: temperature + top-k + top-p + min-p + categorical
        /// in a single C++ call to minimize FFI round-trips.
        /// Input: 2D logits [batch, vocab] (already sliced, penalties applied)
        /// Returns sampled token
        fn fused_sample(
            logits: &MlxArray,
            temperature: f32,
            top_k: i32,
            top_p: f32,
            min_p: f32,
        ) -> UniquePtr<MlxArray>;

        // SSM (State Space Model) primitives for Mamba/Jamba/Nemotron-H.
        /// Cumulative sum along axis
        fn cumsum(a: &MlxArray, axis: i32, reverse: bool, inclusive: bool) -> UniquePtr<MlxArray>;

        /// Lower triangular matrix (keeps elements on and below k-th diagonal)
        fn tril(a: &MlxArray, k: i32) -> UniquePtr<MlxArray>;

        /// Upper triangular matrix (keeps elements on and above k-th diagonal)
        fn triu(a: &MlxArray, k: i32) -> UniquePtr<MlxArray>;

        /// Clip values to range [a_min, a_max]
        fn clip(a: &MlxArray, a_min: &MlxArray, a_max: &MlxArray) -> UniquePtr<MlxArray>;

        /// log(1 + x) - numerically stable for small x
        fn log1p(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Softplus activation: log(1 + exp(x))
        fn softplus(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// 1D convolution with groups support (for depthwise conv when groups=channels)
        fn conv1d(
            input: &MlxArray,
            weight: &MlxArray,
            stride: i32,
            padding: i32,
            dilation: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        /// Swap axes (convenient for SSM attention)
        fn swap_axes(a: &MlxArray, axis1: i32, axis2: i32) -> UniquePtr<MlxArray>;

        // Core ops additions.
        /// Create identity matrix of size n x n
        fn identity(n: i32, dtype: i32) -> UniquePtr<MlxArray>;

        /// Create lower triangular matrix of ones
        fn tri(n: i32, m: i32, k: i32, dtype: i32) -> UniquePtr<MlxArray>;

        /// Unflatten axis into given shape
        fn unflatten(a: &MlxArray, axis: i32, shape: &[i32]) -> UniquePtr<MlxArray>;

        /// View array with given shape and strides
        fn as_strided(
            a: &MlxArray,
            shape: &[i32],
            strides: &[i64],
            offset: usize,
        ) -> UniquePtr<MlxArray>;

        /// Ensure the array memory is contiguous
        fn contiguous(a: &MlxArray, allow_col_major: bool) -> UniquePtr<MlxArray>;

        /// Broadcast a list of arrays against one another (returns i-th result)
        unsafe fn broadcast_arrays_get(
            arrays: &[*const MlxArray],
            index: usize,
        ) -> UniquePtr<MlxArray>;

        /// Number of results from broadcast_arrays
        unsafe fn broadcast_arrays_count(arrays: &[*const MlxArray]) -> usize;

        /// Floor division element-wise
        fn floor_divide(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise equality check (with nan handling option)
        fn array_equal(a: &MlxArray, b: &MlxArray, equal_nan: bool) -> UniquePtr<MlxArray>;

        /// Check if arrays are close within tolerances (returns scalar bool array)
        fn allclose(a: &MlxArray, b: &MlxArray, rtol: f64, atol: f64) -> UniquePtr<MlxArray>;

        /// Element-wise close check within tolerances
        fn isclose(a: &MlxArray, b: &MlxArray, rtol: f64, atol: f64) -> UniquePtr<MlxArray>;

        /// Compute median over all elements
        fn median_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compute median along axis
        fn median_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Cumulative log-sum-exp along axis
        fn logcumsumexp(
            a: &MlxArray,
            axis: i32,
            reverse: bool,
            inclusive: bool,
        ) -> UniquePtr<MlxArray>;

        /// Bitwise invert
        fn bitwise_invert(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Real part of complex array
        fn real_part(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Imaginary part of complex array
        fn imag_part(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Complex conjugate
        fn conjugate(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Reinterpret array as given dtype (bitwise view)
        fn view(a: &MlxArray, dtype: i32) -> UniquePtr<MlxArray>;

        /// Kronecker product of two arrays
        fn kron(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// addmm: beta * c + alpha * (a @ b)
        fn addmm(
            c: &MlxArray,
            a: &MlxArray,
            b: &MlxArray,
            alpha: f32,
            beta: f32,
        ) -> UniquePtr<MlxArray>;

        /// Block-sparse masked matrix multiply
        unsafe fn block_masked_mm(
            a: &MlxArray,
            b: &MlxArray,
            block_size: i32,
            mask_out: *const MlxArray,
            mask_lhs: *const MlxArray,
            mask_rhs: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Segmented matrix multiply
        fn segmented_mm(a: &MlxArray, b: &MlxArray, segments: &MlxArray) -> UniquePtr<MlxArray>;

        /// Hadamard transform
        fn hadamard_transform(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Number of elements along axes as scalar array
        fn number_of_elements(
            a: &MlxArray,
            axes: &[i32],
            inverted: bool,
            dtype: i32,
        ) -> UniquePtr<MlxArray>;

        // Convolution additions.
        /// 3D convolution
        fn conv3d(
            input: &MlxArray,
            weight: &MlxArray,
            stride_d: i32,
            stride_h: i32,
            stride_w: i32,
            padding_d: i32,
            padding_h: i32,
            padding_w: i32,
            dilation_d: i32,
            dilation_h: i32,
            dilation_w: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        /// 1D transposed convolution
        fn conv_transpose1d(
            input: &MlxArray,
            weight: &MlxArray,
            stride: i32,
            padding: i32,
            dilation: i32,
            output_padding: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        /// 2D transposed convolution
        fn conv_transpose2d(
            input: &MlxArray,
            weight: &MlxArray,
            stride_h: i32,
            stride_w: i32,
            padding_h: i32,
            padding_w: i32,
            dilation_h: i32,
            dilation_w: i32,
            output_padding_h: i32,
            output_padding_w: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        /// 3D transposed convolution
        fn conv_transpose3d(
            input: &MlxArray,
            weight: &MlxArray,
            stride_d: i32,
            stride_h: i32,
            stride_w: i32,
            padding_d: i32,
            padding_h: i32,
            padding_w: i32,
            dilation_d: i32,
            dilation_h: i32,
            dilation_w: i32,
            output_padding_d: i32,
            output_padding_h: i32,
            output_padding_w: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        // Einsum.
        /// Einsum contraction
        unsafe fn einsum(subscripts: &str, operands: &[*const MlxArray]) -> UniquePtr<MlxArray>;

        // Linear algebra (linalg).
        /// Vector/matrix norm along axis
        fn linalg_norm(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Vector/matrix norm with numeric ord along axis
        fn linalg_norm_ord(
            a: &MlxArray,
            ord: f64,
            axis: i32,
            keepdims: bool,
        ) -> UniquePtr<MlxArray>;

        /// Vector/matrix norm with string ord (e.g. "fro", "nuc") along axis
        fn linalg_norm_str(
            a: &MlxArray,
            ord: &str,
            axis: i32,
            keepdims: bool,
        ) -> UniquePtr<MlxArray>;

        /// QR decomposition — Q matrix
        fn linalg_qr_q(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// QR decomposition — R matrix
        fn linalg_qr_r(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// SVD — U matrix
        fn linalg_svd_u(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// SVD — singular values S
        fn linalg_svd_s(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// SVD — Vt matrix
        fn linalg_svd_vt(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Matrix inverse
        fn linalg_inv(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Moore-Penrose pseudoinverse
        fn linalg_pinv(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Cholesky decomposition
        fn linalg_cholesky(a: &MlxArray, upper: bool) -> UniquePtr<MlxArray>;

        /// Solve linear system a @ x = b
        fn linalg_solve(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Solve triangular linear system
        fn linalg_solve_triangular(a: &MlxArray, b: &MlxArray, upper: bool) -> UniquePtr<MlxArray>;

        /// LU decomposition — P permutation
        fn linalg_lu_p(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// LU decomposition — L lower triangular
        fn linalg_lu_l(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// LU decomposition — U upper triangular
        fn linalg_lu_u(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// LU factorization — LU combined matrix
        fn linalg_lu_factor_lu(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// LU factorization — pivots
        fn linalg_lu_factor_pivots(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Eigen decomposition — eigenvalues
        fn linalg_eig_values(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Eigen decomposition — eigenvectors
        fn linalg_eig_vectors(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Eigenvalues only
        fn linalg_eigvals(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Symmetric eigen decomposition — eigenvalues (default UPLO="L")
        fn linalg_eigh_values(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Symmetric eigen decomposition — eigenvectors (default UPLO="L")
        fn linalg_eigh_vectors(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Symmetric eigenvalues only
        fn linalg_eigvalsh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Cross product along axis
        fn linalg_cross(a: &MlxArray, b: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Triangular matrix inverse
        fn linalg_tri_inv(a: &MlxArray, upper: bool) -> UniquePtr<MlxArray>;

        /// Cholesky factor inverse
        fn linalg_cholesky_inv(a: &MlxArray, upper: bool) -> UniquePtr<MlxArray>;

        // FFT.
        /// 1D FFT with explicit n and axis
        fn fft(a: &MlxArray, n: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// 1D inverse FFT
        fn ifft(a: &MlxArray, n: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// 1D real FFT
        fn rfft(a: &MlxArray, n: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// 1D inverse real FFT
        fn irfft(a: &MlxArray, n: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// 2D FFT
        fn fft2(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// 2D inverse FFT
        fn ifft2(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// 2D real FFT
        fn rfft2(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// 2D inverse real FFT
        fn irfft2(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// N-dimensional FFT with explicit axes
        fn fftn_axes(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// N-dimensional inverse FFT with explicit axes
        fn ifftn_axes(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// N-dimensional real FFT with explicit axes
        fn rfftn_axes(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// N-dimensional inverse real FFT with explicit axes
        fn irfftn_axes(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// FFT shift along axes
        fn fftshift(a: &MlxArray, axes: &[i32]) -> UniquePtr<MlxArray>;

        /// Inverse FFT shift along axes
        fn ifftshift(a: &MlxArray, axes: &[i32]) -> UniquePtr<MlxArray>;

        // Random.
        /// Create a random key from seed
        fn random_key(seed: u64) -> UniquePtr<MlxArray>;

        /// Split key into num subkeys (returns array of shape [num, 2])
        fn random_split_key(key: &MlxArray, num: i32) -> UniquePtr<MlxArray>;

        /// Uniform random in [low, high)
        unsafe fn random_uniform(
            low: f32,
            high: f32,
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Standard normal random samples
        unsafe fn random_normal(
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Bernoulli random samples with probability p
        unsafe fn random_bernoulli_p(
            p: f32,
            shape: &[i32],
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Random integers in [low, high)
        unsafe fn random_randint(
            low: i32,
            high: i32,
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Truncated normal random samples
        unsafe fn random_truncated_normal(
            lower: f32,
            upper: f32,
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Gumbel random samples
        unsafe fn random_gumbel(
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Laplace random samples
        unsafe fn random_laplace(
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Random permutation of integers 0..x
        unsafe fn random_permutation(x: i32, key: *const MlxArray) -> UniquePtr<MlxArray>;

        /// Random permutation of array along axis
        unsafe fn random_permutation_array(
            a: &MlxArray,
            axis: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Multivariate normal random samples
        unsafe fn random_multivariate_normal(
            mean: &MlxArray,
            cov: &MlxArray,
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        // Quantization additions.
        /// Quantize weights — quantized weights
        fn quantize_weights_w(w: &MlxArray, group_size: i32, bits: i32) -> UniquePtr<MlxArray>;

        /// Quantize weights — scales
        fn quantize_weights_scales(w: &MlxArray, group_size: i32, bits: i32)
            -> UniquePtr<MlxArray>;

        /// Quantize weights — biases
        fn quantize_weights_biases(w: &MlxArray, group_size: i32, bits: i32)
            -> UniquePtr<MlxArray>;

        // Native safetensors loading (MLX-managed mmap, lazy arrays).
        /// Opaque holder for weights loaded via MLX's native load_safetensors()
        type MlxLoadedWeights;

        /// Load safetensors file using MLX's native loader (lazy arrays, MLX-managed mmap).
        /// Returns Result so C++ exceptions (bad path, corrupt file) become recoverable errors.
        fn mlx_load_safetensors(path: &str) -> Result<UniquePtr<MlxLoadedWeights>>;

        /// Number of weight entries in the loaded weights
        fn loaded_weights_len(w: &MlxLoadedWeights) -> usize;

        /// Get the name of the i-th weight entry
        fn loaded_weights_name(w: &MlxLoadedWeights, index: usize) -> String;

        /// Take (move out) the i-th weight array, leaving the slot empty
        fn loaded_weights_take(
            w: Pin<&mut MlxLoadedWeights>,
            index: usize,
        ) -> UniquePtr<MlxArray>;
    }
}

// Re-export the FFI types and functions
pub use ffi::*;

// Re-export cxx::UniquePtr for consumers of this crate
pub use cxx::UniquePtr;
pub use ops::{concatenate, divide_scalar, multiply_scalar, stack, stack_owned};

// High-level layer abstractions
pub mod layers;

// Cache state machines shared by attention families.
// Public so that `server` and `CachePool` consumers can import directly.
pub mod cache;

// Pure-Rust wrappers around frequently used FFI entry points.
mod ops;

// Common utility functions
pub mod utils;

// Weight loading utilities
pub mod weights;

// Token generation
pub mod generate;

// Decode-loop setup helpers shared by standard and speculative generation.
// Public so that the server batch scheduler can reuse seed/EOS/history helpers.
pub mod generation_policy;

// Shared sampling and token-penalty policy helpers.
// Public so that the server batch scheduler can perform step-level sampling.
pub mod sampling;

// Speculative decoding
pub mod speculative;

// Generation-time stream selection and installation wrappers.
// Public so that the server batch scheduler can install its own generation stream.
pub mod streams;

pub mod dtype;

#[cfg(test)]
#[path = "ffi_tests.rs"]
mod ffi_tests;

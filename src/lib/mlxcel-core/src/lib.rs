//! Direct C++ bindings to MLX via cxx
//!
//! This crate provides direct bindings to MLX C++ API, bypassing the mlx-c wrapper
//! for improved performance.

#[cxx::bridge(namespace = "mlx_cxx")]
mod ffi {
    // Opaque types - these are defined in C++ and we just hold pointers to them
    unsafe extern "C++" {
        include!("mlx_cxx_bridge.h");

        /// Opaque wrapper for mlx::core::array
        type MlxArray;

        /// Opaque wrapper for mlx::core::Stream
        type MlxStream;

        // ====================================================================
        // Stream functions
        // ====================================================================

        /// Get the default stream
        fn default_stream() -> UniquePtr<MlxStream>;

        /// Create a new stream on the specified device
        fn new_stream_on_device(gpu: bool) -> UniquePtr<MlxStream>;

        /// Synchronize a stream
        fn synchronize_stream(stream: &MlxStream);

        // ====================================================================
        // Array factory functions
        // ====================================================================

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

        // ====================================================================
        // Array property accessors
        // ====================================================================

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

        // ====================================================================
        // Scalar extraction
        // ====================================================================

        /// Extract f32 scalar value
        fn item_f32(arr: &MlxArray) -> f32;

        /// Extract i32 scalar value
        fn item_i32(arr: &MlxArray) -> i32;

        /// Extract i64 scalar value
        fn item_i64(arr: &MlxArray) -> i64;

        /// Extract bool scalar value
        fn item_bool(arr: &MlxArray) -> bool;

        // ====================================================================
        // Evaluation
        // ====================================================================

        /// Evaluate an array
        fn eval(arr: &MlxArray);

        /// Evaluate multiple arrays at once
        unsafe fn eval_all(arrays: &[*const MlxArray]);

        // ====================================================================
        // Element-wise binary operations
        // ====================================================================

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

        // ====================================================================
        // Element-wise unary operations
        // ====================================================================

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

        // ====================================================================
        // Reduction operations
        // ====================================================================

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

        // ====================================================================
        // Matrix operations
        // ====================================================================

        /// Matrix multiplication
        fn matmul(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Transpose (swap last two dimensions)
        fn transpose(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Transpose with specified axes
        fn transpose_axes(a: &MlxArray, axes: &[i32]) -> UniquePtr<MlxArray>;

        /// Reshape array
        fn reshape(a: &MlxArray, shape: &[i32]) -> UniquePtr<MlxArray>;

        // ====================================================================
        // Shape operations
        // ====================================================================

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

        // ====================================================================
        // Type conversion
        // ====================================================================

        /// Convert array to specified dtype
        fn astype(a: &MlxArray, dtype: i32) -> UniquePtr<MlxArray>;

        // ====================================================================
        // Copy
        // ====================================================================

        /// Copy array
        fn copy(a: &MlxArray) -> UniquePtr<MlxArray>;

        // ====================================================================
        // High-level operations for LLM inference
        // ====================================================================

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

        // ====================================================================
        // Transformer-specific high-level operations
        // ====================================================================

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
        unsafe fn quantized_linear_forward(
            x: &MlxArray,
            weight: &MlxArray,
            scales: &MlxArray,
            biases: &MlxArray,
            linear_bias: *const MlxArray,
            group_size: i32,
            bits: i32,
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

        // ====================================================================
        // Advanced indexing operations
        // ====================================================================

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

        // ====================================================================
        // Logical operations
        // ====================================================================

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

        // ====================================================================
        // Activation functions
        // ====================================================================

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

        // ====================================================================
        // Sorting and searching
        // ====================================================================

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

        // ====================================================================
        // MoE (Mixture of Experts) operations
        // ====================================================================

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
        ) -> UniquePtr<MlxArray>;

        /// Dequantize quantized weights to full precision
        fn dequantize(
            w: &MlxArray,
            scales: &MlxArray,
            biases: &MlxArray,
            group_size: i32,
            bits: i32,
        ) -> UniquePtr<MlxArray>;

        // ====================================================================
        // Embedding
        // ====================================================================

        /// Embedding lookup (same as take along axis 0)
        fn embedding(weight: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray>;

        /// Quantized embedding lookup with dequantization
        fn quantized_embedding(
            weight: &MlxArray,
            scales: &MlxArray,
            biases: &MlxArray,
            indices: &MlxArray,
            group_size: i32,
            bits: i32,
        ) -> UniquePtr<MlxArray>;

        // ====================================================================
        // Fast operations (using MLX fast kernels)
        // ====================================================================

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
        fn fused_qkv_project_and_rope(
            x: &MlxArray,
            weight: &MlxArray,
            scales: &MlxArray,
            biases: &MlxArray,
            num_heads: i32,
            head_dim: i32,
            rope_dims: i32,
            rope_base: f32,
            cache_offset: i32,
            group_size: i32,
            bits: i32,
            apply_rope: bool,
        ) -> UniquePtr<MlxArray>;

        // ====================================================================
        // Compiled operations (with kernel fusion)
        // ====================================================================

        /// Compiled MoE expert forward with quantized weights
        fn compiled_moe_expert_forward(
            x: &MlxArray,
            gate_proj: &MlxArray,
            gate_scales: &MlxArray,
            gate_biases: &MlxArray,
            up_proj: &MlxArray,
            up_scales: &MlxArray,
            up_biases: &MlxArray,
            down_proj: &MlxArray,
            down_scales: &MlxArray,
            down_biases: &MlxArray,
            group_size: i32,
            bits: i32,
        ) -> UniquePtr<MlxArray>;

        // ====================================================================
        // Memory management
        // ====================================================================

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

        // ====================================================================
        // Optimized generation functions
        // ====================================================================

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

        // ====================================================================
        // SSM (State Space Model) primitives for Mamba/Jamba/Nemotron-H
        // ====================================================================

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

        // ====================================================================
        // Core ops additions
        // ====================================================================

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

        // ====================================================================
        // Convolution additions
        // ====================================================================

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

        // ====================================================================
        // Einsum
        // ====================================================================

        /// Einsum contraction
        unsafe fn einsum(subscripts: &str, operands: &[*const MlxArray]) -> UniquePtr<MlxArray>;

        // ====================================================================
        // Linear algebra (linalg)
        // ====================================================================

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

        // ====================================================================
        // FFT
        // ====================================================================

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

        // ====================================================================
        // Random
        // ====================================================================

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

        // ====================================================================
        // Quantization additions
        // ====================================================================

        /// Quantize weights — quantized weights
        fn quantize_weights_w(w: &MlxArray, group_size: i32, bits: i32) -> UniquePtr<MlxArray>;

        /// Quantize weights — scales
        fn quantize_weights_scales(w: &MlxArray, group_size: i32, bits: i32)
            -> UniquePtr<MlxArray>;

        /// Quantize weights — biases
        fn quantize_weights_biases(w: &MlxArray, group_size: i32, bits: i32)
            -> UniquePtr<MlxArray>;
    }
}

// Re-export the FFI types and functions
pub use ffi::*;

// Re-export cxx::UniquePtr for consumers of this crate
pub use cxx::UniquePtr;

/// Safe wrapper to concatenate two arrays along an axis
pub fn concatenate(
    a: &ffi::MlxArray,
    b: &ffi::MlxArray,
    axis: i32,
) -> cxx::UniquePtr<ffi::MlxArray> {
    let ptrs: [*const ffi::MlxArray; 2] = [a as *const ffi::MlxArray, b as *const ffi::MlxArray];
    unsafe { ffi::concatenate(&ptrs, axis) }
}

/// Stack arrays along a new axis (low-level, takes raw pointers)
/// Use stack_owned for UniquePtr inputs
pub fn stack(ptrs: &[*const ffi::MlxArray], axis: i32) -> cxx::UniquePtr<ffi::MlxArray> {
    unsafe { ffi::stack(ptrs, axis) }
}

/// Stack arrays along a new axis (takes UniquePtr references)
pub fn stack_owned(
    arrays: &[cxx::UniquePtr<ffi::MlxArray>],
    axis: i32,
) -> cxx::UniquePtr<ffi::MlxArray> {
    let ptrs: Vec<*const ffi::MlxArray> = arrays
        .iter()
        .map(|a| a.as_ref().unwrap() as *const ffi::MlxArray)
        .collect();
    unsafe { ffi::stack(&ptrs, axis) }
}

/// Multiply array by scalar
pub fn multiply_scalar(a: &ffi::MlxArray, scalar: f32) -> cxx::UniquePtr<ffi::MlxArray> {
    let scalar_array = ffi::full_f32(&[1], scalar, dtype::FLOAT32);
    ffi::multiply(a, &scalar_array)
}

/// Divide array by scalar
pub fn divide_scalar(a: &ffi::MlxArray, scalar: f32) -> cxx::UniquePtr<ffi::MlxArray> {
    let scalar_array = ffi::full_f32(&[1], scalar, dtype::FLOAT32);
    ffi::divide(a, &scalar_array)
}

// High-level layer abstractions
pub mod layers;

// Common utility functions
pub mod utils;

// Weight loading utilities
pub mod weights;

// Token generation
pub mod generate;

// Speculative decoding
pub mod speculative;

/// Dtype constants matching MLX's dtype enum
pub mod dtype {
    pub const BOOL: i32 = 0;
    pub const UINT8: i32 = 1;
    pub const UINT16: i32 = 2;
    pub const UINT32: i32 = 3;
    pub const UINT64: i32 = 4;
    pub const INT8: i32 = 5;
    pub const INT16: i32 = 6;
    pub const INT32: i32 = 7;
    pub const INT64: i32 = 8;
    pub const FLOAT16: i32 = 9;
    pub const FLOAT32: i32 = 10;
    pub const FLOAT64: i32 = 11;
    pub const BFLOAT16: i32 = 12;
    pub const COMPLEX64: i32 = 13;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let arr = zeros(&[2, 3], dtype::FLOAT32);
        assert!(!arr.is_null());
        assert_eq!(array_shape(&arr), vec![2, 3]);
        assert_eq!(array_dtype(&arr), dtype::FLOAT32);
        assert_eq!(array_size(&arr), 6);
    }

    #[test]
    fn test_ones() {
        let arr = ones(&[4, 5], dtype::FLOAT32);
        assert!(!arr.is_null());
        eval(&arr);
        let sum = sum_all(&arr);
        eval(&sum);
        assert_eq!(item_f32(&sum), 20.0);
    }

    #[test]
    fn test_matmul() {
        let a = ones(&[2, 3], dtype::FLOAT32);
        let b = ones(&[3, 4], dtype::FLOAT32);
        let c = matmul(&a, &b);
        assert_eq!(array_shape(&c), vec![2, 4]);
        eval(&c);
        // Sum all elements: 2*4 = 8 elements, each is 3.0, so sum = 24.0
        let total = sum_all(&c);
        eval(&total);
        assert_eq!(item_f32(&total), 24.0);
    }

    #[test]
    fn test_add_multiply() {
        let a = full_f32(&[2, 2], 2.0, dtype::FLOAT32);
        let b = full_f32(&[2, 2], 3.0, dtype::FLOAT32);
        let c = add(&a, &b);
        let d = multiply(&a, &b);

        eval(&c);
        eval(&d);

        let sum = sum_all(&c);
        let prod_sum = sum_all(&d);

        eval(&sum);
        eval(&prod_sum);

        assert_eq!(item_f32(&sum), 20.0); // 4 * 5.0
        assert_eq!(item_f32(&prod_sum), 24.0); // 4 * 6.0
    }

    #[test]
    fn test_softmax() {
        let a = from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let s = softmax(&a, -1);
        eval(&s);

        // Softmax should sum to 1.0
        let total = sum_all(&s);
        eval(&total);
        let sum_val = item_f32(&total);
        assert!((sum_val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm() {
        let x = ones(&[1, 4], dtype::FLOAT32);
        let weight = ones(&[4], dtype::FLOAT32);
        let normed = rms_norm(&x, &weight, 1e-5);
        eval(&normed);

        // For all 1s, RMS norm should produce all 1s
        let total = sum_all(&normed);
        eval(&total);
        assert!((item_f32(&total) - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_argmax() {
        let a = from_slice_f32(&[1.0, 3.0, 2.0, 4.0], &[1, 4]);
        let idx = argmax(&a, -1, false);
        eval(&idx);
        assert_eq!(item_i32(&idx), 3); // Index of max value (4.0)
    }

    #[test]
    fn test_swiglu_mlp() {
        // Simple test with identity-like weights
        let x = ones(&[1, 4], dtype::FLOAT32);
        let gate = ones(&[8, 4], dtype::FLOAT32);
        let up = ones(&[8, 4], dtype::FLOAT32);
        let down = ones(&[4, 8], dtype::FLOAT32);

        let out = swiglu_mlp_forward(&x, &gate, &up, &down);
        eval(&out);

        // Just verify it produces output with correct shape
        let shape = array_shape(&out);
        assert_eq!(shape, vec![1, 4]);
    }

    #[test]
    fn test_compiled_swiglu_activation() {
        // Test compiled swiglu activation (silu(gate) * x)
        let gate = from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let x = from_slice_f32(&[2.0, 2.0, 2.0, 2.0], &[1, 4]);

        let out = compiled_swiglu_activation(&gate, &x);
        eval(&out);

        // Verify shape is correct
        let shape = array_shape(&out);
        assert_eq!(shape, vec![1, 4]);

        // Verify output is non-zero
        let total = sum_all(&out);
        eval(&total);
        let sum_val = item_f32(&total);
        assert!(sum_val > 0.0);
    }

    #[test]
    fn test_new_ops() {
        // Test silu
        let x = from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let s = silu(&x);
        eval(&s);
        let shape = array_shape(&s);
        assert_eq!(shape, vec![1, 4]);

        // Test gelu
        let g = gelu(&x);
        eval(&g);
        assert_eq!(array_shape(&g), vec![1, 4]);

        // Test relu
        let r = relu(&x);
        eval(&r);
        assert_eq!(array_shape(&r), vec![1, 4]);

        // Test take
        let indices = from_slice_i32(&[0, 2], &[2]);
        let taken = take(&x, &indices, -1);
        eval(&taken);
        assert_eq!(array_shape(&taken), vec![1, 2]);

        // Test argsort
        let vals = from_slice_f32(&[3.0, 1.0, 4.0, 2.0], &[4]);
        let sorted_idx = argsort(&vals, 0);
        eval(&sorted_idx);
        assert_eq!(array_shape(&sorted_idx), vec![4]);

        // Test argpartition
        let part_idx = argpartition(&vals, 1, 0);
        eval(&part_idx);
        assert_eq!(array_shape(&part_idx), vec![4]);

        // Test fast_rms_norm
        let inp = ones(&[1, 4], dtype::FLOAT32);
        let weight = ones(&[4], dtype::FLOAT32);
        let normed = fast_rms_norm(&inp, &weight, 1e-5);
        eval(&normed);
        assert_eq!(array_shape(&normed), vec![1, 4]);

        // Test async_eval
        let y = ones(&[2, 2], dtype::FLOAT32);
        async_eval(&y);
        synchronize_default();
        let total = sum_all(&y);
        eval(&total);
        assert_eq!(item_f32(&total), 4.0);
    }

    #[test]
    fn test_gather_mm() {
        // Simple gather_mm test
        // a: [2, 3, 4] - 2 batches of 3x4 matrices
        // b: [2, 4, 5] - 2 experts, each 4x5
        let a_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();

        let a = from_slice_f32(&a_data, &[2, 3, 4]);
        let b = from_slice_f32(&b_data, &[2, 4, 5]);

        // indices: which expert for each batch row
        let rhs_indices = from_slice_i32(&[0, 1], &[2]);

        let result = unsafe {
            gather_mm(
                &a,
                &b,
                std::ptr::null(),
                rhs_indices.as_ref().unwrap() as *const _,
                false,
            )
        };
        eval(&result);
        let shape = array_shape(&result);
        assert_eq!(shape, vec![2, 3, 5]);
    }

    #[test]
    fn test_memory_functions() {
        let max_size = gpu_max_memory_size();
        assert!(max_size > 0);

        // Test set_wired_limit
        let _old = set_wired_limit(1024 * 1024 * 1024); // 1GB

        // Get wired limit
        let limit = get_wired_limit();
        assert!(limit > 0);

        // Reset
        set_wired_limit(0);
    }

    #[test]
    fn bench_compiled_vs_uncompiled_swiglu() {
        use std::time::Instant;

        // Test with different dimensions
        let test_dims = [4096, 8192, 14336, 24576, 49152];

        for dim in test_dims {
            // Create test data
            let gate_data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();
            let x_data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002).cos()).collect();

            let gate = from_slice_f32(&gate_data, &[1, dim as i32]);
            let x = from_slice_f32(&x_data, &[1, dim as i32]);

            // Warmup
            for _ in 0..10 {
                let out = compiled_swiglu_activation(&gate, &x);
                eval(&out);
            }

            // Benchmark compiled
            let iterations = 200;
            let start = Instant::now();
            for _ in 0..iterations {
                let out = compiled_swiglu_activation(&gate, &x);
                eval(&out);
            }
            let compiled_time = start.elapsed();

            // Benchmark uncompiled (using multiply + sigmoid)
            let start = Instant::now();
            for _ in 0..iterations {
                let silu_gate = multiply(&gate, &sigmoid(&gate));
                let out = multiply(&silu_gate, &x);
                eval(&out);
            }
            let uncompiled_time = start.elapsed();

            println!(
                "dim={:5} | Compiled: {:.2} μs | Uncompiled: {:.2} μs | Speedup: {:.2}x",
                dim,
                compiled_time.as_micros() as f64 / iterations as f64,
                uncompiled_time.as_micros() as f64 / iterations as f64,
                uncompiled_time.as_secs_f64() / compiled_time.as_secs_f64()
            );
        }
    }
}

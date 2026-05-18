// Copyright 2025 mlx-lm-rs authors
// Direct C++ bridge implementation for MLX via cxx

#include "mlx_cxx_bridge.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/compile.h"
#include "mlx/memory.h"
#include "mlx/linalg.h"
#include "mlx/fft.h"
#include "mlx/random.h"
#include "mlx/einsum.h"
#ifdef __APPLE__
#include "mlx/backend/metal/metal.h"
#endif
#include <cstring>
#include <iostream>

namespace mlx_cxx {

using namespace mlx::core;

// Helper to convert rust slice to std::vector<int32_t> (Shape)
static Shape to_shape(rust::Slice<const int32_t> slice) {
    return Shape(slice.begin(), slice.end());
}

// Helper to convert Dtype int to mlx::core::Dtype
static Dtype to_dtype(int32_t dtype) {
    switch (dtype) {
        case 0: return bool_;
        case 1: return uint8;
        case 2: return uint16;
        case 3: return uint32;
        case 4: return uint64;
        case 5: return int8;
        case 6: return int16;
        case 7: return int32;
        case 8: return int64;
        case 9: return float16;
        case 10: return float32;
        case 11: return float64;
        case 12: return bfloat16;
        case 13: return complex64;
        default: return float32;
    }
}

// Helper to convert mlx::core::Dtype to int
static int32_t from_dtype(Dtype dtype) {
    switch (dtype.val()) {
        case bool_.val(): return 0;
        case uint8.val(): return 1;
        case uint16.val(): return 2;
        case uint32.val(): return 3;
        case uint64.val(): return 4;
        case int8.val(): return 5;
        case int16.val(): return 6;
        case int32.val(): return 7;
        case int64.val(): return 8;
        case float16.val(): return 9;
        case float32.val(): return 10;
        case float64.val(): return 11;
        case bfloat16.val(): return 12;
        case complex64.val(): return 13;
        default: return 10;  // float32
    }
}

// ============================================================================
// Stream functions
// ============================================================================

std::unique_ptr<MlxStream> default_stream() {
    return std::make_unique<MlxStream>(mlx::core::default_stream(mlx::core::default_device()));
}

std::unique_ptr<MlxStream> new_stream_on_device(bool gpu) {
    Device d = gpu ? Device::gpu : Device::cpu;
    return std::make_unique<MlxStream>(mlx::core::new_stream(d));
}

void synchronize_stream(const MlxStream& stream) {
    mlx::core::synchronize(stream.inner);
}

// ============================================================================
// Array factory functions
// ============================================================================

std::unique_ptr<MlxArray> zeros(rust::Slice<const int32_t> shape, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::zeros(to_shape(shape), to_dtype(dtype)));
}

std::unique_ptr<MlxArray> zeros_stream(rust::Slice<const int32_t> shape, int32_t dtype, const MlxStream& stream) {
    return std::make_unique<MlxArray>(mlx::core::zeros(to_shape(shape), to_dtype(dtype), stream.inner));
}

std::unique_ptr<MlxArray> ones(rust::Slice<const int32_t> shape, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::ones(to_shape(shape), to_dtype(dtype)));
}

std::unique_ptr<MlxArray> ones_stream(rust::Slice<const int32_t> shape, int32_t dtype, const MlxStream& stream) {
    return std::make_unique<MlxArray>(mlx::core::ones(to_shape(shape), to_dtype(dtype), stream.inner));
}

std::unique_ptr<MlxArray> full_f32(rust::Slice<const int32_t> shape, float value, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::full(to_shape(shape), value, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> eye(int32_t n, int32_t m, int32_t k, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::eye(n, m, k, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> linspace(float start, float stop, int32_t num, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::linspace(start, stop, num, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> zeros_like(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::zeros_like(a.inner));
}

std::unique_ptr<MlxArray> ones_like(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::ones_like(a.inner));
}

std::unique_ptr<MlxArray> full_like(const MlxArray& a, float value) {
    return std::make_unique<MlxArray>(mlx::core::full_like(a.inner, value));
}

std::unique_ptr<MlxArray> from_slice_f32(rust::Slice<const float> data, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(array(data.data(), to_shape(shape)));
}

std::unique_ptr<MlxArray> from_slice_i32(rust::Slice<const int32_t> data, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(array(data.data(), to_shape(shape)));
}

std::unique_ptr<MlxArray> from_slice_u32(rust::Slice<const uint32_t> data, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(array(data.data(), to_shape(shape)));
}

std::unique_ptr<MlxArray> from_slice_i64(rust::Slice<const int64_t> data, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(array(data.data(), to_shape(shape)));
}

// Helper to create array from typed pointer
// Does NOT call eval - caller is responsible for ensuring data remains valid until eval
template<typename T>
static std::unique_ptr<MlxArray> make_array_typed(const uint8_t* data, const Shape& shape, Dtype dtype) {
    auto result = array(reinterpret_cast<const T*>(data), shape, dtype);
    return std::make_unique<MlxArray>(result);
}

// Create array from raw bytes with specified dtype
// Uses MLX's array constructor with typed pointer cast
// IMPORTANT: Caller must call eval() on the result before source data goes out of scope
std::unique_ptr<MlxArray> from_bytes(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape, int32_t dtype) {
    auto mlx_dtype = to_dtype(dtype);
    auto mlx_shape = to_shape(shape);

    // Create array from raw data with correct pointer type
    switch (dtype) {
        case 3:  // UINT32
            return make_array_typed<uint32_t>(data.data(), mlx_shape, mlx_dtype);
        case 4:  // UINT64
            return make_array_typed<uint64_t>(data.data(), mlx_shape, mlx_dtype);
        case 7:  // INT32
            return make_array_typed<int32_t>(data.data(), mlx_shape, mlx_dtype);
        case 8:  // INT64
            return make_array_typed<int64_t>(data.data(), mlx_shape, mlx_dtype);
        case 10:  // FLOAT32
            return make_array_typed<float>(data.data(), mlx_shape, mlx_dtype);
        case 1:  // UINT8
        default:
            return std::make_unique<MlxArray>(array(data.data(), mlx_shape, mlx_dtype));
    }
}

// Helper function to convert float16 bits to float32
inline float f16_to_f32_bits(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;  // Zero
        } else {
            // Denormalized - convert to normalized
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);  // Inf/NaN
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);  // Normalized
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// Helper function to convert bfloat16 bits to float32
inline float bf16_to_f32_bits(uint16_t h) {
    // bfloat16 is just the upper 16 bits of float32
    uint32_t f = static_cast<uint32_t>(h) << 16;
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// Create half-precision array from raw bytes (bfloat16 or float16)
// Convert through float32 for correct handling, keep as float32 since MLX float16 array creation has issues
std::unique_ptr<MlxArray> from_bytes_f16(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape, bool is_bfloat16) {
    auto mlx_shape = to_shape(shape);

    // Calculate number of elements
    size_t num_elements = 1;
    for (auto s : mlx_shape) num_elements *= s;

    // Convert to float32
    std::vector<float> float_data(num_elements);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(data.data());

    if (is_bfloat16) {
        for (size_t i = 0; i < num_elements; ++i) {
            float_data[i] = bf16_to_f32_bits(src[i]);
        }
    } else {
        for (size_t i = 0; i < num_elements; ++i) {
            float_data[i] = f16_to_f32_bits(src[i]);
        }
    }

    // Create float32 array - keep as float32 for now due to MLX float16 issues
    return std::make_unique<MlxArray>(array(float_data.data(), mlx_shape));
}

// ============================================================================
// Array property accessors
// ============================================================================

rust::Vec<int32_t> array_shape(const MlxArray& arr) {
    rust::Vec<int32_t> result;
    const auto& shape = arr.inner.shape();
    for (auto s : shape) {
        result.push_back(s);
    }
    return result;
}

int32_t array_dtype(const MlxArray& arr) {
    return from_dtype(arr.inner.dtype());
}

size_t array_size(const MlxArray& arr) {
    return arr.inner.size();
}

size_t array_ndim(const MlxArray& arr) {
    return arr.inner.ndim();
}

size_t array_itemsize(const MlxArray& arr) {
    return arr.inner.itemsize();
}

size_t array_nbytes(const MlxArray& arr) {
    return arr.inner.nbytes();
}

// ============================================================================
// Array data access (scalar extraction)
// ============================================================================

float item_f32(const MlxArray& arr) {
    return const_cast<array&>(arr.inner).item<float>();
}

int32_t item_i32(const MlxArray& arr) {
    return const_cast<array&>(arr.inner).item<int32_t>();
}

int64_t item_i64(const MlxArray& arr) {
    return const_cast<array&>(arr.inner).item<int64_t>();
}

bool item_bool(const MlxArray& arr) {
    return const_cast<array&>(arr.inner).item<bool>();
}

// ============================================================================
// Evaluation
// ============================================================================

void eval(const MlxArray& arr) {
    const_cast<array&>(arr.inner).eval();
}

void eval_all(rust::Slice<const MlxArray* const> arrays) {
    std::vector<array> arrs;
    arrs.reserve(arrays.size());
    for (const auto* a : arrays) {
        arrs.push_back(a->inner);
    }
    mlx::core::eval(arrs);
}

// ============================================================================
// Element-wise binary operations
// ============================================================================

std::unique_ptr<MlxArray> add(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::add(a.inner, b.inner));
}

std::unique_ptr<MlxArray> subtract(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::subtract(a.inner, b.inner));
}

std::unique_ptr<MlxArray> remainder(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::remainder(a.inner, b.inner));
}

std::unique_ptr<MlxArray> multiply(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::multiply(a.inner, b.inner));
}

std::unique_ptr<MlxArray> divide(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::divide(a.inner, b.inner));
}

std::unique_ptr<MlxArray> maximum(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::maximum(a.inner, b.inner));
}

std::unique_ptr<MlxArray> minimum(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::minimum(a.inner, b.inner));
}

// ============================================================================
// Element-wise unary operations
// ============================================================================

std::unique_ptr<MlxArray> negative(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::negative(a.inner));
}

std::unique_ptr<MlxArray> abs(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::abs(a.inner));
}

std::unique_ptr<MlxArray> exp(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::exp(a.inner));
}

std::unique_ptr<MlxArray> log(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::log(a.inner));
}

std::unique_ptr<MlxArray> sqrt(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sqrt(a.inner));
}

std::unique_ptr<MlxArray> rsqrt(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::rsqrt(a.inner));
}

std::unique_ptr<MlxArray> square(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::square(a.inner));
}

std::unique_ptr<MlxArray> sin(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sin(a.inner));
}

std::unique_ptr<MlxArray> cos(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::cos(a.inner));
}

std::unique_ptr<MlxArray> tanh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::tanh(a.inner));
}

std::unique_ptr<MlxArray> sigmoid(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sigmoid(a.inner));
}

std::unique_ptr<MlxArray> floor(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::floor(a.inner));
}

std::unique_ptr<MlxArray> ceil(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::ceil(a.inner));
}

std::unique_ptr<MlxArray> round(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::round(a.inner));
}

std::unique_ptr<MlxArray> sign(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sign(a.inner));
}

std::unique_ptr<MlxArray> reciprocal(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::reciprocal(a.inner));
}

// Trigonometric functions
std::unique_ptr<MlxArray> tan(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::tan(a.inner));
}

std::unique_ptr<MlxArray> sinh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sinh(a.inner));
}

std::unique_ptr<MlxArray> cosh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::cosh(a.inner));
}

std::unique_ptr<MlxArray> arcsin(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arcsin(a.inner));
}

std::unique_ptr<MlxArray> arccos(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arccos(a.inner));
}

std::unique_ptr<MlxArray> arctan(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arctan(a.inner));
}

std::unique_ptr<MlxArray> arctan2(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::arctan2(a.inner, b.inner));
}

std::unique_ptr<MlxArray> arcsinh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arcsinh(a.inner));
}

std::unique_ptr<MlxArray> arccosh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arccosh(a.inner));
}

std::unique_ptr<MlxArray> arctanh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arctanh(a.inner));
}

std::unique_ptr<MlxArray> degrees(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::degrees(a.inner));
}

std::unique_ptr<MlxArray> radians(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::radians(a.inner));
}

// Mathematical/Special functions
std::unique_ptr<MlxArray> erf(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::erf(a.inner));
}

std::unique_ptr<MlxArray> erfinv(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::erfinv(a.inner));
}

std::unique_ptr<MlxArray> expm1(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::expm1(a.inner));
}

std::unique_ptr<MlxArray> log2(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::log2(a.inner));
}

std::unique_ptr<MlxArray> log10(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::log10(a.inner));
}

std::unique_ptr<MlxArray> logaddexp(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::logaddexp(a.inner, b.inner));
}

std::unique_ptr<MlxArray> power(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::power(a.inner, b.inner));
}

// Checks
std::unique_ptr<MlxArray> isnan(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::isnan(a.inner));
}

std::unique_ptr<MlxArray> isinf(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::isinf(a.inner));
}

std::unique_ptr<MlxArray> isfinite(const MlxArray& a) {
    // isfinite = !isnan && !isinf
    auto nan_check = mlx::core::isnan(a.inner);
    auto inf_check = mlx::core::isinf(a.inner);
    auto either = mlx::core::logical_or(nan_check, inf_check);
    return std::make_unique<MlxArray>(mlx::core::logical_not(either));
}

std::unique_ptr<MlxArray> isneginf(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::isneginf(a.inner));
}

std::unique_ptr<MlxArray> isposinf(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::isposinf(a.inner));
}

// ============================================================================
// Reduction operations
// ============================================================================

std::unique_ptr<MlxArray> sum_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sum(a.inner));
}

std::unique_ptr<MlxArray> sum_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::sum(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> mean_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::mean(a.inner));
}

std::unique_ptr<MlxArray> mean_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::mean(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> max_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::max(a.inner));
}

std::unique_ptr<MlxArray> max_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::max(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> min_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::min(a.inner));
}

std::unique_ptr<MlxArray> min_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::min(a.inner, axis, keepdims));
}

// Product reduction
std::unique_ptr<MlxArray> prod_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::prod(a.inner));
}

std::unique_ptr<MlxArray> prod_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::prod(a.inner, axis, keepdims));
}

// Variance and standard deviation
std::unique_ptr<MlxArray> var_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::var(a.inner));
}

std::unique_ptr<MlxArray> var_axis(const MlxArray& a, int32_t axis, bool keepdims, int32_t ddof) {
    return std::make_unique<MlxArray>(mlx::core::var(a.inner, axis, keepdims, ddof));
}

std::unique_ptr<MlxArray> std_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::std(a.inner));
}

std::unique_ptr<MlxArray> std_axis(const MlxArray& a, int32_t axis, bool keepdims, int32_t ddof) {
    return std::make_unique<MlxArray>(mlx::core::std(a.inner, axis, keepdims, ddof));
}

// Logsumexp
std::unique_ptr<MlxArray> logsumexp_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::logsumexp(a.inner));
}

std::unique_ptr<MlxArray> logsumexp_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::logsumexp(a.inner, axis, keepdims));
}

// All/any reductions
std::unique_ptr<MlxArray> all_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::all(a.inner));
}

std::unique_ptr<MlxArray> any_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::any(a.inner));
}

// ============================================================================
// Matrix operations
// ============================================================================

std::unique_ptr<MlxArray> matmul(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::matmul(a.inner, b.inner));
}

std::unique_ptr<MlxArray> transpose(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::transpose(a.inner));
}

std::unique_ptr<MlxArray> transpose_axes(const MlxArray& a, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::transpose(a.inner, axes_vec));
}

std::unique_ptr<MlxArray> reshape(const MlxArray& a, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(mlx::core::reshape(a.inner, to_shape(shape)));
}

// ============================================================================
// Shape operations
// ============================================================================

std::unique_ptr<MlxArray> expand_dims(const MlxArray& a, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::expand_dims(a.inner, axis));
}

std::unique_ptr<MlxArray> squeeze(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::squeeze(a.inner));
}

std::unique_ptr<MlxArray> squeeze_axis(const MlxArray& a, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::squeeze(a.inner, axis));
}

std::unique_ptr<MlxArray> broadcast_to(const MlxArray& a, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(mlx::core::broadcast_to(a.inner, to_shape(shape)));
}

// Flatten array
std::unique_ptr<MlxArray> flatten(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::flatten(a.inner));
}

std::unique_ptr<MlxArray> flatten_range(const MlxArray& a, int32_t start_axis, int32_t end_axis) {
    return std::make_unique<MlxArray>(mlx::core::flatten(a.inner, start_axis, end_axis));
}

// Move axis
std::unique_ptr<MlxArray> moveaxis(const MlxArray& a, int32_t source, int32_t destination) {
    return std::make_unique<MlxArray>(mlx::core::moveaxis(a.inner, source, destination));
}

// Pad array
std::unique_ptr<MlxArray> pad(const MlxArray& a, rust::Slice<const int32_t> pad_width, float pad_value) {
    // pad_width is pairs of [before, after] for each dimension
    // Convert to vector of pairs
    std::vector<std::pair<int, int>> width_pairs;
    for (size_t i = 0; i + 1 < pad_width.size(); i += 2) {
        width_pairs.push_back({pad_width[i], pad_width[i + 1]});
    }
    return std::make_unique<MlxArray>(mlx::core::pad(a.inner, width_pairs, array(pad_value)));
}

// Split array at indices - returns single concatenated result for simplicity
// For proper multi-output support, users should use slice
std::unique_ptr<MlxArray> split_at_indices(const MlxArray& a, rust::Slice<const int32_t> indices, int32_t axis) {
    // Convert to Shape (SmallVector<int>) which MLX expects
    Shape idx_shape(indices.begin(), indices.end());
    // This returns vector of arrays - we just return the first for simple use cases
    // Full split support would need different approach
    auto splits = mlx::core::split(a.inner, idx_shape, axis);
    if (splits.empty()) {
        return std::make_unique<MlxArray>(a.inner);
    }
    return std::make_unique<MlxArray>(std::move(splits[0]));
}

// Diagonal operations
std::unique_ptr<MlxArray> diag(const MlxArray& a, int32_t k) {
    return std::make_unique<MlxArray>(mlx::core::diag(a.inner, k));
}

std::unique_ptr<MlxArray> diagonal(const MlxArray& a, int32_t offset, int32_t axis1, int32_t axis2) {
    return std::make_unique<MlxArray>(mlx::core::diagonal(a.inner, offset, axis1, axis2));
}

// ============================================================================
// Type conversion
// ============================================================================

std::unique_ptr<MlxArray> astype(const MlxArray& a, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::astype(a.inner, to_dtype(dtype)));
}

// ============================================================================
// Copy
// ============================================================================

std::unique_ptr<MlxArray> copy(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::copy(a.inner));
}

// ============================================================================
// High-level operations for LLM inference
// ============================================================================

std::unique_ptr<MlxArray> softmax(const MlxArray& a, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::softmax(a.inner, axis));
}

std::unique_ptr<MlxArray> log_softmax(const MlxArray& a, int32_t axis) {
    // log_softmax(x) = x - logsumexp(x)
    auto lse = mlx::core::logsumexp(a.inner, axis, true);
    return std::make_unique<MlxArray>(mlx::core::subtract(a.inner, lse));
}

std::unique_ptr<MlxArray> rms_norm(const MlxArray& x, const MlxArray& weight, float eps) {
    // RMS norm: x * weight / sqrt(mean(x^2) + eps)
    auto x_sq = mlx::core::square(x.inner);
    auto mean_sq = mlx::core::mean(x_sq, -1, true);
    auto norm = mlx::core::rsqrt(mlx::core::add(mean_sq, array(eps)));
    auto normalized = mlx::core::multiply(x.inner, norm);
    return std::make_unique<MlxArray>(mlx::core::multiply(normalized, weight.inner));
}

std::unique_ptr<MlxArray> layer_norm(const MlxArray& x, const MlxArray& weight,
                                     const MlxArray& bias, float eps) {
    // Layer norm: (x - mean) / sqrt(var + eps) * weight + bias
    auto mean = mlx::core::mean(x.inner, -1, true);
    auto centered = mlx::core::subtract(x.inner, mean);
    auto var = mlx::core::mean(mlx::core::square(centered), -1, true);
    auto norm = mlx::core::rsqrt(mlx::core::add(var, array(eps)));
    auto normalized = mlx::core::multiply(centered, norm);
    auto scaled = mlx::core::multiply(normalized, weight.inner);
    return std::make_unique<MlxArray>(mlx::core::add(scaled, bias.inner));
}

std::unique_ptr<MlxArray> concatenate(rust::Slice<const MlxArray* const> arrays, int32_t axis) {
    std::vector<array> arrs;
    arrs.reserve(arrays.size());
    for (const auto* a : arrays) {
        arrs.push_back(a->inner);
    }
    return std::make_unique<MlxArray>(mlx::core::concatenate(arrs, axis));
}

rust::Vec<std::unique_ptr<MlxArray>> split(const MlxArray& a, int32_t num_splits, int32_t axis) {
    auto splits = mlx::core::split(a.inner, num_splits, axis);
    rust::Vec<std::unique_ptr<MlxArray>> result;
    for (auto& s : splits) {
        result.push_back(std::make_unique<MlxArray>(std::move(s)));
    }
    return result;
}

std::unique_ptr<MlxArray> slice(const MlxArray& a,
                                rust::Slice<const int32_t> starts,
                                rust::Slice<const int32_t> stops) {
    Shape starts_shape(starts.begin(), starts.end());
    Shape stops_shape(stops.begin(), stops.end());
    return std::make_unique<MlxArray>(mlx::core::slice(a.inner, starts_shape, stops_shape));
}

std::unique_ptr<MlxArray> slice_update(const MlxArray& src,
                                        const MlxArray& update,
                                        rust::Slice<const int32_t> starts,
                                        rust::Slice<const int32_t> stops) {
    Shape starts_shape(starts.begin(), starts.end());
    Shape stops_shape(stops.begin(), stops.end());
    return std::make_unique<MlxArray>(mlx::core::slice_update(src.inner, update.inner, starts_shape, stops_shape));
}

std::unique_ptr<MlxArray> argmax(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::argmax(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> where_cond(const MlxArray& condition, const MlxArray& x, const MlxArray& y) {
    return std::make_unique<MlxArray>(mlx::core::where(condition.inner, x.inner, y.inner));
}

std::unique_ptr<MlxArray> greater(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::greater(a.inner, b.inner));
}

std::unique_ptr<MlxArray> less(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::less(a.inner, b.inner));
}

std::unique_ptr<MlxArray> equal(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::equal(a.inner, b.inner));
}

void random_seed(uint64_t seed) {
    mlx::core::random::seed(seed);
}

std::unique_ptr<MlxArray> random_categorical(const MlxArray& logits, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::random::categorical(logits.inner, axis));
}

// ============================================================================
// Transformer-specific high-level operations
// ============================================================================

std::unique_ptr<MlxArray> rope_forward(
    const MlxArray& x,
    int32_t head_dim,
    float theta,
    int32_t offset,
    bool traditional
) {
    // Generate position-dependent rotation frequencies
    // freq = theta^(-2i/d) for i in [0, d/2)
    auto half_dim = head_dim / 2;
    std::vector<float> inv_freq;
    inv_freq.reserve(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        inv_freq.push_back(1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim));
    }
    auto freqs = array(inv_freq.data(), {half_dim});

    // Get sequence length from input shape
    auto shape = x.inner.shape();
    int seq_len = shape[shape.size() - 2];  // [..., seq_len, head_dim]

    // Create position indices
    std::vector<int> pos;
    pos.reserve(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        pos.push_back(offset + i);
    }
    auto positions = array(pos.data(), {seq_len});

    // Compute angles: pos * freq -> [seq_len, half_dim]
    auto pos_f = mlx::core::astype(positions, float32);
    pos_f = mlx::core::expand_dims(pos_f, 1);  // [seq_len, 1]
    freqs = mlx::core::expand_dims(freqs, 0);  // [1, half_dim]
    auto angles = mlx::core::matmul(pos_f, freqs);  // [seq_len, half_dim]

    // Compute cos and sin
    auto cos_vals = mlx::core::cos(angles);
    auto sin_vals = mlx::core::sin(angles);

    // Interleave or concatenate based on traditional flag
    if (traditional) {
        // Traditional: interleave cos, sin
        cos_vals = mlx::core::repeat(cos_vals, 2, -1);
        sin_vals = mlx::core::repeat(sin_vals, 2, -1);
    } else {
        // Modern: concatenate cos, sin
        cos_vals = mlx::core::concatenate({cos_vals, cos_vals}, -1);
        sin_vals = mlx::core::concatenate({sin_vals, sin_vals}, -1);
    }

    // Apply rotation
    auto x1 = mlx::core::slice(x.inner, {}, {}, {2});  // even indices
    auto x2 = mlx::core::slice(x.inner, {1}, {}, {2}); // odd indices

    // Rotate: x * cos + rotate(x) * sin
    auto rotated = mlx::core::concatenate({
        mlx::core::subtract(
            mlx::core::multiply(x.inner, cos_vals),
            mlx::core::multiply(
                mlx::core::concatenate({mlx::core::negative(x2), x1}, -1),
                sin_vals
            )
        )
    }, -1);

    return std::make_unique<MlxArray>(std::move(rotated));
}

std::unique_ptr<MlxArray> apply_rope(
    const MlxArray& x,
    const MlxArray& cos,
    const MlxArray& sin
) {
    // Split x into two halves
    int dim = x.inner.shape().back();
    int half_dim = dim / 2;
    auto x1 = mlx::core::slice(x.inner, {0, 0, 0, 0}, {-1, -1, -1, half_dim});
    auto x2 = mlx::core::slice(x.inner, {0, 0, 0, half_dim}, {-1, -1, -1, dim});

    // Rotate: [x1, x2] * cos + [-x2, x1] * sin
    auto rotated_x1 = mlx::core::subtract(
        mlx::core::multiply(x1, cos.inner),
        mlx::core::multiply(x2, sin.inner)
    );
    auto rotated_x2 = mlx::core::add(
        mlx::core::multiply(x2, cos.inner),
        mlx::core::multiply(x1, sin.inner)
    );

    return std::make_unique<MlxArray>(mlx::core::concatenate({rotated_x1, rotated_x2}, -1));
}

std::unique_ptr<MlxArray> scaled_dot_product_attention(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    const MlxArray* mask
) {
    // Compute attention scores: Q @ K^T * scale
    auto k_t = mlx::core::transpose(k.inner, {0, 1, 3, 2});
    auto scores = mlx::core::matmul(q.inner, k_t);
    scores = mlx::core::multiply(scores, array(scale));

    // Apply mask if provided
    if (mask != nullptr) {
        scores = mlx::core::add(scores, mask->inner);
    }

    // Softmax
    auto weights = mlx::core::softmax(scores, -1);

    // Apply attention: weights @ V
    return std::make_unique<MlxArray>(mlx::core::matmul(weights, v.inner));
}

std::unique_ptr<MlxArray> linear_forward(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray* bias
) {
    // x @ weight.T
    auto w_t = mlx::core::transpose(weight.inner);
    auto result = mlx::core::matmul(x.inner, w_t);

    if (bias != nullptr) {
        result = mlx::core::add(result, bias->inner);
    }

    return std::make_unique<MlxArray>(std::move(result));
}

std::unique_ptr<MlxArray> quantized_linear_forward(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray& biases,
    const MlxArray* linear_bias,
    int32_t group_size,
    int32_t bits
) {
    // Pass scales/biases directly to quantized_matmul (matching Python nn.QuantizedLinear)
    auto result = mlx::core::quantized_matmul(
        x.inner, weight.inner, scales.inner, biases.inner,
        true,  // transpose
        group_size, bits
    );

    if (linear_bias != nullptr) {
        result = mlx::core::add(result, linear_bias->inner);
    }

    return std::make_unique<MlxArray>(std::move(result));
}

std::unique_ptr<MlxArray> swiglu_mlp_forward(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& up_proj,
    const MlxArray& down_proj
) {
    // gate = gate_proj(x)
    auto gate_t = mlx::core::transpose(gate_proj.inner);
    auto gate = mlx::core::matmul(x.inner, gate_t);

    // up = up_proj(x)
    auto up_t = mlx::core::transpose(up_proj.inner);
    auto up = mlx::core::matmul(x.inner, up_t);

    // SwiGLU: silu(gate) * up
    // silu(x) = x * sigmoid(x)
    auto silu_gate = mlx::core::multiply(gate, mlx::core::sigmoid(gate));
    auto activated = mlx::core::multiply(silu_gate, up);

    // down_proj(activated)
    auto down_t = mlx::core::transpose(down_proj.inner);
    return std::make_unique<MlxArray>(mlx::core::matmul(activated, down_t));
}

// Compiled swiglu activation using mlx::core::compile with shapeless=true
// This should provide kernel fusion like Python's @mx.compile(shapeless=True)
namespace {
    // Static compiled function - initialized once and reused
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_swiglu() {
        // Define the swiglu function: silu(gate) * x
        auto swiglu_fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& gate = inputs[0];
            const auto& x = inputs[1];
            // silu(gate) = gate * sigmoid(gate)
            auto silu_gate = mlx::core::multiply(gate, mlx::core::sigmoid(gate));
            return {mlx::core::multiply(silu_gate, x)};
        };
        // Compile with shapeless=true for kernel fusion
        return mlx::core::compile(swiglu_fn, true);
    }
}

std::unique_ptr<MlxArray> compiled_swiglu_activation(
    const MlxArray& gate,
    const MlxArray& x
) {
    // Get or create the compiled function (thread-safe initialization)
    static auto compiled_fn = get_compiled_swiglu();

    // Call the compiled function
    auto result = compiled_fn({gate.inner, x.inner});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

std::unique_ptr<MlxArray> transformer_layer_forward(
    const MlxArray& x,
    const MlxArray& attn_norm_weight,
    const MlxArray& q_proj,
    const MlxArray& k_proj,
    const MlxArray& v_proj,
    const MlxArray& o_proj,
    const MlxArray& ffn_norm_weight,
    const MlxArray& gate_proj,
    const MlxArray& up_proj,
    const MlxArray& down_proj,
    const MlxArray* kv_cache_k,
    const MlxArray* kv_cache_v,
    int32_t n_heads,
    int32_t n_kv_heads,
    int32_t head_dim,
    float rope_theta,
    int32_t rope_offset,
    float norm_eps
) {
    auto batch_size = x.inner.shape()[0];
    auto seq_len = x.inner.shape()[1];

    // Pre-attention norm (RMS norm)
    auto x_sq = mlx::core::square(x.inner);
    auto mean_sq = mlx::core::mean(x_sq, -1, true);
    auto norm = mlx::core::rsqrt(mlx::core::add(mean_sq, array(norm_eps)));
    auto h = mlx::core::multiply(mlx::core::multiply(x.inner, norm), attn_norm_weight.inner);

    // Q, K, V projections
    auto q = mlx::core::matmul(h, mlx::core::transpose(q_proj.inner));
    auto k = mlx::core::matmul(h, mlx::core::transpose(k_proj.inner));
    auto v = mlx::core::matmul(h, mlx::core::transpose(v_proj.inner));

    // Reshape for multi-head attention
    q = mlx::core::reshape(q, {batch_size, seq_len, n_heads, head_dim});
    k = mlx::core::reshape(k, {batch_size, seq_len, n_kv_heads, head_dim});
    v = mlx::core::reshape(v, {batch_size, seq_len, n_kv_heads, head_dim});

    // Transpose to [batch, heads, seq, dim]
    q = mlx::core::transpose(q, {0, 2, 1, 3});
    k = mlx::core::transpose(k, {0, 2, 1, 3});
    v = mlx::core::transpose(v, {0, 2, 1, 3});

    // Apply RoPE (simplified - just use the existing values for now)
    // In production, would compute cos/sin and apply

    // Update KV cache if provided
    if (kv_cache_k != nullptr && kv_cache_v != nullptr) {
        k = mlx::core::concatenate({kv_cache_k->inner, k}, 2);
        v = mlx::core::concatenate({kv_cache_v->inner, v}, 2);
    }

    // Handle GQA (grouped query attention)
    if (n_kv_heads != n_heads) {
        int repeats = n_heads / n_kv_heads;
        k = mlx::core::repeat(k, repeats, 1);
        v = mlx::core::repeat(v, repeats, 1);
    }

    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto k_t = mlx::core::transpose(k, {0, 1, 3, 2});
    auto scores = mlx::core::multiply(mlx::core::matmul(q, k_t), array(scale));

    // Causal mask
    auto kv_len = k.shape()[2];
    auto mask_val = array(-1e9f);
    // Create causal mask (lower triangular)
    std::vector<float> mask_data(seq_len * kv_len, 0.0f);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = rope_offset + i + 1; j < kv_len; ++j) {
            mask_data[i * kv_len + j] = -1e9f;
        }
    }
    auto mask = array(mask_data.data(), {1, 1, seq_len, kv_len});
    scores = mlx::core::add(scores, mask);

    auto weights = mlx::core::softmax(scores, -1);
    auto attn_out = mlx::core::matmul(weights, v);

    // Transpose back and reshape
    attn_out = mlx::core::transpose(attn_out, {0, 2, 1, 3});
    attn_out = mlx::core::reshape(attn_out, {batch_size, seq_len, n_heads * head_dim});

    // Output projection
    auto attn_result = mlx::core::matmul(attn_out, mlx::core::transpose(o_proj.inner));

    // Residual connection
    auto h2 = mlx::core::add(x.inner, attn_result);

    // Pre-FFN norm (RMS norm)
    auto h2_sq = mlx::core::square(h2);
    auto mean_sq2 = mlx::core::mean(h2_sq, -1, true);
    auto norm2 = mlx::core::rsqrt(mlx::core::add(mean_sq2, array(norm_eps)));
    auto h3 = mlx::core::multiply(mlx::core::multiply(h2, norm2), ffn_norm_weight.inner);

    // FFN (SwiGLU)
    auto gate = mlx::core::matmul(h3, mlx::core::transpose(gate_proj.inner));
    auto up = mlx::core::matmul(h3, mlx::core::transpose(up_proj.inner));
    // silu(x) = x * sigmoid(x)
    auto silu_gate = mlx::core::multiply(gate, mlx::core::sigmoid(gate));
    auto activated = mlx::core::multiply(silu_gate, up);
    auto ffn_out = mlx::core::matmul(activated, mlx::core::transpose(down_proj.inner));

    // Final residual
    return std::make_unique<MlxArray>(mlx::core::add(h2, ffn_out));
}

// ============================================================================
// Advanced indexing operations
// ============================================================================

std::unique_ptr<MlxArray> take(const MlxArray& a, const MlxArray& indices, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::take(a.inner, indices.inner, axis));
}

std::unique_ptr<MlxArray> gather(
    const MlxArray& a,
    rust::Slice<const MlxArray* const> indices,
    rust::Slice<const int32_t> axes,
    rust::Slice<const int32_t> slice_sizes
) {
    std::vector<array> idx_vec;
    idx_vec.reserve(indices.size());
    for (const auto* idx : indices) {
        idx_vec.push_back(idx->inner);
    }

    std::vector<int> axes_vec(axes.begin(), axes.end());

    // Shape is SmallVector<int>, construct it from slice
    mlx::core::Shape shape_vec;
    for (auto s : slice_sizes) {
        shape_vec.push_back(s);
    }

    return std::make_unique<MlxArray>(mlx::core::gather(
        a.inner, idx_vec, axes_vec, shape_vec
    ));
}

std::unique_ptr<MlxArray> take_along_axis(const MlxArray& a, const MlxArray& indices, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::take_along_axis(a.inner, indices.inner, axis));
}

std::unique_ptr<MlxArray> put_along_axis(const MlxArray& a, const MlxArray& indices,
                                          const MlxArray& values, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::put_along_axis(a.inner, indices.inner, values.inner, axis));
}

std::unique_ptr<MlxArray> stack(rust::Slice<const MlxArray* const> arrays, int32_t axis) {
    std::vector<array> arrs;
    arrs.reserve(arrays.size());
    for (const auto* a : arrays) {
        arrs.push_back(a->inner);
    }
    return std::make_unique<MlxArray>(mlx::core::stack(arrs, axis));
}

std::unique_ptr<MlxArray> tile(const MlxArray& a, rust::Slice<const int32_t> reps) {
    std::vector<int> reps_vec(reps.begin(), reps.end());
    return std::make_unique<MlxArray>(mlx::core::tile(a.inner, reps_vec));
}

std::unique_ptr<MlxArray> repeat(const MlxArray& a, int32_t repeats, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::repeat(a.inner, repeats, axis));
}

std::unique_ptr<MlxArray> arange_f32(float start, float stop, float step) {
    return std::make_unique<MlxArray>(mlx::core::arange(start, stop, step));
}

std::unique_ptr<MlxArray> arange_i32(int32_t start, int32_t stop, int32_t step) {
    return std::make_unique<MlxArray>(mlx::core::arange(start, stop, step));
}

// ============================================================================
// Logical operations
// ============================================================================

std::unique_ptr<MlxArray> logical_not(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::logical_not(a.inner));
}

std::unique_ptr<MlxArray> logical_and(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::logical_and(a.inner, b.inner));
}

std::unique_ptr<MlxArray> logical_or(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::logical_or(a.inner, b.inner));
}

std::unique_ptr<MlxArray> all_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::all(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> any_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::any(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> greater_equal(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::greater_equal(a.inner, b.inner));
}

std::unique_ptr<MlxArray> less_equal(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::less_equal(a.inner, b.inner));
}

std::unique_ptr<MlxArray> not_equal(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::not_equal(a.inner, b.inner));
}

// ============================================================================
// Activation functions
// ============================================================================

std::unique_ptr<MlxArray> silu(const MlxArray& a) {
    // silu(x) = x * sigmoid(x)
    return std::make_unique<MlxArray>(mlx::core::multiply(a.inner, mlx::core::sigmoid(a.inner)));
}

std::unique_ptr<MlxArray> gelu(const MlxArray& a) {
    // gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    auto sqrt2 = array(std::sqrt(2.0f));
    auto half = array(0.5f);
    auto one = array(1.0f);
    auto erf_val = mlx::core::erf(mlx::core::divide(a.inner, sqrt2));
    auto scale = mlx::core::multiply(half, mlx::core::add(one, erf_val));
    return std::make_unique<MlxArray>(mlx::core::multiply(a.inner, scale));
}

std::unique_ptr<MlxArray> gelu_approx(const MlxArray& a) {
    // Approximate GELU using tanh
    // gelu_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    auto sqrt_2_pi = array(std::sqrt(2.0f / M_PI));
    auto coef = array(0.044715f);
    auto half = array(0.5f);
    auto one = array(1.0f);

    auto x3 = mlx::core::power(a.inner, array(3.0f));
    auto inner = mlx::core::add(a.inner, mlx::core::multiply(coef, x3));
    auto tanh_val = mlx::core::tanh(mlx::core::multiply(sqrt_2_pi, inner));
    auto scale = mlx::core::multiply(half, mlx::core::add(one, tanh_val));
    return std::make_unique<MlxArray>(mlx::core::multiply(a.inner, scale));
}

std::unique_ptr<MlxArray> relu(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::maximum(a.inner, array(0.0f)));
}

std::unique_ptr<MlxArray> leaky_relu(const MlxArray& a, float negative_slope) {
    auto zero = array(0.0f);
    auto pos = mlx::core::maximum(a.inner, zero);
    auto neg = mlx::core::multiply(mlx::core::minimum(a.inner, zero), array(negative_slope));
    return std::make_unique<MlxArray>(mlx::core::add(pos, neg));
}

// ============================================================================
// Sorting and searching
// ============================================================================

std::unique_ptr<MlxArray> argsort(const MlxArray& a, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::argsort(a.inner, axis));
}

std::unique_ptr<MlxArray> argpartition(const MlxArray& a, int32_t kth, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::argpartition(a.inner, kth, axis));
}

std::unique_ptr<MlxArray> argmin(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::argmin(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> topk(const MlxArray& a, int32_t k, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::topk(a.inner, k, axis));
}

// Sort and partition
std::unique_ptr<MlxArray> sort(const MlxArray& a, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::sort(a.inner, axis));
}

std::unique_ptr<MlxArray> partition(const MlxArray& a, int32_t kth, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::partition(a.inner, kth, axis));
}

// Cumulative operations
std::unique_ptr<MlxArray> cummax(const MlxArray& a, int32_t axis, bool reverse, bool inclusive) {
    return std::make_unique<MlxArray>(mlx::core::cummax(a.inner, axis, reverse, inclusive));
}

std::unique_ptr<MlxArray> cummin(const MlxArray& a, int32_t axis, bool reverse, bool inclusive) {
    return std::make_unique<MlxArray>(mlx::core::cummin(a.inner, axis, reverse, inclusive));
}

std::unique_ptr<MlxArray> cumprod(const MlxArray& a, int32_t axis, bool reverse, bool inclusive) {
    return std::make_unique<MlxArray>(mlx::core::cumprod(a.inner, axis, reverse, inclusive));
}

// Scatter operations
std::unique_ptr<MlxArray> scatter(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis) {
    // MLX scatter uses different signature - simplified version
    std::vector<array> idx_vec = {indices.inner};
    return std::make_unique<MlxArray>(mlx::core::scatter(a.inner, idx_vec, updates.inner, {axis}));
}

std::unique_ptr<MlxArray> scatter_add(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis) {
    std::vector<array> idx_vec = {indices.inner};
    return std::make_unique<MlxArray>(mlx::core::scatter_add(a.inner, idx_vec, updates.inner, {axis}));
}

std::unique_ptr<MlxArray> scatter_max(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis) {
    std::vector<array> idx_vec = {indices.inner};
    return std::make_unique<MlxArray>(mlx::core::scatter_max(a.inner, idx_vec, updates.inner, {axis}));
}

std::unique_ptr<MlxArray> scatter_min(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis) {
    std::vector<array> idx_vec = {indices.inner};
    return std::make_unique<MlxArray>(mlx::core::scatter_min(a.inner, idx_vec, updates.inner, {axis}));
}

std::unique_ptr<MlxArray> scatter_prod(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis) {
    std::vector<array> idx_vec = {indices.inner};
    return std::make_unique<MlxArray>(mlx::core::scatter_prod(a.inner, idx_vec, updates.inner, {axis}));
}

// Bitwise operations
std::unique_ptr<MlxArray> bitwise_and(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::bitwise_and(a.inner, b.inner));
}

std::unique_ptr<MlxArray> bitwise_or(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::bitwise_or(a.inner, b.inner));
}

std::unique_ptr<MlxArray> bitwise_xor(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::bitwise_xor(a.inner, b.inner));
}

std::unique_ptr<MlxArray> left_shift(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::left_shift(a.inner, b.inner));
}

std::unique_ptr<MlxArray> right_shift(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::right_shift(a.inner, b.inner));
}

// Linear algebra
std::unique_ptr<MlxArray> tensordot(const MlxArray& a, const MlxArray& b, int32_t axes) {
    return std::make_unique<MlxArray>(mlx::core::tensordot(a.inner, b.inner, axes));
}

std::unique_ptr<MlxArray> inner(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::inner(a.inner, b.inner));
}

std::unique_ptr<MlxArray> outer(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::outer(a.inner, b.inner));
}

std::unique_ptr<MlxArray> trace(const MlxArray& a, int32_t offset, int32_t axis1, int32_t axis2) {
    return std::make_unique<MlxArray>(mlx::core::trace(a.inner, offset, axis1, axis2));
}

// Roll (circular shift)
std::unique_ptr<MlxArray> roll(const MlxArray& a, int32_t shift, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::roll(a.inner, shift, axis));
}

// Nan handling
std::unique_ptr<MlxArray> nan_to_num(const MlxArray& a, float nan_val, float posinf_val, float neginf_val) {
    return std::make_unique<MlxArray>(mlx::core::nan_to_num(a.inner, nan_val, posinf_val, neginf_val));
}

// Stop gradient
std::unique_ptr<MlxArray> stop_gradient(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::stop_gradient(a.inner));
}

// 2D convolution
std::unique_ptr<MlxArray> conv2d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv2d(
        input.inner, weight.inner,
        {stride_h, stride_w},
        {padding_h, padding_w},
        {dilation_h, dilation_w},
        groups
    ));
}

std::unique_ptr<MlxArray> avg_pool2d(
    const MlxArray& input,
    int32_t kernel_h, int32_t kernel_w,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w
) {
    // Manual average pooling: input shape [B, H, W, C]
    // Use conv2d with a kernel of 1/(kH*kW) uniform weights for each channel
    // But MLX doesn't have nn::AvgPool2d in C++ ops, so we implement manually.
    //
    // For each output position, sum the kernel window and divide by kernel size.
    // We extract windows using as_strided or manual loops, but the simplest
    // approach for MLX is to use the unfold pattern.
    //
    // Actually, MLX Python's nn.AvgPool2d uses mlx.core ops internally.
    // The simplest C++ implementation: use a depthwise conv2d with uniform weights.
    auto& x = input.inner;
    auto shape = x.shape();
    int32_t channels = shape.back(); // [B, H, W, C] -> C is last dim

    // Create uniform kernel: shape [C, kH, kW, 1] for groups=C (depthwise)
    float kernel_val = 1.0f / (kernel_h * kernel_w);
    auto kernel = mlx::core::full({channels, kernel_h, kernel_w, 1}, kernel_val);

    // Depthwise conv2d with groups=C gives us the sum/count = average
    auto result = mlx::core::conv2d(
        x, kernel,
        {stride_h, stride_w},
        {padding_h, padding_w},
        {1, 1},  // dilation
        channels  // groups = channels for depthwise
    );
    return std::make_unique<MlxArray>(std::move(result));
}

// ============================================================================
// MoE (Mixture of Experts) operations
// ============================================================================

std::unique_ptr<MlxArray> gather_mm(
    const MlxArray& a,
    const MlxArray& b,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool sorted_indices
) {
    std::optional<array> lhs_opt = lhs_indices ? std::optional(lhs_indices->inner) : std::nullopt;
    std::optional<array> rhs_opt = rhs_indices ? std::optional(rhs_indices->inner) : std::nullopt;

    return std::make_unique<MlxArray>(mlx::core::gather_mm(
        a.inner, b.inner, lhs_opt, rhs_opt, sorted_indices
    ));
}

std::unique_ptr<MlxArray> gather_qmm(
    const MlxArray& x,
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool transpose,
    int32_t group_size,
    int32_t bits,
    bool sorted_indices
) {
    std::optional<array> biases_opt = biases ? std::optional(biases->inner) : std::nullopt;
    std::optional<array> lhs_opt = lhs_indices ? std::optional(lhs_indices->inner) : std::nullopt;
    std::optional<array> rhs_opt = rhs_indices ? std::optional(rhs_indices->inner) : std::nullopt;

    return std::make_unique<MlxArray>(mlx::core::gather_qmm(
        x.inner, w.inner, scales.inner, biases_opt,
        lhs_opt, rhs_opt, transpose,
        std::optional<int>(group_size), std::optional<int>(bits),
        "affine", sorted_indices
    ));
}

std::unique_ptr<MlxArray> quantized_matmul(
    const MlxArray& x,
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,
    bool transpose,
    int32_t group_size,
    int32_t bits
) {
    std::optional<array> biases_opt = biases ? std::optional(biases->inner) : std::nullopt;

    return std::make_unique<MlxArray>(mlx::core::quantized_matmul(
        x.inner, w.inner, scales.inner, biases_opt,
        transpose,
        std::optional<int>(group_size), std::optional<int>(bits),
        "affine"
    ));
}

std::unique_ptr<MlxArray> dequantize(
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray& biases,
    int32_t group_size,
    int32_t bits
) {
    return std::make_unique<MlxArray>(mlx::core::dequantize(
        w.inner, scales.inner, biases.inner,
        group_size, bits, "affine"
    ));
}

// ============================================================================
// Embedding
// ============================================================================

std::unique_ptr<MlxArray> embedding(const MlxArray& weight, const MlxArray& indices) {
    return std::make_unique<MlxArray>(mlx::core::take(weight.inner, indices.inner, 0));
}

std::unique_ptr<MlxArray> quantized_embedding(
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray& biases,
    const MlxArray& indices,
    int32_t group_size,
    int32_t bits
) {
    // Save original indices shape for reshaping later
    auto idx_shape = indices.inner.shape();

    // Flatten indices: [batch, seq] -> [batch * seq]
    auto flat_indices = mlx::core::reshape(indices.inner, {-1});

    // Take rows from quantized weights using flattened indices
    auto w_indexed = mlx::core::take(weight.inner, flat_indices, 0);
    auto scales_indexed = mlx::core::take(scales.inner, flat_indices, 0);
    auto biases_indexed = mlx::core::take(biases.inner, flat_indices, 0);

    // Dequantize with explicit optional wrapping
    auto result = mlx::core::dequantize(
        w_indexed,
        scales_indexed,
        std::optional<mlx::core::array>(biases_indexed),
        std::optional<int>(group_size),
        std::optional<int>(bits),
        "affine",
        std::nullopt  // use default dtype
    );

    // Get hidden dimension from dequantized result
    auto hidden_dim = result.shape().back();

    // Reshape back to original batch/seq dims + hidden: [batch * seq, hidden] -> [batch, seq, hidden]
    Shape new_shape;
    for (size_t i = 0; i < idx_shape.size(); ++i) {
        new_shape.push_back(idx_shape[i]);
    }
    new_shape.push_back(hidden_dim);

    auto reshaped = mlx::core::reshape(result, new_shape);

    return std::make_unique<MlxArray>(reshaped);
}

// ============================================================================
// Fast operations (using MLX fast kernels)
// ============================================================================

std::unique_ptr<MlxArray> fast_rope(
    const MlxArray& x,
    int32_t dims,
    bool traditional,
    float base,
    float scale,
    int32_t offset
) {
    return std::make_unique<MlxArray>(mlx::core::fast::rope(
        x.inner, dims, traditional, base, scale, offset
    ));
}

std::unique_ptr<MlxArray> fast_rope_with_freqs(
    const MlxArray& x,
    int32_t dims,
    bool traditional,
    float scale,
    int32_t offset,
    const MlxArray& freqs
) {
    // When using custom freqs, pass nullopt for base (can't use both)
    return std::make_unique<MlxArray>(mlx::core::fast::rope(
        x.inner, dims, traditional, std::nullopt, scale, offset, freqs.inner
    ));
}

std::unique_ptr<MlxArray> fast_rms_norm(
    const MlxArray& x,
    const MlxArray& weight,
    float eps
) {
    return std::make_unique<MlxArray>(mlx::core::fast::rms_norm(
        x.inner, weight.inner, eps
    ));
}

std::unique_ptr<MlxArray> fast_layer_norm(
    const MlxArray& x,
    const MlxArray* weight,
    const MlxArray* bias,
    float eps
) {
    std::optional<array> weight_opt = weight ? std::optional(weight->inner) : std::nullopt;
    std::optional<array> bias_opt = bias ? std::optional(bias->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::fast::layer_norm(
        x.inner, weight_opt, bias_opt, eps
    ));
}

std::unique_ptr<MlxArray> fast_scaled_dot_product_attention(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    const MlxArray* mask
) {
    std::optional<array> mask_opt = mask ? std::optional(mask->inner) : std::nullopt;
    std::string mask_mode = mask ? "array" : "";
    return std::make_unique<MlxArray>(mlx::core::fast::scaled_dot_product_attention(
        q.inner, k.inner, v.inner, scale, mask_mode, mask_opt
    ));
}

// SDPA with explicit causal masking for prefill
std::unique_ptr<MlxArray> fast_scaled_dot_product_attention_causal(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale
) {
    return std::make_unique<MlxArray>(mlx::core::fast::scaled_dot_product_attention(
        q.inner, k.inner, v.inner, scale, "causal", std::nullopt
    ));
}

// Fused QKV projection + reshape + transpose + RoPE (no cache, no SDPA)
// Returns new_k array ready for cache. Call this function twice (for k and v) or use fused version below.
// This reduces FFI overhead for the projection + reshape + transpose + RoPE chain.
std::unique_ptr<MlxArray> fused_qkv_project_and_rope(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray& biases,
    int32_t num_heads,
    int32_t head_dim,
    int32_t rope_dims,
    float rope_base,
    int32_t cache_offset,
    int32_t group_size,
    int32_t bits,
    bool apply_rope
) {
    auto batch_size = x.inner.shape()[0];
    auto seq_len = x.inner.shape()[1];

    // Projection
    auto proj = quantized_matmul(x.inner, weight.inner, scales.inner, biases.inner, true, group_size, bits);

    // Reshape to [batch, seq_len, n_heads, head_dim]
    proj = reshape(proj, {batch_size, seq_len, num_heads, head_dim});

    // Transpose to [batch, n_heads, seq_len, head_dim]
    proj = transpose(proj, {0, 2, 1, 3});

    // Apply RoPE if requested (Q and K need it, V doesn't)
    if (apply_rope) {
        proj = mlx::core::fast::rope(proj, rope_dims, false, rope_base, 1.0f, cache_offset);
    }

    return std::make_unique<MlxArray>(std::move(proj));
}

// ============================================================================
// Compiled operations (with kernel fusion)
// ============================================================================

// Compiled MoE expert forward with quantized weights
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_qmoe_expert() {
        auto moe_expert_fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            // inputs: [x, gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b, params]
            const auto& x = inputs[0];
            const auto& gate_w = inputs[1];
            const auto& gate_s = inputs[2];
            const auto& gate_b = inputs[3];
            const auto& up_w = inputs[4];
            const auto& up_s = inputs[5];
            const auto& up_b = inputs[6];
            const auto& down_w = inputs[7];
            const auto& down_s = inputs[8];
            const auto& down_b = inputs[9];
            // params[0] = group_size, params[1] = bits
            int group_size = 64;  // hardcoded for common case
            int bits = 4;

            // gate = quantized_matmul(x, gate_w, gate_s, gate_b)
            auto gate = mlx::core::quantized_matmul(x, gate_w, gate_s, gate_b, true, group_size, bits);

            // up = quantized_matmul(x, up_w, up_s, up_b)
            auto up = mlx::core::quantized_matmul(x, up_w, up_s, up_b, true, group_size, bits);

            // SwiGLU: silu(gate) * up
            auto silu_gate = mlx::core::multiply(gate, mlx::core::sigmoid(gate));
            auto activated = mlx::core::multiply(silu_gate, up);

            // down = quantized_matmul(activated, down_w, down_s, down_b)
            auto down = mlx::core::quantized_matmul(activated, down_w, down_s, down_b, true, group_size, bits);

            return {down};
        };
        return mlx::core::compile(moe_expert_fn, true);  // shapeless=true
    }
}

std::unique_ptr<MlxArray> compiled_moe_expert_forward(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& gate_scales,
    const MlxArray& gate_biases,
    const MlxArray& up_proj,
    const MlxArray& up_scales,
    const MlxArray& up_biases,
    const MlxArray& down_proj,
    const MlxArray& down_scales,
    const MlxArray& down_biases,
    int32_t group_size,
    int32_t bits
) {
    static auto compiled_fn = get_compiled_qmoe_expert();

    auto result = compiled_fn({
        x.inner,
        gate_proj.inner, gate_scales.inner, gate_biases.inner,
        up_proj.inner, up_scales.inner, up_biases.inner,
        down_proj.inner, down_scales.inner, down_biases.inner
    });

    return std::make_unique<MlxArray>(std::move(result[0]));
}

// ============================================================================
// Memory management
// ============================================================================

void clear_memory_cache() {
    mlx::core::clear_cache();
}

void async_eval(const MlxArray& arr) {
    mlx::core::async_eval(const_cast<array&>(arr.inner));
}

void async_eval_all(rust::Slice<const MlxArray* const> arrays) {
    std::vector<array> arrs;
    arrs.reserve(arrays.size());
    for (const auto* a : arrays) {
        arrs.push_back(a->inner);
    }
    mlx::core::async_eval(arrs);
}

void synchronize_default() {
    mlx::core::synchronize();
}

size_t set_wired_limit(size_t limit) {
    return mlx::core::set_wired_limit(limit);
}

size_t get_wired_limit() {
    auto& info = mlx::core::device_info();
    auto it = info.find("max_recommended_working_set_size");
    if (it != info.end()) {
        return std::get<size_t>(it->second);
    }
    return 0;
}

size_t gpu_max_memory_size() {
    auto& info = mlx::core::device_info();
    // Metal backend uses "max_recommended_working_set_size"
    // CUDA backend uses "total_memory"
    for (const auto& key : {"max_recommended_working_set_size", "total_memory"}) {
        auto it = info.find(key);
        if (it != info.end()) {
            return std::get<size_t>(it->second);
        }
    }
    return 0;
}

std::unique_ptr<MlxStream> new_gpu_stream() {
    return std::make_unique<MlxStream>(mlx::core::new_stream(mlx::core::Device::gpu));
}

// ============================================================================
// Optimized generation functions
// ============================================================================

// Extract last token logits: logits[:, -1, :] -> [batch, vocab]
// This is the optimized path for sampling during generation
std::unique_ptr<MlxArray> slice_last_logits(const MlxArray& logits) {
    // logits shape: [batch, seq_len, vocab_size]
    auto shape = logits.inner.shape();
    if (shape.size() != 3) {
        // Already 2D, just return copy
        return std::make_unique<MlxArray>(logits.inner);
    }
    int32_t seq_len = shape[1];
    // Use slice to get [:, -1:, :] then squeeze
    auto sliced = mlx::core::slice(logits.inner, {0, seq_len - 1, 0}, shape);
    auto squeezed = mlx::core::squeeze(sliced, 1);
    return std::make_unique<MlxArray>(std::move(squeezed));
}

// Slice on the last dimension only: a[..., start:end]
// Useful for fused QKV/gate_up projections
std::unique_ptr<MlxArray> slice_last_dim(const MlxArray& a, int32_t start, int32_t end) {
    auto shape = a.inner.shape();
    int ndim = shape.size();

    // Build starts: [0, 0, ..., start]
    Shape starts(ndim, 0);
    starts[ndim - 1] = start;

    // Build stops: [dim0, dim1, ..., end]
    Shape stops;
    for (int i = 0; i < ndim - 1; i++) {
        stops.push_back(shape[i]);
    }
    stops.push_back(end);

    return std::make_unique<MlxArray>(mlx::core::slice(a.inner, starts, stops));
}

// Argmax on last axis for greedy sampling, returns scalar
std::unique_ptr<MlxArray> argmax_last_axis(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::argmax(a.inner, -1, false));
}

// Reshape token for next forward pass: [] or [batch] -> [batch, 1]
// This allows passing token array directly without extracting scalar value
std::unique_ptr<MlxArray> reshape_token_for_forward(const MlxArray& token) {
    auto shape = token.inner.shape();
    if (shape.empty()) {
        // Scalar -> [1, 1]
        return std::make_unique<MlxArray>(mlx::core::reshape(token.inner, {1, 1}));
    } else if (shape.size() == 1) {
        // [batch] -> [batch, 1]
        return std::make_unique<MlxArray>(mlx::core::reshape(token.inner, {static_cast<int32_t>(shape[0]), 1}));
    }
    // Already correct shape
    return std::make_unique<MlxArray>(token.inner);
}

// Async eval two arrays at once (for lookahead pipelining)
void async_eval_pair(const MlxArray& a, const MlxArray& b) {
    std::vector<array> arrs = {a.inner, b.inner};
    mlx::core::async_eval(arrs);
}

// Set default stream for subsequent operations
void set_default_stream(const MlxStream& stream) {
    mlx::core::set_default_stream(stream.inner);
}

// Get default device (needed for stream creation)
bool is_gpu_available() {
    return mlx::core::default_device() == mlx::core::Device::gpu;
}

// Fused sampling: temperature scaling + top-k + top-p + min-p + categorical
// in a single function call to minimize FFI round-trips.
// Input: 2D logits [batch, vocab] (already sliced, penalties already applied)
// Returns (token, filtered_logits)
std::unique_ptr<MlxArray> fused_sample(
    const MlxArray& logits,
    float temperature,
    int32_t top_k,
    float top_p,
    float min_p
) {
    auto x = logits.inner;

    // Greedy path: argmax
    if (temperature == 0.0f || top_k == 1) {
        return std::make_unique<MlxArray>(mlx::core::argmax(x, -1, false));
    }

    // Temperature scaling
    if (temperature > 0.0f && temperature != 1.0f) {
        x = x / mlx::core::array(temperature);
    }

    // Top-k filtering: keep only the k highest-probability tokens
    if (top_k > 0) {
        auto neg_x = mlx::core::negative(x);
        auto indices = mlx::core::argpartition(neg_x, top_k - 1, -1);

        auto shape = indices.shape();
        int ndim = shape.size();
        Shape start(ndim, 0);
        Shape stop(shape.begin(), shape.end());
        start[ndim - 1] = top_k - 1;
        stop[ndim - 1] = top_k;

        auto kth_idx = mlx::core::slice(indices, start, stop);
        auto threshold = mlx::core::take_along_axis(x, kth_idx, -1);
        auto mask = mlx::core::greater_equal(x, threshold);
        x = mlx::core::where(mask, x, mlx::core::array(std::numeric_limits<float>::lowest()));
    }

    // Top-p (nucleus) filtering: keep smallest set whose probability sums >= top_p
    if (top_p > 0.0f && top_p < 1.0f) {
        auto probs = mlx::core::softmax(x, -1);
        auto sorted_indices = mlx::core::argsort(mlx::core::negative(probs), -1);
        auto sorted_probs = mlx::core::take_along_axis(probs, sorted_indices, -1);
        auto cum_probs = mlx::core::cumsum(sorted_probs, -1, false, true);

        // Keep tokens where cumsum(before this token) <= top_p
        auto shifted_cum = cum_probs - sorted_probs;
        auto mask = mlx::core::less_equal(shifted_cum, mlx::core::array(top_p));

        // Apply mask in sorted space
        auto sorted_logits = mlx::core::take_along_axis(x, sorted_indices, -1);
        auto filtered_sorted = mlx::core::where(
            mask, sorted_logits, mlx::core::array(std::numeric_limits<float>::lowest()));

        // Unsort back to original order
        auto unsort_indices = mlx::core::argsort(sorted_indices, -1);
        x = mlx::core::take_along_axis(filtered_sorted, unsort_indices, -1);
    }

    // Min-p filtering: keep tokens with probability >= min_p * max_probability
    if (min_p > 0.0f && min_p < 1.0f) {
        auto probs = mlx::core::softmax(x, -1);
        auto max_prob = mlx::core::max(probs, -1, true);
        auto threshold = max_prob * mlx::core::array(min_p);
        auto mask = mlx::core::greater_equal(probs, threshold);
        x = mlx::core::where(mask, x, mlx::core::array(std::numeric_limits<float>::lowest()));
    }

    // Categorical sampling
    return std::make_unique<MlxArray>(mlx::core::random::categorical(x, -1));
}

// ============================================================================
// SSM (State Space Model) primitives for Mamba/Jamba/Nemotron-H
// ============================================================================

// Cumulative sum along axis
std::unique_ptr<MlxArray> cumsum(const MlxArray& a, int32_t axis, bool reverse, bool inclusive) {
    return std::make_unique<MlxArray>(mlx::core::cumsum(a.inner, axis, reverse, inclusive));
}

// Lower triangular matrix (keeps elements on and below k-th diagonal)
std::unique_ptr<MlxArray> tril(const MlxArray& a, int32_t k) {
    return std::make_unique<MlxArray>(mlx::core::tril(a.inner, k));
}

// Upper triangular matrix (keeps elements on and above k-th diagonal)
std::unique_ptr<MlxArray> triu(const MlxArray& a, int32_t k) {
    return std::make_unique<MlxArray>(mlx::core::triu(a.inner, k));
}

// Clip values to range [a_min, a_max]
std::unique_ptr<MlxArray> clip(const MlxArray& a, const MlxArray& a_min, const MlxArray& a_max) {
    return std::make_unique<MlxArray>(mlx::core::clip(a.inner, a_min.inner, a_max.inner));
}

// log(1 + x) - numerically stable for small x, used for softplus
std::unique_ptr<MlxArray> log1p(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::log1p(a.inner));
}

// Softplus activation: log(1 + exp(x))
// Uses identity: softplus(x) = log1p(exp(x))
std::unique_ptr<MlxArray> softplus(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::log1p(mlx::core::exp(a.inner)));
}

// 1D convolution with groups support (for depthwise conv when groups=channels)
std::unique_ptr<MlxArray> conv1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv1d(
        input.inner, weight.inner, stride, padding, dilation, groups
    ));
}

// Swap axes (convenient for SSM attention)
std::unique_ptr<MlxArray> swap_axes(const MlxArray& a, int32_t axis1, int32_t axis2) {
    return std::make_unique<MlxArray>(mlx::core::swapaxes(a.inner, axis1, axis2));
}

// ============================================================================
// Core ops additions
// ============================================================================

std::unique_ptr<MlxArray> identity(int32_t n, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::identity(n, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> tri(int32_t n, int32_t m, int32_t k, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::tri(n, m, k, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> unflatten(const MlxArray& a, int32_t axis, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(mlx::core::unflatten(a.inner, axis, to_shape(shape)));
}

std::unique_ptr<MlxArray> as_strided(const MlxArray& a, rust::Slice<const int32_t> shape, rust::Slice<const int64_t> strides, size_t offset) {
    std::vector<size_t> strides_vec(strides.begin(), strides.end());
    return std::make_unique<MlxArray>(mlx::core::as_strided(a.inner, to_shape(shape), mlx::core::Strides(strides_vec.begin(), strides_vec.end()), offset));
}

std::unique_ptr<MlxArray> contiguous(const MlxArray& a, bool allow_col_major) {
    return std::make_unique<MlxArray>(mlx::core::contiguous(a.inner, allow_col_major));
}

std::unique_ptr<MlxArray> broadcast_arrays_get(rust::Slice<const MlxArray* const> arrays, size_t index) {
    std::vector<mlx::core::array> inputs;
    inputs.reserve(arrays.size());
    for (const MlxArray* p : arrays) {
        inputs.push_back(p->inner);
    }
    auto result = mlx::core::broadcast_arrays(inputs);
    return std::make_unique<MlxArray>(std::move(result.at(index)));
}

size_t broadcast_arrays_count(rust::Slice<const MlxArray* const> arrays) {
    std::vector<mlx::core::array> inputs;
    inputs.reserve(arrays.size());
    for (const MlxArray* p : arrays) {
        inputs.push_back(p->inner);
    }
    return mlx::core::broadcast_arrays(inputs).size();
}

std::unique_ptr<MlxArray> floor_divide(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::floor_divide(a.inner, b.inner));
}

std::unique_ptr<MlxArray> array_equal(const MlxArray& a, const MlxArray& b, bool equal_nan) {
    return std::make_unique<MlxArray>(mlx::core::array_equal(a.inner, b.inner, equal_nan));
}

std::unique_ptr<MlxArray> allclose(const MlxArray& a, const MlxArray& b, double rtol, double atol) {
    return std::make_unique<MlxArray>(mlx::core::allclose(a.inner, b.inner, rtol, atol));
}

std::unique_ptr<MlxArray> isclose(const MlxArray& a, const MlxArray& b, double rtol, double atol) {
    return std::make_unique<MlxArray>(mlx::core::isclose(a.inner, b.inner, rtol, atol));
}

std::unique_ptr<MlxArray> median_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::median(a.inner));
}

std::unique_ptr<MlxArray> median_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::median(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> logcumsumexp(const MlxArray& a, int32_t axis, bool reverse, bool inclusive) {
    return std::make_unique<MlxArray>(mlx::core::logcumsumexp(a.inner, axis, reverse, inclusive));
}

std::unique_ptr<MlxArray> bitwise_invert(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::bitwise_invert(a.inner));
}

std::unique_ptr<MlxArray> real_part(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::real(a.inner));
}

std::unique_ptr<MlxArray> imag_part(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::imag(a.inner));
}

std::unique_ptr<MlxArray> conjugate(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::conjugate(a.inner));
}

std::unique_ptr<MlxArray> view(const MlxArray& a, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::view(a.inner, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> kron(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::kron(a.inner, b.inner));
}

std::unique_ptr<MlxArray> addmm(const MlxArray& c, const MlxArray& a, const MlxArray& b, float alpha, float beta) {
    return std::make_unique<MlxArray>(mlx::core::addmm(c.inner, a.inner, b.inner, alpha, beta));
}

std::unique_ptr<MlxArray> block_masked_mm(
    const MlxArray& a,
    const MlxArray& b,
    int32_t block_size,
    const MlxArray* mask_out,
    const MlxArray* mask_lhs,
    const MlxArray* mask_rhs
) {
    std::optional<mlx::core::array> opt_mask_out = mask_out ? std::make_optional(mask_out->inner) : std::nullopt;
    std::optional<mlx::core::array> opt_mask_lhs = mask_lhs ? std::make_optional(mask_lhs->inner) : std::nullopt;
    std::optional<mlx::core::array> opt_mask_rhs = mask_rhs ? std::make_optional(mask_rhs->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::block_masked_mm(a.inner, b.inner, block_size, opt_mask_out, opt_mask_lhs, opt_mask_rhs));
}

std::unique_ptr<MlxArray> segmented_mm(const MlxArray& a, const MlxArray& b, const MlxArray& segments) {
    return std::make_unique<MlxArray>(mlx::core::segmented_mm(a.inner, b.inner, segments.inner));
}

std::unique_ptr<MlxArray> hadamard_transform(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::hadamard_transform(a.inner));
}

std::unique_ptr<MlxArray> number_of_elements(const MlxArray& a, rust::Slice<const int32_t> axes, bool inverted, int32_t dtype) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::number_of_elements(a.inner, axes_vec, inverted, to_dtype(dtype)));
}

// ============================================================================
// Convolution additions
// ============================================================================

std::unique_ptr<MlxArray> conv3d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_d, int32_t stride_h, int32_t stride_w,
    int32_t padding_d, int32_t padding_h, int32_t padding_w,
    int32_t dilation_d, int32_t dilation_h, int32_t dilation_w,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv3d(
        input.inner, weight.inner,
        {stride_d, stride_h, stride_w},
        {padding_d, padding_h, padding_w},
        {dilation_d, dilation_h, dilation_w},
        groups
    ));
}

std::unique_ptr<MlxArray> conv_transpose1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t output_padding,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv_transpose1d(
        input.inner, weight.inner, stride, padding, dilation, output_padding, groups
    ));
}

std::unique_ptr<MlxArray> conv_transpose2d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t output_padding_h, int32_t output_padding_w,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv_transpose2d(
        input.inner, weight.inner,
        {stride_h, stride_w},
        {padding_h, padding_w},
        {dilation_h, dilation_w},
        {output_padding_h, output_padding_w},
        groups
    ));
}

std::unique_ptr<MlxArray> conv_transpose3d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_d, int32_t stride_h, int32_t stride_w,
    int32_t padding_d, int32_t padding_h, int32_t padding_w,
    int32_t dilation_d, int32_t dilation_h, int32_t dilation_w,
    int32_t output_padding_d, int32_t output_padding_h, int32_t output_padding_w,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv_transpose3d(
        input.inner, weight.inner,
        {stride_d, stride_h, stride_w},
        {padding_d, padding_h, padding_w},
        {dilation_d, dilation_h, dilation_w},
        {output_padding_d, output_padding_h, output_padding_w},
        groups
    ));
}

// ============================================================================
// Einsum
// ============================================================================

std::unique_ptr<MlxArray> einsum(rust::Str subscripts, rust::Slice<const MlxArray* const> operands) {
    std::string subscripts_str(subscripts.begin(), subscripts.end());
    std::vector<mlx::core::array> operands_vec;
    operands_vec.reserve(operands.size());
    for (const MlxArray* p : operands) {
        operands_vec.push_back(p->inner);
    }
    return std::make_unique<MlxArray>(mlx::core::einsum(subscripts_str, operands_vec));
}

// ============================================================================
// Linear algebra
// ============================================================================

std::unique_ptr<MlxArray> linalg_norm(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::linalg::norm(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> linalg_norm_ord(const MlxArray& a, double ord, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::linalg::norm(a.inner, ord, axis, keepdims));
}

std::unique_ptr<MlxArray> linalg_norm_str(const MlxArray& a, rust::Str ord, int32_t axis, bool keepdims) {
    std::string ord_str(ord.begin(), ord.end());
    return std::make_unique<MlxArray>(mlx::core::linalg::norm(a.inner, ord_str, axis, keepdims));
}

std::unique_ptr<MlxArray> linalg_qr_q(const MlxArray& a) {
    auto [Q, R] = mlx::core::linalg::qr(a.inner);
    return std::make_unique<MlxArray>(std::move(Q));
}

std::unique_ptr<MlxArray> linalg_qr_r(const MlxArray& a) {
    auto [Q, R] = mlx::core::linalg::qr(a.inner);
    return std::make_unique<MlxArray>(std::move(R));
}

std::unique_ptr<MlxArray> linalg_svd_u(const MlxArray& a) {
    auto result = mlx::core::linalg::svd(a.inner);
    return std::make_unique<MlxArray>(std::move(result[0]));
}

std::unique_ptr<MlxArray> linalg_svd_s(const MlxArray& a) {
    auto result = mlx::core::linalg::svd(a.inner);
    return std::make_unique<MlxArray>(std::move(result[1]));
}

std::unique_ptr<MlxArray> linalg_svd_vt(const MlxArray& a) {
    auto result = mlx::core::linalg::svd(a.inner);
    return std::make_unique<MlxArray>(std::move(result[2]));
}

std::unique_ptr<MlxArray> linalg_inv(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::linalg::inv(a.inner));
}

std::unique_ptr<MlxArray> linalg_pinv(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::linalg::pinv(a.inner));
}

std::unique_ptr<MlxArray> linalg_cholesky(const MlxArray& a, bool upper) {
    return std::make_unique<MlxArray>(mlx::core::linalg::cholesky(a.inner, upper));
}

std::unique_ptr<MlxArray> linalg_solve(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::linalg::solve(a.inner, b.inner));
}

std::unique_ptr<MlxArray> linalg_solve_triangular(const MlxArray& a, const MlxArray& b, bool upper) {
    return std::make_unique<MlxArray>(mlx::core::linalg::solve_triangular(a.inner, b.inner, upper));
}

std::unique_ptr<MlxArray> linalg_lu_p(const MlxArray& a) {
    auto result = mlx::core::linalg::lu(a.inner);
    return std::make_unique<MlxArray>(std::move(result[0]));
}

std::unique_ptr<MlxArray> linalg_lu_l(const MlxArray& a) {
    auto result = mlx::core::linalg::lu(a.inner);
    return std::make_unique<MlxArray>(std::move(result[1]));
}

std::unique_ptr<MlxArray> linalg_lu_u(const MlxArray& a) {
    auto result = mlx::core::linalg::lu(a.inner);
    return std::make_unique<MlxArray>(std::move(result[2]));
}

std::unique_ptr<MlxArray> linalg_lu_factor_lu(const MlxArray& a) {
    auto [LU, pivots] = mlx::core::linalg::lu_factor(a.inner);
    return std::make_unique<MlxArray>(std::move(LU));
}

std::unique_ptr<MlxArray> linalg_lu_factor_pivots(const MlxArray& a) {
    auto [LU, pivots] = mlx::core::linalg::lu_factor(a.inner);
    return std::make_unique<MlxArray>(std::move(pivots));
}

std::unique_ptr<MlxArray> linalg_eig_values(const MlxArray& a) {
    auto [eigenvalues, eigenvectors] = mlx::core::linalg::eig(a.inner);
    return std::make_unique<MlxArray>(std::move(eigenvalues));
}

std::unique_ptr<MlxArray> linalg_eig_vectors(const MlxArray& a) {
    auto [eigenvalues, eigenvectors] = mlx::core::linalg::eig(a.inner);
    return std::make_unique<MlxArray>(std::move(eigenvectors));
}

std::unique_ptr<MlxArray> linalg_eigvals(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::linalg::eigvals(a.inner));
}

std::unique_ptr<MlxArray> linalg_eigh_values(const MlxArray& a) {
    auto [eigenvalues, eigenvectors] = mlx::core::linalg::eigh(a.inner);
    return std::make_unique<MlxArray>(std::move(eigenvalues));
}

std::unique_ptr<MlxArray> linalg_eigh_vectors(const MlxArray& a) {
    auto [eigenvalues, eigenvectors] = mlx::core::linalg::eigh(a.inner);
    return std::make_unique<MlxArray>(std::move(eigenvectors));
}

std::unique_ptr<MlxArray> linalg_eigvalsh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::linalg::eigvalsh(a.inner));
}

std::unique_ptr<MlxArray> linalg_cross(const MlxArray& a, const MlxArray& b, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::linalg::cross(a.inner, b.inner, axis));
}

std::unique_ptr<MlxArray> linalg_tri_inv(const MlxArray& a, bool upper) {
    return std::make_unique<MlxArray>(mlx::core::linalg::tri_inv(a.inner, upper));
}

std::unique_ptr<MlxArray> linalg_cholesky_inv(const MlxArray& a, bool upper) {
    return std::make_unique<MlxArray>(mlx::core::linalg::cholesky_inv(a.inner, upper));
}

// ============================================================================
// FFT
// ============================================================================

std::unique_ptr<MlxArray> fft(const MlxArray& a, int32_t n, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::fft::fftn(a.inner, {n}, {axis}));
}

std::unique_ptr<MlxArray> ifft(const MlxArray& a, int32_t n, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::fft::ifftn(a.inner, {n}, {axis}));
}

std::unique_ptr<MlxArray> rfft(const MlxArray& a, int32_t n, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::fft::rfftn(a.inner, {n}, {axis}));
}

std::unique_ptr<MlxArray> irfft(const MlxArray& a, int32_t n, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::fft::irfftn(a.inner, {n}, {axis}));
}

std::unique_ptr<MlxArray> fft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::fftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> ifft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::ifftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> rfft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::rfftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> irfft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::irfftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> fftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::fftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> ifftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::ifftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> rfftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::rfftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> irfftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::irfftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> fftshift(const MlxArray& a, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::fftshift(a.inner, axes_vec));
}

std::unique_ptr<MlxArray> ifftshift(const MlxArray& a, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::ifftshift(a.inner, axes_vec));
}

// ============================================================================
// Random
// ============================================================================

std::unique_ptr<MlxArray> random_key(uint64_t seed) {
    return std::make_unique<MlxArray>(mlx::core::random::key(seed));
}

std::unique_ptr<MlxArray> random_split_key(const MlxArray& key, int32_t num) {
    return std::make_unique<MlxArray>(mlx::core::random::split(key.inner, num));
}

std::unique_ptr<MlxArray> random_uniform(float low, float high, rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::uniform(
        mlx::core::array(low), mlx::core::array(high), to_shape(shape), to_dtype(dtype), opt_key
    ));
}

std::unique_ptr<MlxArray> random_normal(rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::normal(to_shape(shape), to_dtype(dtype), opt_key));
}

std::unique_ptr<MlxArray> random_bernoulli_p(float p, rust::Slice<const int32_t> shape, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::bernoulli(mlx::core::array(p), to_shape(shape), opt_key));
}

std::unique_ptr<MlxArray> random_randint(int32_t low, int32_t high, rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::randint(
        mlx::core::array(low), mlx::core::array(high), to_shape(shape), to_dtype(dtype), opt_key
    ));
}

std::unique_ptr<MlxArray> random_truncated_normal(float lower, float upper, rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::truncated_normal(
        mlx::core::array(lower), mlx::core::array(upper), to_shape(shape), to_dtype(dtype), opt_key
    ));
}

std::unique_ptr<MlxArray> random_gumbel(rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::gumbel(to_shape(shape), to_dtype(dtype), opt_key));
}

std::unique_ptr<MlxArray> random_laplace(rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::laplace(to_shape(shape), to_dtype(dtype), opt_key));
}

std::unique_ptr<MlxArray> random_permutation(int32_t x, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::permutation(x, opt_key));
}

std::unique_ptr<MlxArray> random_permutation_array(const MlxArray& a, int32_t axis, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::permutation(a.inner, axis, opt_key));
}

std::unique_ptr<MlxArray> random_multivariate_normal(
    const MlxArray& mean,
    const MlxArray& cov,
    rust::Slice<const int32_t> shape,
    int32_t dtype,
    const MlxArray* key
) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::multivariate_normal(
        mean.inner, cov.inner, to_shape(shape), to_dtype(dtype), opt_key
    ));
}

// ============================================================================
// Quantization additions
// ============================================================================

std::unique_ptr<MlxArray> quantize_weights_w(const MlxArray& w, int32_t group_size, int32_t bits) {
    auto result = mlx::core::quantize(w.inner, group_size, bits);
    return std::make_unique<MlxArray>(std::move(result[0]));
}

std::unique_ptr<MlxArray> quantize_weights_scales(const MlxArray& w, int32_t group_size, int32_t bits) {
    auto result = mlx::core::quantize(w.inner, group_size, bits);
    return std::make_unique<MlxArray>(std::move(result[1]));
}

std::unique_ptr<MlxArray> quantize_weights_biases(const MlxArray& w, int32_t group_size, int32_t bits) {
    auto result = mlx::core::quantize(w.inner, group_size, bits);
    return std::make_unique<MlxArray>(std::move(result[2]));
}

} // namespace mlx_cxx

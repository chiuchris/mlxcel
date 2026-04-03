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

//! Attention cache state machines shared by text and VLM families.
//!
//! These types keep cache growth, rewinding, and sliding-window semantics in
//! one place so `layers.rs` can focus on layer math while models continue to
//! import the same cache types via `mlxcel_core::layers`.
//!
//! # KV Cache Quantization
//!
//! `KVCache` optionally stores keys/values in INT8 to reduce memory by ~50%.
//! Enable via `KVCacheMode::Int8` at construction time. The `update_and_fetch`
//! method always returns FP16 tensors (dequantized on read), so the attention
//! computation is unaffected.

use crate::concatenate;
use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::ops::divide_scalar;
use cxx::UniquePtr;

/// Storage mode for KV cache tensors.
///
/// Controls the on-device representation of accumulated key/value tensors.
/// The public `update_and_fetch` interface always returns FP16 regardless of
/// the chosen mode, so attention kernels are unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KVCacheMode {
    /// Standard half-precision storage (default). No quantization overhead.
    #[default]
    Fp16,
    /// Per-token INT8 absmax quantization. Reduces KV cache memory by ~50%
    /// at the cost of small quantization error per token.
    Int8,
}

impl std::str::FromStr for KVCacheMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "fp16" | "float16" => Ok(Self::Fp16),
            "int8" | "i8" => Ok(Self::Int8),
            other => Err(format!(
                "unknown kv-cache-mode \"{other}\"; expected \"fp16\" or \"int8\""
            )),
        }
    }
}

impl std::fmt::Display for KVCacheMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp16 => f.write_str("fp16"),
            Self::Int8 => f.write_str("int8"),
        }
    }
}

// ---------------------------------------------------------------------------
// INT8 quantization helpers
// ---------------------------------------------------------------------------

/// Quantize a tensor to INT8 using per-token absmax scaling.
///
/// `x` has shape `[B, H, T, D]` where T is typically 1 (one new token).
/// Returns `(x_int8, scale)` where:
/// - `x_int8`: `[B, H, T, D]` INT8
/// - `scale`:  `[B, H, T, 1]` FP16 — the absmax / 127.0 for each token
///
/// Used by: QuantizedKVCache (INT8 mode of KVCache)
fn quantize_per_token(x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Compute per-token absmax: reduce over last dim (head_dim), keepdims
    let abs_x = ffi::abs(x);
    let absmax = ffi::max_axis(&abs_x, -1, true); // [B, H, T, 1]

    // scale = absmax / 127.0  (FP16 to match cache dtype)
    let scale = divide_scalar(&absmax, 127.0); // [B, H, T, 1]

    // Avoid divide-by-zero: replace zero scales with 1.0
    let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT16);
    let safe_scale = ffi::maximum(&scale, &one);

    // x_int8 = round(x / safe_scale).clamp(-128, 127)
    let x_div = ffi::divide(x, &safe_scale);
    let x_rounded = ffi::round(&x_div);
    let lo = ffi::full_f32(&[1], -128.0, ffi::array_dtype(x));
    let hi = ffi::full_f32(&[1], 127.0, ffi::array_dtype(x));
    let x_clipped = ffi::clip(&x_rounded, &lo, &hi);
    let x_int8 = ffi::astype(&x_clipped, dtype::INT8);

    (x_int8, safe_scale)
}

/// Dequantize INT8 tensor back to FP16 for attention computation.
///
/// `x_int8`: `[B, H, L, D]` INT8
/// `scale`:  `[B, H, L, 1]` FP16
/// Returns:  `[B, H, L, D]` FP16
///
/// Used by: QuantizedKVCache (INT8 mode of KVCache)
fn dequantize(x_int8: &MlxArray, scale: &MlxArray) -> UniquePtr<MlxArray> {
    let x_fp16 = ffi::astype(x_int8, dtype::FLOAT16);
    ffi::multiply(&x_fp16, scale)
}

/// KV Cache for attention layers.
///
/// Uses pre-allocated buffers with slice_update for O(1) per-token updates,
/// matching Python mlx-lm's KVCache implementation. Buffers grow by `step`
/// slots at a time (default 256) to amortize allocation cost.
///
/// When `mode` is `KVCacheMode::Int8`, keys and values are stored as INT8
/// tensors with per-token scale factors. `update_and_fetch` always returns
/// FP16 (dequantized) so attention kernels see standard tensors.
///
/// Used by: All transformer models (Llama, Qwen, Gemma, etc.)
pub struct KVCache {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    pub offset: i32,
    step: i32,
    /// Quantization mode for stored keys/values.
    pub mode: KVCacheMode,
    // INT8-mode scale factors: [B, H, L, 1] FP16, None when mode == Fp16
    key_scales: Option<UniquePtr<MlxArray>>,
    val_scales: Option<UniquePtr<MlxArray>>,
}

impl KVCache {
    /// Create a new empty KV cache with default step size (256) and FP16 mode.
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            step: 256,
            mode: KVCacheMode::Fp16,
            key_scales: None,
            val_scales: None,
        }
    }

    /// Create a new empty KV cache with the specified quantization mode.
    ///
    /// Use `KVCacheMode::Int8` to store accumulated keys/values in INT8 format.
    /// The `update_and_fetch` method will transparently quantize incoming
    /// tensors and dequantize them on read, so callers receive standard FP16.
    pub fn new_with_mode(mode: KVCacheMode) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            step: 256,
            mode,
            key_scales: None,
            val_scales: None,
        }
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }

    /// Get current sequence length in cache
    pub fn seq_len(&self) -> i32 {
        self.offset
    }

    /// Get the allocated buffer size (sequence dimension)
    fn buffer_seq_len(&self) -> i32 {
        match &self.keys {
            Some(k) => {
                let shape = ffi::array_shape(k);
                if shape.len() >= 3 {
                    shape[2]
                } else {
                    0
                }
            }
            None => 0,
        }
    }

    /// Update cache with new key/value using pre-allocated buffer + slice_update.
    ///
    /// In `KVCacheMode::Int8` the incoming tensors are quantized to INT8 before
    /// storage; scale factors are accumulated in a parallel `[B, H, L, 1]`
    /// buffer. In `KVCacheMode::Fp16` this behaves identically to the original
    /// implementation.
    pub fn update(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
        if self.mode == KVCacheMode::Int8 {
            self.update_int8(new_keys, new_values);
        } else {
            self.update_fp16(new_keys, new_values);
        }
    }

    /// FP16 (standard) update path — original pre-allocated buffer logic.
    fn update_fp16(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
        let key_shape = ffi::array_shape(&new_keys);
        let new_seq_len = key_shape[2];
        let prev = self.offset;

        if self.keys.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_head_dim = key_shape[3];
            let val_shape = ffi::array_shape(&new_values);
            let v_head_dim = val_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let k_dtype = ffi::array_dtype(&new_keys);
            let v_dtype = ffi::array_dtype(&new_values);
            let new_k = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], k_dtype);
            let new_v = ffi::zeros(&[b, n_kv_heads, buf_size, v_head_dim], v_dtype);

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_head_dim],
                    ));
                    self.values = Some(ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_head_dim],
                    ));
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k, 2));
                self.values = Some(concatenate(self.values.as_ref().unwrap(), &new_v, 2));
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        self.offset += new_seq_len;

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &new_keys,
            &[0, 0, prev, 0],
            &[k_shape[0], k_shape[1], self.offset, k_shape[3]],
        ));
        self.values = Some(ffi::slice_update(
            self.values.as_ref().unwrap(),
            &new_values,
            &[0, 0, prev, 0],
            &[v_shape[0], v_shape[1], self.offset, v_shape[3]],
        ));
    }

    /// INT8 update path — quantizes incoming K/V tokens and accumulates into
    /// INT8 key/value buffers alongside FP16 per-token scale buffers.
    ///
    /// Layout of stored buffers (step-aligned, grown lazily):
    /// - `keys`/`values`: `[B, H, capacity, D]` INT8
    /// - `key_scales`/`val_scales`: `[B, H, capacity, 1]` FP16
    fn update_int8(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
        // Cast incoming tensors to FP16 before quantization so scale
        // computation operates in a consistent dtype.
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        let (k_int8, k_scale) = quantize_per_token(&new_keys_f16);
        let (v_int8, v_scale) = quantize_per_token(&new_values_f16);

        let key_shape = ffi::array_shape(&k_int8);
        let new_seq_len = key_shape[2];
        let prev = self.offset;

        if self.keys.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_head_dim = key_shape[3];
            let val_shape = ffi::array_shape(&v_int8);
            let v_head_dim = val_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let new_k_buf = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], dtype::INT8);
            let new_v_buf = ffi::zeros(&[b, n_kv_heads, buf_size, v_head_dim], dtype::INT8);
            let new_ks_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            let new_vs_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_head_dim],
                    ));
                    self.values = Some(ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_head_dim],
                    ));
                    self.key_scales = Some(ffi::slice(
                        self.key_scales.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                    self.val_scales = Some(ffi::slice(
                        self.val_scales.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k_buf, 2));
                self.values = Some(concatenate(self.values.as_ref().unwrap(), &new_v_buf, 2));
                self.key_scales = Some(concatenate(
                    self.key_scales.as_ref().unwrap(),
                    &new_ks_buf,
                    2,
                ));
                self.val_scales = Some(concatenate(
                    self.val_scales.as_ref().unwrap(),
                    &new_vs_buf,
                    2,
                ));
            } else {
                self.keys = Some(new_k_buf);
                self.values = Some(new_v_buf);
                self.key_scales = Some(new_ks_buf);
                self.val_scales = Some(new_vs_buf);
            }
        }

        self.offset += new_seq_len;

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
        let ks_shape = ffi::array_shape(self.key_scales.as_ref().unwrap());
        let vs_shape = ffi::array_shape(self.val_scales.as_ref().unwrap());

        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &k_int8,
            &[0, 0, prev, 0],
            &[k_shape[0], k_shape[1], self.offset, k_shape[3]],
        ));
        self.values = Some(ffi::slice_update(
            self.values.as_ref().unwrap(),
            &v_int8,
            &[0, 0, prev, 0],
            &[v_shape[0], v_shape[1], self.offset, v_shape[3]],
        ));
        self.key_scales = Some(ffi::slice_update(
            self.key_scales.as_ref().unwrap(),
            &k_scale,
            &[0, 0, prev, 0],
            &[ks_shape[0], ks_shape[1], self.offset, 1],
        ));
        self.val_scales = Some(ffi::slice_update(
            self.val_scales.as_ref().unwrap(),
            &v_scale,
            &[0, 0, prev, 0],
            &[vs_shape[0], vs_shape[1], self.offset, 1],
        ));
    }

    /// Trim the last `n` entries from the cache.
    ///
    /// Returns the number of entries actually trimmed.
    /// In INT8 mode the corresponding scale buffers are also trimmed.
    /// Used by: speculative decoding cache rewinds
    pub fn trim(&mut self, n: i32) -> i32 {
        let n = n.min(self.offset);
        if n <= 0 {
            return 0;
        }
        self.offset -= n;
        if self.offset == 0 {
            self.keys = None;
            self.values = None;
            self.key_scales = None;
            self.val_scales = None;
        } else {
            let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
            let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
            self.keys = Some(ffi::slice(
                self.keys.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[k_shape[0], k_shape[1], self.offset, k_shape[3]],
            ));
            self.values = Some(ffi::slice(
                self.values.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[v_shape[0], v_shape[1], self.offset, v_shape[3]],
            ));
            // Also trim scale buffers in INT8 mode
            if self.mode == KVCacheMode::Int8 {
                if let Some(ref ks) = self.key_scales {
                    let ks_shape = ffi::array_shape(ks);
                    self.key_scales = Some(ffi::slice(
                        ks,
                        &[0, 0, 0, 0],
                        &[ks_shape[0], ks_shape[1], self.offset, 1],
                    ));
                }
                if let Some(ref vs) = self.val_scales {
                    let vs_shape = ffi::array_shape(vs);
                    self.val_scales = Some(ffi::slice(
                        vs,
                        &[0, 0, 0, 0],
                        &[vs_shape[0], vs_shape[1], self.offset, 1],
                    ));
                }
            }
        }
        n
    }

    /// Update cache and return view of filled portion.
    ///
    /// In `KVCacheMode::Fp16` returns sliced FP16 keys/values directly.
    /// In `KVCacheMode::Int8` dequantizes the accumulated INT8 buffers back to
    /// FP16 before returning, so attention kernels always receive FP16 tensors.
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update(new_keys, new_values);

        if self.mode == KVCacheMode::Int8 {
            // Dequantize the filled portion of the INT8 buffers
            let k_int8 = self.keys.as_ref().unwrap();
            let v_int8 = self.values.as_ref().unwrap();
            let k_scales = self.key_scales.as_ref().unwrap();
            let v_scales = self.val_scales.as_ref().unwrap();

            let ks = ffi::array_shape(k_int8);
            let vs = ffi::array_shape(v_int8);
            let kss = ffi::array_shape(k_scales);
            let vss = ffi::array_shape(v_scales);

            let k_slice = ffi::slice(k_int8, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]);
            let v_slice = ffi::slice(v_int8, &[0, 0, 0, 0], &[vs[0], vs[1], self.offset, vs[3]]);
            let ks_slice = ffi::slice(k_scales, &[0, 0, 0, 0], &[kss[0], kss[1], self.offset, 1]);
            let vs_slice = ffi::slice(v_scales, &[0, 0, 0, 0], &[vss[0], vss[1], self.offset, 1]);

            (
                dequantize(&k_slice, &ks_slice),
                dequantize(&v_slice, &vs_slice),
            )
        } else {
            let k = self.keys.as_ref().unwrap();
            let v = self.values.as_ref().unwrap();
            let ks = ffi::array_shape(k);
            let vs = ffi::array_shape(v);
            (
                ffi::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]),
                ffi::slice(v, &[0, 0, 0, 0], &[vs[0], vs[1], self.offset, vs[3]]),
            )
        }
    }

    /// Get the total memory size of the cached keys and values in bytes.
    ///
    /// In INT8 mode this includes both the INT8 buffers and the scale tensors.
    pub fn nbytes(&self) -> usize {
        let k_bytes = self.keys.as_ref().map_or(0, |k| ffi::array_nbytes(k));
        let v_bytes = self.values.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let ks_bytes = self.key_scales.as_ref().map_or(0, |k| ffi::array_nbytes(k));
        let vs_bytes = self.val_scales.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        k_bytes + v_bytes + ks_bytes + vs_bytes
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Rotating KV Cache for sliding window attention (e.g. Gemma 3, Ministral 3).
///
/// Maintains a fixed-size circular buffer for keys/values. Oversized prefill
/// is linearized before single-token decode so wraparound stays well-defined.
pub struct RotatingKVCache {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    pub max_size: i32,
    pub offset: i32,
    /// Current write position in the buffer (separate from offset to handle trim correctly)
    idx: i32,
}

impl RotatingKVCache {
    /// Create a new rotating KV cache with specified maximum size
    pub fn new(max_size: i32) -> Self {
        Self {
            keys: None,
            values: None,
            max_size,
            offset: 0,
            idx: 0,
        }
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }

    /// Get current sequence length in cache
    pub fn seq_len(&self) -> i32 {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            if shape.len() >= 3 {
                shape[2]
            } else {
                0
            }
        } else {
            0
        }
    }

    /// Update cache with new key/value, rotating if necessary.
    ///
    /// Returns the full cached keys/values.
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let new_seq_len = {
            let shape = ffi::array_shape(&new_keys);
            shape[2]
        };

        if new_seq_len > 1 {
            return self.update_concat(new_keys, new_values, new_seq_len);
        }

        self.update_in_place(new_keys, new_values)
    }

    fn update_concat(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        new_seq_len: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        if self.keys.is_none() {
            self.offset += new_seq_len;
            self.idx = new_seq_len;
            self.keys = Some(ffi::contiguous(&new_keys, false));
            self.values = Some(ffi::contiguous(&new_values, false));
            return (new_keys, new_values);
        }

        let current_seq_len = {
            let shape = ffi::array_shape(self.keys.as_ref().unwrap());
            shape[2]
        };

        let concat_k = concatenate(self.keys.as_ref().unwrap(), &new_keys, 2);
        let concat_v = concatenate(self.values.as_ref().unwrap(), &new_values, 2);

        let total_len = current_seq_len + new_seq_len;
        self.offset += new_seq_len;

        if total_len > self.max_size {
            let start = total_len - self.max_size;
            let k = ffi::slice(
                &concat_k,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            let v = ffi::slice(
                &concat_v,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            self.idx = self.max_size;
            self.keys = Some(ffi::contiguous(&k, false));
            self.values = Some(ffi::contiguous(&v, false));
            (k, v)
        } else {
            self.idx = total_len;
            self.keys = Some(ffi::contiguous(&concat_k, false));
            self.values = Some(ffi::contiguous(&concat_v, false));
            (concat_k, concat_v)
        }
    }

    fn update_in_place(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            let buffer_size = shape[2];
            if buffer_size > self.max_size {
                let start = buffer_size - self.max_size;
                let ks = ffi::array_shape(self.keys.as_ref().unwrap());
                let vs = ffi::array_shape(self.values.as_ref().unwrap());
                self.keys = Some(ffi::contiguous(
                    &ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, start, 0],
                        &[ks[0], ks[1], buffer_size, ks[3]],
                    ),
                    false,
                ));
                self.values = Some(ffi::contiguous(
                    &ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, start, 0],
                        &[vs[0], vs[1], buffer_size, vs[3]],
                    ),
                    false,
                ));
                self.idx = self.max_size;
            }
        }

        if self.keys.is_none() {
            let shape = ffi::array_shape(&new_keys);
            let batch = shape[0];
            let heads = shape[1];
            let head_dim = shape[3];

            let k_zeros = ffi::zeros(
                &[batch, heads, self.max_size, head_dim],
                ffi::array_dtype(&new_keys),
            );
            let v_zeros = ffi::zeros(
                &[batch, heads, self.max_size, head_dim],
                ffi::array_dtype(&new_values),
            );

            let k = ffi::slice_update(
                &k_zeros,
                &new_keys,
                &[0, 0, 0, 0],
                &[batch, heads, 1, head_dim],
            );
            let v = ffi::slice_update(
                &v_zeros,
                &new_values,
                &[0, 0, 0, 0],
                &[batch, heads, 1, head_dim],
            );

            self.offset = 1;
            self.idx = 1;
            self.keys = Some(ffi::contiguous(&k, false));
            self.values = Some(ffi::contiguous(&v, false));

            let k_out = ffi::slice(&k, &[0, 0, 0, 0], &[batch, heads, 1, head_dim]);
            let v_out = ffi::slice(&v, &[0, 0, 0, 0], &[batch, heads, 1, head_dim]);
            return (k_out, v_out);
        }

        let k_buffer = self.keys.take().unwrap();
        let v_buffer = self.values.take().unwrap();

        let shape = ffi::array_shape(&k_buffer);
        let batch = shape[0];
        let heads = shape[1];
        let buffer_size = shape[2];
        let head_dim = shape[3];

        if self.idx >= buffer_size && buffer_size < self.max_size {
            let k_concat = concatenate(&k_buffer, &new_keys, 2);
            let v_concat = concatenate(&v_buffer, &new_values, 2);

            self.offset += 1;
            self.idx += 1;
            self.keys = Some(ffi::contiguous(&k_concat, false));
            self.values = Some(ffi::contiguous(&v_concat, false));
            return (k_concat, v_concat);
        }

        if self.idx >= self.max_size {
            self.idx = 0;
        }

        let pos = self.idx;
        let k_buffer = ffi::slice_update(
            &k_buffer,
            &new_keys,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, head_dim],
        );
        let v_buffer = ffi::slice_update(
            &v_buffer,
            &new_values,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, head_dim],
        );

        self.offset += 1;
        self.idx += 1;

        if self.offset < self.max_size {
            let k_out = ffi::slice(
                &k_buffer,
                &[0, 0, 0, 0],
                &[batch, heads, self.offset, head_dim],
            );
            let v_out = ffi::slice(
                &v_buffer,
                &[0, 0, 0, 0],
                &[batch, heads, self.offset, head_dim],
            );
            self.keys = Some(k_buffer);
            self.values = Some(v_buffer);
            (k_out, v_out)
        } else {
            let k_out = ffi::contiguous(&k_buffer, false);
            let v_out = ffi::contiguous(&v_buffer, false);
            self.keys = Some(k_buffer);
            self.values = Some(v_buffer);
            (k_out, v_out)
        }
    }

    /// Get the current offset
    pub fn get_offset(&self) -> i32 {
        self.offset
    }
}

impl Default for RotatingKVCache {
    fn default() -> Self {
        Self::new(4096)
    }
}

/// Chunked KV Cache for Llama 4's iGQA (Interleaved GQA) attention.
///
/// Maintains a sliding window cache that trims from the front when exceeding
/// `chunk_size`, while still tracking the global start position for mask logic.
pub struct ChunkedKVCache {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    pub chunk_size: i32,
    pub offset: i32,
    pub start_position: i32,
    step: i32,
}

impl ChunkedKVCache {
    /// Create a new chunked KV cache with specified chunk size
    pub fn new(chunk_size: i32) -> Self {
        Self {
            keys: None,
            values: None,
            chunk_size,
            offset: 0,
            start_position: 0,
            step: 256,
        }
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }

    /// Get the global offset (total tokens processed)
    pub fn get_offset(&self) -> i32 {
        self.offset
    }

    /// Get the start position (where visible window begins)
    pub fn get_start_position(&self) -> i32 {
        self.start_position
    }

    /// Trim the front of the cache if it exceeds chunk_size.
    ///
    /// This should be called before processing each layer.
    pub fn maybe_trim_front(&mut self) {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            let seq_len = (self.offset - self.start_position).min(shape[2]);

            if seq_len > self.chunk_size {
                let trim_amount = seq_len - self.chunk_size;
                self.start_position += trim_amount;

                let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
                let v_shape = ffi::array_shape(self.values.as_ref().unwrap());

                self.keys = Some(ffi::slice(
                    self.keys.as_ref().unwrap(),
                    &[0, 0, trim_amount, 0],
                    &[k_shape[0], k_shape[1], seq_len, k_shape[3]],
                ));
                self.values = Some(ffi::slice(
                    self.values.as_ref().unwrap(),
                    &[0, 0, trim_amount, 0],
                    &[v_shape[0], v_shape[1], seq_len, v_shape[3]],
                ));
            }
        }
    }

    /// Update cache with new key/value and return the visible portion
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let new_shape = ffi::array_shape(&new_keys);
        let new_seq_len = new_shape[2];
        let prev = self.offset - self.start_position;

        if self.keys.is_none() || (prev + new_seq_len) > self.get_buffer_size() {
            let b = new_shape[0];
            let n_kv_heads = new_shape[1];
            let k_head_dim = new_shape[3];
            let v_shape = ffi::array_shape(&new_values);
            let v_head_dim = v_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let new_buffer_size = n_steps * self.step;

            let new_k = ffi::zeros(
                &[b, n_kv_heads, new_buffer_size, k_head_dim],
                ffi::array_dtype(&new_keys),
            );
            let new_v = ffi::zeros(
                &[b, n_kv_heads, new_buffer_size, v_head_dim],
                ffi::array_dtype(&new_values),
            );

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
                    let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[k_shape[0], k_shape[1], prev, k_shape[3]],
                    ));
                    self.values = Some(ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[v_shape[0], v_shape[1], prev, v_shape[3]],
                    ));
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k, 2));
                self.values = Some(concatenate(self.values.as_ref().unwrap(), &new_v, 2));
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        self.offset += new_seq_len;
        let end = self.offset - self.start_position;

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());

        if prev > 0 {
            let k_before = ffi::slice(
                self.keys.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[k_shape[0], k_shape[1], prev, k_shape[3]],
            );
            let v_before = ffi::slice(
                self.values.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[v_shape[0], v_shape[1], prev, v_shape[3]],
            );

            if end < k_shape[2] {
                let k_after = ffi::slice(
                    self.keys.as_ref().unwrap(),
                    &[0, 0, end, 0],
                    &[k_shape[0], k_shape[1], k_shape[2], k_shape[3]],
                );
                let v_after = ffi::slice(
                    self.values.as_ref().unwrap(),
                    &[0, 0, end, 0],
                    &[v_shape[0], v_shape[1], v_shape[2], v_shape[3]],
                );
                self.keys = Some(concatenate(
                    &concatenate(&k_before, &new_keys, 2),
                    &k_after,
                    2,
                ));
                self.values = Some(concatenate(
                    &concatenate(&v_before, &new_values, 2),
                    &v_after,
                    2,
                ));
            } else {
                self.keys = Some(concatenate(&k_before, &new_keys, 2));
                self.values = Some(concatenate(&v_before, &new_values, 2));
            }
        } else if end < k_shape[2] {
            let k_after = ffi::slice(
                self.keys.as_ref().unwrap(),
                &[0, 0, end, 0],
                &[k_shape[0], k_shape[1], k_shape[2], k_shape[3]],
            );
            let v_after = ffi::slice(
                self.values.as_ref().unwrap(),
                &[0, 0, end, 0],
                &[v_shape[0], v_shape[1], v_shape[2], v_shape[3]],
            );
            self.keys = Some(concatenate(&new_keys, &k_after, 2));
            self.values = Some(concatenate(&new_values, &v_after, 2));
        } else {
            self.keys = Some(ffi::contiguous(&new_keys, false));
            self.values = Some(ffi::contiguous(&new_values, false));
        }

        (
            ffi::slice(
                self.keys.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[k_shape[0], k_shape[1], end, k_shape[3]],
            ),
            ffi::slice(
                self.values.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[v_shape[0], v_shape[1], end, v_shape[3]],
            ),
        )
    }

    fn get_buffer_size(&self) -> i32 {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            shape[2]
        } else {
            0
        }
    }
}

impl Default for ChunkedKVCache {
    fn default() -> Self {
        Self::new(8192)
    }
}

// --- Per-sequence cache isolation for continuous batching ---

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Unique identifier for a sequence in the batch.
///
/// Each active generation sequence receives a unique monotonically increasing
/// ID from the owning `CachePool`. The inner `u64` never wraps within any
/// reasonable server lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(u64);

impl SequenceId {
    /// Construct a `SequenceId` from a raw `u64` value.
    ///
    /// In production code, IDs are assigned by `CachePool::allocate`. This
    /// constructor is provided for tests, builders, and deserialization.
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Return the raw numeric identifier.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SequenceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "seq-{}", self.0)
    }
}

/// One sequence's full set of layer caches.
///
/// Created by `CachePool::allocate` and tied to a single generation request.
/// The caller owns mutable access while the sequence is active and must call
/// `CachePool::release` when generation finishes.
pub struct SequenceCacheSet {
    /// Per-layer KV caches (one entry per model layer).
    pub caches: Vec<KVCache>,
    /// Unique identifier assigned by the pool.
    pub seq_id: SequenceId,
    /// Number of prompt tokens originally prefilled.
    pub prompt_len: usize,
    /// Current generation position (incremented during decode).
    pub current_offset: i32,
    /// Wall-clock time when this cache set was allocated.
    pub created_at: Instant,
}

impl SequenceCacheSet {
    /// Total memory footprint of all layer caches in bytes.
    pub fn nbytes(&self) -> usize {
        self.caches.iter().map(|c| c.nbytes()).sum()
    }
}

/// Pool that allocates and recycles per-sequence cache sets.
///
/// Designed for use by a continuous-batching scheduler. The pool assigns
/// monotonically increasing `SequenceId` values and enforces a hard upper
/// bound on concurrent active sequences.
///
/// Thread safety: `CachePool` itself is **not** `Sync`; callers in async
/// server code should wrap it in an appropriate lock (`Mutex` or `RwLock`).
pub struct CachePool {
    next_id: AtomicU64,
    active: HashMap<SequenceId, SequenceCacheSet>,
    max_sequences: usize,
}

impl CachePool {
    /// Create a new pool allowing up to `max_sequences` concurrent cache sets.
    pub fn new(max_sequences: usize) -> Self {
        Self {
            next_id: AtomicU64::new(0),
            active: HashMap::new(),
            max_sequences,
        }
    }

    /// Allocate a fresh cache set for a new sequence.
    ///
    /// Calls `model.make_caches()` to build per-layer caches, assigns a
    /// unique `SequenceId`, and stores the set in the pool.
    ///
    /// Returns `Err` if the model does not support batching (i.e.
    /// `supports_batching()` returns `false`) or the pool already has
    /// `max_sequences` active entries.
    pub fn allocate(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
    ) -> Result<SequenceId, String> {
        if !model.supports_batching() {
            return Err(
                "CachePool: model does not support batching (internal recurrent/SSM state)".into(),
            );
        }
        if self.active.len() >= self.max_sequences {
            return Err(format!(
                "CachePool: max capacity ({}) reached, cannot allocate new sequence",
                self.max_sequences
            ));
        }

        let id = SequenceId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let caches = model.make_caches();
        let entry = SequenceCacheSet {
            caches,
            seq_id: id,
            prompt_len: 0,
            current_offset: 0,
            created_at: Instant::now(),
        };
        self.active.insert(id, entry);
        Ok(id)
    }

    /// Return a mutable reference to the full `SequenceCacheSet` for the
    /// given sequence, or `None` if the ID is not active.
    pub fn get_mut(&mut self, id: SequenceId) -> Option<&mut SequenceCacheSet> {
        self.active.get_mut(&id)
    }

    /// Return a mutable slice of the per-layer KV caches for direct use
    /// in `model.forward()`, or `None` if the ID is not active.
    pub fn get_caches_mut(&mut self, id: SequenceId) -> Option<&mut [KVCache]> {
        self.active.get_mut(&id).map(|s| s.caches.as_mut_slice())
    }

    /// Release a sequence's caches, reclaiming the pool slot.
    ///
    /// This is a no-op if `id` is not currently active.
    pub fn release(&mut self, id: SequenceId) {
        self.active.remove(&id);
    }

    /// Number of sequences currently holding active cache sets.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Sum of `nbytes()` across all active cache sets.
    pub fn memory_usage_bytes(&self) -> usize {
        self.active.values().map(|s| s.nbytes()).sum()
    }

    /// Maximum number of concurrent sequences this pool allows.
    pub fn max_sequences(&self) -> usize {
        self.max_sequences
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_trim_clears_storage_when_fully_rewound() {
        let mut cache = KVCache::new();
        let keys = ffi::from_slice_f32(&[1.0, 2.0], &[1, 1, 2, 1]);
        let values = ffi::from_slice_f32(&[3.0, 4.0], &[1, 1, 2, 1]);
        cache.update(keys, values);

        assert_eq!(cache.seq_len(), 2);
        assert!(!cache.is_empty());
        assert_eq!(cache.trim(5), 2);
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn rotating_cache_wraps_single_token_updates_to_window_size() {
        let mut cache = RotatingKVCache::new(2);
        let first = ffi::from_slice_f32(&[1.0], &[1, 1, 1, 1]);
        let second = ffi::from_slice_f32(&[2.0], &[1, 1, 1, 1]);
        let third = ffi::from_slice_f32(&[3.0], &[1, 1, 1, 1]);
        let values = |x| ffi::from_slice_f32(&[x], &[1, 1, 1, 1]);

        cache.update_and_fetch(first, values(1.0));
        cache.update_and_fetch(second, values(2.0));
        let (keys, _values) = cache.update_and_fetch(third, values(3.0));

        assert_eq!(cache.get_offset(), 3);
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(ffi::array_shape(&keys), vec![1, 1, 2, 1]);
    }

    #[test]
    fn chunked_cache_trim_front_advances_visible_window() {
        let mut cache = ChunkedKVCache::new(2);
        let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 1, 3, 1]);
        let values = ffi::from_slice_f32(&[4.0, 5.0, 6.0], &[1, 1, 3, 1]);
        cache.update_and_fetch(keys, values);
        cache.maybe_trim_front();

        assert_eq!(cache.get_offset(), 3);
        assert_eq!(cache.get_start_position(), 1);
        assert_eq!(
            ffi::array_shape(cache.keys.as_ref().unwrap()),
            vec![1, 1, 2, 1]
        );
    }

    // --- CachePool tests ---

    /// Minimal model stub for CachePool tests. Produces N empty KVCaches.
    struct StubModel {
        num_layers: usize,
    }

    impl crate::generate::LanguageModel for StubModel {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::zeros(&[1], 0)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            (0..self.num_layers).map(|_| KVCache::new()).collect()
        }

        fn num_layers(&self) -> usize {
            self.num_layers
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }
    }

    #[test]
    fn cache_pool_allocate_and_release() {
        let model = StubModel { num_layers: 4 };
        let mut pool = CachePool::new(8);

        let id1 = pool.allocate(&model).expect("should allocate");
        let id2 = pool.allocate(&model).expect("should allocate");

        assert_ne!(id1, id2);
        assert_eq!(pool.active_count(), 2);

        // Each sequence should have 4 layer caches
        assert_eq!(pool.get_caches_mut(id1).unwrap().len(), 4);
        assert_eq!(pool.get_caches_mut(id2).unwrap().len(), 4);

        pool.release(id1);
        assert_eq!(pool.active_count(), 1);
        assert!(pool.get_mut(id1).is_none());
        assert!(pool.get_mut(id2).is_some());

        pool.release(id2);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn cache_pool_refuses_allocation_when_full() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(2);

        pool.allocate(&model).expect("first");
        pool.allocate(&model).expect("second");

        let result = pool.allocate(&model);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("max capacity"));
    }

    #[test]
    fn cache_pool_release_reopens_slot() {
        let model = StubModel { num_layers: 1 };
        let mut pool = CachePool::new(1);

        let id = pool.allocate(&model).expect("first");
        assert!(pool.allocate(&model).is_err());

        pool.release(id);
        assert_eq!(pool.active_count(), 0);

        // Slot should be available again
        let id2 = pool.allocate(&model).expect("after release");
        assert_ne!(id, id2); // IDs are monotonic, never reused
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn cache_pool_independent_state() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(4);

        let id1 = pool.allocate(&model).unwrap();
        let id2 = pool.allocate(&model).unwrap();

        // Mutate caches for sequence 1 only
        {
            let caches = pool.get_caches_mut(id1).unwrap();
            let keys = ffi::from_slice_f32(&[1.0, 2.0], &[1, 1, 2, 1]);
            let values = ffi::from_slice_f32(&[3.0, 4.0], &[1, 1, 2, 1]);
            caches[0].update(keys, values);
        }

        // Sequence 2 caches should still be empty
        {
            let caches = pool.get_caches_mut(id2).unwrap();
            assert!(caches[0].is_empty());
            assert!(caches[1].is_empty());
        }

        // Sequence 1 cache should have data
        {
            let caches = pool.get_caches_mut(id1).unwrap();
            assert!(!caches[0].is_empty());
            assert_eq!(caches[0].seq_len(), 2);
            // Second layer still empty
            assert!(caches[1].is_empty());
        }
    }

    #[test]
    fn cache_pool_memory_usage() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(4);

        // Empty pool
        assert_eq!(pool.memory_usage_bytes(), 0);

        let id1 = pool.allocate(&model).unwrap();

        // Freshly allocated caches have no data
        assert_eq!(pool.memory_usage_bytes(), 0);

        // Add some data to one cache
        {
            let caches = pool.get_caches_mut(id1).unwrap();
            let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
            let values = ffi::from_slice_f32(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
            caches[0].update(keys, values);
        }

        let mem_after = pool.memory_usage_bytes();
        assert!(mem_after > 0);

        // Release should bring memory tracking back to zero
        pool.release(id1);
        assert_eq!(pool.memory_usage_bytes(), 0);
    }

    #[test]
    fn cache_pool_sequence_metadata() {
        let model = StubModel { num_layers: 1 };
        let mut pool = CachePool::new(4);

        let id = pool.allocate(&model).unwrap();
        let entry = pool.get_mut(id).unwrap();

        assert_eq!(entry.seq_id, id);
        assert_eq!(entry.prompt_len, 0);
        assert_eq!(entry.current_offset, 0);

        // Simulate prefill state update
        entry.prompt_len = 42;
        entry.current_offset = 42;

        let entry = pool.get_mut(id).unwrap();
        assert_eq!(entry.prompt_len, 42);
        assert_eq!(entry.current_offset, 42);
    }

    #[test]
    fn cache_pool_release_nonexistent_is_noop() {
        let mut pool = CachePool::new(4);
        let fake_id = SequenceId(9999);
        pool.release(fake_id); // should not panic
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn sequence_id_display() {
        let id = SequenceId(42);
        assert_eq!(format!("{id}"), "seq-42");
        assert_eq!(id.as_u64(), 42);
    }

    #[test]
    fn cache_pool_rejects_non_batching_model() {
        struct NonBatchModel;

        impl crate::generate::LanguageModel for NonBatchModel {
            fn forward(
                &self,
                _input_ids: &MlxArray,
                _caches: &mut [KVCache],
                _mask: Option<&MlxArray>,
            ) -> UniquePtr<MlxArray> {
                ffi::zeros(&[1], 0)
            }

            fn make_caches(&self) -> Vec<KVCache> {
                vec![KVCache::new()]
            }

            fn num_layers(&self) -> usize {
                1
            }

            fn eos_token_ids(&self) -> Vec<i32> {
                vec![0]
            }

            fn supports_batching(&self) -> bool {
                false
            }
        }

        let model = NonBatchModel;
        let mut pool = CachePool::new(8);

        let result = pool.allocate(&model);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not support batching"));
        assert_eq!(pool.active_count(), 0);
    }
}

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

use crate::concatenate;
use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;

/// KV Cache for attention layers.
///
/// Uses pre-allocated buffers with slice_update for O(1) per-token updates,
/// matching Python mlx-lm's KVCache implementation. Buffers grow by `step`
/// slots at a time (default 256) to amortize allocation cost.
///
/// Used by: All transformer models (Llama, Qwen, Gemma, etc.)
pub struct KVCache {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    pub offset: i32,
    step: i32,
}

impl KVCache {
    /// Create a new empty KV cache with default step size (256)
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            step: 256,
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

    /// Update cache with new key/value using pre-allocated buffer + slice_update
    pub fn update(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
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

    /// Trim the last `n` entries from the cache.
    ///
    /// Returns the number of entries actually trimmed.
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
        }
        n
    }

    /// Update cache and return view of filled portion.
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update(new_keys, new_values);

        let k = self.keys.as_ref().unwrap();
        let v = self.values.as_ref().unwrap();
        let ks = ffi::array_shape(k);
        let vs = ffi::array_shape(v);
        (
            ffi::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]),
            ffi::slice(v, &[0, 0, 0, 0], &[vs[0], vs[1], self.offset, vs[3]]),
        )
    }

    /// Get the total memory size of the cached keys and values in bytes
    pub fn nbytes(&self) -> usize {
        let k_bytes = self.keys.as_ref().map_or(0, |k| ffi::array_nbytes(k));
        let v_bytes = self.values.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        k_bytes + v_bytes
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
}

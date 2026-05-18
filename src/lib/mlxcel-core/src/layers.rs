//! High-level model layer implementations using mlx-cxx
//!
//! This module provides Rust wrappers for common neural network layers
//! using the mlx-cxx bindings for optimal performance.

use crate::concatenate;
use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;

/// Quantized weight structure for 4-bit/8-bit quantized layers
pub struct QuantizedWeight {
    pub weight: UniquePtr<MlxArray>,
    pub scales: UniquePtr<MlxArray>,
    pub biases: UniquePtr<MlxArray>,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedWeight {
    /// Create a new quantized weight from raw components
    pub fn new(
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        }
    }
}

// =============================================================================
// Embedding Layers
// =============================================================================

/// Non-quantized embedding layer
pub struct Embedding {
    pub weight: UniquePtr<MlxArray>,
}

impl Embedding {
    /// Create a new embedding layer
    pub fn new(weight: UniquePtr<MlxArray>) -> Self {
        Self { weight }
    }

    /// Load from weight map
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
        Ok(Self { weight })
    }

    /// Embedding lookup: indices -> embeddings
    pub fn forward(&self, indices: &MlxArray) -> UniquePtr<MlxArray> {
        ffi::embedding(&self.weight, indices)
    }

    /// Use embedding as linear projection (for tied embeddings/lm_head)
    /// y = x @ W (no transpose needed since embedding weight is [vocab, dim])
    pub fn as_linear(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let wt = ffi::transpose(&self.weight);
        ffi::matmul(x, &wt)
    }
}

/// Quantized embedding layer (4-bit/8-bit)
pub struct QuantizedEmbedding {
    pub weight: UniquePtr<MlxArray>,
    pub scales: UniquePtr<MlxArray>,
    pub biases: UniquePtr<MlxArray>,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedEmbedding {
    /// Create a new quantized embedding layer
    pub fn new(
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        }
    }

    /// Load from weight map
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let scales_name = format!("{}.scales", prefix);
        let biases_name = format!("{}.biases", prefix);

        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
        let scales = weights
            .get(&scales_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Scales not found: {}", scales_name))?;
        let biases = weights
            .get(&biases_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Biases not found: {}", biases_name))?;

        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    /// Quantized embedding lookup with dequantization
    pub fn forward(&self, indices: &MlxArray) -> UniquePtr<MlxArray> {
        ffi::quantized_embedding(
            &self.weight,
            &self.scales,
            &self.biases,
            indices,
            self.group_size,
            self.bits,
        )
    }

    /// Use embedding as linear projection (for tied embeddings/lm_head)
    pub fn as_linear(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.weight,
                &self.scales,
                &self.biases,
                std::ptr::null(),
                self.group_size,
                self.bits,
            )
        }
    }
}

/// Unified embedding that auto-detects quantization
pub enum UnifiedEmbedding {
    Quantized(QuantizedEmbedding),
    Regular(Embedding),
}

impl UnifiedEmbedding {
    /// Load from weight map, auto-detecting quantization
    ///
    /// Detects quantization by checking for `.scales` key in weights.
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let scales_name = format!("{}.scales", prefix);

        if weights.contains_key(&scales_name) {
            // Quantized embedding
            Ok(Self::Quantized(QuantizedEmbedding::from_weights(
                weights, prefix, group_size, bits,
            )?))
        } else {
            // Regular embedding
            Ok(Self::Regular(Embedding::from_weights(weights, prefix)?))
        }
    }

    /// Embedding lookup
    pub fn forward(&self, indices: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized(e) => e.forward(indices),
            Self::Regular(e) => e.forward(indices),
        }
    }

    /// Use embedding as linear projection
    pub fn as_linear(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized(e) => e.as_linear(x),
            Self::Regular(e) => e.as_linear(x),
        }
    }

    /// Check if this is a quantized embedding
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }
}

// =============================================================================
// Cache Layers
// =============================================================================

/// KV Cache for attention layers
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
                if shape.len() >= 3 { shape[2] } else { 0 }
            }
            None => 0,
        }
    }

    /// Update cache with new key/value using pre-allocated buffer + slice_update
    pub fn update(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
        let key_shape = ffi::array_shape(&new_keys);
        let new_seq_len = key_shape[2]; // [batch, heads, seq_len, head_dim]
        let prev = self.offset;

        // Grow buffer if needed
        if self.keys.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_head_dim = key_shape[3];
            let val_shape = ffi::array_shape(&new_values);
            let v_head_dim = val_shape[3];

            // Allocate in steps (matching Python's step=256)
            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let k_dtype = ffi::array_dtype(&new_keys);
            let v_dtype = ffi::array_dtype(&new_values);
            let new_k = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], k_dtype);
            let new_v = ffi::zeros(&[b, n_kv_heads, buf_size, v_head_dim], v_dtype);

            if self.keys.is_some() {
                // Trim existing buffer to exact offset length if not aligned
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
                // Concatenate existing data with new buffer space
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k, 2));
                self.values = Some(concatenate(self.values.as_ref().unwrap(), &new_v, 2));
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        self.offset += new_seq_len;

        // Slice assignment: buffer[..., prev:offset, :] = new_data
        // Uses MLX slice_update which can reuse the buffer in-place when refcount == 1
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

    /// Trim the last `n` entries from the cache
    /// Returns the number of entries actually trimmed
    /// Used by: Speculative decoding (cache rewinding for rejected draft tokens)
    pub fn trim(&mut self, n: i32) -> i32 {
        let n = n.min(self.offset);
        if n <= 0 {
            return 0;
        }
        self.offset -= n;
        if self.offset == 0 {
            self.keys = None;
            self.values = None;
        }
        // No need to slice the buffer; offset tracks the valid range
        n
    }

    /// Update cache and return view of filled portion
    /// Takes ownership of new k/v and returns sliced views of the buffer
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update(new_keys, new_values);

        // Return slice view of filled portion: buffer[..., :offset, :]
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

// =============================================================================
// Rotating KV Cache (for sliding window attention)
// =============================================================================

/// Rotating KV Cache for sliding window attention (e.g., Gemma 3, Ministral 3)
/// Maintains a fixed-size circular buffer for keys/values
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
                shape[2] // [batch, heads, seq_len, head_dim]
            } else {
                0
            }
        } else {
            0
        }
    }

    /// Update cache with new key/value, rotating if necessary
    /// Returns the full cached keys/values
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let new_seq_len = {
            let shape = ffi::array_shape(&new_keys);
            shape[2] // [batch, heads, seq_len, head_dim]
        };

        // For prefill (new_seq_len > 1), use concat path
        if new_seq_len > 1 {
            return self.update_concat(new_keys, new_values, new_seq_len);
        }

        // Single token generation: use in-place circular buffer
        self.update_in_place(new_keys, new_values)
    }

    /// Prefill path: concatenate new tokens and trim if exceeding max_size
    fn update_concat(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        new_seq_len: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        if self.keys.is_none() {
            // First prefill, just store
            self.offset += new_seq_len;
            self.idx = new_seq_len;
            self.keys = Some(ffi::contiguous(&new_keys, false));
            self.values = Some(ffi::contiguous(&new_values, false));
            return (new_keys, new_values);
        }

        // Subsequent prefill: concatenate and trim if needed
        let current_seq_len = {
            let shape = ffi::array_shape(self.keys.as_ref().unwrap());
            shape[2]
        };

        let concat_k = concatenate(self.keys.as_ref().unwrap(), &new_keys, 2);
        let concat_v = concatenate(self.values.as_ref().unwrap(), &new_values, 2);

        let total_len = current_seq_len + new_seq_len;
        self.offset += new_seq_len;

        if total_len > self.max_size {
            // Trim from the beginning (keep last max_size tokens)
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

    /// Single-token generation path: in-place update with circular buffer
    fn update_in_place(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // If buffer is larger than max_size (from an oversized first prefill),
        // trim to the last max_size tokens so circular indexing works correctly.
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            let buffer_size = shape[2];
            if buffer_size > self.max_size {
                let start = buffer_size - self.max_size;
                let ks = ffi::array_shape(self.keys.as_ref().unwrap());
                let vs = ffi::array_shape(self.values.as_ref().unwrap());
                self.keys = Some(ffi::contiguous(&ffi::slice(
                    self.keys.as_ref().unwrap(),
                    &[0, 0, start, 0],
                    &[ks[0], ks[1], buffer_size, ks[3]],
                ), false));
                self.values = Some(ffi::contiguous(&ffi::slice(
                    self.values.as_ref().unwrap(),
                    &[0, 0, start, 0],
                    &[vs[0], vs[1], buffer_size, vs[3]],
                ), false));
                // Buffer is now linearized to max_size; idx should wrap to 0
                self.idx = self.max_size;
            }
        }

        if self.keys.is_none() {
            // First token, initialize buffer
            let shape = ffi::array_shape(&new_keys);
            let batch = shape[0];
            let heads = shape[1];
            let head_dim = shape[3];

            let k_zeros = ffi::zeros(&[batch, heads, self.max_size, head_dim], ffi::array_dtype(&new_keys));
            let v_zeros = ffi::zeros(&[batch, heads, self.max_size, head_dim], ffi::array_dtype(&new_values));

            let k = ffi::slice_update(&k_zeros, &new_keys, &[0, 0, 0, 0], &[batch, heads, 1, head_dim]);
            let v = ffi::slice_update(&v_zeros, &new_values, &[0, 0, 0, 0], &[batch, heads, 1, head_dim]);

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

        // Buffer growth: if buffer is smaller than max_size, grow by concatenation
        if self.idx >= buffer_size && buffer_size < self.max_size {
            let k_concat = concatenate(&k_buffer, &new_keys, 2);
            let v_concat = concatenate(&v_buffer, &new_values, 2);

            self.offset += 1;
            self.idx += 1;
            self.keys = Some(ffi::contiguous(&k_concat, false));
            self.values = Some(ffi::contiguous(&v_concat, false));
            return (k_concat, v_concat);
        }

        // Wrap around: when idx reaches max_size, reset to 0
        if self.idx >= self.max_size {
            self.idx = 0;
        }

        // Write at current idx position
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

        // Return the valid portion of the cache
        if self.offset < self.max_size {
            let k_out = ffi::slice(&k_buffer, &[0, 0, 0, 0], &[batch, heads, self.offset, head_dim]);
            let v_out = ffi::slice(&v_buffer, &[0, 0, 0, 0], &[batch, heads, self.offset, head_dim]);
            self.keys = Some(k_buffer);
            self.values = Some(v_buffer);
            (k_out, v_out)
        } else {
            // Buffer is full, return all
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
        Self::new(4096) // Default sliding window size
    }
}

// =============================================================================
// Chunked KV Cache (for Llama 4 iGQA)
// =============================================================================

/// Chunked KV Cache for Llama 4's iGQA (Interleaved GQA) attention
/// Maintains a sliding window cache that trims from the front when exceeding chunk_size
pub struct ChunkedKVCache {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    pub chunk_size: i32,
    pub offset: i32,
    pub start_position: i32,
    step: i32, // Buffer growth step
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

    /// Trim the front of the cache if it exceeds chunk_size
    /// This should be called before processing each layer
    pub fn maybe_trim_front(&mut self) {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            let seq_len = shape[2]; // [batch, heads, seq_len, head_dim]

            if seq_len >= self.chunk_size {
                // Keep only the last chunk_size tokens
                let trim_amount = seq_len - self.chunk_size;
                self.start_position += trim_amount;

                // Slice to keep only [-chunk_size:] along axis 2
                let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
                let v_shape = ffi::array_shape(self.values.as_ref().unwrap());

                // Slice keys: [..., -chunk_size:, :]
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

        // Calculate current position in the buffer
        let prev = self.offset - self.start_position;

        if self.keys.is_none() || (prev + new_seq_len) > self.get_buffer_size() {
            // Need to grow the buffer
            let b = new_shape[0];
            let n_kv_heads = new_shape[1];
            let k_head_dim = new_shape[3];
            let v_shape = ffi::array_shape(&new_values);
            let v_head_dim = v_shape[3];

            // Calculate new buffer size (round up to step)
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
                // Concatenate existing with new buffer
                if prev % self.step != 0 {
                    // Trim existing to exact prev length
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

        // Update offset
        self.offset += new_seq_len;
        let end = self.offset - self.start_position;

        // Write new keys/values to buffer using scatter/index update
        // For now, use concatenation approach similar to KVCache
        // TODO: Optimize with in-place scatter if needed
        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());

        // Slice before prev, new values, slice after end
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
                self.keys = Some(concatenate(&concatenate(&k_before, &new_keys, 2), &k_after, 2));
                self.values =
                    Some(concatenate(&concatenate(&v_before, &new_values, 2), &v_after, 2));
            } else {
                self.keys = Some(concatenate(&k_before, &new_keys, 2));
                self.values = Some(concatenate(&v_before, &new_values, 2));
            }
        } else {
            // prev == 0, just update from the start
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
                self.keys = Some(concatenate(&new_keys, &k_after, 2));
                self.values = Some(concatenate(&new_values, &v_after, 2));
            } else {
                self.keys = Some(ffi::contiguous(&new_keys, false));
                self.values = Some(ffi::contiguous(&new_values, false));
            }
        }

        // Return visible portion: [:end]
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

    /// Get current buffer size
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
        Self::new(8192) // Default Llama 4 chunk size
    }
}

/// RMS Normalization layer
pub struct RMSNorm {
    pub weight: UniquePtr<MlxArray>,
    pub eps: f32,
}

impl RMSNorm {
    /// Create a new RMS norm layer
    pub fn new(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Forward pass using fast RMS norm kernel
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        ffi::fast_rms_norm(x, &self.weight, self.eps)
    }
}

/// Gemma-style RMS Normalization layer with (1 + weight) pattern
pub struct GemmaRMSNorm {
    pub weight: UniquePtr<MlxArray>,
    pub eps: f32,
}

impl GemmaRMSNorm {
    /// Create a new Gemma RMS norm layer
    pub fn new(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Forward pass using fast RMS norm kernel with (1 + weight)
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Gemma models use (1 + weight) instead of just weight
        let ones = ffi::ones(&[ffi::array_shape(&self.weight)[0]], ffi::array_dtype(&self.weight));
        let adjusted_weight = ffi::add(&ones, &self.weight);
        ffi::fast_rms_norm(x, &adjusted_weight, self.eps)
    }
}

/// Layer Normalization layer (standard LayerNorm with weight and optional bias)
pub struct LayerNorm {
    pub weight: UniquePtr<MlxArray>,
    pub bias: Option<UniquePtr<MlxArray>>,
    pub eps: f32,
}

impl LayerNorm {
    /// Create a new layer norm
    pub fn new(weight: UniquePtr<MlxArray>, bias: Option<UniquePtr<MlxArray>>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Forward pass using fast layer norm kernel
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let weight_ptr = self.weight.as_ref().unwrap() as *const MlxArray;
        let bias_ptr = self
            .bias
            .as_ref()
            .map(|b| b.as_ref().unwrap() as *const MlxArray)
            .unwrap_or(std::ptr::null());

        unsafe { ffi::fast_layer_norm(x, weight_ptr, bias_ptr, self.eps) }
    }
}

/// Regular (non-quantized) Linear layer
pub struct Linear {
    pub weight: UniquePtr<MlxArray>,
    pub bias: Option<UniquePtr<MlxArray>>,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(weight: UniquePtr<MlxArray>, bias: Option<UniquePtr<MlxArray>>) -> Self {
        Self { weight, bias }
    }

    /// Load from weight map
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);

        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;

        // Check for optional bias
        let bias_name = format!("{}.bias", prefix);
        let bias = weights.get(&bias_name).map(|w| ffi::copy(w));

        Ok(Self { weight, bias })
    }

    /// Forward pass: y = x @ W.T + bias
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Transpose weight from [out, in] to [in, out]
        let wt = ffi::transpose(&self.weight);
        let result = ffi::matmul(x, &wt);

        match &self.bias {
            Some(b) => ffi::add(&result, b),
            None => result,
        }
    }
}

/// Unified Linear layer that auto-detects quantization
///
/// Checks for `.scales` key in weight map to determine whether to use
/// quantized or regular linear operations. Replaces both the old
/// `QuantizedLinear` and `UnifiedLinear` types.
///
/// Used by: all text/VLM model implementations
pub enum UnifiedLinear {
    Quantized {
        weight: QuantizedWeight,
        bias: Option<UniquePtr<MlxArray>>,
    },
    Regular(Linear),
}

/// Backward-compatible alias
pub type QuantizedLinear = UnifiedLinear;

impl UnifiedLinear {
    /// Create a new quantized linear layer (explicit construction)
    pub fn new(weight: QuantizedWeight, bias: Option<UniquePtr<MlxArray>>) -> Self {
        Self::Quantized { weight, bias }
    }

    /// Load from weight map, auto-detecting quantization
    ///
    /// Detects quantization by checking for `.scales` key in weights.
    /// Falls back to regular Linear if scales are absent.
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let scales_name = format!("{}.scales", prefix);

        if weights.contains_key(&scales_name) {
            // Quantized path
            let weight_name = format!("{}.weight", prefix);
            let biases_name = format!("{}.biases", prefix);

            let weight = weights
                .get(&weight_name)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
            let scales = weights
                .get(&scales_name)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Scales not found: {}", scales_name))?;
            let biases = weights
                .get(&biases_name)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Biases not found: {}", biases_name))?;

            let qweight = QuantizedWeight {
                weight,
                scales,
                biases,
                group_size,
                bits,
            };

            let bias_name = format!("{}.bias", prefix);
            let bias = weights.get(&bias_name).map(|w| ffi::copy(w));

            Ok(Self::Quantized {
                weight: qweight,
                bias,
            })
        } else {
            // Fallback to regular linear (non-quantized model)
            Ok(Self::Regular(Linear::from_weights(weights, prefix)?))
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized { weight, bias } => {
                let bias_ptr = bias
                    .as_ref()
                    .map(|b| b.as_ref().unwrap() as *const MlxArray)
                    .unwrap_or(std::ptr::null());

                unsafe {
                    ffi::quantized_linear_forward(
                        x,
                        &weight.weight,
                        &weight.scales,
                        &weight.biases,
                        bias_ptr,
                        weight.group_size,
                        weight.bits,
                    )
                }
            }
            Self::Regular(linear) => linear.forward(x),
        }
    }

    /// Check if this is a quantized linear layer
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }
}

/// Quantized per-head linear layer for MLA (Multi-head Latent Attention)
/// Weight shape: [num_heads, output_dim, input_dim_packed]
/// Used in GLM4 MoE Lite, DeepSeek-V2, etc.
pub struct QuantizedMultiLinear {
    pub weight: UniquePtr<MlxArray>,
    pub scales: UniquePtr<MlxArray>,
    pub biases: Option<UniquePtr<MlxArray>>,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedMultiLinear {
    /// Create a new quantized multi-linear layer
    pub fn new(
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: Option<UniquePtr<MlxArray>>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        }
    }

    /// Load from weight map
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let scales_name = format!("{}.scales", prefix);
        let biases_name = format!("{}.biases", prefix);

        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
        let scales = weights
            .get(&scales_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Scales not found: {}", scales_name))?;
        let biases = weights.get(&biases_name).map(|w| ffi::copy(w));

        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    /// Forward pass: per-head linear projection
    /// x: [batch, heads, seq, input_dim]
    /// Returns: [batch, heads, seq, output_dim]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Get raw pointer to biases if present
        let biases_ptr: *const MlxArray = match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        };

        unsafe {
            ffi::quantized_matmul(
                x,
                &self.weight,
                &self.scales,
                biases_ptr,
                true, // transpose
                self.group_size,
                self.bits,
            )
        }
    }

    /// Forward pass without transpose: x @ weight
    /// Used by MLA embed_q(kv_latent, transpose=False) for projecting latent to K
    pub fn forward_no_transpose(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let biases_ptr: *const MlxArray = match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        };

        unsafe {
            ffi::quantized_matmul(
                x,
                &self.weight,
                &self.scales,
                biases_ptr,
                false, // no transpose
                self.group_size,
                self.bits,
            )
        }
    }

    /// Dequantize weights to full precision
    /// Returns: [num_heads, output_dim, input_dim]
    pub fn dequantize(&self) -> UniquePtr<MlxArray> {
        let biases = self
            .biases
            .as_ref()
            .map(|b| ffi::copy(b))
            .unwrap_or_else(|| ffi::zeros(&[1], crate::dtype::FLOAT16));

        ffi::dequantize(&self.weight, &self.scales, &biases, self.group_size, self.bits)
    }
}

/// SwiGLU MLP layer with optional compilation for kernel fusion
pub struct SwiGLUMLP {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
    pub use_compiled: bool,
}

impl SwiGLUMLP {
    /// Create a new SwiGLU MLP layer
    pub fn new(
        gate_proj: QuantizedWeight,
        up_proj: QuantizedWeight,
        down_proj: QuantizedWeight,
        use_compiled: bool,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
            use_compiled,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if self.use_compiled {
            // Use compiled version with kernel fusion
            ffi::compiled_moe_expert_forward(
                x,
                &self.gate_proj.weight,
                &self.gate_proj.scales,
                &self.gate_proj.biases,
                &self.up_proj.weight,
                &self.up_proj.scales,
                &self.up_proj.biases,
                &self.down_proj.weight,
                &self.down_proj.scales,
                &self.down_proj.biases,
                self.gate_proj.group_size,
                self.gate_proj.bits,
            )
        } else {
            // Non-compiled version
            // gate = quantized_matmul(x, gate_proj)
            let gate = unsafe {
                ffi::quantized_linear_forward(
                    x,
                    &self.gate_proj.weight,
                    &self.gate_proj.scales,
                    &self.gate_proj.biases,
                    std::ptr::null(),
                    self.gate_proj.group_size,
                    self.gate_proj.bits,
                )
            };

            // up = quantized_matmul(x, up_proj)
            let up = unsafe {
                ffi::quantized_linear_forward(
                    x,
                    &self.up_proj.weight,
                    &self.up_proj.scales,
                    &self.up_proj.biases,
                    std::ptr::null(),
                    self.up_proj.group_size,
                    self.up_proj.bits,
                )
            };

            // activated = silu(gate) * up
            let silu_gate = ffi::silu(&gate);
            let activated = ffi::multiply(&silu_gate, &up);

            // down = quantized_matmul(activated, down_proj)
            unsafe {
                ffi::quantized_linear_forward(
                    &activated,
                    &self.down_proj.weight,
                    &self.down_proj.scales,
                    &self.down_proj.biases,
                    std::ptr::null(),
                    self.down_proj.group_size,
                    self.down_proj.bits,
                )
            }
        }
    }
}

/// MoE Switch layer using gather_qmm for efficient expert routing
pub struct MoESwitch {
    /// Expert weights: [num_experts, ...]
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
    pub num_experts: i32,
}

impl MoESwitch {
    /// Create a new MoE switch layer
    pub fn new(
        gate_proj: QuantizedWeight,
        up_proj: QuantizedWeight,
        down_proj: QuantizedWeight,
        num_experts: i32,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
            num_experts,
        }
    }

    /// Forward pass with expert indices
    /// x: [batch, seq_len, hidden_dim]
    /// indices: [batch, seq_len] - expert index for each token
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        // Use gather_qmm for efficient MoE computation
        // gate = gather_qmm(x, gate_proj, indices)
        let gate = unsafe {
            ffi::gather_qmm(
                x,
                &self.gate_proj.weight,
                &self.gate_proj.scales,
                self.gate_proj
                    .biases
                    .as_ref()
                    .map(|b| b as *const _)
                    .unwrap_or(std::ptr::null()),
                std::ptr::null(),    // lhs_indices
                indices as *const _, // rhs_indices
                true,                // transpose
                self.gate_proj.group_size,
                self.gate_proj.bits,
                false, // sorted_indices
            )
        };

        // up = gather_qmm(x, up_proj, indices)
        let up = unsafe {
            ffi::gather_qmm(
                x,
                &self.up_proj.weight,
                &self.up_proj.scales,
                self.up_proj
                    .biases
                    .as_ref()
                    .map(|b| b as *const _)
                    .unwrap_or(std::ptr::null()),
                std::ptr::null(),
                indices as *const _,
                true,
                self.up_proj.group_size,
                self.up_proj.bits,
                false,
            )
        };

        // activated = silu(gate) * up (using compiled version for kernel fusion)
        let activated = ffi::compiled_swiglu_activation(&gate, &up);

        // down = gather_qmm(activated, down_proj, indices)
        unsafe {
            ffi::gather_qmm(
                &activated,
                &self.down_proj.weight,
                &self.down_proj.scales,
                self.down_proj
                    .biases
                    .as_ref()
                    .map(|b| b as *const _)
                    .unwrap_or(std::ptr::null()),
                std::ptr::null(),
                indices as *const _,
                true,
                self.down_proj.group_size,
                self.down_proj.bits,
                false,
            )
        }
    }
}

/// Attention layer with RoPE and KV cache
pub struct Attention {
    pub q_proj: QuantizedWeight,
    pub k_proj: QuantizedWeight,
    pub v_proj: QuantizedWeight,
    pub o_proj: QuantizedWeight,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub rope_scale: f32,
}

impl Attention {
    /// Create a new attention layer
    pub fn new(
        q_proj: QuantizedWeight,
        k_proj: QuantizedWeight,
        v_proj: QuantizedWeight,
        o_proj: QuantizedWeight,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rope_dims: i32,
        rope_base: f32,
        rope_scale: f32,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dims,
            rope_base,
            rope_scale,
        }
    }

    /// Forward pass
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(x);
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let q = unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.q_proj.weight,
                &self.q_proj.scales,
                &self.q_proj.biases,
                std::ptr::null(),
                self.q_proj.group_size,
                self.q_proj.bits,
            )
        };

        let k = unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.k_proj.weight,
                &self.k_proj.scales,
                &self.k_proj.biases,
                std::ptr::null(),
                self.k_proj.group_size,
                self.k_proj.bits,
            )
        };

        let v = unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.v_proj.weight,
                &self.v_proj.scales,
                &self.v_proj.biases,
                std::ptr::null(),
                self.v_proj.group_size,
                self.v_proj.bits,
            )
        };

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = ffi::reshape(&q, &[batch_size, seq_len, self.n_heads, self.head_dim]);
        let k = ffi::reshape(&k, &[batch_size, seq_len, self.n_kv_heads, self.head_dim]);
        let v = ffi::reshape(&v, &[batch_size, seq_len, self.n_kv_heads, self.head_dim]);

        // Apply RoPE
        let offset = cache.offset;
        let q = ffi::fast_rope(
            &q,
            self.rope_dims,
            false,
            self.rope_base,
            self.rope_scale,
            offset,
        );
        let k = ffi::fast_rope(
            &k,
            self.rope_dims,
            false,
            self.rope_base,
            self.rope_scale,
            offset,
        );

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = ffi::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = ffi::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = ffi::transpose_axes(&v, &[0, 2, 1, 3]);

        // Update KV cache and get sliced views
        let (k, v) = cache.update_and_fetch(k, v);

        // Compute attention scale
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Scaled dot-product attention
        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let attn_out =
            unsafe { ffi::fast_scaled_dot_product_attention(&q, &k, &v, scale, mask_ptr) };

        // Transpose back and reshape
        let attn_out = ffi::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = ffi::reshape(
            &attn_out,
            &[batch_size, seq_len, self.n_heads * self.head_dim],
        );

        // Output projection
        unsafe {
            ffi::quantized_linear_forward(
                &attn_out,
                &self.o_proj.weight,
                &self.o_proj.scales,
                &self.o_proj.biases,
                std::ptr::null(),
                self.o_proj.group_size,
                self.o_proj.bits,
            )
        }
    }
}

// =============================================================================
// MultiLinear Layer (for MLA attention: embed_q, unembed_out)
// =============================================================================

/// MultiLinear layer for per-head linear projections.
///
/// Weight shape: `[num_heads, output_dims, input_dims]`
///
/// Used by MLA attention (DeepSeek V3/V3.2, GLM4 MoE Lite) for:
/// - `embed_q`: projects Q_nope into KV latent space
/// - `unembed_out`: projects attention output from latent space to V dimensions
///
/// Supports both quantized and non-quantized weights.
/// Used by: DeepSeek V3, DeepSeek V3.2, GLM4 MoE Lite
pub enum MultiLinear {
    Quantized(QuantizedMultiLinear),
    Regular(RegularMultiLinear),
}

/// Non-quantized multi-head linear layer.
/// Weight shape: `[num_heads, output_dims, input_dims]`
pub struct RegularMultiLinear {
    pub weight: UniquePtr<MlxArray>,
}

impl MultiLinear {
    /// Load from weight map, auto-detecting quantization.
    ///
    /// Checks for `.scales` key to determine if weights are quantized.
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let scales_name = format!("{}.scales", prefix);

        if weights.contains_key(&scales_name) {
            // Quantized: use existing QuantizedMultiLinear
            Ok(MultiLinear::Quantized(QuantizedMultiLinear::from_weights(
                weights, prefix, group_size, bits,
            )?))
        } else {
            // Non-quantized: regular weight
            let weight = weights
                .get(&weight_name)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
            Ok(MultiLinear::Regular(RegularMultiLinear { weight }))
        }
    }

    /// Forward pass with transpose (default behavior).
    ///
    /// Computes `x @ weight.swapaxes(-1, -2)`.
    /// Input x: `[..., num_heads, seq_len, input_dims]`
    /// Output: `[..., num_heads, seq_len, output_dims]`
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MultiLinear::Quantized(q) => q.forward(x),
            MultiLinear::Regular(r) => {
                // weight is [num_heads, output_dims, input_dims]
                // swapaxes(-1, -2) → [num_heads, input_dims, output_dims]
                let wt = ffi::transpose_axes(&r.weight, &[0, 2, 1]);
                ffi::matmul(x, &wt)
            }
        }
    }

    /// Forward pass without transpose.
    ///
    /// Computes `x @ weight`.
    /// Input x: `[..., 1_or_heads, seq_len, output_dims]`
    /// Output: `[..., num_heads, seq_len, input_dims]`
    pub fn forward_no_transpose(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MultiLinear::Quantized(q) => q.forward_no_transpose(x),
            MultiLinear::Regular(r) => ffi::matmul(x, &r.weight),
        }
    }
}

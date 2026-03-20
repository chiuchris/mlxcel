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

//! Rust-native serializable representations of KV cache state.
//!
//! These types bridge the gap between the FFI-bound cache types
//! (`KVCache`, `RotatingKVCache`, `ChunkedKVCache`) and the wire format.
//! They contain only plain Rust data (no `UniquePtr<MlxArray>`) and can
//! be freely serialized, sent over the network, and deserialized.

use serde::{Deserialize, Serialize};

use super::super::tensor_protocol::TensorDtype;

/// Discriminant for the cache variant being serialized.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CacheType {
    /// Standard KV cache (pre-allocated buffer with offset).
    Standard = 0,
    /// Rotating (sliding-window) KV cache.
    Rotating = 1,
    /// Chunked KV cache (Llama 4 iGQA).
    Chunked = 2,
}

impl TryFrom<u8> for CacheType {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> anyhow::Result<Self> {
        match value {
            0 => Ok(Self::Standard),
            1 => Ok(Self::Rotating),
            2 => Ok(Self::Chunked),
            other => anyhow::bail!("unknown cache type: {other}"),
        }
    }
}

/// Raw tensor data extracted from an `MlxArray`.
///
/// Contains the byte contents, shape, and dtype needed to reconstruct
/// the tensor on the receiving side via `ffi::from_bytes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTensorData {
    /// Raw bytes of the evaluated, contiguous tensor.
    pub data: Vec<u8>,
    /// Shape of the tensor (e.g. `[batch, n_kv_heads, seq_len, head_dim]`).
    pub shape: Vec<i32>,
    /// MLX dtype code (matches `mlxcel_core::dtype` constants).
    pub dtype: i32,
}

/// One layer's key/value pair in serializable form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCacheEntry {
    /// Serialized key tensor (None if cache layer is empty).
    pub keys: Option<RawTensorData>,
    /// Serialized value tensor (None if cache layer is empty).
    pub values: Option<RawTensorData>,
}

/// Metadata for the serialized cache state.
///
/// Serialized as JSON within the wire format so it can evolve without
/// breaking the binary framing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Number of prompt tokens that were originally prefilled.
    pub prompt_len: usize,
    /// Current generation offset (total tokens processed).
    pub current_offset: i32,
    /// Number of layers in the cache.
    pub num_layers: usize,
    /// Per-layer offsets (for Standard/Chunked caches).
    pub layer_offsets: Vec<i32>,

    // -- Rotating-specific fields --
    /// Maximum window size (only meaningful for Rotating caches).
    pub max_size: Option<i32>,
    /// Per-layer write indices (only meaningful for Rotating caches).
    pub layer_indices: Option<Vec<i32>>,

    // -- Chunked-specific fields --
    /// Chunk size (only meaningful for Chunked caches).
    pub chunk_size: Option<i32>,
    /// Per-layer start positions (only meaningful for Chunked caches).
    pub start_positions: Option<Vec<i32>>,
}

/// Serializable sampling state, mirroring the live `SamplingConfig` fields
/// that are relevant to decode continuation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSamplingState {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: Option<u64>,
    pub repetition_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_penalty_last_n: usize,
    pub dry_sequence_breakers: Vec<i32>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop_token_ids: Vec<i32>,
}

/// Complete serializable cache state for one sequence.
///
/// This is the top-level type that gets serialized to the wire format
/// and transferred from prefill node to decode node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCacheState {
    /// Cache variant discriminant.
    pub cache_type: CacheType,
    /// Per-layer cache entries.
    pub entries: Vec<SerializableCacheEntry>,
    /// Cache and sequence metadata.
    pub metadata: CacheMetadata,
    /// Sampling parameters for decode continuation.
    pub sampling_state: Option<SerializableSamplingState>,
    /// Token history for repetition/DRY penalties.
    pub token_history: Vec<i32>,
    /// Unique sequence ID (from CachePool).
    pub sequence_id: u64,
}

/// Convert MLX dtype code to `TensorDtype` for the tensor protocol.
///
/// Used when serializing cache tensors through the existing tensor
/// wire format.
pub fn mlx_dtype_to_tensor_dtype(mlx_dtype: i32) -> anyhow::Result<TensorDtype> {
    // MLX dtype codes from mlxcel_core::dtype
    match mlx_dtype {
        0 => Ok(TensorDtype::Bool),      // BOOL
        5 => Ok(TensorDtype::Int8),      // INT8
        6 => Ok(TensorDtype::Int16),     // INT16
        7 => Ok(TensorDtype::Int32),     // INT32
        9 => Ok(TensorDtype::Float16),   // FLOAT16
        10 => Ok(TensorDtype::Float32),  // FLOAT32
        12 => Ok(TensorDtype::BFloat16), // BFLOAT16
        other => anyhow::bail!("unsupported MLX dtype code for serialization: {other}"),
    }
}

/// Return the element size in bytes for an MLX dtype code.
///
/// Used by: deserialization validation to verify tensor data buffer sizes.
pub fn mlx_dtype_element_size(mlx_dtype: i32) -> anyhow::Result<usize> {
    match mlx_dtype {
        0 => Ok(1),               // BOOL
        1 => Ok(1),               // UINT8
        5 => Ok(1),               // INT8
        2 | 6 | 9 => Ok(2),       // UINT16, INT16, FLOAT16
        3 | 7 | 10 | 12 => Ok(4), // UINT32, INT32, FLOAT32, BFLOAT16
        4 | 8 | 11 => Ok(8),      // UINT64, INT64, FLOAT64
        13 => Ok(8),              // COMPLEX64
        other => anyhow::bail!("unknown MLX dtype code: {other}"),
    }
}

/// Validate that a `RawTensorData` is internally consistent.
///
/// Checks:
/// - All shape dimensions are non-negative.
/// - `data.len()` matches the expected byte count from shape and dtype.
///
/// Used by: deserialization to reject malformed tensors before passing
/// to `ffi::from_bytes`.
pub fn validate_raw_tensor(tensor: &RawTensorData) -> anyhow::Result<()> {
    // Reject negative shape dimensions
    for (i, &dim) in tensor.shape.iter().enumerate() {
        if dim < 0 {
            anyhow::bail!("invalid shape dimension at axis {i}: {dim} (must be non-negative)");
        }
    }

    let element_size = mlx_dtype_element_size(tensor.dtype)?;

    let num_elements: usize = tensor.shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim as usize)
            .ok_or_else(|| anyhow::anyhow!("shape overflow computing total elements"))
    })?;

    let expected_bytes = num_elements
        .checked_mul(element_size)
        .ok_or_else(|| anyhow::anyhow!("byte count overflow for tensor"))?;

    if tensor.data.len() != expected_bytes {
        anyhow::bail!(
            "tensor data size mismatch: shape {:?} with dtype {} expects {} bytes, got {}",
            tensor.shape,
            tensor.dtype,
            expected_bytes,
            tensor.data.len()
        );
    }

    Ok(())
}

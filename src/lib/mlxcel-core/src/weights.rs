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

//! Weight loading utilities for mlx-cxx
//!
//! This module provides functions to load model weights from safetensors files
//! using MLX's native C++ `load_safetensors()`. Arrays are lazy and MLX manages
//! the file mmap internally, eliminating the need for eager materialization and
//! Rust-side mmap lifetime management.

use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;
use std::collections::HashMap;
use std::path::Path;

/// Loaded model weights as a map of tensor names to mlx-cxx arrays
pub type WeightMap = HashMap<String, UniquePtr<MlxArray>>;

/// Load all weights from a directory containing safetensors files.
///
/// Uses MLX's native `load_safetensors()` which returns lazy arrays with
/// MLX-managed mmap. No `synchronize_default()` barrier is needed because
/// MLX owns the file mappings and materializes tensors on demand.
pub fn load_weights_from_dir<P: AsRef<Path>>(dir: P) -> Result<WeightMap, String> {
    let dir = dir.as_ref();
    let mut weights = HashMap::new();

    // Collect and sort shard paths for deterministic ordering
    let mut shard_paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory: {e}"))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    shard_paths.sort(); // deterministic order

    for path in &shard_paths {
        let path_str = path
            .to_str()
            .ok_or_else(|| format!("Non-UTF8 path: {}", path.display()))?;
        let mut loaded = ffi::mlx_load_safetensors(path_str)
            .map_err(|e| format!("Failed to load {}: {e}", path.display()))?;
        let len = ffi::loaded_weights_len(&loaded);
        for i in 0..len {
            let name = ffi::loaded_weights_name(&loaded, i);
            let array = ffi::loaded_weights_take(loaded.pin_mut(), i);
            weights.insert(name, array);
        }
    }

    Ok(weights)
}

/// Load weights from a single safetensors file.
///
/// Uses MLX's native `load_safetensors()` which returns lazy arrays with
/// MLX-managed mmap. No synchronization barrier is needed.
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<WeightMap, String> {
    let path = path.as_ref();
    let path_str = path
        .to_str()
        .ok_or_else(|| format!("Non-UTF8 path: {}", path.display()))?;
    let mut loaded = ffi::mlx_load_safetensors(path_str)
        .map_err(|e| format!("Failed to load {}: {e}", path.display()))?;
    let len = ffi::loaded_weights_len(&loaded);
    let mut weights = HashMap::with_capacity(len);
    for i in 0..len {
        let name = ffi::loaded_weights_name(&loaded, i);
        let array = ffi::loaded_weights_take(loaded.pin_mut(), i);
        weights.insert(name, array);
    }
    Ok(weights)
}

/// Get a weight from the weight map, with optional prefix
pub fn get_weight<'a>(weights: &'a WeightMap, name: &str) -> Option<&'a UniquePtr<MlxArray>> {
    weights.get(name)
}

/// Get a weight with a prefix (e.g., "model.layers.0.self_attn.q_proj.weight")
pub fn get_weight_with_prefix<'a>(
    weights: &'a WeightMap,
    prefix: &str,
    suffix: &str,
) -> Option<&'a UniquePtr<MlxArray>> {
    let full_name = format!("{prefix}.{suffix}");
    weights.get(&full_name)
}

/// Check if a weight exists in the weight map
pub fn has_weight(weights: &WeightMap, name: &str) -> bool {
    weights.contains_key(name)
}

/// Clone a weight from the map (creates a copy of the array)
pub fn clone_weight(weights: &WeightMap, name: &str) -> Option<UniquePtr<MlxArray>> {
    weights.get(name).map(|w| ffi::copy(w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;

    #[test]
    fn test_weight_map_operations() {
        let mut weights = WeightMap::new();

        // Create a test array
        let arr = ffi::ones(&[4, 4], dtype::FLOAT32);
        weights.insert("test.weight".to_string(), arr);

        // Check operations
        assert!(has_weight(&weights, "test.weight"));
        assert!(!has_weight(&weights, "nonexistent"));

        let w = get_weight(&weights, "test.weight").unwrap();
        let shape = ffi::array_shape(w);
        assert_eq!(shape, vec![4, 4]);

        // Clone
        let cloned = clone_weight(&weights, "test.weight").unwrap();
        let cloned_shape = ffi::array_shape(&cloned);
        assert_eq!(cloned_shape, vec![4, 4]);
    }
}

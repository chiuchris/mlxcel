//! Weight loading utilities for mlx-cxx
//!
//! This module provides functions to load model weights from safetensors files
//! and convert them to mlx-cxx arrays.

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Loaded model weights as a map of tensor names to mlx-cxx arrays
pub type WeightMap = HashMap<String, UniquePtr<MlxArray>>;

/// Load all weights from a directory containing safetensors files
pub fn load_weights_from_dir<P: AsRef<Path>>(dir: P) -> Result<WeightMap, String> {
    let dir = dir.as_ref();
    let mut weights = HashMap::new();

    // Find all safetensors files in the directory
    let entries = std::fs::read_dir(dir).map_err(|e| format!("Failed to read directory: {}", e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "safetensors") {
            let file_weights = load_safetensors(&path)?;
            weights.extend(file_weights);
        }
    }

    Ok(weights)
}

/// Load weights from a single safetensors file
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<WeightMap, String> {
    let path = path.as_ref();
    let file =
        File::open(path).map_err(|e| format!("Failed to open file {}: {}", path.display(), e))?;

    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("Failed to mmap file: {}", e))?;

    let tensors = SafeTensors::deserialize(&mmap)
        .map_err(|e| format!("Failed to deserialize safetensors: {}", e))?;

    let mut weights = HashMap::new();

    // First pass: create all arrays (lazy, referencing mmap)
    for (name, tensor_view) in tensors.tensors() {
        let array = tensor_to_mlx_array(&tensor_view)?;
        // Use async_eval to queue the copy without blocking
        ffi::async_eval(&array);
        weights.insert(name.to_string(), array);
    }

    // Synchronize to ensure all arrays are materialized before mmap goes away
    ffi::synchronize_default();

    Ok(weights)
}

/// Convert a safetensors tensor view to an mlx-cxx array
fn tensor_to_mlx_array(
    tensor: &safetensors::tensor::TensorView,
) -> Result<UniquePtr<MlxArray>, String> {
    let shape: Vec<i32> = tensor.shape().iter().map(|&d| d as i32).collect();
    let data = tensor.data();

    match tensor.dtype() {
        safetensors::Dtype::F32 => Ok(ffi::from_bytes(data, &shape, dtype::FLOAT32)),
        safetensors::Dtype::F16 => Ok(ffi::from_bytes_f16(data, &shape, false)),
        safetensors::Dtype::BF16 => Ok(ffi::from_bytes_f16(data, &shape, true)),
        safetensors::Dtype::I32 => Ok(ffi::from_bytes(data, &shape, dtype::INT32)),
        safetensors::Dtype::I64 => Ok(ffi::from_bytes(data, &shape, dtype::INT64)),
        safetensors::Dtype::U32 => Ok(ffi::from_bytes(data, &shape, dtype::UINT32)),
        safetensors::Dtype::U8 => Ok(ffi::from_bytes(data, &shape, dtype::UINT8)),
        dtype => Err(format!("Unsupported dtype: {:?}", dtype)),
    }
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
    let full_name = format!("{}.{}", prefix, suffix);
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

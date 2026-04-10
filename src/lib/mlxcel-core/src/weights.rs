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

/// Parse a `model.safetensors.index.json` file and return the set of unique shard filenames.
///
/// The index JSON format is:
/// ```json
/// {
///   "metadata": {"total_size": 123456},
///   "weight_map": {
///     "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
///     ...
///   }
/// }
/// ```
///
/// Returns `Ok(Some(shards))` if the file exists and is valid, `Ok(None)` if the file
/// does not exist, or `Err(...)` if the file exists but cannot be parsed.
pub fn parse_shard_index<P: AsRef<Path>>(dir: P) -> Result<Option<Vec<String>>, String> {
    let index_path = dir.as_ref().join("model.safetensors.index.json");
    if !index_path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&index_path)
        .map_err(|e| format!("Failed to read {}: {e}", index_path.display()))?;

    let shards = extract_shards_from_index_json(&content)
        .map_err(|e| format!("Failed to parse {}: {e}", index_path.display()))?;

    Ok(Some(shards))
}

/// Extract unique shard filenames from a HuggingFace safetensors index JSON string.
fn extract_shards_from_index_json(json: &str) -> Result<Vec<String>, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("Invalid JSON: {e}"))?;

    let weight_map = parsed
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "Missing \"weight_map\" key or not an object".to_string())?;

    let mut seen = std::collections::HashSet::new();
    let mut shards = Vec::new();
    for value in weight_map.values() {
        if let Some(s) = value.as_str() {
            if seen.insert(s.to_string()) {
                shards.push(s.to_string());
            }
        }
    }

    if shards.is_empty() {
        return Err("No shard filenames found in weight_map".to_string());
    }

    shards.sort();
    Ok(shards)
}

/// Check if a path is a broken symlink (exists as a symlink but target is missing).
///
/// Returns `(is_symlink, target_exists)`.
fn check_symlink(path: &Path) -> (bool, bool) {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => {
            // It is a symlink — check if the target resolves
            let target_exists = path.exists(); // follows symlink
            (true, target_exists)
        }
        _ => (false, true), // not a symlink, or metadata error → treat as not symlink
    }
}

/// Load all weights from a directory containing safetensors files.
///
/// Uses MLX's native `load_safetensors()` which returns lazy arrays with
/// MLX-managed mmap. No `synchronize_default()` barrier is needed because
/// MLX owns the file mappings and materializes tensors on demand.
///
/// If `model.safetensors.index.json` is present, it is parsed to obtain the
/// exact set of shard filenames. Otherwise all `*.safetensors` files in the
/// directory are loaded. Broken symlinks are detected and skipped with a
/// warning; if every candidate file is a broken symlink an error is returned.
pub fn load_weights_from_dir<P: AsRef<Path>>(dir: P) -> Result<WeightMap, String> {
    let dir = dir.as_ref();
    let mut weights = HashMap::new();

    // Determine which shard files to load
    let shard_paths = collect_shard_paths(dir)?;

    if shard_paths.is_empty() {
        return Err(format!(
            "No safetensors files found in directory: {}",
            dir.display()
        ));
    }

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

/// Collect and validate the list of shard paths to load from a model directory.
///
/// Uses the index JSON when present; otherwise globs all `*.safetensors` files.
/// Broken symlinks are skipped with a warning message. Returns an error if all
/// candidate files are broken symlinks or if the index references missing shards.
fn collect_shard_paths(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    // Try to parse the index file first
    let index_shards = parse_shard_index(dir)?;

    let candidates: Vec<std::path::PathBuf> = if let Some(shard_names) = index_shards {
        // Use the shard list from the index — deterministic and validated
        validate_index_shards(dir, &shard_names)?
    } else {
        // Fallback: glob all *.safetensors in the directory
        glob_safetensors(dir)?
    };

    // Filter out broken symlinks with warnings
    let mut valid_paths = Vec::new();
    let mut broken_count = 0usize;

    for path in candidates {
        let (is_symlink, target_exists) = check_symlink(&path);
        if is_symlink && !target_exists {
            broken_count += 1;
            let target = std::fs::read_link(&path)
                .map(|t| t.display().to_string())
                .unwrap_or_else(|_| "<unknown>".to_string());
            eprintln!(
                "Warning: skipping broken symlink {} -> {}\n  \
                 Hint: re-download with `huggingface-cli download --local-dir` \
                 to get real files instead of cache symlinks.",
                path.display(),
                target
            );
        } else {
            valid_paths.push(path);
        }
    }

    if valid_paths.is_empty() && broken_count > 0 {
        return Err(format!(
            "All {broken_count} safetensors file(s) in {} are broken symlinks.\n\
             Re-download the model with:\n  \
             huggingface-cli download <model-id> --local-dir {}",
            dir.display(),
            dir.display()
        ));
    }

    Ok(valid_paths)
}

/// Validate shard filenames from the index, returning their full paths.
/// Returns an error listing any missing shards.
///
/// Shard filenames are validated to be plain filenames (no path separators or
/// parent-directory components) to prevent path traversal attacks via malicious
/// index JSON files.
fn validate_index_shards(
    dir: &Path,
    shard_names: &[String],
) -> Result<Vec<std::path::PathBuf>, String> {
    let mut missing = Vec::new();
    let mut paths = Vec::new();

    for name in shard_names {
        // Security: reject any shard name that is not a plain filename.
        // This prevents path traversal via entries like "../secret.safetensors"
        // or absolute paths in a malicious index.json from an untrusted model repo.
        let shard_path = Path::new(name);
        if shard_path.is_absolute()
            || shard_path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            || name.contains('\0')
            || shard_path.components().count() != 1
        {
            return Err(format!(
                "Invalid shard filename in model.safetensors.index.json: \"{}\"\n\
                 Shard names must be plain filenames without path separators.",
                name
            ));
        }

        let path = dir.join(name);
        // Check raw existence (symlink_metadata to avoid following broken links)
        let meta = std::fs::symlink_metadata(&path);
        if meta.is_err() {
            missing.push(name.clone());
        } else {
            paths.push(path);
        }
    }

    if !missing.is_empty() {
        return Err(format!(
            "Missing shard file(s) referenced in model.safetensors.index.json:\n  {}\n\
             Re-download the model with:\n  \
             huggingface-cli download <model-id> --local-dir {}",
            missing.join("\n  "),
            dir.display()
        ));
    }

    paths.sort();
    Ok(paths)
}

/// Glob all `*.safetensors` files in a directory, sorted for deterministic order.
fn glob_safetensors(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
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
    paths.sort();
    Ok(paths)
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

    #[test]
    fn test_extract_shards_from_index_json_basic() {
        let json = r#"{
            "metadata": {"total_size": 123456},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00003.safetensors",
                "model.norm.weight": "model-00003-of-00003.safetensors"
            }
        }"#;

        let shards = extract_shards_from_index_json(json).expect("should parse");
        assert_eq!(shards.len(), 3);
        assert!(shards.contains(&"model-00001-of-00003.safetensors".to_string()));
        assert!(shards.contains(&"model-00002-of-00003.safetensors".to_string()));
        assert!(shards.contains(&"model-00003-of-00003.safetensors".to_string()));
    }

    #[test]
    fn test_extract_shards_deduplicates() {
        let json = r#"{
            "weight_map": {
                "a.weight": "shard-1.safetensors",
                "b.weight": "shard-1.safetensors",
                "c.weight": "shard-2.safetensors"
            }
        }"#;
        let shards = extract_shards_from_index_json(json).expect("should parse");
        assert_eq!(shards.len(), 2);
    }

    #[test]
    fn test_extract_shards_missing_weight_map() {
        let json = r#"{"metadata": {"total_size": 0}}"#;
        let result = extract_shards_from_index_json(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("weight_map"));
    }

    #[test]
    fn test_parse_shard_index_no_file() {
        // A temp dir with no index file should return Ok(None)
        let dir = std::env::temp_dir();
        // We just verify that a dir without the file returns None
        // (the actual index file likely doesn't exist in temp dir)
        let result = parse_shard_index(&dir);
        assert!(result.is_ok());
        // Result could be Some or None depending on whether the file happens to exist;
        // we at minimum assert it does not error on a valid directory.
    }

    #[test]
    fn test_parse_shard_index_valid_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("create temp dir");
        let index_path = dir.path().join("model.safetensors.index.json");
        let mut f = std::fs::File::create(&index_path).unwrap();
        writeln!(
            f,
            r#"{{"metadata": {{}}, "weight_map": {{"x.weight": "shard-1.safetensors"}}}}"#
        )
        .unwrap();

        let result = parse_shard_index(dir.path()).expect("should succeed");
        assert!(result.is_some());
        let shards = result.unwrap();
        assert_eq!(shards, vec!["shard-1.safetensors"]);
    }

    #[test]
    fn test_check_symlink_regular_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("real.safetensors");
        std::fs::File::create(&file)
            .unwrap()
            .write_all(b"")
            .unwrap();

        let (is_sym, exists) = check_symlink(&file);
        assert!(!is_sym);
        assert!(exists);
    }

    #[test]
    fn test_check_symlink_broken() {
        let dir = tempfile::tempdir().unwrap();
        let link = dir.path().join("broken.safetensors");
        // Create a symlink pointing to a non-existent target
        #[cfg(unix)]
        std::os::unix::fs::symlink("/nonexistent/target.safetensors", &link).unwrap();
        #[cfg(not(unix))]
        {
            // Skip on non-unix; symlinks may require elevated permissions on Windows
            return;
        }

        let (is_sym, exists) = check_symlink(&link);
        assert!(is_sym);
        assert!(!exists);
    }

    #[test]
    fn test_weight_map_operations() {
        use crate::dtype;
        use crate::ffi;
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

    #[test]
    fn test_validate_index_shards_rejects_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        // Parent directory traversal
        let result = validate_index_shards(dir.path(), &["../secret.safetensors".to_string()]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid shard filename"));

        // Absolute path
        let result = validate_index_shards(dir.path(), &["/etc/passwd".to_string()]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid shard filename"));

        // Subdirectory traversal
        let result = validate_index_shards(dir.path(), &["subdir/model.safetensors".to_string()]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid shard filename"));
    }

    #[test]
    fn test_extract_shards_rejects_path_traversal_in_json() {
        // Even though extract_shards_from_index_json doesn't validate paths itself,
        // the downstream validate_index_shards will catch these. Verify the full flow.
        let json = r#"{
            "weight_map": {
                "x.weight": "../../../etc/shadow"
            }
        }"#;
        // extract_shards succeeds (it just extracts strings)
        let shards = extract_shards_from_index_json(json).expect("should parse");
        assert_eq!(shards, vec!["../../../etc/shadow"]);

        // But validate_index_shards must reject it
        let dir = tempfile::tempdir().unwrap();
        let result = validate_index_shards(dir.path(), &shards);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid shard filename"));
    }
}

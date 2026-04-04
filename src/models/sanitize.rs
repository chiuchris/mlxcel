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

//! Shared config and weight sanitization helpers.
//!
//! These helpers support both model `load()` implementations and higher-level
//! loading code, so they live beside the model registry but outside
//! `models/mod.rs`.

use memmap2::MmapOptions;
use safetensors::{Dtype as SafeTensorDtype, SafeTensors, tensor::TensorView};
use serde_json::Value;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectiveLoadMode {
    Materialize,
    Borrowed,
}

#[derive(Debug, Default)]
pub(crate) struct Gemma4WeightBacking {
    pub mmaps: Vec<memmap2::Mmap>,
    pub owned_buffers: Vec<Vec<u8>>,
}

fn is_gemma4_model_config(config: &Value) -> bool {
    config.get("model_type").and_then(Value::as_str) == Some("gemma4")
        || config
            .get("text_config")
            .and_then(|text| text.get("model_type"))
            .and_then(Value::as_str)
            == Some("gemma4")
}

fn is_gemma4_text_weight(name: &str) -> bool {
    name.starts_with("language_model.")
        || name.starts_with("model.")
        || name.starts_with("lm_head.")
}

fn is_gemma4_vlm_weight(name: &str) -> bool {
    is_gemma4_text_weight(name)
        || name.starts_with("vision_tower.")
        || name.starts_with("embed_vision.")
}

fn tensor_view_to_array(
    name: &str,
    tensor: &TensorView<'_>,
    mode: SelectiveLoadMode,
    mut owned_buffers: Option<&mut Vec<Vec<u8>>>,
) -> Result<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>, String> {
    let shape: Vec<i32> = tensor
        .shape()
        .iter()
        .map(|&dim| {
            i32::try_from(dim)
                .map_err(|_| format!("Tensor {name} has dimension {dim} that exceeds i32"))
        })
        .collect::<Result<_, _>>()?;

    let array = match tensor.dtype() {
        SafeTensorDtype::BF16 => {
            if mode == SelectiveLoadMode::Borrowed {
                let owned_buffers = owned_buffers.as_mut().ok_or_else(|| {
                    format!("Missing owned buffer storage for borrowed tensor {name}")
                })?;
                let mut buffer = Vec::with_capacity(tensor.data().len() * 2);
                for chunk in tensor.data().chunks_exact(std::mem::size_of::<u16>()) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                    let value = f32::from_bits(bits << 16);
                    buffer.extend_from_slice(&value.to_le_bytes());
                }
                owned_buffers.push(buffer);
                let backing = owned_buffers
                    .last()
                    .ok_or_else(|| format!("Failed to retain borrowed buffer for tensor {name}"))?;
                mlxcel_core::from_bytes_nocopy(backing, &shape, mlxcel_core::dtype::FLOAT32)
            } else if should_convert_bf16_to_f16() {
                let values = tensor
                    .data()
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                        f32::from_bits(bits << 16)
                    })
                    .collect::<Vec<_>>();
                let array = mlxcel_core::from_slice_f32(&values, &shape);
                mlxcel_core::astype(&array, mlxcel_core::dtype::FLOAT16)
            } else {
                let values = tensor
                    .data()
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                        f32::from_bits(bits << 16)
                    })
                    .collect::<Vec<_>>();
                let array = mlxcel_core::from_slice_f32(&values, &shape);
                mlxcel_core::astype(&array, mlxcel_core::dtype::BFLOAT16)
            }
        }
        SafeTensorDtype::F16 => {
            if mode == SelectiveLoadMode::Borrowed {
                mlxcel_core::from_bytes_nocopy(tensor.data(), &shape, mlxcel_core::dtype::FLOAT16)
            } else {
                mlxcel_core::from_bytes_f16(tensor.data(), &shape, false)
            }
        }
        SafeTensorDtype::F32 => {
            if mode == SelectiveLoadMode::Borrowed {
                mlxcel_core::from_bytes_nocopy(tensor.data(), &shape, mlxcel_core::dtype::FLOAT32)
            } else {
                mlxcel_core::from_bytes(tensor.data(), &shape, mlxcel_core::dtype::FLOAT32)
            }
        }
        SafeTensorDtype::U32 => {
            if mode == SelectiveLoadMode::Borrowed {
                mlxcel_core::from_bytes_nocopy(tensor.data(), &shape, mlxcel_core::dtype::UINT32)
            } else {
                let bytes = tensor.data();
                let words: Vec<u32>;
                let slice = {
                    let (prefix, aligned, suffix) = unsafe { bytes.align_to::<u32>() };
                    if prefix.is_empty() && suffix.is_empty() {
                        aligned
                    } else {
                        words = bytes
                            .chunks_exact(std::mem::size_of::<u32>())
                            .map(|chunk| {
                                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();
                        words.as_slice()
                    }
                };
                mlxcel_core::from_slice_u32(slice, &shape)
            }
        }
        SafeTensorDtype::U64 => {
            if mode == SelectiveLoadMode::Borrowed {
                mlxcel_core::from_bytes_nocopy(tensor.data(), &shape, mlxcel_core::dtype::UINT64)
            } else {
                mlxcel_core::from_bytes(tensor.data(), &shape, mlxcel_core::dtype::UINT64)
            }
        }
        SafeTensorDtype::I32 => {
            if mode == SelectiveLoadMode::Borrowed {
                mlxcel_core::from_bytes_nocopy(tensor.data(), &shape, mlxcel_core::dtype::INT32)
            } else {
                mlxcel_core::from_bytes(tensor.data(), &shape, mlxcel_core::dtype::INT32)
            }
        }
        SafeTensorDtype::I64 => {
            if mode == SelectiveLoadMode::Borrowed {
                mlxcel_core::from_bytes_nocopy(tensor.data(), &shape, mlxcel_core::dtype::INT64)
            } else {
                mlxcel_core::from_bytes(tensor.data(), &shape, mlxcel_core::dtype::INT64)
            }
        }
        SafeTensorDtype::U8 => {
            if mode == SelectiveLoadMode::Borrowed {
                mlxcel_core::from_bytes_nocopy(tensor.data(), &shape, mlxcel_core::dtype::UINT8)
            } else {
                mlxcel_core::from_bytes(tensor.data(), &shape, mlxcel_core::dtype::UINT8)
            }
        }
        SafeTensorDtype::I8 => {
            if mode == SelectiveLoadMode::Borrowed {
                mlxcel_core::from_bytes_nocopy(tensor.data(), &shape, mlxcel_core::dtype::INT8)
            } else {
                mlxcel_core::from_bytes(tensor.data(), &shape, mlxcel_core::dtype::INT8)
            }
        }
        dtype => {
            return Err(format!(
                "Unsupported safetensors dtype {dtype:?} for selectively loaded tensor {name}"
            ));
        }
    };

    if mode == SelectiveLoadMode::Materialize {
        // from_bytes() borrows the source mmap until evaluation, so
        // materialized selective loads must force realization before the
        // shard mapping is dropped.
        mlxcel_core::eval(&array);
    }
    Ok(array)
}

fn load_filtered_shard<F>(
    path: &Path,
    weights: &mut mlxcel_core::weights::WeightMap,
    keep: F,
    prefer_native_full_shard_load: bool,
    mode: SelectiveLoadMode,
    backing_mmaps: Option<&mut Vec<memmap2::Mmap>>,
    mut owned_buffers: Option<&mut Vec<Vec<u8>>>,
) -> Result<(), String>
where
    F: Fn(&str) -> bool + Copy,
{
    let debug_gemma4 = std::env::var_os("MLXCEL_DEBUG_GEMMA4_LOAD").is_some();
    let file = File::open(path)
        .map_err(|e| format!("Failed to open safetensors shard {}: {e}", path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .map_err(|e| format!("Failed to mmap safetensors shard {}: {e}", path.display()))?;
    let tensors = SafeTensors::deserialize(&mmap)
        .map_err(|e| format!("Failed to parse safetensors shard {}: {e}", path.display()))?;

    let selected_names: Vec<String> = tensors
        .names()
        .into_iter()
        .filter(|name| keep(name))
        .map(str::to_string)
        .collect();

    if selected_names.is_empty() {
        return Ok(());
    }

    if mode == SelectiveLoadMode::Materialize
        && prefer_native_full_shard_load
        && selected_names.len() == tensors.len()
    {
        drop(tensors);
        drop(mmap);
        weights.extend(mlxcel_core::weights::load_safetensors(path)?);
        return Ok(());
    }

    if debug_gemma4 {
        eprintln!(
            "gemma4 selective shard {} (selected {} / total {})",
            path.display(),
            selected_names.len(),
            tensors.len()
        );
    }

    for name in selected_names {
        let tensor = tensors
            .tensor(&name)
            .map_err(|e| format!("Failed to read tensor {name} from {}: {e}", path.display()))?;
        if debug_gemma4 {
            eprintln!("  loading {name} {:?} {:?}", tensor.dtype(), tensor.shape());
        }
        let array = tensor_view_to_array(&name, &tensor, mode, owned_buffers.as_deref_mut())?;
        if debug_gemma4 {
            eprintln!("  loaded {name}");
        }
        weights.insert(name, array);
    }

    drop(tensors);
    if let Some(backing_mmaps) = backing_mmaps {
        backing_mmaps.push(mmap);
    }

    Ok(())
}

fn load_weights_from_dir_with_filter<P, F>(
    model_dir: P,
    keep: F,
    prefer_native_full_shard_load: bool,
) -> Result<mlxcel_core::weights::WeightMap, String>
where
    P: AsRef<Path>,
    F: Fn(&str) -> bool + Copy,
{
    let model_dir = model_dir.as_ref();
    let mut weights = mlxcel_core::weights::WeightMap::new();

    let mut shard_paths: Vec<_> = std::fs::read_dir(model_dir)
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
    shard_paths.sort();

    for path in shard_paths {
        load_filtered_shard(
            &path,
            &mut weights,
            keep,
            prefer_native_full_shard_load,
            SelectiveLoadMode::Materialize,
            None,
            None,
        )?;
    }

    Ok(weights)
}

fn load_gemma4_text_weights<P: AsRef<Path>>(
    model_dir: P,
) -> Result<mlxcel_core::weights::WeightMap, String> {
    // MLX's native load_safetensors currently crashes on Gemma 4 shards,
    // including pure language-model shards. Keep Gemma 4 on the selective
    // mmap + eager materialization path until the native loader can handle
    // these checkpoints.
    load_weights_from_dir_with_filter(model_dir, is_gemma4_text_weight, false)
}

pub(crate) fn load_gemma4_text_weights_with_backing<P: AsRef<Path>>(
    model_dir: P,
) -> Result<(mlxcel_core::weights::WeightMap, Gemma4WeightBacking), String> {
    let model_dir = model_dir.as_ref();
    let mut weights = mlxcel_core::weights::WeightMap::new();
    let mut backing = Gemma4WeightBacking::default();

    let mut shard_paths: Vec<_> = std::fs::read_dir(model_dir)
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
    shard_paths.sort();

    for path in shard_paths {
        load_filtered_shard(
            &path,
            &mut weights,
            is_gemma4_text_weight,
            false,
            SelectiveLoadMode::Materialize,
            Some(&mut backing.mmaps),
            Some(&mut backing.owned_buffers),
        )?;
    }

    Ok((weights, backing))
}

pub(crate) fn load_gemma4_vlm_weights<P: AsRef<Path>>(
    model_dir: P,
) -> Result<mlxcel_core::weights::WeightMap, String> {
    load_weights_from_dir_with_filter(model_dir, is_gemma4_vlm_weight, false)
}

/// Ensure lm_head weights exist for models with tied embeddings.
///
/// Many models share embedding weights for the output projection (lm_head).
/// When tie_word_embeddings is true (or omitted), lm_head.weight may not be
/// saved in safetensors. This function auto-detects the missing weight and
/// copies model.embed_tokens.* → lm_head.* so model loaders work uniformly.
///
/// Auto-detection: if tie_word_embeddings is explicitly false, do nothing.
/// Otherwise (true or absent), copy if lm_head.weight is missing.
///
/// Used by: all VLM loaders, load_model_from_weights, load_and_sanitize_weights
pub fn sanitize_tied_embeddings(
    weights: &mut mlxcel_core::weights::WeightMap,
    config: &serde_json::Value,
) {
    let tie = config
        .get("tie_word_embeddings")
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("tie_word_embeddings"))
        })
        .and_then(|v| v.as_bool());

    if tie == Some(false) {
        return;
    }

    if !weights.contains_key("lm_head.weight") {
        for suffix in &["weight", "scales", "biases"] {
            let src = format!("model.embed_tokens.{}", suffix);
            let dst = format!("lm_head.{}", suffix);
            if let Some(w) = weights.get(&src) {
                weights.insert(dst, mlxcel_core::copy(w));
            }
        }
    }

    if !weights.contains_key("language_model.lm_head.weight") {
        for suffix in &["weight", "scales", "biases"] {
            let src = format!("language_model.model.embed_tokens.{}", suffix);
            let dst = format!("language_model.lm_head.{}", suffix);
            if let Some(w) = weights.get(&src) {
                weights.insert(dst, mlxcel_core::copy(w));
            }
        }
    }
}

/// Load weights from a model directory with automatic tied-embedding sanitization.
///
/// This is the common weight loading entry point for text model `load()`
/// functions. It reads safetensors, parses config.json, and ensures lm_head
/// weights exist.
///
/// On Apple Silicon, bf16 tensors are automatically converted to f16 for
/// performance.  No Apple GPU (M1–M5) has native bf16 ALU hardware — bf16
/// arithmetic is emulated via f32 upcast/truncate, yielding f32 throughput.
/// f16 is strictly better: on M3/M4 it unlocks ~2x compute throughput via
/// fp16 co-issue, and on M1/M2 there is no penalty.  Non-Apple backends
/// keep bf16 as-is since they may support it natively.
pub fn load_and_sanitize_weights<P: AsRef<std::path::Path>>(
    model_dir: P,
) -> Result<mlxcel_core::weights::WeightMap, String> {
    let model_dir = model_dir.as_ref();
    let config_path = model_dir.join("config.json");
    let parsed_config = std::fs::read_to_string(&config_path)
        .ok()
        .map(|config_str| sanitize_config_json(&config_str))
        .and_then(|config_str| serde_json::from_str::<Value>(&config_str).ok());

    let mut weights = if parsed_config.as_ref().is_some_and(is_gemma4_model_config) {
        load_gemma4_text_weights(model_dir)?
    } else {
        mlxcel_core::weights::load_weights_from_dir(model_dir)?
    };

    let mut is_quantized = false;
    if let Some(config) = parsed_config.as_ref() {
        sanitize_tied_embeddings(&mut weights, config);
        is_quantized = config.get("quantization").is_some()
            || config
                .get("text_config")
                .and_then(|tc| tc.get("quantization"))
                .is_some();
    }

    // Convert bf16 → f16 on all Apple Silicon for performance.  No Apple GPU
    // has native bf16 ALU, so f16 is strictly better.  Only for non-quantized
    // models — quantized models use bf16 scales/biases in quantized_matmul
    // which handles bf16 natively.
    if !is_quantized && should_convert_bf16_to_f16() {
        let had_bf16 = convert_bf16_weights(&mut weights);
        if had_bf16 {
            warn_bf16_precision();
        }
    }

    Ok(weights)
}

/// Returns true when bf16 tensors should be cast to f16 at load time.
///
/// All Apple Silicon GPUs (M1–M5) lack native bf16 ALU hardware.  Metal's
/// `bfloat` type is storage-only — arithmetic is emulated via f32
/// upcast/truncate, yielding f32 throughput.  f16 is strictly better:
/// - M3/M4: fp16 co-issue provides ~2x compute throughput over bf16/f32.
/// - M1/M2: fp16 and fp32 have identical throughput, no penalty from converting.
/// - M5: already benefits from conversion (crash avoidance + performance).
///
/// Non-Apple backends (Unknown silicon_gen) keep bf16 as-is.
fn should_convert_bf16_to_f16() -> bool {
    let hw = mlxcel_core::hardware::get_hardware();
    hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown
}

/// Emit a one-line stderr note when a full-precision bf16 model is loaded,
/// unless suppressed by `MLXCEL_NO_PRECISION_WARNING` env var.
///
/// Used by: load_and_sanitize_weights, load_vlm_weights
pub fn warn_bf16_precision() {
    if std::env::var("MLXCEL_NO_PRECISION_WARNING").is_err() {
        eprintln!(
            "Note: This model uses bf16 weights. On Apple Silicon, quantized models (4bit/8bit) are significantly faster. Consider using a quantized variant from mlx-community."
        );
    }
}

/// Cast every bf16 tensor in the weight map to f16.
///
/// Returns `true` if any bf16 tensors were found and converted, `false` otherwise.
///
/// Used by: load_and_sanitize_weights, load_vlm_weights
#[must_use]
pub fn convert_bf16_weights(weights: &mut mlxcel_core::weights::WeightMap) -> bool {
    let bf16_keys: Vec<String> = weights
        .iter()
        .filter(|(_, v)| mlxcel_core::array_dtype(v) == mlxcel_core::dtype::BFLOAT16)
        .map(|(k, _)| k.clone())
        .collect();

    if !bf16_keys.is_empty() {
        eprintln!(
            "Converting {} bf16 weight tensors to f16 for Apple Silicon fp16 optimization.",
            bf16_keys.len()
        );
        for key in bf16_keys {
            if let Some(tensor) = weights.get(&key) {
                let converted = mlxcel_core::astype(tensor, mlxcel_core::dtype::FLOAT16);
                weights.insert(key, converted);
            }
        }
        true
    } else {
        false
    }
}

/// Sanitize config JSON string by replacing non-standard JSON values.
pub fn sanitize_config_json(config_str: &str) -> String {
    config_str
        .replace("Infinity", "1e38")
        .replace("-Infinity", "-1e38")
        .replace("NaN", "0.0")
}

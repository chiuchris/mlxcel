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
        || name.starts_with("audio_tower.")
        || name.starts_with("embed_audio.")
}

/// Convert a single F8_E4M3 byte to f32.
///
/// F8_E4M3FN format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
/// No infinity representation; the all-ones exponent with non-zero mantissa encodes NaN.
/// Range: ±448.0.
fn f8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF; // 4-bit exponent
    let mant = bits & 0x7; // 3-bit mantissa

    // NaN: exponent all-ones AND mantissa all-ones (no infinity in E4M3FN).
    // Only the single pattern exp=0xF, mant=0x7 is NaN; other exp=0xF values are valid normals.
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }

    let f_sign = if sign != 0 { -1.0f32 } else { 1.0f32 };

    if exp == 0 {
        // Subnormal: value = (-1)^sign * 2^(1-7) * (mant / 8)
        //                   = (-1)^sign * mant * 2^(-9)
        if mant == 0 {
            return f_sign * 0.0;
        }
        f_sign * (mant as f32) * (2.0f32).powi(-9)
    } else {
        // Normal: value = (-1)^sign * 2^(exp-7) * (1 + mant/8)
        f_sign * (2.0f32).powi(exp as i32 - 7) * (1.0 + mant as f32 / 8.0)
    }
}

/// Convert a single F8_E5M2 byte to f32.
///
/// F8_E5M2 format: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits.
/// Supports infinity and NaN (all-ones exponent with non-zero mantissa).
fn f8_e5m2_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 2) & 0x1F; // 5-bit exponent
    let mant = bits & 0x3; // 2-bit mantissa

    let f_sign = if sign != 0 { -1.0f32 } else { 1.0f32 };

    if exp == 0x1F {
        // Special values: infinity or NaN
        if mant == 0 {
            return f_sign * f32::INFINITY;
        } else {
            return f32::NAN;
        }
    }

    if exp == 0 {
        // Subnormal: value = (-1)^sign * 2^(1-15) * (mant / 4)
        //                   = (-1)^sign * mant * 2^(-16)
        if mant == 0 {
            return f_sign * 0.0;
        }
        f_sign * (mant as f32) * (2.0f32).powi(-16)
    } else {
        // Normal: value = (-1)^sign * 2^(exp-15) * (1 + mant/4)
        f_sign * (2.0f32).powi(exp as i32 - 15) * (1.0 + mant as f32 / 4.0)
    }
}

/// Convert a 4-bit FP4 E2M1 nibble to f32.
///
/// FP4 E2M1: 1 sign bit, 2 exponent bits (bias=1), 1 mantissa bit.
/// Values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} x {+1, -1}
fn fp4_e2m1_to_f32(nibble: u8) -> f32 {
    let sign = (nibble >> 3) & 1;
    let exp = (nibble >> 1) & 0x3; // 2-bit exponent
    let mant = nibble & 0x1; // 1-bit mantissa

    let f_sign = if sign != 0 { -1.0f32 } else { 1.0f32 };

    if exp == 0 {
        // Subnormal: (-1)^sign * 2^(1-1) * (0 + mant/2) = mant * 0.5
        if mant == 0 {
            return f_sign * 0.0;
        }
        f_sign * 0.5
    } else {
        // Normal: (-1)^sign * 2^(exp-1) * (1 + mant/2)
        f_sign * (2.0f32).powi(exp as i32 - 1) * (1.0 + mant as f32 * 0.5)
    }
}

/// Convert f16 bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal f16
        let mut m = mant;
        let mut e = 0i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
    } else if exp == 0x1F {
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::NAN
        }
    } else {
        let f32_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Remap NVFP4-style weight keys to the MLX-community naming convention.
///
/// nvfp4 Gemma 4 checkpoints use `model.language_model.X` prefixes while the
/// model code expects `language_model.model.X`. This function performs the
/// following remapping:
///
/// - `model.language_model.X` → `language_model.model.X`
/// - `model.embed_vision.X`   → `embed_vision.X`
/// - `model.lm_head.X`        → `lm_head.X`
///
/// If no keys matching the nvfp4 pattern are found, this is a no-op.
fn normalize_nvfp4_keys(weights: &mut mlxcel_core::weights::WeightMap) {
    let nvfp4_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.starts_with("model.language_model."))
        .cloned()
        .collect();

    if nvfp4_keys.is_empty() {
        return;
    }

    eprintln!(
        "Remapping {} NVFP4-style weight keys to MLX-community convention...",
        nvfp4_keys.len()
    );

    // Collect all key-value pairs that need remapping, then reinsert.
    let remappings: Vec<(String, String)> = weights
        .keys()
        .filter_map(|k| {
            let new_key = if let Some(rest) = k.strip_prefix("model.language_model.") {
                format!("language_model.model.{rest}")
            } else if let Some(rest) = k.strip_prefix("model.embed_vision.") {
                format!("embed_vision.{rest}")
            } else if let Some(rest) = k.strip_prefix("model.lm_head.") {
                format!("lm_head.{rest}")
            } else {
                return None; // No remapping needed
            };
            Some((k.clone(), new_key))
        })
        .collect();

    for (old_key, new_key) in remappings {
        if let Some(arr) = weights.remove(&old_key) {
            weights.insert(new_key, arr);
        }
    }
}

/// Dequantize NVFP4-packed weights in-place.
///
/// Detects weight groups by the presence of `{prefix}.weight_scale_2` keys.
/// For each group, unpacks FP4 E2M1 nibbles from U8 storage and applies
/// per-block (weight_scale) and global (weight_scale_2) scale factors to
/// produce dequantized f16 weights.
///
/// After dequantization the auxiliary keys `weight_scale`, `weight_scale_2`,
/// and `input_scale` are removed from the weight map.
fn dequantize_nvfp4_weights(weights: &mut mlxcel_core::weights::WeightMap) {
    // Collect prefixes first to avoid borrowing conflicts during mutation.
    let fp4_prefixes: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".weight_scale_2"))
        .map(|k| k.strip_suffix(".weight_scale_2").unwrap().to_string())
        .collect();

    if fp4_prefixes.is_empty() {
        return;
    }

    eprintln!(
        "Dequantizing {} NVFP4 weight groups to f16...",
        fp4_prefixes.len()
    );

    for prefix in fp4_prefixes {
        let weight_key = format!("{prefix}.weight");
        let scale_key = format!("{prefix}.weight_scale");
        let scale2_key = format!("{prefix}.weight_scale_2");
        let input_scale_key = format!("{prefix}.input_scale");

        // Verify all required keys exist before proceeding.
        if !weights.contains_key(&weight_key) || !weights.contains_key(&scale_key) {
            // Remove orphaned scale2 key and continue.
            weights.remove(&scale2_key);
            continue;
        }

        let (weight_shape, weight_bytes, scale_bytes, scale2_val) = {
            let weight_arr = weights.get(&weight_key).unwrap();
            let scale_arr = weights.get(&scale_key).unwrap();
            let scale2_arr = weights.get(&scale2_key).unwrap();

            mlxcel_core::eval(weight_arr);
            mlxcel_core::eval(scale_arr);
            mlxcel_core::eval(scale2_arr);

            let weight_shape = mlxcel_core::array_shape(weight_arr);
            let weight_bytes = mlxcel_core::array_to_raw_bytes(weight_arr);
            let scale_bytes = mlxcel_core::array_to_raw_bytes(scale_arr);
            let scale2_val = mlxcel_core::item_f32(scale2_arr);

            (weight_shape, weight_bytes, scale_bytes, scale2_val)
        };

        // Validate weight tensor is 2-D with positive dimensions.
        if weight_shape.len() < 2 {
            eprintln!(
                "Skipping NVFP4 dequant for {prefix}: weight tensor is {}-D (expected 2-D)",
                weight_shape.len()
            );
            weights.remove(&scale2_key);
            continue;
        }
        if weight_shape[0] <= 0 || weight_shape[1] <= 0 {
            eprintln!(
                "Skipping NVFP4 dequant for {prefix}: non-positive dimensions [{}, {}]",
                weight_shape[0], weight_shape[1]
            );
            weights.remove(&scale2_key);
            continue;
        }

        // weight_shape = [out_dim, in_dim/2] (packed U8 — 2 FP4 nibbles per byte)
        let out_dim = weight_shape[0] as usize;
        let packed_dim = weight_shape[1] as usize; // in_dim / 2
        let in_dim = packed_dim * 2;

        let group_size: usize = 16;

        // in_dim must be a multiple of group_size for scale indexing to be valid.
        if !in_dim.is_multiple_of(group_size) {
            eprintln!(
                "Skipping NVFP4 dequant for {prefix}: in_dim {in_dim} is not a multiple of group_size {group_size}"
            );
            weights.remove(&scale2_key);
            continue;
        }
        let num_groups = in_dim / group_size;

        // Validate raw byte buffer lengths match expected sizes before indexing.
        let expected_weight_bytes = out_dim * packed_dim;
        let expected_scale_bytes = out_dim * num_groups * 2; // F16 = 2 bytes each
        if weight_bytes.len() < expected_weight_bytes {
            eprintln!(
                "Skipping NVFP4 dequant for {prefix}: weight_bytes length {} < expected {}",
                weight_bytes.len(),
                expected_weight_bytes
            );
            weights.remove(&scale2_key);
            continue;
        }
        if scale_bytes.len() < expected_scale_bytes {
            eprintln!(
                "Skipping NVFP4 dequant for {prefix}: scale_bytes length {} < expected {}",
                scale_bytes.len(),
                expected_scale_bytes
            );
            weights.remove(&scale2_key);
            continue;
        }

        let mut dequant_f32 = Vec::with_capacity(out_dim * in_dim);

        for row in 0..out_dim {
            for col in 0..in_dim {
                let byte_idx = row * packed_dim + col / 2;
                let nibble = if col % 2 == 0 {
                    weight_bytes[byte_idx] & 0x0F // low nibble
                } else {
                    (weight_bytes[byte_idx] >> 4) & 0x0F // high nibble
                };
                let fp4_val = fp4_e2m1_to_f32(nibble);

                // Block scale (F16 stored as 2-byte little-endian).
                let group_idx = col / group_size;
                let scale_flat_idx = row * num_groups + group_idx;
                let scale_f16_bits = u16::from_le_bytes([
                    scale_bytes[scale_flat_idx * 2],
                    scale_bytes[scale_flat_idx * 2 + 1],
                ]);
                let scale_val = f16_to_f32(scale_f16_bits);

                dequant_f32.push(fp4_val * scale_val * scale2_val);
            }
        }

        // Create a new f16 array with shape [out_dim, in_dim].
        let new_shape = vec![out_dim as i32, in_dim as i32];
        let new_arr = mlxcel_core::from_slice_f32(&dequant_f32, &new_shape);
        let new_arr_f16 = mlxcel_core::astype(&new_arr, mlxcel_core::dtype::FLOAT16);
        mlxcel_core::eval(&new_arr_f16);

        // Replace the packed weight and remove auxiliary keys.
        weights.insert(weight_key, new_arr_f16);
        weights.remove(&scale_key);
        weights.remove(&scale2_key);
        weights.remove(&input_scale_key); // may not exist; remove is a no-op then
    }
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
        SafeTensorDtype::F8_E4M3 => {
            // MLX has no native float8 dtype; convert F8_E4M3 → f16 at load time.
            // Used by nvfp4 Gemma 4 checkpoints (weight_scale tensors).
            if mode == SelectiveLoadMode::Borrowed {
                let owned_buffers = owned_buffers.as_mut().ok_or_else(|| {
                    format!("Missing owned buffer storage for borrowed F8_E4M3 tensor {name}")
                })?;
                let values: Vec<f32> = tensor.data().iter().map(|&b| f8_e4m3_to_f32(b)).collect();
                let mut buffer = Vec::with_capacity(values.len() * 4);
                for v in &values {
                    buffer.extend_from_slice(&v.to_le_bytes());
                }
                owned_buffers.push(buffer);
                let backing = owned_buffers
                    .last()
                    .ok_or_else(|| format!("Failed to retain buffer for F8_E4M3 tensor {name}"))?;
                let array =
                    mlxcel_core::from_bytes_nocopy(backing, &shape, mlxcel_core::dtype::FLOAT32);
                mlxcel_core::astype(&array, mlxcel_core::dtype::FLOAT16)
            } else {
                let values: Vec<f32> = tensor.data().iter().map(|&b| f8_e4m3_to_f32(b)).collect();
                let array = mlxcel_core::from_slice_f32(&values, &shape);
                mlxcel_core::astype(&array, mlxcel_core::dtype::FLOAT16)
            }
        }
        SafeTensorDtype::F8_E5M2 => {
            // MLX has no native float8 dtype; convert F8_E5M2 → f16 at load time.
            if mode == SelectiveLoadMode::Borrowed {
                let owned_buffers = owned_buffers.as_mut().ok_or_else(|| {
                    format!("Missing owned buffer storage for borrowed F8_E5M2 tensor {name}")
                })?;
                let values: Vec<f32> = tensor.data().iter().map(|&b| f8_e5m2_to_f32(b)).collect();
                let mut buffer = Vec::with_capacity(values.len() * 4);
                for v in &values {
                    buffer.extend_from_slice(&v.to_le_bytes());
                }
                owned_buffers.push(buffer);
                let backing = owned_buffers
                    .last()
                    .ok_or_else(|| format!("Failed to retain buffer for F8_E5M2 tensor {name}"))?;
                let array =
                    mlxcel_core::from_bytes_nocopy(backing, &shape, mlxcel_core::dtype::FLOAT32);
                mlxcel_core::astype(&array, mlxcel_core::dtype::FLOAT16)
            } else {
                let values: Vec<f32> = tensor.data().iter().map(|&b| f8_e5m2_to_f32(b)).collect();
                let array = mlxcel_core::from_slice_f32(&values, &shape);
                mlxcel_core::astype(&array, mlxcel_core::dtype::FLOAT16)
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

    let is_gemma4 = parsed_config.as_ref().is_some_and(is_gemma4_model_config);

    let mut weights = if is_gemma4 {
        load_gemma4_text_weights(model_dir)?
    } else {
        mlxcel_core::weights::load_weights_from_dir(model_dir)?
    };

    // Apply NVFP4 key normalization and dequantization for Gemma 4 nvfp4
    // checkpoints before tied-embedding sanitization so that lookups succeed.
    if is_gemma4 {
        normalize_nvfp4_keys(&mut weights);
        dequantize_nvfp4_weights(&mut weights);
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    // --- f8_e4m3_to_f32 tests ---

    #[test]
    fn f8_e4m3_positive_zero() {
        // 0b0_0000_000 = 0x00
        assert_eq!(f8_e4m3_to_f32(0x00), 0.0f32);
    }

    #[test]
    fn f8_e4m3_negative_zero() {
        // 0b1_0000_000 = 0x80
        let v = f8_e4m3_to_f32(0x80);
        assert_eq!(v, 0.0f32);
        // Negative zero: sign bit set
        assert!(v.is_sign_negative() || v == 0.0);
    }

    #[test]
    fn f8_e4m3_one() {
        // 1.0 = (-1)^0 * 2^(7-7) * (1 + 0/8) = 2^0 * 1.0 = 1.0
        // exp=7 (0b0111), mant=0 (0b000) => 0b0_0111_000 = 0x38
        assert!((f8_e4m3_to_f32(0x38) - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn f8_e4m3_negative_one() {
        // -1.0: sign=1, exp=7, mant=0 => 0b1_0111_000 = 0xB8
        assert!((f8_e4m3_to_f32(0xB8) - (-1.0f32)).abs() < 1e-6);
    }

    #[test]
    fn f8_e4m3_max_value() {
        // Max normal: exp=14 (0b1110), mant=7 (0b111), sign=0
        // value = 2^(14-7) * (1 + 7/8) = 128 * 1.875 = 240.0
        // Wait: E4M3FN max is 448. Let me recalculate:
        // max exp for normal = 0b1110 = 14 (0b1111 with mant=0b111 is NaN only if mant != 0)
        // Actually 0b1111 with mant=0 gives 2^(15-7)*(1+0/8) = 256 — but spec says max=448
        // E4M3FN: exp=0b1111=15, mant=0b110=6 => 2^(15-7)*(1+6/8) = 256*1.75 = 448
        // exp=0b1111=15, mant=0b111=7 => NaN (special case for E4M3FN)
        // 0b0_1111_110 = 0x7E
        let v = f8_e4m3_to_f32(0x7E);
        assert!((v - 448.0f32).abs() < 1e-3, "Expected 448.0, got {v}");
    }

    #[test]
    fn f8_e4m3_subnormal() {
        // Subnormal: exp=0, mant=1 => value = 1 * 2^(-9) = 1/512
        // 0b0_0000_001 = 0x01
        let expected = 1.0f32 / 512.0;
        let v = f8_e4m3_to_f32(0x01);
        assert!((v - expected).abs() < 1e-8, "Expected {expected}, got {v}");
    }

    #[test]
    fn f8_e4m3_nan() {
        // NaN: exp=0b1111=15, mant non-zero — 0b0_1111_111 = 0x7F
        assert!(f8_e4m3_to_f32(0x7F).is_nan());
        // Also negative NaN: 0b1_1111_111 = 0xFF
        assert!(f8_e4m3_to_f32(0xFF).is_nan());
    }

    #[test]
    fn f8_e4m3_two() {
        // 2.0: exp=8 (0b1000), mant=0 => 2^(8-7)*(1+0) = 2.0
        // 0b0_1000_000 = 0x40
        assert!((f8_e4m3_to_f32(0x40) - 2.0f32).abs() < 1e-6);
    }

    // --- f8_e5m2_to_f32 tests ---

    #[test]
    fn f8_e5m2_positive_zero() {
        // 0b0_00000_00 = 0x00
        assert_eq!(f8_e5m2_to_f32(0x00), 0.0f32);
    }

    #[test]
    fn f8_e5m2_negative_zero() {
        // 0b1_00000_00 = 0x80
        let v = f8_e5m2_to_f32(0x80);
        assert_eq!(v, 0.0f32);
    }

    #[test]
    fn f8_e5m2_one() {
        // 1.0: exp=15 (bias=15 => 2^0=1), mant=0
        // 0b0_01111_00 = 0x3C
        assert!((f8_e5m2_to_f32(0x3C) - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn f8_e5m2_negative_one() {
        // -1.0: sign=1, exp=15, mant=0 => 0b1_01111_00 = 0xBC
        assert!((f8_e5m2_to_f32(0xBC) - (-1.0f32)).abs() < 1e-6);
    }

    #[test]
    fn f8_e5m2_positive_infinity() {
        // +inf: exp=0b11111=31, mant=0, sign=0 => 0b0_11111_00 = 0x7C
        assert!(f8_e5m2_to_f32(0x7C).is_infinite());
        assert!(f8_e5m2_to_f32(0x7C).is_sign_positive());
    }

    #[test]
    fn f8_e5m2_negative_infinity() {
        // -inf: sign=1, exp=31, mant=0 => 0b1_11111_00 = 0xFC
        assert!(f8_e5m2_to_f32(0xFC).is_infinite());
        assert!(f8_e5m2_to_f32(0xFC).is_sign_negative());
    }

    #[test]
    fn f8_e5m2_nan() {
        // NaN: exp=31, mant non-zero => 0b0_11111_01 = 0x7D
        assert!(f8_e5m2_to_f32(0x7D).is_nan());
        // 0b0_11111_10 = 0x7E
        assert!(f8_e5m2_to_f32(0x7E).is_nan());
        // 0b0_11111_11 = 0x7F
        assert!(f8_e5m2_to_f32(0x7F).is_nan());
    }

    #[test]
    fn f8_e5m2_subnormal() {
        // Subnormal: exp=0, mant=1, sign=0 => value = 1 * 2^(-16)
        // 0b0_00000_01 = 0x01
        let expected = 1.0f32 / 65536.0;
        let v = f8_e5m2_to_f32(0x01);
        assert!((v - expected).abs() < 1e-10, "Expected {expected}, got {v}");
    }

    #[test]
    fn f8_e5m2_two() {
        // 2.0: exp=16 (2^(16-15)=2), mant=0 => 0b0_10000_00 = 0x40
        assert!((f8_e5m2_to_f32(0x40) - 2.0f32).abs() < 1e-6);
    }

    #[test]
    fn f8_e5m2_1_25() {
        // 1.25: exp=15, mant=1 => 2^0 * (1 + 1/4) = 1.25
        // 0b0_01111_01 = 0x3D
        assert!((f8_e5m2_to_f32(0x3D) - 1.25f32).abs() < 1e-6);
    }

    // --- fp4_e2m1_to_f32 tests (all 16 nibble values) ---

    #[test]
    fn fp4_e2m1_all_positive_values() {
        // FP4 E2M1: sign|exp[1:0]|mant
        // Positive: sign=0
        // 0b0_00_0 = 0x0 → 0.0 (subnormal, mant=0)
        assert_eq!(fp4_e2m1_to_f32(0x0), 0.0f32);
        // 0b0_00_1 = 0x1 → 0.5 (subnormal, mant=1)
        assert!((fp4_e2m1_to_f32(0x1) - 0.5f32).abs() < 1e-6);
        // 0b0_01_0 = 0x2 → 2^0 * 1.0 = 1.0
        assert!((fp4_e2m1_to_f32(0x2) - 1.0f32).abs() < 1e-6);
        // 0b0_01_1 = 0x3 → 2^0 * 1.5 = 1.5
        assert!((fp4_e2m1_to_f32(0x3) - 1.5f32).abs() < 1e-6);
        // 0b0_10_0 = 0x4 → 2^1 * 1.0 = 2.0
        assert!((fp4_e2m1_to_f32(0x4) - 2.0f32).abs() < 1e-6);
        // 0b0_10_1 = 0x5 → 2^1 * 1.5 = 3.0
        assert!((fp4_e2m1_to_f32(0x5) - 3.0f32).abs() < 1e-6);
        // 0b0_11_0 = 0x6 → 2^2 * 1.0 = 4.0
        assert!((fp4_e2m1_to_f32(0x6) - 4.0f32).abs() < 1e-6);
        // 0b0_11_1 = 0x7 → 2^2 * 1.5 = 6.0
        assert!((fp4_e2m1_to_f32(0x7) - 6.0f32).abs() < 1e-6);
    }

    #[test]
    fn fp4_e2m1_all_negative_values() {
        // Negative: sign=1 (bit 3 set)
        // 0b1_00_0 = 0x8 → -0.0
        assert_eq!(fp4_e2m1_to_f32(0x8), -0.0f32);
        // 0b1_00_1 = 0x9 → -0.5
        assert!((fp4_e2m1_to_f32(0x9) - (-0.5f32)).abs() < 1e-6);
        // 0b1_01_0 = 0xA → -1.0
        assert!((fp4_e2m1_to_f32(0xA) - (-1.0f32)).abs() < 1e-6);
        // 0b1_01_1 = 0xB → -1.5
        assert!((fp4_e2m1_to_f32(0xB) - (-1.5f32)).abs() < 1e-6);
        // 0b1_10_0 = 0xC → -2.0
        assert!((fp4_e2m1_to_f32(0xC) - (-2.0f32)).abs() < 1e-6);
        // 0b1_10_1 = 0xD → -3.0
        assert!((fp4_e2m1_to_f32(0xD) - (-3.0f32)).abs() < 1e-6);
        // 0b1_11_0 = 0xE → -4.0
        assert!((fp4_e2m1_to_f32(0xE) - (-4.0f32)).abs() < 1e-6);
        // 0b1_11_1 = 0xF → -6.0
        assert!((fp4_e2m1_to_f32(0xF) - (-6.0f32)).abs() < 1e-6);
    }

    // --- f16_to_f32 tests ---

    #[test]
    fn f16_to_f32_positive_zero() {
        // +0.0: 0x0000
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
    }

    #[test]
    fn f16_to_f32_negative_zero() {
        // -0.0: 0x8000
        let v = f16_to_f32(0x8000);
        assert!(v == 0.0f32 && v.is_sign_negative());
    }

    #[test]
    fn f16_to_f32_one() {
        // 1.0: sign=0, exp=15 (0b01111), mant=0 → 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn f16_to_f32_negative_one() {
        // -1.0: sign=1, exp=15, mant=0 → 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0f32)).abs() < 1e-6);
    }

    #[test]
    fn f16_to_f32_positive_infinity() {
        // +inf: exp=0x1F, mant=0, sign=0 → 0x7C00
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00).is_sign_positive());
    }

    #[test]
    fn f16_to_f32_nan() {
        // NaN: exp=0x1F, mant≠0 → e.g. 0x7E00
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn f16_to_f32_subnormal() {
        // Smallest positive subnormal f16: exp=0, mant=1 → 0x0001
        // value = 2^(-14) * (1/1024) ≈ 5.96e-8
        let v = f16_to_f32(0x0001);
        let expected = 2.0f32.powi(-24);
        assert!((v - expected).abs() < 1e-10, "Expected {expected}, got {v}");
    }

    #[test]
    fn f16_to_f32_two() {
        // 2.0: exp=16, mant=0 → 0x4000
        assert!((f16_to_f32(0x4000) - 2.0f32).abs() < 1e-6);
    }

    // --- normalize_nvfp4_keys tests ---

    #[test]
    fn normalize_nvfp4_keys_remaps_language_model_prefix() {
        let mut weights = mlxcel_core::weights::WeightMap::new();
        weights.insert(
            "model.language_model.layers.0.mlp.gate_proj.weight".to_string(),
            mlxcel_core::from_slice_f32(&[1.0f32], &[1]),
        );
        weights.insert(
            "model.language_model.embed_tokens.weight".to_string(),
            mlxcel_core::from_slice_f32(&[2.0f32], &[1]),
        );
        normalize_nvfp4_keys(&mut weights);
        assert!(
            weights.contains_key("language_model.model.layers.0.mlp.gate_proj.weight"),
            "Expected remapped key not found"
        );
        assert!(
            weights.contains_key("language_model.model.embed_tokens.weight"),
            "Expected remapped embed_tokens key not found"
        );
        assert!(
            !weights.contains_key("model.language_model.layers.0.mlp.gate_proj.weight"),
            "Old key should be removed"
        );
    }

    #[test]
    fn normalize_nvfp4_keys_remaps_embed_vision_and_lm_head() {
        let mut weights = mlxcel_core::weights::WeightMap::new();
        weights.insert(
            "model.language_model.norm.weight".to_string(),
            mlxcel_core::from_slice_f32(&[1.0f32], &[1]),
        );
        weights.insert(
            "model.embed_vision.proj.weight".to_string(),
            mlxcel_core::from_slice_f32(&[1.0f32], &[1]),
        );
        weights.insert(
            "model.lm_head.weight".to_string(),
            mlxcel_core::from_slice_f32(&[1.0f32], &[1]),
        );
        normalize_nvfp4_keys(&mut weights);
        assert!(weights.contains_key("embed_vision.proj.weight"));
        assert!(weights.contains_key("lm_head.weight"));
        assert!(!weights.contains_key("model.embed_vision.proj.weight"));
        assert!(!weights.contains_key("model.lm_head.weight"));
    }

    #[test]
    fn normalize_nvfp4_keys_noop_when_no_nvfp4_keys() {
        let mut weights = mlxcel_core::weights::WeightMap::new();
        weights.insert(
            "language_model.model.layers.0.mlp.gate_proj.weight".to_string(),
            mlxcel_core::from_slice_f32(&[1.0f32], &[1]),
        );
        normalize_nvfp4_keys(&mut weights);
        // The existing key should remain unchanged.
        assert!(weights.contains_key("language_model.model.layers.0.mlp.gate_proj.weight"));
    }
}

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
    let mut weights = mlxcel_core::weights::load_weights_from_dir(model_dir)?;

    let config_path = model_dir.join("config.json");
    let mut is_quantized = false;
    if let Ok(config_str) = std::fs::read_to_string(&config_path) {
        let config_str = sanitize_config_json(&config_str);
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
            sanitize_tied_embeddings(&mut weights, &config);
            is_quantized = config.get("quantization").is_some()
                || config
                    .get("text_config")
                    .and_then(|tc| tc.get("quantization"))
                    .is_some();
        }
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

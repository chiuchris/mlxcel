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

//! Shared VLM loading router and helper utilities.
//!
//! This module owns the common config/weight helpers used by family-specific
//! VLM loaders. Real construction logic stays in sibling `vlm_<family>.rs`
//! modules so this file can remain a router plus a small toolbox.
//!
//! Families:
//! - `vlm_gemma.rs`: Gemma3 / Gemma3n
//! - `vlm_llava.rs`: LLaVA / Bunny
//! - `vlm_pixtral.rs`: Pixtral / Mistral3
//! - `vlm_qwen.rs`: Qwen2 / 2.5 / 3 / 3.5-VL
//! - `vlm_siglip.rs`: Aya Vision / PaliGemma
//! - `vlm_special.rs`: Llama4 / MiniCPM-o / Phi4MM / Phi4-SigLIP / Phi3V / Molmo2

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use models::sanitize_config_json;

#[path = "vlm_gemma.rs"]
mod gemma;
#[path = "vlm_llava.rs"]
mod llava;
#[path = "vlm_pixtral.rs"]
mod pixtral;
#[path = "vlm_qwen.rs"]
mod qwen;
#[path = "vlm_siglip.rs"]
mod siglip;
#[path = "vlm_special.rs"]
mod special;

pub(crate) use gemma::{load_gemma3_vlm, load_gemma3n_vlm, load_gemma4_vlm};
pub(crate) use llava::{load_llava_bunny_vlm, load_llava_vlm};
pub(crate) use pixtral::{load_mistral3_vlm, load_pixtral_vlm};
pub(crate) use qwen::{
    load_qwen2_5_vl, load_qwen2_vl, load_qwen3_5_moe_vlm, load_qwen3_5_vlm, load_qwen3_vl,
    load_qwen3_vl_moe,
};
pub(crate) use siglip::{load_aya_vision_vlm, load_paligemma_vlm};
pub(crate) use special::{
    load_llama4_vlm, load_minicpmo_vlm, load_molmo_point_vlm, load_molmo2_vlm, load_moondream3_vlm,
    load_phi3_vlm, load_phi4_siglip_vlm, load_phi4mm_vlm,
};

fn read_sanitized_vlm_config(model_path: &Path) -> Result<(String, Value)> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let full_config: Value = parse_vlm_config(&config_str, "config")?;
    Ok((config_str, full_config))
}

fn parse_vlm_config<T: DeserializeOwned>(config_str: &str, label: &str) -> Result<T> {
    serde_json::from_str(config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", label, e))
}

fn parse_required_vlm_subconfig<T: DeserializeOwned>(
    full_config: &Value,
    key: &str,
    label: &str,
) -> Result<T> {
    let value = full_config
        .get(key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing {} in config.json", key))?;
    serde_json::from_value(value).map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", label, e))
}

pub(super) fn require_object_mut<'a>(
    value: &'a mut Value,
    label: &str,
) -> Result<&'a mut serde_json::Map<String, Value>> {
    value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Expected {} to be a JSON object", label))
}

fn load_vlm_weights(model_path: &Path) -> Result<WeightMap> {
    let mut weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // On Apple Silicon, convert bf16 → f16 for performance (no Apple GPU has
    // native bf16 ALU).  Quantized models keep bf16 scales/biases as-is.
    let hw = mlxcel_core::hardware::get_hardware();
    if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown {
        let is_quantized = is_model_quantized(model_path);
        if !is_quantized {
            let had_bf16 = crate::models::convert_bf16_weights(&mut weights);
            if had_bf16 {
                crate::models::warn_bf16_precision();
            }
        }
    }

    Ok(weights)
}

/// Check if a model is quantized by looking for `.scales` keys in weights
/// or a `quantization` field in config.json.
fn is_model_quantized(model_path: &Path) -> bool {
    let config_path = model_path.join("config.json");
    if let Ok(config_str) = std::fs::read_to_string(&config_path) {
        let config_str = crate::models::sanitize_config_json(&config_str);
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
            if config.get("quantization").is_some() {
                return true;
            }
            if let Some(tc) = config.get("text_config")
                && tc.get("quantization").is_some()
            {
                return true;
            }
        }
    }
    false
}

fn read_optional_model_json(model_path: &Path, file_name: &str) -> Option<Value> {
    let path = model_path.join(file_name);
    let json = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&json).ok()
}

pub(super) fn strip_language_model_prefix(raw_weights: WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("language_model.") {
            stripped.to_string()
        } else {
            key
        };
        weights.insert(new_key, value);
    }
    weights
}

trait QwenVisionConfigExt {
    fn quant_group_size(&self) -> i32;
    fn quant_bits(&self) -> i32;
    fn set_quant_group_size(&mut self, value: i32);
    fn set_quant_bits(&mut self, value: i32);
    fn patch_size(&self) -> usize;
    fn temporal_patch_size(&self) -> usize;
    fn spatial_merge_size(&self) -> usize;
}

impl QwenVisionConfigExt for vision::encoders::qwen2_vl::Qwen2VLVisionConfig {
    fn quant_group_size(&self) -> i32 {
        self.quant_group_size
    }

    fn quant_bits(&self) -> i32 {
        self.quant_bits
    }

    fn set_quant_group_size(&mut self, value: i32) {
        self.quant_group_size = value;
    }

    fn set_quant_bits(&mut self, value: i32) {
        self.quant_bits = value;
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }
}

impl QwenVisionConfigExt for vision::encoders::qwen2_5_vl::Qwen25VLVisionConfig {
    fn quant_group_size(&self) -> i32 {
        self.quant_group_size
    }

    fn quant_bits(&self) -> i32 {
        self.quant_bits
    }

    fn set_quant_group_size(&mut self, value: i32) {
        self.quant_group_size = value;
    }

    fn set_quant_bits(&mut self, value: i32) {
        self.quant_bits = value;
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }
}

impl QwenVisionConfigExt for vision::encoders::qwen3_vl::Qwen3VLVisionConfig {
    fn quant_group_size(&self) -> i32 {
        self.quant_group_size
    }

    fn quant_bits(&self) -> i32 {
        self.quant_bits
    }

    fn set_quant_group_size(&mut self, value: i32) {
        self.quant_group_size = value;
    }

    fn set_quant_bits(&mut self, value: i32) {
        self.quant_bits = value;
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }
}

fn inherit_qwen_vision_quantization<T: QwenVisionConfigExt>(
    vision_config: &mut T,
    full_config: &Value,
) {
    if (vision_config.quant_group_size() == 0 || vision_config.quant_bits() == 0)
        && let Some(q) = full_config.get("quantization")
    {
        vision_config.set_quant_group_size(
            q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32,
        );
        vision_config.set_quant_bits(q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32);
    }
}

trait QwenTextQuantizationExt {
    fn has_quantization(&self) -> bool;
    fn set_quantization_from_value(&mut self, value: &Value);
}

impl QwenTextQuantizationExt for models::qwen3_vl::Qwen3VLConfig {
    fn has_quantization(&self) -> bool {
        self.quantization.is_some()
    }

    fn set_quantization_from_value(&mut self, value: &Value) {
        self.quantization = serde_json::from_value(value.clone()).ok();
    }
}

impl QwenTextQuantizationExt for models::qwen3_vl_moe::Qwen3VLMoeConfig {
    fn has_quantization(&self) -> bool {
        self.quantization.is_some()
    }

    fn set_quantization_from_value(&mut self, value: &Value) {
        self.quantization = serde_json::from_value(value.clone()).ok();
    }
}

fn inherit_qwen_text_quantization<T: QwenTextQuantizationExt>(
    text_config: &mut T,
    full_config: &Value,
) {
    if !text_config.has_quantization()
        && let Some(q) = full_config.get("quantization")
    {
        text_config.set_quantization_from_value(q);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QwenVisionTokenIds {
    image_token_id: i32,
    video_token_id: i32,
    vision_start_token_id: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35VlmVariant {
    Dense,
    Moe,
}

fn qwen_vl_token_ids(full_config: &Value, defaults: QwenVisionTokenIds) -> QwenVisionTokenIds {
    QwenVisionTokenIds {
        image_token_id: full_config
            .get("image_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(defaults.image_token_id as i64) as i32,
        video_token_id: full_config
            .get("video_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(defaults.video_token_id as i64) as i32,
        vision_start_token_id: full_config
            .get("vision_start_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(defaults.vision_start_token_id as i64) as i32,
    }
}

fn qwen35_vlm_token_defaults() -> QwenVisionTokenIds {
    QwenVisionTokenIds {
        image_token_id: 248056,
        video_token_id: 248057,
        vision_start_token_id: 248045,
    }
}

fn wrap_qwen35_vlm(vlm: vision::Qwen35VLModel, variant: Qwen35VlmVariant) -> LoadedModel {
    match variant {
        Qwen35VlmVariant::Dense => LoadedModel::Qwen35VLM(vlm),
        Qwen35VlmVariant::Moe => LoadedModel::Qwen35MoeVLM(vlm),
    }
}

fn qwen_vl_processor<T: QwenVisionConfigExt>(
    vision_config: &T,
) -> vision::processors::qwen2_vl::Qwen2VLProcessor {
    vision::processors::qwen2_vl::Qwen2VLProcessor::new(
        vision_config.patch_size(),
        vision_config.temporal_patch_size(),
        vision_config.spatial_merge_size(),
    )
}

fn qwen_vl_processor_with_norm<T: QwenVisionConfigExt>(
    vision_config: &T,
    mean: [f32; 3],
    std: [f32; 3],
) -> vision::processors::qwen2_vl::Qwen2VLProcessor {
    vision::processors::qwen2_vl::Qwen2VLProcessor::new_with_norm(
        vision_config.patch_size(),
        vision_config.temporal_patch_size(),
        vision_config.spatial_merge_size(),
        mean,
        std,
    )
}

fn rewrite_qwen3_vl_weight_key(key: String, moe_experts: bool) -> String {
    let new_key = if key.starts_with("model.language_model.") {
        key.replacen("model.language_model.", "model.", 1)
    } else if key.starts_with("model.visual.") {
        key.replacen("model.visual.", "vision_tower.", 1)
    } else if key.starts_with("language_model.") {
        key.replacen("language_model.", "", 1)
    } else {
        key
    };

    if moe_experts && new_key.contains(".mlp.experts.") {
        if let Some((_, after_experts)) = new_key.rsplit_once(".mlp.experts.") {
            if !after_experts.contains('.') {
                format!(
                    "{}.weight",
                    new_key.replace(".mlp.experts.", ".mlp.switch_mlp.")
                )
            } else {
                new_key.replace(".mlp.experts.", ".mlp.switch_mlp.")
            }
        } else {
            new_key
        }
    } else {
        new_key
    }
}

fn remap_qwen3_vl_weights(raw_weights: WeightMap, moe_experts: bool) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        weights.insert(rewrite_qwen3_vl_weight_key(key, moe_experts), value);
    }
    weights
}

#[cfg(test)]
#[path = "vlm_tests.rs"]
mod tests;

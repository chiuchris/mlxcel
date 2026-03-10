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

//! Special-case VLM loaders that do not fit another family bucket.
//!
//! Families:
//! - Llama4 VLM
//! - MiniCPM-o
//! - Phi4-SigLIP
//! - Phi3V
//! - Molmo2
//!
//! These architectures need custom config shaping or weight remapping that is
//! distinct from the LLaVA/Qwen/Gemma/SigLIP families, so they are grouped here.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{
    load_vlm_weights, parse_required_vlm_subconfig, parse_vlm_config, read_optional_model_json,
    read_sanitized_vlm_config,
};

fn phi3_vision_config() -> vision::config::VisionConfig {
    vision::config::VisionConfig {
        model_type: "phi3_v".to_string(),
        hidden_size: 1024,
        image_size: 336,
        intermediate_size: 4096,
        num_attention_heads: 16,
        num_hidden_layers: 24,
        num_channels: 3,
        patch_size: 14,
        layer_norm_eps: 1e-5,
    }
}

const PHI4_SIGLIP_TEXT_FIELDS: &[&str] = &[
    "vocab_size",
    "num_hidden_layers",
    "intermediate_size",
    "num_attention_heads",
    "rms_norm_eps",
    "hidden_size",
    "num_key_value_heads",
    "rope_theta",
    "rope_traditional",
    "partial_rotary_factor",
    "rope_scaling",
    "model_type",
    "quantization",
    "tie_word_embeddings",
    "max_position_embeddings",
];

fn parse_quantization_params(full_config: &Value) -> (i32, i32) {
    let group_size = full_config
        .get("quantization")
        .and_then(|value| value.get("group_size"))
        .and_then(|value| value.as_i64())
        .unwrap_or(0) as i32;
    let bits = full_config
        .get("quantization")
        .and_then(|value| value.get("bits"))
        .and_then(|value| value.as_i64())
        .unwrap_or(0) as i32;
    (group_size, bits)
}

pub(super) fn remap_minicpmo_text_weights(raw_weights: &WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("language_model.") {
            stripped.to_string()
        } else {
            key.clone()
        };
        weights.insert(new_key, mlxcel_core::copy(value));
    }
    weights
}

fn minicpmo_processor_config_value(model_path: &Path, full_config: &Value) -> Option<Value> {
    read_optional_model_json(model_path, "preprocessor_config.json")
        .or_else(|| {
            read_optional_model_json(model_path, "processor_config.json")
                .and_then(|value| value.get("image_processor").cloned().or(Some(value)))
        })
        .or_else(|| Some(full_config.clone()))
}

pub(crate) fn load_minicpmo_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::minicpmo::{
        MiniCPMOResampler, MiniCPMOVisionConfig, MiniCPMOVisionModel,
    };
    use vision::processors::minicpmo::MiniCPMOProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let text_config: models::qwen3_vl::Qwen3VLConfig = serde_json::from_value(full_config.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse MiniCPM-o text config: {}", e))?;
    let vision_config: MiniCPMOVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "MiniCPM-o vision config")?;
    let processor_config = minicpmo_processor_config_value(model_path, &full_config)
        .unwrap_or_else(|| full_config.clone());

    let raw_weights = load_vlm_weights(model_path)?;
    let text_weights = remap_minicpmo_text_weights(&raw_weights);
    let (group_size, bits) = parse_quantization_params(&full_config);

    let text_model = models::Qwen3VLModel::from_weights(&text_weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load MiniCPM-o text model: {}", e))?;
    let vision_tower = MiniCPMOVisionModel::from_weights(
        &raw_weights,
        &vision_config,
        "vision_tower",
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load MiniCPM-o vision tower: {}", e))?;
    let resampler = MiniCPMOResampler::from_weights(
        &raw_weights,
        "resampler",
        text_config.hidden_size,
        (text_config.hidden_size / 128).max(1),
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load MiniCPM-o resampler: {}", e))?;

    let processor = MiniCPMOProcessor::new(
        processor_config
            .get("patch_size")
            .and_then(|value| value.as_u64())
            .unwrap_or(14) as usize,
        processor_config
            .get("scale_resolution")
            .and_then(|value| value.as_u64())
            .unwrap_or(448) as usize,
        processor_config
            .get("image_feature_size")
            .and_then(|value| value.as_u64())
            .unwrap_or(
                full_config
                    .get("query_num")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(64),
            ) as usize,
    );

    let eos_token_ids = match full_config.get("eos_token_id") {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_i64().map(|id| id as i32))
            .collect(),
        Some(Value::Number(value)) => value.as_i64().map(|id| vec![id as i32]).unwrap_or_default(),
        _ => Vec::new(),
    };

    Ok(LoadedModel::MiniCPMOVLM(vision::MiniCPMOVLModel {
        text_model,
        vision_tower,
        resampler,
        processor,
        eos_token_ids,
    }))
}

pub(super) fn rewrite_phi4_siglip_weight_key(key: &str) -> Option<String> {
    if key.contains("position_ids") || key.contains("vision_model.head.") {
        None
    } else if let Some(rest) = key.strip_prefix("model.vision_tower.") {
        Some(format!("vision_tower.{}", rest))
    } else if let Some(rest) = key.strip_prefix("model.mm_projector.0.") {
        Some(format!("mm_projector_linear1.{}", rest))
    } else if let Some(rest) = key.strip_prefix("model.mm_projector.2.") {
        Some(format!("mm_projector_linear2.{}", rest))
    } else {
        Some(key.to_string())
    }
}

fn remap_phi4_siglip_weights(raw_weights: WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        let Some(new_key) = rewrite_phi4_siglip_weight_key(&key) else {
            continue;
        };
        weights.insert(new_key, value);
    }
    weights
}

pub(super) fn phi4_siglip_text_config_value(full_config: &Value) -> Result<Value> {
    let mut text_config = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    let text_obj = super::require_object_mut(&mut text_config, "Phi4-SigLIP text_config")?;
    for &field in PHI4_SIGLIP_TEXT_FIELDS {
        if let Some(value) = full_config.get(field) {
            text_obj
                .entry(field.to_string())
                .or_insert_with(|| value.clone());
        }
    }

    Ok(text_config)
}

pub(crate) fn load_phi4_siglip_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::phi4_siglip::{Phi4SigLipVisionConfig, Phi4SigLipVisionEncoder};
    use vision::processors::phi4_siglip::Phi4SigLipProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut text_config_value = phi4_siglip_text_config_value(&full_config)?;
    inherit_quantization_if_missing(&mut text_config_value, &full_config)?;
    let text_config: models::phi3::ModelArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Phi4-SigLIP text config: {}", e))?;

    let vision_config: Phi4SigLipVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Phi4-SigLIP vision config")?;

    let mm_hidden_size = full_config
        .get("mm_hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(vision_config.hidden_size as i64) as usize;
    let select_layer = full_config
        .get("mm_vision_select_layer")
        .and_then(|v| v.as_i64())
        .unwrap_or(-2) as isize;
    let min_num_patches = full_config
        .get("min_num_patches")
        .and_then(|v| v.as_u64())
        .unwrap_or(vision_config.num_patches as u64) as usize;
    let max_num_patches = full_config
        .get("max_num_patches")
        .and_then(|v| v.as_u64())
        .unwrap_or((vision_config.image_size / vision_config.patch_size).pow(2) as u64)
        as usize;

    let mut weights = remap_phi4_siglip_weights(load_vlm_weights(model_path)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Phi3Model::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Phi4-SigLIP text model: {}", e))?;
    let vision_tower = Phi4SigLipVisionEncoder::from_weights(
        &weights,
        &vision_config,
        "vision_tower.vision_tower.vision_model",
        text_config.group_size(),
        text_config.bits(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Phi4-SigLIP vision tower: {}", e))?;

    let mm_projector_linear1 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "mm_projector_linear1",
        text_config.group_size(),
        text_config.bits(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Phi4-SigLIP mm_projector_linear1: {}", e))?;
    let mm_projector_linear2 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "mm_projector_linear2",
        text_config.group_size(),
        text_config.bits(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Phi4-SigLIP mm_projector_linear2: {}", e))?;

    let processor =
        Phi4SigLipProcessor::new(vision_config.patch_size, min_num_patches, max_num_patches);

    let eos_token_ids = match full_config.get("eos_token_id") {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_i64().map(|id| id as i32))
            .collect(),
        Some(Value::Number(value)) => value
            .as_i64()
            .map(|id| vec![id as i32])
            .unwrap_or_else(Vec::new),
        _ => Vec::new(),
    };
    let _ = mm_hidden_size;

    let vlm = vision::Phi4SigLipVLModel {
        text_model,
        vision_tower,
        mm_projector_linear1,
        mm_projector_linear2,
        processor,
        select_layer,
        eos_token_ids,
    };

    Ok(LoadedModel::Phi4SigLipVLM(vlm))
}

pub(super) fn rewrite_phi3_weight_key(key: &str) -> Option<String> {
    if key.contains("position_ids") {
        None
    } else if let Some(rest) = key.strip_prefix("model.vision_embed_tokens.img_processor.") {
        Some(format!("vision_tower.{}", rest))
    } else if let Some(rest) = key.strip_prefix("model.vision_embed_tokens.img_projection.0.") {
        Some(format!("img_projection.0.{}", rest))
    } else if let Some(rest) = key.strip_prefix("model.vision_embed_tokens.img_projection.2.") {
        Some(format!("img_projection.2.{}", rest))
    } else if key == "model.vision_embed_tokens.glb_GN" {
        Some("glb_GN".to_string())
    } else if key == "model.vision_embed_tokens.sub_GN" {
        Some("sub_GN".to_string())
    } else {
        Some(key.to_string())
    }
}

pub(super) fn should_transpose_phi3_patch_embedding(shape: &[i32]) -> bool {
    if shape.len() != 4 {
        return false;
    }
    let out_ch = shape[0];
    let dim1 = shape[1];
    let dim2 = shape[2];
    !(out_ch >= dim1 && out_ch >= dim2 && dim1 == dim2)
}

fn remap_phi3_weights(raw_weights: WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        let Some(new_key) = rewrite_phi3_weight_key(&key) else {
            continue;
        };

        let mapped_value = if new_key.contains("patch_embedding.weight")
            && should_transpose_phi3_patch_embedding(&mlxcel_core::array_shape(&value))
        {
            mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1])
        } else {
            value
        };

        weights.insert(new_key, mapped_value);
    }
    weights
}

pub(super) fn phi3_num_crops(full_config: &Value, preprocessor_config: Option<&Value>) -> usize {
    if let Some(config) = preprocessor_config {
        return config
            .get("num_crops")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;
    }

    full_config
        .get("vision_config")
        .and_then(|vc| vc.get("num_crops"))
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize
}

pub(crate) fn load_phi3_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::processors::phi3_v::Phi3VProcessor;

    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let text_args: models::phi3::ModelArgs = parse_vlm_config(&config_str, "text config")?;

    let image_dim_out = phi3_vision_config().hidden_size;
    let mut weights = remap_phi3_weights(load_vlm_weights(model_path)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Phi3Model::from_weights(&weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;

    let vision_encoder = SigLipVisionModel::from_weights(
        &weights,
        &phi3_vision_config(),
        "vision_tower.vision_model",
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?
    .with_feature_selection(-2, "default".to_string());

    let glb_gn = weights
        .get("glb_GN")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("glb_GN weight not found"))?;
    let sub_gn = weights
        .get("sub_GN")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("sub_GN weight not found"))?;

    let group_size = text_args.group_size();
    let bits = text_args.bits();
    let img_proj_linear1 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "img_projection.0",
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load img_projection.0: {}", e))?;
    let img_proj_linear2 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "img_projection.2",
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load img_projection.2: {}", e))?;

    let preprocessor_config = read_optional_model_json(model_path, "preprocessor_config.json");
    let processor = Phi3VProcessor::new(phi3_num_crops(&full_config, preprocessor_config.as_ref()));

    let vlm = vision::Phi3VLModel {
        text_model,
        vision_encoder,
        glb_gn,
        sub_gn,
        img_proj_linear1,
        img_proj_linear2,
        processor,
        image_dim_out,
    };

    Ok(LoadedModel::Phi3VLM(vlm))
}

pub(super) fn cap_molmo2_vit_num_layers(num_layers: usize) -> usize {
    num_layers.min(25)
}

pub(super) fn parse_molmo2_vit_layers(adapter_config: &Value) -> Vec<i32> {
    adapter_config
        .get("vit_layers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect()
        })
        .unwrap_or_else(|| vec![-3, -9])
}

pub(super) fn rewrite_molmo2_weight_key(key: &str) -> String {
    let mut new_key = key.to_string();
    if new_key.starts_with("model.transformer.") {
        new_key = new_key.replacen("model.transformer.", "language_model.model.", 1);
    }
    if new_key.starts_with("model.vision_backbone.") {
        new_key = new_key.replacen("model.vision_backbone.", "vision_tower.", 1);
    }
    if new_key.starts_with("lm_head.") {
        new_key = new_key.replacen("lm_head.", "language_model.lm_head.", 1);
    }
    new_key.replace(".transformer.resblocks.", ".transformer.")
}

fn remap_molmo2_weights(raw_weights: WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        weights.insert(rewrite_molmo2_weight_key(&key), value);
    }
    weights
}

pub(super) fn molmo2_max_crops(preprocessor_config: Option<&Value>) -> usize {
    preprocessor_config
        .and_then(|config| config.get("max_crops"))
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as usize
}

pub(crate) fn load_molmo2_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::molmo2::Molmo2VisionModel;
    use vision::processors::molmo2::Molmo2Processor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| full_config.clone());
    inherit_quantization_if_missing(&mut text_config_value, &full_config)?;
    let text_config: models::molmo2::Molmo2TextConfig =
        serde_json::from_value(text_config_value)
            .map_err(|e| anyhow::anyhow!("Failed to parse text config: {}", e))?;

    let vision_config = full_config.get("vision_config").unwrap_or(&full_config);
    let vit_config = vision_config.get("vit_config").unwrap_or(vision_config);
    let adapter_config = vision_config.get("adapter_config").unwrap_or(vision_config);

    let vit_num_layers = cap_molmo2_vit_num_layers(
        vit_config
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(25) as usize,
    );
    let vit_hidden_size = vit_config
        .get("hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(1152) as i32;
    let vit_intermediate_size = vit_config
        .get("intermediate_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(4304) as i32;
    let vit_num_heads = vit_config
        .get("num_attention_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let vit_num_kv_heads = vit_config
        .get("num_key_value_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let vit_head_dim = vit_config
        .get("head_dim")
        .and_then(|v| v.as_i64())
        .unwrap_or(72) as i32;
    let vit_image_num_pos = vit_config
        .get("image_num_pos")
        .and_then(|v| v.as_u64())
        .unwrap_or(729) as usize;
    let vit_layer_norm_eps = vit_config
        .get("layer_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let vit_float32_attention = vit_config
        .get("float32_attention")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let adapter_hidden_size = adapter_config
        .get("hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(1152) as i32;
    let adapter_intermediate_size = adapter_config
        .get("intermediate_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(9728) as i32;
    let adapter_text_hidden_size = adapter_config
        .get("text_hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(text_config.hidden_size as i64) as i32;
    let adapter_num_heads = adapter_config
        .get("num_attention_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let adapter_num_kv_heads = adapter_config
        .get("num_key_value_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let adapter_head_dim = adapter_config
        .get("head_dim")
        .and_then(|v| v.as_i64())
        .unwrap_or(72) as i32;
    let adapter_float32_attention = adapter_config
        .get("float32_attention")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let pooling_attention_mask = adapter_config
        .get("pooling_attention_mask")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let image_patch_id = full_config
        .get("image_patch_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151938) as i32;
    let image_end_token_id = full_config
        .get("image_end_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151937) as i32;

    let mut weights = remap_molmo2_weights(load_vlm_weights(model_path)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model =
        models::Molmo2Model::from_weights(&weights, &text_config, "language_model.model")
            .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;

    let vision_tower = Molmo2VisionModel::from_weights(
        &weights,
        "vision_tower",
        vit_num_layers,
        vit_hidden_size,
        vit_intermediate_size,
        vit_num_heads,
        vit_num_kv_heads,
        vit_head_dim,
        vit_image_num_pos,
        vit_layer_norm_eps,
        vit_float32_attention,
        adapter_hidden_size,
        adapter_intermediate_size,
        adapter_text_hidden_size,
        adapter_num_heads,
        adapter_num_kv_heads,
        adapter_head_dim,
        adapter_float32_attention,
        &parse_molmo2_vit_layers(adapter_config),
        pooling_attention_mask,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision model: {}", e))?;

    let preprocessor_config = read_optional_model_json(model_path, "preprocessor_config.json");
    let processor = Molmo2Processor::new(
        molmo2_max_crops(preprocessor_config.as_ref()),
        None,
        None,
        None,
        None,
    );

    let vlm = vision::Molmo2VLModel {
        text_model,
        vision_tower,
        processor,
        image_patch_id,
        image_end_token_id,
    };

    Ok(LoadedModel::Molmo2VLM(vlm))
}

pub(super) fn inherit_quantization_if_missing(
    text_config: &mut Value,
    full_config: &Value,
) -> Result<()> {
    if text_config.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        super::require_object_mut(text_config, "special VLM text_config")?
            .insert("quantization".to_string(), q.clone());
    }
    Ok(())
}

fn language_model_only_weights(weights: &WeightMap) -> WeightMap {
    let mut text_weights = WeightMap::new();
    for (key, value) in weights {
        if key.starts_with("language_model.") {
            text_weights.insert(key.clone(), mlxcel_core::copy(value));
        }
    }
    text_weights
}

pub(super) fn llama4_vision_prefix(weights: &WeightMap) -> &'static str {
    if weights.contains_key("vision_model.patch_embedding.linear.weight") {
        "vision_model"
    } else {
        "vision_tower"
    }
}

pub(super) fn llama4_quantization_params(full_config: &Value) -> (i32, i32) {
    let group_size = full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(64) as i32;
    let bits = full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;
    (group_size, bits)
}

pub(super) fn llama4_token_ids(full_config: &Value) -> (i32, i32) {
    let image_token_id = full_config
        .get("image_token_index")
        .or_else(|| full_config.get("image_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(200092) as i32;
    let pad_token_id = full_config
        .get("text_config")
        .and_then(|tc| tc.get("pad_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(200018) as i32;
    (image_token_id, pad_token_id)
}

pub(super) fn llama4_mm_tokens_per_image(
    vision_config: &vision::encoders::llama4::Llama4VisionConfig,
) -> usize {
    let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
    (num_patches as f32 * vision_config.pixel_shuffle_ratio.powi(2)) as usize
}

pub(crate) fn load_llama4_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::linear::LinearProjector;
    use vision::encoders::llama4::{Llama4VisionConfig, Llama4VisionModel};
    use vision::processors::siglip::SigLipProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let text_config_value = if let Some(tc) = full_config.get("text_config") {
        let mut tc = tc.clone();
        inherit_quantization_if_missing(&mut tc, &full_config)?;
        tc
    } else {
        full_config.clone()
    };

    let text_args: models::llama4::TextArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Llama4 text config: {}", e))?;
    let vision_config: Llama4VisionConfig = serde_json::from_value(
        full_config
            .get("vision_config")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing vision_config in config.json"))?,
    )
    .map_err(|e| anyhow::anyhow!("Failed to parse vision_config: {}", e))?;

    let weights = load_vlm_weights(model_path)?;
    let mut text_weights = language_model_only_weights(&weights);
    models::sanitize_tied_embeddings(&mut text_weights, &full_config);

    let text_model = models::Llama4CxxModel::from_weights(&text_weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load Llama4 text model: {}", e))?;
    let text_wrapper = models::Llama4Wrapper::new(text_model);

    let (quant_group_size, quant_bits) = llama4_quantization_params(&full_config);
    let vision_encoder = Llama4VisionModel::from_weights(
        &weights,
        &vision_config,
        llama4_vision_prefix(&weights),
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Llama4 vision encoder: {}", e))?;

    let connector = LinearProjector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Llama4 projector: {}", e))?;

    let (image_token_id, pad_token_id) = llama4_token_ids(&full_config);
    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(SigLipProcessor::new(vision_config.image_size)),
        image_token_id,
        pad_token_id,
        hidden_size: text_args.hidden_size,
        boi_token_id: 0,
        eoi_token_id: 0,
        mm_tokens_per_image: llama4_mm_tokens_per_image(&vision_config),
        merge_strategy: vision::MergeStrategy::LLaVA,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(LoadedModel::Llama4(text_wrapper)),
        vision: vision_module,
    };

    Ok(LoadedModel::Llama4VLM(vlm))
}

#[cfg(test)]
#[path = "vlm_special_tests.rs"]
mod tests;

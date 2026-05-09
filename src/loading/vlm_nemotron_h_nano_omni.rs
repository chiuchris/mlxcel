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

//! Nemotron H Nano Omni VLM loader (vision-only scope, issue #554).
//!
//! Constructs [`crate::vision::NemotronHNanoOmniVlModel`] from a HF
//! `mlx-community` checkpoint. Audio weights present in the checkpoint
//! are filtered out — see issue #554's audio follow-up.
//!
//! Wire-up:
//! - `config.json` carries `text_config` (parsed by
//!   [`crate::models::nemotron_h::NemotronHConfig`]) plus `vision_config`
//!   (parsed by
//!   [`crate::vision::encoders::nemotron_h_nano_omni::NemotronHNanoOmniVisionConfig`])
//!   and the top-level projector knobs (`projector_hidden_size`,
//!   `vit_hidden_size`, `downsample_ratio`, `ps_version`,
//!   `img_context_token_id`).
//! - Weight remap mirrors upstream `Model.sanitize`: rename
//!   `mlp1.{0,1,3}` → `mlp1.layers.{0,1,3}` and strip the
//!   `language_model.` prefix from text weights so the existing
//!   [`crate::models::NemotronHModel`] sees the same names it does for
//!   the text-only Nemotron-H checkpoint.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::path::Path;

use super::{load_vlm_weights, read_sanitized_vlm_config};
use crate::LoadedModel;
use crate::models::NemotronHModel;
use crate::models::nemotron_h::{BlockType, NemotronHConfig};
use crate::vision::encoders::nemotron_h_nano_omni::{
    NemotronHNanoOmniVisionConfig, NemotronHNanoOmniVisionModel,
};
use crate::vision::nemotron_h_nano_omni_vl::{
    NemotronHNanoOmniProjector, NemotronHNanoOmniVlConfig, NemotronHNanoOmniVlModel,
};
use crate::vision::processors::nemotron_h_nano_omni::{
    NemotronHNanoOmniImageProcessor, NemotronHNanoOmniProcessorConfig,
};

/// Strip audio-only weights from the checkpoint. Vision-only PR per
/// issue #554; audio support is tracked separately.
fn filter_out_audio_weights(weights: WeightMap) -> WeightMap {
    weights
        .into_iter()
        .filter(|(key, _)| {
            !(key.starts_with("sound_encoder.")
                || key.starts_with("sound_projection.")
                || key.starts_with("sound_feature_extractor.")
                || key.starts_with("audio_tower."))
        })
        .collect()
}

/// Mirror upstream `Model.sanitize` mlp1 layer rename:
/// `mlp1.{0,1,3}.` → `mlp1.layers.{0,1,3}.`.
fn rename_projector_weights(raw: WeightMap) -> WeightMap {
    let mut out = WeightMap::new();
    for (key, value) in raw {
        let new_key = if key.starts_with("mlp1.0.") {
            key.replacen("mlp1.0.", "mlp1.layers.0.", 1)
        } else if key.starts_with("mlp1.1.") {
            key.replacen("mlp1.1.", "mlp1.layers.1.", 1)
        } else if key.starts_with("mlp1.3.") {
            key.replacen("mlp1.3.", "mlp1.layers.3.", 1)
        } else {
            key
        };
        out.insert(new_key, value);
    }
    out
}

/// Split the unified weight map into text-model weights (with
/// `language_model.` prefix removed) and the remainder (vision tower +
/// projector + everything else). The text path can then call
/// [`NemotronHModel::sanitize_weights`] and feed the result into the
/// existing config-backed builder.
fn split_text_and_others(raw: WeightMap) -> (WeightMap, WeightMap) {
    let mut text = WeightMap::new();
    let mut others = WeightMap::new();
    for (key, value) in raw {
        if let Some(rest) = key.strip_prefix("language_model.") {
            text.insert(rest.to_string(), value);
        } else {
            others.insert(key, value);
        }
    }
    (text, others)
}

fn parse_text_config(full_config: &Value) -> Result<NemotronHConfig> {
    let text_value = full_config
        .get("text_config")
        .or_else(|| full_config.get("llm_config"))
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!("Missing text_config in nemotron_h_nano_omni config.json")
        })?;
    let mut config: NemotronHConfig = serde_json::from_value(text_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Nemotron-H text config: {e}"))?;
    config
        .post_init()
        .map_err(|e| anyhow::anyhow!("Nemotron-H text config post_init failed: {e}"))?;
    Ok(config)
}

fn parse_vision_config(full_config: &Value) -> Result<NemotronHNanoOmniVisionConfig> {
    let vision_value = full_config
        .get("vision_config")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    serde_json::from_value(vision_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Nemotron H Nano Omni vision config: {e}"))
}

fn parse_processor_config(
    model_path: &Path,
    full_config: &Value,
) -> NemotronHNanoOmniProcessorConfig {
    // The released checkpoint stores image-processor knobs in
    // `preprocessor_config.json` (or the nested `image_processor` block
    // in `processor_config.json` on older exports). Fall back to
    // sensible defaults when the file is missing — the loader is
    // resilient to partial metadata.
    let mut config = NemotronHNanoOmniProcessorConfig::default();

    let mut sources: Vec<Value> = Vec::new();
    if let Ok(raw) = std::fs::read_to_string(model_path.join("preprocessor_config.json"))
        && let Ok(value) = serde_json::from_str::<Value>(&raw)
    {
        sources.push(value);
    }
    if let Ok(raw) = std::fs::read_to_string(model_path.join("processor_config.json"))
        && let Ok(value) = serde_json::from_str::<Value>(&raw)
    {
        if let Some(nested) = value.get("image_processor").cloned() {
            sources.push(nested);
        }
        sources.push(value);
    }

    // Final fallback to the top-level config so global overrides like
    // `downsample_ratio` propagate to the processor.
    sources.push(full_config.clone());

    fn array_of_3<T: serde::de::DeserializeOwned + Copy + Default>(
        value: Option<&Value>,
    ) -> Option<[T; 3]> {
        let arr = value?.as_array()?;
        if arr.len() != 3 {
            return None;
        }
        let mut out = [T::default(); 3];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = serde_json::from_value(arr[i].clone()).ok()?;
        }
        Some(out)
    }

    for source in &sources {
        if let Some(value) = array_of_3::<f32>(source.get("norm_mean")) {
            config.norm_mean = value;
        }
        if let Some(value) = array_of_3::<f32>(source.get("norm_std")) {
            config.norm_std = value;
        }
        if let Some(value) = source.get("patch_size").and_then(Value::as_u64) {
            config.patch_size = value as usize;
        }
        if let Some(value) = source.get("downsample_ratio").and_then(Value::as_f64) {
            config.downsample_ratio = value as f32;
        }
        if let Some(value) = source.get("min_num_patches").and_then(Value::as_u64) {
            config.min_num_patches = value as usize;
        }
        if let Some(value) = source.get("max_num_patches").and_then(Value::as_u64) {
            config.max_num_patches = value as usize;
        }
        if let Some(value) = source.get("max_model_len").and_then(Value::as_u64) {
            config.max_model_len = value as usize;
        }
    }
    config
}

fn parse_eos_token_ids(model_path: &Path, full_config: &Value) -> Vec<i32> {
    if let Ok(raw) = std::fs::read_to_string(model_path.join("generation_config.json"))
        && let Ok(value) = serde_json::from_str::<Value>(&raw)
        && let Some(ids) = read_eos(&value)
    {
        return ids;
    }
    read_eos(full_config).unwrap_or_default()
}

fn read_eos(value: &Value) -> Option<Vec<i32>> {
    let entry = value.get("eos_token_id")?;
    match entry {
        Value::Array(arr) => Some(
            arr.iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect(),
        ),
        Value::Number(n) => n.as_i64().map(|n| vec![n as i32]),
        _ => None,
    }
}

fn parse_block_types(config: &NemotronHConfig) -> Result<Vec<BlockType>> {
    let pattern = config.hybrid_override_pattern.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "NemotronH: hybrid_override_pattern must be set in text_config for nemotron_h_nano_omni"
        )
    })?;
    Ok(pattern
        .iter()
        .map(|name| BlockType::from_str(name))
        .collect())
}

pub(crate) fn load_nemotron_h_nano_omni_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_, full_config) = read_sanitized_vlm_config(model_path)?;

    // 1. Parse all configs up front so we can fail early on missing fields.
    let text_config = parse_text_config(&full_config)?;
    let block_types = parse_block_types(&text_config)?;
    let vision_config = parse_vision_config(&full_config)?;
    let processor_config = parse_processor_config(model_path, &full_config);
    let processor = NemotronHNanoOmniImageProcessor::new(processor_config);

    let vit_hidden_size = full_config
        .get("vit_hidden_size")
        .and_then(Value::as_u64)
        .unwrap_or(vision_config.hidden_size as u64) as usize;
    let projector_hidden_size = full_config
        .get("projector_hidden_size")
        .and_then(Value::as_u64)
        .unwrap_or(4096) as usize;
    let downsample_ratio = full_config
        .get("downsample_ratio")
        .and_then(Value::as_f64)
        .unwrap_or(0.5) as f32;
    let ps_version = full_config
        .get("ps_version")
        .and_then(Value::as_str)
        .unwrap_or("v1")
        .to_string();
    let img_context_token_id = full_config
        .get("img_context_token_id")
        .or_else(|| full_config.get("image_token_index"))
        .and_then(Value::as_i64)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Missing img_context_token_id (or image_token_index) in nemotron_h_nano_omni config"
            )
        })? as i32;
    let image_start_token_id = full_config
        .get("image_start_token_id")
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;
    let image_end_token_id = full_config
        .get("image_end_token_id")
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;

    let eos_token_ids = parse_eos_token_ids(model_path, &full_config);

    // 2. Load weights, drop audio, rename projector layers, then split
    // text vs. other weights for downstream component construction.
    let raw_weights = load_vlm_weights(model_path)?;
    let raw_weights = filter_out_audio_weights(raw_weights);
    let raw_weights = rename_projector_weights(raw_weights);
    let (text_weights, other_weights) = split_text_and_others(raw_weights);

    // 3. Build the text model using the existing Nemotron-H sanitizer
    // and `from_weights` builder. This guarantees parity with the
    // text-only Nemotron-H path — any future text-side improvement
    // automatically benefits the VLM.
    let text_weights = NemotronHModel::sanitize_weights(text_weights, &text_config);
    let text_model = NemotronHModel::from_weights(text_config, text_weights, block_types)
        .map_err(|err| anyhow::anyhow!("Failed to load Nemotron H Nano Omni text model: {err}"))?;

    // 4. Build the vision tower and the multimodal projector. The vision
    // tower lives at `vision_model.radio_model.*`; the projector at
    // `mlp1.*` (after the layer rename above).
    let group_size = group_size_from_config(&full_config);
    let bits = bits_from_config(&full_config);
    let vision_tower = NemotronHNanoOmniVisionModel::from_weights(
        &other_weights,
        "vision_model.radio_model",
        &vision_config,
        group_size,
        bits,
    )
    .map_err(|err| anyhow::anyhow!("Failed to load Nemotron H Nano Omni vision tower: {err}"))?;
    let projector =
        NemotronHNanoOmniProjector::from_weights(&other_weights, "mlp1", group_size, bits)
            .map_err(|err| {
                anyhow::anyhow!("Failed to load Nemotron H Nano Omni projector: {err}")
            })?;

    // 5. Compose the VLM wrapper.
    let text_hidden_size = text_model.hidden_size();
    let vl_config = NemotronHNanoOmniVlConfig {
        vit_hidden_size,
        projector_hidden_size,
        text_hidden_size,
        downsample_ratio,
        ps_version,
        img_context_token_id,
        image_start_token_id,
        image_end_token_id,
        eos_token_ids,
    };
    let model =
        NemotronHNanoOmniVlModel::new(text_model, vision_tower, projector, processor, vl_config);
    Ok(LoadedModel::NemotronHNanoOmniVLM(model))
}

fn group_size_from_config(full_config: &Value) -> i32 {
    full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(Value::as_i64)
        .unwrap_or(64) as i32
}

fn bits_from_config(full_config: &Value) -> i32 {
    full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(Value::as_i64)
        .unwrap_or(4) as i32
}

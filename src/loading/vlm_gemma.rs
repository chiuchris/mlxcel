use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{
    load_vlm_weights, parse_vlm_config, read_sanitized_vlm_config, strip_language_model_prefix,
};

struct Gemma3nMetadata {
    vision_hidden_size: usize,
    image_size: usize,
    image_token_id: i32,
    boi_token_id: i32,
    eoi_token_id: i32,
    vision_rms_eps: f32,
}

fn gemma3n_metadata(config: &Value) -> Gemma3nMetadata {
    Gemma3nMetadata {
        vision_hidden_size: config
            .get("vision_config")
            .and_then(|vc| vc.get("hidden_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize,
        image_size: config
            .get("vision_config")
            .and_then(|vc| vc.get("image_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize,
        image_token_id: config
            .get("image_token_id")
            .or_else(|| config.get("image_token_index"))
            .and_then(|v| v.as_i64())
            .unwrap_or(262_145) as i32,
        boi_token_id: config
            .get("boi_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(255_999) as i32,
        eoi_token_id: config
            .get("eoi_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(262_144) as i32,
        vision_rms_eps: config
            .get("vision_config")
            .and_then(|vc| vc.get("rms_norm_eps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32,
    }
}

fn gemma3n_needs_conv_transpose(raw_weights: &WeightMap) -> bool {
    raw_weights
        .get("model.vision_tower.timm_model.blocks.0.0.conv_exp.weight")
        .map(|w| {
            let shape = mlxcel_core::array_shape(w);
            shape.len() == 4 && shape[1] > shape[2]
        })
        .unwrap_or(false)
}

fn gemma3n_language_model_prefix(weights: &WeightMap) -> &'static str {
    if weights.contains_key("language_model.model.embed_tokens.weight") {
        "language_model.model"
    } else {
        "language_model"
    }
}

fn sanitize_gemma3n_weights(raw_weights: WeightMap) -> WeightMap {
    let needs_transpose = gemma3n_needs_conv_transpose(&raw_weights);
    let mut weights = WeightMap::new();

    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("model.") {
            stripped.to_string()
        } else {
            key
        };

        let value = if needs_transpose {
            let shape = mlxcel_core::array_shape(&value);
            if shape.len() == 4 {
                mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1])
            } else {
                mlxcel_core::copy(&value)
            }
        } else {
            mlxcel_core::copy(&value)
        };
        weights.insert(new_key, value);
    }

    weights
}

/// Load a Gemma3 VLM model (text + vision tower + projector).
pub(crate) fn load_gemma3_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::config::VLMConfig;
    use vision::connectors::avg_pool::AvgPoolProjector;
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::processors::siglip::SigLipProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let vlm_config: VLMConfig = serde_json::from_value(full_config.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse VLM config: {}", e))?;
    let text_config: models::gemma3::ModelArgs =
        serde_json::from_value(vlm_config.text_config.clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse text_config: {}", e))?;

    let mut weights = strip_language_model_prefix(load_vlm_weights(model_path)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Gemma3Model::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;
    let text_wrapper = models::Gemma3Wrapper::new(text_model);

    let vision_encoder = SigLipVisionModel::from_weights(
        &weights,
        &vlm_config.vision_config,
        "vision_tower.vision_model",
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?;

    let mm_tokens_per_image = vlm_config.get_mm_tokens_per_image();
    let connector = AvgPoolProjector::from_weights(
        &weights,
        "multi_modal_projector",
        vlm_config.vision_config.hidden_size,
        vlm_config.vision_config.image_size,
        vlm_config.vision_config.patch_size,
        mm_tokens_per_image,
        vlm_config.vision_config.layer_norm_eps,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load projector: {}", e))?;

    let processor = SigLipProcessor::new(vlm_config.vision_config.image_size);
    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: vlm_config.image_token_index,
        pad_token_id: vlm_config.pad_token_id,
        hidden_size: if vlm_config.hidden_size > 0 {
            vlm_config.hidden_size
        } else {
            text_config.hidden_size
        },
        boi_token_id: vlm_config.boi_token_index,
        eoi_token_id: vlm_config.eoi_token_index,
        mm_tokens_per_image,
        merge_strategy: vision::MergeStrategy::Gemma3,
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(LoadedModel::Gemma3(text_wrapper)),
        vision: vision_module,
    };

    Ok(LoadedModel::Gemma3VLM(vlm))
}

/// Load a Gemma3n VLM model.
pub(crate) fn load_gemma3n_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::gemma3n::load_gemma3n_vision;
    use vision::processors::siglip::SigLipProcessor;

    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let top_args: models::gemma3n::ModelArgs = parse_vlm_config(&config_str, "Gemma3n config")?;
    let text_config = top_args.text_args();
    let metadata = gemma3n_metadata(&full_config);

    let mut weights = sanitize_gemma3n_weights(load_vlm_weights(model_path)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let language_model = models::gemma3n::Gemma3nLanguageModel::from_weights(
        &weights,
        &text_config,
        gemma3n_language_model_prefix(&weights),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;

    let text_model = models::Gemma3nModel {
        language_model,
        config: text_config.clone(),
    };

    let vision_tower = load_gemma3n_vision(&weights, "vision_tower.timm_model")
        .map_err(|e| anyhow::anyhow!("Failed to load vision tower: {}", e))?;

    let group_size = text_config
        .quantization
        .as_ref()
        .map(|q| q.group_size as i32)
        .unwrap_or(64);
    let bits = text_config
        .quantization
        .as_ref()
        .map(|q| q.bits as i32)
        .unwrap_or(4);

    let embed_vision = models::gemma3n::Gemma3nMultimodalEmbedder::from_weights(
        &weights,
        "embed_vision",
        metadata.vision_hidden_size,
        text_config.hidden_size,
        metadata.vision_rms_eps,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load multimodal embedder: {}", e))?;

    let processor = SigLipProcessor::new_rescale_only(metadata.image_size);

    let vlm = vision::Gemma3nVLModel::new(
        text_model,
        vision_tower,
        embed_vision,
        processor,
        metadata.image_token_id,
        metadata.boi_token_id,
        metadata.eoi_token_id,
        metadata.vision_hidden_size,
    );

    Ok(LoadedModel::Gemma3nVLM(vlm))
}

#[cfg(test)]
#[path = "vlm_gemma_tests.rs"]
mod tests;

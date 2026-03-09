use anyhow::Result;
use serde::de::DeserializeOwned;
use std::fmt::Display;
use std::path::{Path, PathBuf};

use crate::LoadedModel;
use crate::lora;
use crate::models::{self, ModelType, get_model_type, sanitize_config_json};
use crate::tokenizer::{self, MlxcelTokenizer};
use mlxcel_core::weights::WeightMap;

mod config_backed;
mod nonstandard;
mod special;
mod vlm;

use self::config_backed::{
    try_load_config_backed_model_from_dir, try_load_config_backed_model_from_weights,
};
use self::nonstandard::try_load_nonstandard_model_from_dir;
use self::special::{adapter_loading_unsupported_message, try_load_special_model_from_weights};
use self::vlm::*;

/// Resolve model path: if a file is given, use its parent directory.
///
/// This provides compatibility with callers that pass a specific model file
/// (e.g. `/path/to/model.safetensors`) instead of the model directory.
fn resolve_model_dir(model_path: &Path) -> PathBuf {
    if model_path.is_file() {
        model_path.parent().unwrap_or(model_path).to_path_buf()
    } else {
        model_path.to_path_buf()
    }
}

fn parse_eos_token_ids(config: &serde_json::Value) -> Vec<i32> {
    match config.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => {
            if let Some(id) = n.as_i64() {
                vec![id as i32]
            } else {
                Vec::new()
            }
        }
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_i64().map(|n| n as i32))
            .collect(),
        _ => Vec::new(),
    }
}

pub(super) fn parse_model_config<T: DeserializeOwned>(config_str: &str) -> Result<T> {
    serde_json::from_str(config_str).map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))
}

fn is_ministral3_config(config: &serde_json::Value) -> bool {
    config
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .and_then(|mt| mt.as_str())
        .map(|mt| mt == "ministral3")
        .unwrap_or(false)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35VlmKind {
    Dense,
    Moe,
}

fn qwen35_vlm_kind(model_type: ModelType) -> Option<Qwen35VlmKind> {
    match model_type {
        ModelType::Qwen35VLM => Some(Qwen35VlmKind::Dense),
        ModelType::Qwen35MoeVLM => Some(Qwen35VlmKind::Moe),
        _ => None,
    }
}

fn try_load_vlm_model_from_dir(
    model_type: ModelType,
    model_path: &Path,
) -> Result<Option<LoadedModel>> {
    Ok(match model_type {
        ModelType::Llama4VLM => Some(load_llama4_vlm(model_path)?),
        ModelType::Qwen35VLM | ModelType::Qwen35MoeVLM => Some(
            match qwen35_vlm_kind(model_type)
                .expect("Qwen3.5 VLM variants should map to a concrete loader")
            {
                Qwen35VlmKind::Dense => load_qwen3_5_vlm(model_path)?,
                Qwen35VlmKind::Moe => load_qwen3_5_moe_vlm(model_path)?,
            },
        ),
        ModelType::Gemma3VLM => Some(load_gemma3_vlm(model_path)?),
        ModelType::LlavaVLM => Some(load_llava_vlm(model_path)?),
        ModelType::LlavaBunnyVLM => Some(load_llava_bunny_vlm(model_path)?),
        ModelType::AyaVisionVLM => Some(load_aya_vision_vlm(model_path)?),
        ModelType::PaliGemmaVLM => Some(load_paligemma_vlm(model_path)?),
        ModelType::PixtralVLM => Some(load_pixtral_vlm(model_path)?),
        ModelType::Mistral3VLM => Some(load_mistral3_vlm(model_path)?),
        ModelType::Qwen2VL => Some(load_qwen2_vl(model_path)?),
        ModelType::Qwen25VL => Some(load_qwen2_5_vl(model_path)?),
        ModelType::Qwen3VL => Some(load_qwen3_vl(model_path)?),
        ModelType::Qwen3VLMoe => Some(load_qwen3_vl_moe(model_path)?),
        ModelType::Gemma3nVLM => Some(load_gemma3n_vlm(model_path)?),
        ModelType::Phi3VLM => Some(load_phi3_vlm(model_path)?),
        ModelType::Molmo2VLM => Some(load_molmo2_vlm(model_path)?),
        _ => None,
    })
}

fn load_pair_from_dir<T, U, E, F>(path_str: &str, load: F) -> Result<T>
where
    F: FnOnce(String) -> std::result::Result<(T, U), E>,
    E: Display,
{
    let (model, _) = load(path_str.to_owned()).map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(model)
}

/// Read EOS token IDs from generation_config.json
///
/// Returns the token IDs from the `eos_token_id` field, which can be either
/// a single integer or an array of integers. Returns empty vec if not found.
pub fn read_eos_token_ids(model_dir: &Path) -> Vec<i32> {
    let config_path = model_dir.join("generation_config.json");
    let Ok(content) = std::fs::read_to_string(&config_path) else {
        return Vec::new();
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) else {
        return Vec::new();
    };
    parse_eos_token_ids(&config)
}

/// Load a model from a directory (or file — parent directory will be used)
pub fn load_model(model_path: &Path) -> Result<(LoadedModel, MlxcelTokenizer)> {
    let model_path = resolve_model_dir(model_path);
    let model_path = model_path.as_path();
    let model_type = get_model_type(model_path)?;
    let path_str = model_path.to_str().unwrap();

    let model = if model_type == ModelType::Mistral3 {
        // Mistral3 is a VLM wrapper - check text_config for inner model type
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = parse_model_config(&config_str)?;

        if is_ministral3_config(&config) {
            // Load as Ministral3 with text_config extracted
            LoadedModel::Ministral3(models::Ministral3Wrapper::new(load_pair_from_dir(
                path_str,
                models::Ministral3Model::load_from_text_config,
            )?))
        } else {
            // Load as standard Llama
            LoadedModel::Llama(load_pair_from_dir(path_str, models::Llama3Model::load)?)
        }
    } else if let Some(model) = try_load_vlm_model_from_dir(model_type, model_path)? {
        model
    } else if let Some(model) =
        try_load_nonstandard_model_from_dir(model_type, model_path, path_str)?
    {
        model
    } else {
        try_load_config_backed_model_from_dir(model_type, path_str)?.ok_or_else(|| {
            anyhow::anyhow!("Missing directory loader for model type: {:?}", model_type)
        })?
    };

    let tokenizer = tokenizer::load_tokenizer(model_path)?;
    Ok((model, tokenizer))
}

/// Load a model with LoRA adapter weights fused in
///
/// This loads the base model weights, fuses them with LoRA adapter weights,
/// then constructs the model from the fused weights.
pub fn load_model_with_adapter(
    model_path: &Path,
    adapter_path: &Path,
) -> Result<(LoadedModel, MlxcelTokenizer)> {
    let model_path = resolve_model_dir(model_path);
    let model_path = model_path.as_path();
    // Load base weights
    let base_weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Fuse with adapter weights
    let mut fused_weights = lora::apply_lora_adapters(&base_weights, adapter_path)?;

    // Build model from fused weights
    let model = load_model_from_weights(model_path, &mut fused_weights)?;

    let tokenizer = tokenizer::load_tokenizer(model_path)?;
    Ok((model, tokenizer))
}

/// Build a model from pre-loaded weights (used by adapter loading)
fn load_model_from_weights(model_path: &Path, weights: &mut WeightMap) -> Result<LoadedModel> {
    let model_type = get_model_type(model_path)?;
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config_str = sanitize_config_json(&config_str);

    // Sanitize tied embeddings: copy embed_tokens → lm_head if needed
    let config_value: serde_json::Value = parse_model_config(&config_str)?;
    models::sanitize_tied_embeddings(weights, &config_value);

    if let Some(message) = adapter_loading_unsupported_message(model_type) {
        return Err(anyhow::anyhow!(message));
    }

    let model = if matches!(model_type, ModelType::Llama | ModelType::Mistral3) {
        // Check for ministral3 sub-type
        let config: serde_json::Value = parse_model_config(&config_str)?;
        if model_type == ModelType::Mistral3 && is_ministral3_config(&config) {
            // Load as Ministral3 with text_config
            let text_config = config
                .get("text_config")
                .ok_or_else(|| anyhow::anyhow!("Missing text_config for Ministral3"))?;
            let args: models::ministral3::ModelArgs =
                serde_json::from_value(text_config.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to parse text_config: {}", e))?;
            let m = models::Ministral3Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Ministral3(models::Ministral3Wrapper::new(m))
        } else {
            let args: models::llama3::ModelArgs = parse_model_config(&config_str)?;
            let m = models::Llama3Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Llama(m)
        }
    } else if let Some(model) =
        try_load_special_model_from_weights(model_type, &config_str, weights)?
    {
        model
    } else {
        try_load_config_backed_model_from_weights(model_type, &config_str, weights)?.ok_or_else(
            || anyhow::anyhow!("Missing weight loader for model type: {:?}", model_type),
        )?
    };

    Ok(model)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

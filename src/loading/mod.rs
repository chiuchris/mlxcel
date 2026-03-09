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
mod vlm;

use self::config_backed::{
    try_load_config_backed_model_from_dir, try_load_config_backed_model_from_weights,
};
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

fn copy_weight_map(weights: &WeightMap) -> WeightMap {
    weights
        .iter()
        .map(|(key, value)| (key.clone(), mlxcel_core::copy(value)))
        .collect()
}

fn is_ministral3_config(config: &serde_json::Value) -> bool {
    config
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .and_then(|mt| mt.as_str())
        .map(|mt| mt == "ministral3")
        .unwrap_or(false)
}

fn qwen35_text_config(config: &serde_json::Value) -> Result<serde_json::Value> {
    let mut text_config = config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| config.clone());

    if text_config.get("quantization").is_none() && config.get("quantization").is_some() {
        let text_config_obj = text_config.as_object_mut().ok_or_else(|| {
            anyhow::anyhow!("Failed to merge quantization into non-object text_config")
        })?;
        text_config_obj.insert("quantization".to_string(), config["quantization"].clone());
    }

    Ok(text_config)
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

macro_rules! load_owned_model_from_config {
    ($config_str:expr, $weights:expr, $args_ty:ty, $builder:path, $wrap:expr) => {{
        let args: $args_ty = parse_model_config($config_str)?;
        let owned = copy_weight_map($weights);
        let model = $builder(args, owned).map_err(|err| anyhow::anyhow!("{}", err))?;
        ($wrap)(model)
    }};
}

fn adapter_loading_unsupported_message(model_type: ModelType) -> Option<&'static str> {
    match model_type {
        ModelType::Llama4VLM => Some("Llama4 VLM cannot be loaded with LoRA adapters yet"),
        ModelType::Qwen35VLM | ModelType::Qwen35MoeVLM => {
            Some("Qwen3.5 VLM does not support adapter loading")
        }
        ModelType::Gemma3VLM => Some("Gemma3 VLM cannot be loaded with LoRA adapters yet"),
        ModelType::LlavaVLM | ModelType::LlavaBunnyVLM => {
            Some("LLaVA VLM cannot be loaded with LoRA adapters yet")
        }
        ModelType::AyaVisionVLM => Some("Aya Vision VLM cannot be loaded with LoRA adapters yet"),
        ModelType::PaliGemmaVLM => Some("PaliGemma VLM cannot be loaded with LoRA adapters yet"),
        ModelType::PixtralVLM | ModelType::Mistral3VLM => {
            Some("Pixtral/Mistral3 VLM cannot be loaded with LoRA adapters yet")
        }
        ModelType::Qwen2VL => Some("Qwen2-VL cannot be loaded with LoRA adapters yet"),
        ModelType::Qwen25VL | ModelType::Qwen3VL | ModelType::Qwen3VLMoe => {
            Some("Qwen VL models cannot be loaded with LoRA adapters yet")
        }
        ModelType::Gemma3nVLM => Some("Gemma3n VLM cannot be loaded with LoRA adapters yet"),
        ModelType::Phi3VLM => {
            Some("Phi3V VLM does not support adapter loading; use load_model() instead")
        }
        ModelType::Molmo2VLM => {
            Some("Molmo2 VLM does not support adapter loading; use load_model() instead")
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecialWeightLoaderKind {
    Qwen35,
    Gemma3n,
    OwnedConfig,
    NemotronH,
    KimiLinear,
    Longcat,
    Rwkv7,
}

fn special_weight_loader_kind(model_type: ModelType) -> Option<SpecialWeightLoaderKind> {
    match model_type {
        ModelType::Qwen35 | ModelType::Qwen35Moe => Some(SpecialWeightLoaderKind::Qwen35),
        ModelType::Gemma3n => Some(SpecialWeightLoaderKind::Gemma3n),
        ModelType::Mamba
        | ModelType::Mamba2
        | ModelType::Jamba
        | ModelType::NemotronNAS
        | ModelType::RecurrentGemma => Some(SpecialWeightLoaderKind::OwnedConfig),
        ModelType::NemotronH => Some(SpecialWeightLoaderKind::NemotronH),
        ModelType::KimiLinear => Some(SpecialWeightLoaderKind::KimiLinear),
        ModelType::LongcatFlash | ModelType::LongcatFlashNgram => {
            Some(SpecialWeightLoaderKind::Longcat)
        }
        ModelType::Rwkv7 => Some(SpecialWeightLoaderKind::Rwkv7),
        _ => None,
    }
}

fn try_load_special_model_from_weights(
    model_type: ModelType,
    config_str: &str,
    weights: &mut WeightMap,
) -> Result<Option<LoadedModel>> {
    let Some(kind) = special_weight_loader_kind(model_type) else {
        return Ok(None);
    };

    Ok(Some(match kind {
        SpecialWeightLoaderKind::Qwen35 => {
            let v: serde_json::Value = parse_model_config(config_str)?;
            let text_config = qwen35_text_config(&v)?;
            let args: models::qwen3_5::Qwen35Config = serde_json::from_value(text_config)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let owned = copy_weight_map(weights);
            let owned = models::qwen3_5::sanitize_moe_weights(owned, &args);
            let m = models::Qwen35Model::from_weights(&owned, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            if model_type == ModelType::Qwen35Moe {
                LoadedModel::Qwen35Moe(m)
            } else {
                LoadedModel::Qwen35(m)
            }
        }
        SpecialWeightLoaderKind::Gemma3n => {
            let top_args: models::gemma3n::ModelArgs = parse_model_config(config_str)?;
            let config = top_args.text_args();
            let language_model = models::gemma3n::Gemma3nLanguageModel::from_weights(
                weights,
                &config,
                "language_model.model",
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Gemma3n(models::Gemma3nModel {
                language_model,
                config,
            })
        }
        SpecialWeightLoaderKind::OwnedConfig => match model_type {
            ModelType::Mamba => load_owned_model_from_config!(
                config_str,
                weights,
                models::mamba::MambaConfig,
                models::MambaModel::from_weights,
                LoadedModel::Mamba
            ),
            ModelType::Mamba2 => load_owned_model_from_config!(
                config_str,
                weights,
                models::mamba2::Mamba2Config,
                models::Mamba2Model::from_weights,
                LoadedModel::Mamba2
            ),
            ModelType::Jamba => load_owned_model_from_config!(
                config_str,
                weights,
                models::jamba::JambaConfig,
                models::JambaModel::from_weights,
                LoadedModel::Jamba
            ),
            ModelType::NemotronNAS => load_owned_model_from_config!(
                config_str,
                weights,
                models::nemotron_nas::NemotronNASConfig,
                models::NemotronNASModel::from_weights,
                LoadedModel::NemotronNAS
            ),
            ModelType::RecurrentGemma => load_owned_model_from_config!(
                config_str,
                weights,
                models::recurrent_gemma::GriffinConfig,
                models::GriffinModel::from_weights,
                LoadedModel::RecurrentGemma
            ),
            _ => unreachable!(
                "owned-config helper called for non-owned model: {:?}",
                model_type
            ),
        },
        SpecialWeightLoaderKind::NemotronH => {
            let args: models::nemotron_h::NemotronHConfig = parse_model_config(config_str)?;
            let block_types: Vec<models::nemotron_h::BlockType> = args
                .hybrid_override_pattern
                .iter()
                .map(|s| models::nemotron_h::BlockType::from_str(s))
                .collect();
            let owned = copy_weight_map(weights);
            let owned = models::NemotronHModel::sanitize_weights(owned, &args);
            let m = models::NemotronHModel::from_weights(args, owned, block_types)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::NemotronH(m)
        }
        SpecialWeightLoaderKind::KimiLinear => {
            let args: models::kimi_linear::KimiLinearConfig = parse_model_config(config_str)?;
            let mut owned = copy_weight_map(weights);
            owned = models::KimiLinearModel::sanitize_weights(owned, &args);
            let m = models::KimiLinearModel::from_weights(&owned, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::KimiLinear(m)
        }
        SpecialWeightLoaderKind::Longcat => {
            let args: models::longcat_flash_ngram::LongcatFlashNgramConfig =
                parse_model_config(config_str)?;
            let mut owned = copy_weight_map(weights);
            owned = models::longcat_flash_ngram::sanitize_weights(owned, &args);
            let m = models::LongcatFlashNgramModel::from_weights(&owned, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            if model_type == ModelType::LongcatFlashNgram {
                LoadedModel::LongcatFlashNgram(m)
            } else {
                LoadedModel::LongcatFlash(m)
            }
        }
        SpecialWeightLoaderKind::Rwkv7 => {
            let args: models::rwkv7::Rwkv7Config = parse_model_config(config_str)?;
            let m =
                models::Rwkv7::from_weights(weights, args).map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Rwkv7(m)
        }
    }))
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

fn try_load_nonstandard_model_from_dir(
    model_type: ModelType,
    model_path: &Path,
    path_str: &str,
) -> Result<Option<LoadedModel>> {
    Ok(match model_type {
        ModelType::Qwen35 => {
            Some(load_pair_from_dir(path_str, models::Qwen35Model::load).map(LoadedModel::Qwen35)?)
        }
        ModelType::Qwen35Moe => Some(
            load_pair_from_dir(path_str, models::Qwen35Model::load).map(LoadedModel::Qwen35Moe)?,
        ),
        ModelType::Gemma3n => {
            Some(load_from_dir(path_str, models::Gemma3nModel::load).map(LoadedModel::Gemma3n)?)
        }
        ModelType::Mamba => Some(
            load_pair_from_dir(path_str, |path| models::MambaModel::load(&path))
                .map(LoadedModel::Mamba)?,
        ),
        ModelType::Mamba2 => Some(
            load_pair_from_dir(path_str, |path| models::Mamba2Model::load(&path))
                .map(LoadedModel::Mamba2)?,
        ),
        ModelType::Jamba => Some(
            load_pair_from_dir(path_str, |path| models::JambaModel::load(&path))
                .map(LoadedModel::Jamba)?,
        ),
        ModelType::NemotronH => Some(
            load_pair_from_dir(path_str, |path| models::NemotronHModel::load(&path))
                .map(LoadedModel::NemotronH)?,
        ),
        ModelType::NemotronNAS => Some(
            load_pair_from_dir(path_str, |path| models::NemotronNASModel::load(&path))
                .map(LoadedModel::NemotronNAS)?,
        ),
        ModelType::KimiLinear => Some(
            load_pair_from_dir(path_str, models::KimiLinearModel::load)
                .map(LoadedModel::KimiLinear)?,
        ),
        ModelType::LongcatFlash => Some(
            load_pair_from_dir(path_str, |path| models::LongcatFlashNgramModel::load(&path))
                .map(LoadedModel::LongcatFlash)?,
        ),
        ModelType::LongcatFlashNgram => Some(
            load_pair_from_dir(path_str, |path| models::LongcatFlashNgramModel::load(&path))
                .map(LoadedModel::LongcatFlashNgram)?,
        ),
        ModelType::Rwkv7 => {
            Some(load_from_path(model_path, models::Rwkv7::load).map(LoadedModel::Rwkv7)?)
        }
        ModelType::RecurrentGemma => Some(
            load_pair_from_dir(path_str, |path| models::GriffinModel::load(&path))
                .map(LoadedModel::RecurrentGemma)?,
        ),
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

fn load_from_dir<T, E, F>(path_str: &str, load: F) -> Result<T>
where
    F: FnOnce(String) -> std::result::Result<T, E>,
    E: Display,
{
    load(path_str.to_owned()).map_err(|e| anyhow::anyhow!("{}", e))
}

fn load_from_path<T, E, F>(path: &Path, load: F) -> Result<T>
where
    F: FnOnce(&Path) -> std::result::Result<T, E>,
    E: Display,
{
    load(path).map_err(|e| anyhow::anyhow!("{}", e))
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

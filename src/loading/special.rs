//! Special-case weight loading strategies.
//!
//! These families diverge from the standard config-backed path because they
//! need owned weights, extra sanitization, or architecture-specific config
//! reshaping before model construction.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;

use crate::LoadedModel;
use crate::models::{self, ModelType};

fn copy_weight_map(weights: &WeightMap) -> WeightMap {
    weights
        .iter()
        .map(|(key, value)| (key.clone(), mlxcel_core::copy(value)))
        .collect()
}

macro_rules! load_owned_model_from_config {
    ($config_str:expr, $weights:expr, $args_ty:ty, $builder:path, $wrap:expr) => {{
        let args: $args_ty = super::parse_model_config($config_str)?;
        let owned = copy_weight_map($weights);
        let model = $builder(args, owned).map_err(|err| anyhow::anyhow!("{}", err))?;
        ($wrap)(model)
    }};
}

pub(crate) fn qwen35_text_config(config: &serde_json::Value) -> Result<serde_json::Value> {
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

pub(crate) fn adapter_loading_unsupported_message(model_type: ModelType) -> Option<&'static str> {
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

pub(crate) fn is_special_weight_model_type(model_type: ModelType) -> bool {
    special_weight_loader_kind(model_type).is_some()
}

pub(crate) fn try_load_special_model_from_weights(
    model_type: ModelType,
    config_str: &str,
    weights: &mut WeightMap,
) -> Result<Option<LoadedModel>> {
    let Some(kind) = special_weight_loader_kind(model_type) else {
        return Ok(None);
    };

    Ok(Some(match kind {
        SpecialWeightLoaderKind::Qwen35 => {
            let value: serde_json::Value = super::parse_model_config(config_str)?;
            let text_config = qwen35_text_config(&value)?;
            let args: models::qwen3_5::Qwen35Config = serde_json::from_value(text_config)
                .map_err(|err| anyhow::anyhow!("Failed to parse config: {}", err))?;
            let owned = copy_weight_map(weights);
            let owned = models::qwen3_5::sanitize_moe_weights(owned, &args);
            let model = models::Qwen35Model::from_weights(&owned, &args)
                .map_err(|err| anyhow::anyhow!("{}", err))?;
            if model_type == ModelType::Qwen35Moe {
                LoadedModel::Qwen35Moe(model)
            } else {
                LoadedModel::Qwen35(model)
            }
        }
        SpecialWeightLoaderKind::Gemma3n => {
            let top_args: models::gemma3n::ModelArgs = super::parse_model_config(config_str)?;
            let config = top_args.text_args();
            let language_model = models::gemma3n::Gemma3nLanguageModel::from_weights(
                weights,
                &config,
                "language_model.model",
            )
            .map_err(|err| anyhow::anyhow!("{}", err))?;
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
            let args: models::nemotron_h::NemotronHConfig = super::parse_model_config(config_str)?;
            let block_types: Vec<models::nemotron_h::BlockType> = args
                .hybrid_override_pattern
                .iter()
                .map(|name| models::nemotron_h::BlockType::from_str(name))
                .collect();
            let owned = copy_weight_map(weights);
            let owned = models::NemotronHModel::sanitize_weights(owned, &args);
            let model = models::NemotronHModel::from_weights(args, owned, block_types)
                .map_err(|err| anyhow::anyhow!("{}", err))?;
            LoadedModel::NemotronH(model)
        }
        SpecialWeightLoaderKind::KimiLinear => {
            let args: models::kimi_linear::KimiLinearConfig =
                super::parse_model_config(config_str)?;
            let mut owned = copy_weight_map(weights);
            owned = models::KimiLinearModel::sanitize_weights(owned, &args);
            let model = models::KimiLinearModel::from_weights(&owned, &args)
                .map_err(|err| anyhow::anyhow!("{}", err))?;
            LoadedModel::KimiLinear(model)
        }
        SpecialWeightLoaderKind::Longcat => {
            let args: models::longcat_flash_ngram::LongcatFlashNgramConfig =
                super::parse_model_config(config_str)?;
            let mut owned = copy_weight_map(weights);
            owned = models::longcat_flash_ngram::sanitize_weights(owned, &args);
            let model = models::LongcatFlashNgramModel::from_weights(&owned, &args)
                .map_err(|err| anyhow::anyhow!("{}", err))?;
            if model_type == ModelType::LongcatFlashNgram {
                LoadedModel::LongcatFlashNgram(model)
            } else {
                LoadedModel::LongcatFlash(model)
            }
        }
        SpecialWeightLoaderKind::Rwkv7 => {
            let args: models::rwkv7::Rwkv7Config = super::parse_model_config(config_str)?;
            let model = models::Rwkv7::from_weights(weights, args)
                .map_err(|err| anyhow::anyhow!("{}", err))?;
            LoadedModel::Rwkv7(model)
        }
    }))
}

#[cfg(test)]
#[path = "special_tests.rs"]
mod tests;

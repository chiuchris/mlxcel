use anyhow::Result;
use serde::de::DeserializeOwned;
use std::fmt::Display;
use std::path::{Path, PathBuf};

use crate::LoadedModel;
use crate::loader_vlm::*;
use crate::lora;
use crate::models::{self, ModelType, get_model_type, sanitize_config_json};
use crate::tokenizer::{self, MlxcelTokenizer};
use mlxcel_core::weights::WeightMap;

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

fn parse_model_config<T: DeserializeOwned>(config_str: &str) -> Result<T> {
    serde_json::from_str(config_str).map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))
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

    let model = match model_type {
        ModelType::Llama => {
            LoadedModel::Llama(load_pair_from_dir(path_str, models::Llama3Model::load)?)
        }
        ModelType::Mistral3 => {
            // Mistral3 is a VLM wrapper - check text_config for inner model type
            let config_path = model_path.join("config.json");
            let config_str = std::fs::read_to_string(&config_path)?;
            let config: serde_json::Value = serde_json::from_str(&config_str)?;

            // Check if text_config.model_type is "ministral3"
            let is_ministral3 = config
                .get("text_config")
                .and_then(|tc| tc.get("model_type"))
                .and_then(|mt| mt.as_str())
                .map(|mt| mt == "ministral3")
                .unwrap_or(false);

            if is_ministral3 {
                // Load as Ministral3 with text_config extracted
                LoadedModel::Ministral3(models::Ministral3Wrapper::new(load_pair_from_dir(
                    path_str,
                    models::Ministral3Model::load_from_text_config,
                )?))
            } else {
                // Load as standard Llama
                LoadedModel::Llama(load_pair_from_dir(path_str, models::Llama3Model::load)?)
            }
        }
        ModelType::Llama4 => LoadedModel::Llama4(models::Llama4Wrapper::new(load_pair_from_dir(
            path_str,
            models::Llama4CxxModel::load,
        )?)),
        ModelType::Llama4VLM => load_llama4_vlm(model_path)?,
        ModelType::Qwen2 => {
            LoadedModel::Qwen2(load_pair_from_dir(path_str, models::Qwen2Model::load)?)
        }
        ModelType::Qwen3 => {
            LoadedModel::Qwen3(load_pair_from_dir(path_str, models::Qwen3Model::load)?)
        }
        ModelType::Qwen3Moe => {
            LoadedModel::Qwen3Moe(load_pair_from_dir(path_str, models::Qwen3MoeModel::load)?)
        }
        ModelType::Qwen3Next => {
            LoadedModel::Qwen3Next(load_pair_from_dir(path_str, models::Qwen3NextModel::load)?)
        }
        ModelType::Qwen35 => {
            LoadedModel::Qwen35(load_pair_from_dir(path_str, models::Qwen35Model::load)?)
        }
        ModelType::Qwen35VLM => {
            return Ok((
                load_qwen3_5_vlm(model_path)?,
                tokenizer::load_tokenizer(model_path)?,
            ));
        }
        ModelType::Qwen35Moe => {
            LoadedModel::Qwen35Moe(load_pair_from_dir(path_str, models::Qwen35Model::load)?)
        }
        ModelType::Qwen35MoeVLM => {
            return Ok((
                load_qwen3_5_vlm(model_path)?,
                tokenizer::load_tokenizer(model_path)?,
            ));
        }
        ModelType::Qwen2Moe => {
            LoadedModel::Qwen2Moe(load_pair_from_dir(path_str, models::Qwen2MoeModel::load)?)
        }
        ModelType::Gemma => {
            LoadedModel::Gemma(load_pair_from_dir(path_str, models::GemmaModel::load)?)
        }
        ModelType::Gemma2 => {
            LoadedModel::Gemma2(load_pair_from_dir(path_str, models::Gemma2Model::load)?)
        }
        ModelType::Gemma3 => LoadedModel::Gemma3(models::Gemma3Wrapper::new(load_pair_from_dir(
            path_str,
            models::Gemma3Model::load,
        )?)),
        ModelType::Gemma3VLM => load_gemma3_vlm(model_path)?,
        ModelType::LlavaVLM => load_llava_vlm(model_path)?,
        ModelType::LlavaBunnyVLM => load_llava_bunny_vlm(model_path)?,
        ModelType::AyaVisionVLM => load_aya_vision_vlm(model_path)?,
        ModelType::PaliGemmaVLM => load_paligemma_vlm(model_path)?,
        ModelType::PixtralVLM => load_pixtral_vlm(model_path)?,
        ModelType::Mistral3VLM => load_mistral3_vlm(model_path)?,
        ModelType::Qwen2VL => load_qwen2_vl(model_path)?,
        ModelType::Qwen25VL => load_qwen2_5_vl(model_path)?,
        ModelType::Qwen3VL => load_qwen3_vl(model_path)?,
        ModelType::Qwen3VLMoe => load_qwen3_vl_moe(model_path)?,
        ModelType::Gemma3n => {
            LoadedModel::Gemma3n(load_from_dir(path_str, models::Gemma3nModel::load)?)
        }
        ModelType::Gemma3nVLM => load_gemma3n_vlm(model_path)?,
        ModelType::Phi => LoadedModel::Phi(load_pair_from_dir(path_str, models::PhiModel::load)?),
        ModelType::Phi3 => {
            LoadedModel::Phi3(load_pair_from_dir(path_str, models::Phi3Model::load)?)
        }
        ModelType::Phi3VLM => {
            return Ok((
                load_phi3_vlm(model_path)?,
                tokenizer::load_tokenizer(model_path)?,
            ));
        }
        ModelType::Molmo2VLM => {
            return Ok((
                load_molmo2_vlm(model_path)?,
                tokenizer::load_tokenizer(model_path)?,
            ));
        }
        ModelType::Phi3Small => {
            LoadedModel::Phi3Small(load_pair_from_dir(path_str, models::Phi3SmallModel::load)?)
        }
        ModelType::PhiMoe => {
            LoadedModel::PhiMoe(load_pair_from_dir(path_str, models::PhiMoeModel::load)?)
        }
        ModelType::Mixtral => {
            LoadedModel::Mixtral(load_pair_from_dir(path_str, models::MixtralModel::load)?)
        }
        ModelType::OLMoE => {
            LoadedModel::OLMoE(load_pair_from_dir(path_str, models::OlmoeModel::load)?)
        }
        ModelType::DeepSeek => {
            LoadedModel::DeepSeek(load_pair_from_dir(path_str, models::DeepSeekModel::load)?)
        }
        ModelType::DeepSeekV2 => {
            LoadedModel::DeepSeekV2(load_pair_from_dir(path_str, models::DeepSeekV2Model::load)?)
        }
        ModelType::DeepSeekV3 => {
            LoadedModel::DeepSeekV3(load_pair_from_dir(path_str, models::DeepSeekV3Model::load)?)
        }
        ModelType::DeepSeekV32 => LoadedModel::DeepSeekV32(load_pair_from_dir(
            path_str,
            models::DeepSeekV32Model::load,
        )?),
        ModelType::Cohere => {
            LoadedModel::Cohere(load_pair_from_dir(path_str, models::CohereModel::load)?)
        }
        ModelType::Cohere2 => {
            LoadedModel::Cohere2(load_pair_from_dir(path_str, models::Cohere2Model::load)?)
        }
        ModelType::InternLM2 => {
            LoadedModel::InternLM2(load_pair_from_dir(path_str, models::InternLM2Model::load)?)
        }
        ModelType::InternLM3 => {
            LoadedModel::InternLM3(load_pair_from_dir(path_str, models::InternLM3Model::load)?)
        }
        ModelType::Baichuan => {
            LoadedModel::Baichuan(load_pair_from_dir(path_str, models::BaichuanModel::load)?)
        }
        ModelType::Glm4 => {
            LoadedModel::Glm4(load_pair_from_dir(path_str, models::Glm4Model::load)?)
        }
        ModelType::Glm4Moe => {
            LoadedModel::Glm4Moe(load_pair_from_dir(path_str, models::Glm4MoeModel::load)?)
        }
        ModelType::Glm4MoeLite => LoadedModel::Glm4MoeLite(load_pair_from_dir(
            path_str,
            models::Glm4MoeLiteModel::load,
        )?),
        ModelType::GlmMoeDsa => {
            LoadedModel::GlmMoeDsa(load_pair_from_dir(path_str, models::GlmMoeDsaModel::load)?)
        }
        ModelType::Ernie45 => {
            LoadedModel::Ernie45(load_pair_from_dir(path_str, models::Ernie45Model::load)?)
        }
        ModelType::Ernie45Moe => {
            LoadedModel::Ernie45Moe(load_pair_from_dir(path_str, models::Ernie45MoeModel::load)?)
        }
        ModelType::HunyuanMoe => {
            LoadedModel::HunyuanMoe(load_pair_from_dir(path_str, models::HunyuanMoeModel::load)?)
        }
        ModelType::HunyuanV1Dense => LoadedModel::HunyuanV1Dense(load_pair_from_dir(
            path_str,
            models::HunyuanV1DenseModel::load,
        )?),
        ModelType::MiMo => {
            LoadedModel::MiMo(load_pair_from_dir(path_str, models::MiMoModel::load)?)
        }
        ModelType::ExaOne => {
            LoadedModel::ExaOne(load_pair_from_dir(path_str, models::ExaOneModel::load)?)
        }
        ModelType::ExaOne4 => LoadedModel::ExaOne4(models::ExaOne4Wrapper::new(
            load_pair_from_dir(path_str, models::ExaOne4Model::load)?,
        )),
        ModelType::ExaOneMoe => {
            LoadedModel::ExaOneMoe(load_pair_from_dir(path_str, models::ExaoneMoeModel::load)?)
        }
        ModelType::Olmo => {
            LoadedModel::Olmo(load_pair_from_dir(path_str, models::OlmoModel::load)?)
        }
        ModelType::Olmo2 => {
            LoadedModel::Olmo2(load_pair_from_dir(path_str, models::OLMo2Model::load)?)
        }
        ModelType::Olmo3 => {
            LoadedModel::Olmo3(load_pair_from_dir(path_str, models::OLMo3Model::load)?)
        }
        ModelType::StarCoder2 => {
            LoadedModel::StarCoder2(load_pair_from_dir(path_str, models::StarCoder2Model::load)?)
        }
        ModelType::MiniCPM => {
            LoadedModel::MiniCPM(load_pair_from_dir(path_str, models::MiniCPMModel::load)?)
        }
        ModelType::MiniCPM3 => {
            LoadedModel::MiniCPM3(load_pair_from_dir(path_str, models::MiniCPM3Model::load)?)
        }
        ModelType::StableLM => {
            LoadedModel::StableLM(load_pair_from_dir(path_str, models::StableLMModel::load)?)
        }
        ModelType::SmolLM3 => {
            LoadedModel::SmolLM3(load_pair_from_dir(path_str, models::SmolLM3Model::load)?)
        }
        ModelType::Ministral3 => LoadedModel::Ministral3(models::Ministral3Wrapper::new(
            load_pair_from_dir(path_str, models::Ministral3Model::load)?,
        )),
        ModelType::Nemotron => {
            LoadedModel::Nemotron(load_pair_from_dir(path_str, models::NemotronModel::load)?)
        }
        ModelType::Mamba => LoadedModel::Mamba(load_pair_from_dir(path_str, |path| {
            models::MambaModel::load(&path)
        })?),
        ModelType::Mamba2 => LoadedModel::Mamba2(load_pair_from_dir(path_str, |path| {
            models::Mamba2Model::load(&path)
        })?),
        ModelType::Jamba => LoadedModel::Jamba(load_pair_from_dir(path_str, |path| {
            models::JambaModel::load(&path)
        })?),
        ModelType::NemotronH => LoadedModel::NemotronH(load_pair_from_dir(path_str, |path| {
            models::NemotronHModel::load(&path)
        })?),
        ModelType::NemotronNAS => LoadedModel::NemotronNAS(load_pair_from_dir(path_str, |path| {
            models::NemotronNASModel::load(&path)
        })?),
        ModelType::Step3p5 => {
            LoadedModel::Step3p5(load_pair_from_dir(path_str, models::Step3p5Model::load)?)
        }
        ModelType::KimiLinear => {
            LoadedModel::KimiLinear(load_pair_from_dir(path_str, models::KimiLinearModel::load)?)
        }
        ModelType::LongcatFlash => {
            LoadedModel::LongcatFlash(load_pair_from_dir(path_str, |path| {
                models::LongcatFlashNgramModel::load(&path)
            })?)
        }
        ModelType::LongcatFlashNgram => {
            LoadedModel::LongcatFlashNgram(load_pair_from_dir(path_str, |path| {
                models::LongcatFlashNgramModel::load(&path)
            })?)
        }
        ModelType::Rwkv7 => LoadedModel::Rwkv7(load_from_path(model_path, models::Rwkv7::load)?),
        ModelType::RecurrentGemma => {
            LoadedModel::RecurrentGemma(load_pair_from_dir(path_str, |path| {
                models::GriffinModel::load(&path)
            })?)
        }
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

    let model = match model_type {
        ModelType::Llama | ModelType::Mistral3 => {
            // Check for ministral3 sub-type
            let config: serde_json::Value = parse_model_config(&config_str)?;
            let is_ministral3 = config
                .get("text_config")
                .and_then(|tc| tc.get("model_type"))
                .and_then(|mt| mt.as_str())
                .map(|mt| mt == "ministral3")
                .unwrap_or(false);

            if model_type == ModelType::Mistral3 && is_ministral3 {
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
        }
        ModelType::Llama4 => {
            let args: models::llama4::TextArgs = parse_model_config(&config_str)?;
            let m = models::Llama4CxxModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Llama4(models::Llama4Wrapper::new(m))
        }
        ModelType::Llama4VLM => {
            return Err(anyhow::anyhow!(
                "Llama4 VLM cannot be loaded with LoRA adapters yet"
            ));
        }
        ModelType::Qwen2 => {
            let args: models::llama3::ModelArgs = parse_model_config(&config_str)?;
            let m = models::Qwen2Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Qwen2(m)
        }
        ModelType::Qwen3 => {
            let args: models::qwen3::ModelArgs = parse_model_config(&config_str)?;
            let m = models::Qwen3Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Qwen3(m)
        }
        ModelType::Qwen3Moe => {
            let args: models::qwen3_moe::ModelArgs = parse_model_config(&config_str)?;
            let m = models::Qwen3MoeModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Qwen3Moe(m)
        }
        ModelType::Qwen3Next => {
            let args: models::qwen3_next::Qwen3NextConfig = parse_model_config(&config_str)?;
            let m = models::Qwen3NextModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Qwen3Next(m)
        }
        ModelType::Qwen35VLM | ModelType::Qwen35MoeVLM => {
            return Err(anyhow::anyhow!(
                "Qwen3.5 VLM does not support adapter loading"
            ));
        }
        ModelType::Qwen35 | ModelType::Qwen35Moe => {
            let v: serde_json::Value = serde_json::from_str(&config_str)?;
            let mut text_config = if let Some(tc) = v.get("text_config") {
                tc.clone()
            } else {
                v.clone()
            };
            // Merge quantization from top level if not in text_config
            if text_config.get("quantization").is_none() && v.get("quantization").is_some() {
                text_config
                    .as_object_mut()
                    .unwrap()
                    .insert("quantization".to_string(), v["quantization"].clone());
            }
            let args: models::qwen3_5::Qwen35Config = serde_json::from_value(text_config)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let owned: WeightMap = weights
                .iter()
                .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
                .collect();
            let owned = models::qwen3_5::sanitize_moe_weights(owned, &args);
            let m = models::Qwen35Model::from_weights(&owned, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            if model_type == ModelType::Qwen35Moe {
                LoadedModel::Qwen35Moe(m)
            } else {
                LoadedModel::Qwen35(m)
            }
        }
        ModelType::Qwen2Moe => {
            let args: models::qwen2_moe::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Qwen2MoeModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Qwen2Moe(m)
        }
        ModelType::Gemma => {
            let args: models::gemma::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::GemmaModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Gemma(m)
        }
        ModelType::Gemma2 => {
            let args: models::gemma2::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Gemma2Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Gemma2(m)
        }
        ModelType::Gemma3 => {
            let args: models::gemma3::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Gemma3Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Gemma3(models::Gemma3Wrapper::new(m))
        }
        ModelType::Gemma3VLM => {
            return Err(anyhow::anyhow!(
                "Gemma3 VLM cannot be loaded with LoRA adapters yet"
            ));
        }
        ModelType::LlavaVLM | ModelType::LlavaBunnyVLM => {
            return Err(anyhow::anyhow!(
                "LLaVA VLM cannot be loaded with LoRA adapters yet"
            ));
        }
        ModelType::AyaVisionVLM => {
            return Err(anyhow::anyhow!(
                "Aya Vision VLM cannot be loaded with LoRA adapters yet"
            ));
        }
        ModelType::PaliGemmaVLM => {
            return Err(anyhow::anyhow!(
                "PaliGemma VLM cannot be loaded with LoRA adapters yet"
            ));
        }
        ModelType::PixtralVLM | ModelType::Mistral3VLM => {
            return Err(anyhow::anyhow!(
                "Pixtral/Mistral3 VLM cannot be loaded with LoRA adapters yet"
            ));
        }
        ModelType::Qwen2VL => {
            return Err(anyhow::anyhow!(
                "Qwen2-VL cannot be loaded with LoRA adapters yet"
            ));
        }
        ModelType::Qwen25VL | ModelType::Qwen3VL | ModelType::Qwen3VLMoe => {
            return Err(anyhow::anyhow!(
                "Qwen VL models cannot be loaded with LoRA adapters yet"
            ));
        }
        ModelType::Gemma3n => {
            let top_args: models::gemma3n::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
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
        ModelType::Gemma3nVLM => {
            return Err(anyhow::anyhow!(
                "Gemma3n VLM cannot be loaded with LoRA adapters yet"
            ));
        }
        ModelType::Phi => {
            let args: models::phi::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::PhiModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Phi(m)
        }
        ModelType::Phi3 => {
            let args: models::phi3::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Phi3Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Phi3(m)
        }
        ModelType::Phi3VLM => {
            return Err(anyhow::anyhow!(
                "Phi3V VLM does not support adapter loading; use load_model() instead"
            ));
        }
        ModelType::Molmo2VLM => {
            return Err(anyhow::anyhow!(
                "Molmo2 VLM does not support adapter loading; use load_model() instead"
            ));
        }
        ModelType::Phi3Small => {
            let args: models::phi3small::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Phi3SmallModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Phi3Small(m)
        }
        ModelType::PhiMoe => {
            let args: models::phimoe::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::PhiMoeModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::PhiMoe(m)
        }
        ModelType::Mixtral => {
            let args: models::mixtral::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::MixtralModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Mixtral(m)
        }
        ModelType::OLMoE => {
            let args: models::olmoe::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::OlmoeModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::OLMoE(m)
        }
        ModelType::DeepSeek => {
            let args: models::deepseek::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::DeepSeekModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::DeepSeek(m)
        }
        ModelType::DeepSeekV2 => {
            let args: models::deepseek_v2::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::DeepSeekV2Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::DeepSeekV2(m)
        }
        ModelType::DeepSeekV3 => {
            let args: models::deepseek_v3::DeepSeekV3Config = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::DeepSeekV3Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::DeepSeekV3(m)
        }
        ModelType::DeepSeekV32 => {
            let args: models::deepseek_v32::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::DeepSeekV32Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::DeepSeekV32(m)
        }
        ModelType::Cohere => {
            let args: models::cohere::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::CohereModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Cohere(m)
        }
        ModelType::Cohere2 => {
            let args: models::cohere2::Cohere2Config = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Cohere2Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Cohere2(m)
        }
        ModelType::InternLM2 => {
            let args: models::internlm2::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::InternLM2Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::InternLM2(m)
        }
        ModelType::InternLM3 => {
            let args: models::internlm3::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::InternLM3Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::InternLM3(m)
        }
        ModelType::Baichuan => {
            let args: models::baichuan::BaichuanConfig = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::BaichuanModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Baichuan(m)
        }
        ModelType::Glm4 => {
            let args: models::glm4::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Glm4Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Glm4(m)
        }
        ModelType::Glm4Moe => {
            let args: models::glm4_moe::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Glm4MoeModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Glm4Moe(m)
        }
        ModelType::Glm4MoeLite => {
            let args: models::glm4_moe_lite::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Glm4MoeLiteModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Glm4MoeLite(m)
        }
        ModelType::GlmMoeDsa => {
            let args: models::glm_moe_dsa::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::GlmMoeDsaModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::GlmMoeDsa(m)
        }
        ModelType::Ernie45 => {
            let args: models::ernie4_5::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Ernie45Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Ernie45(m)
        }
        ModelType::Ernie45Moe => {
            let args: models::ernie4_5_moe::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Ernie45MoeModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Ernie45Moe(m)
        }
        ModelType::HunyuanMoe => {
            let args: models::hunyuan_moe::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::HunyuanMoeModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::HunyuanMoe(m)
        }
        ModelType::HunyuanV1Dense => {
            let args: models::hunyuan_v1_dense::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::HunyuanV1DenseModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::HunyuanV1Dense(m)
        }
        ModelType::MiMo => {
            let args: models::mimo::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::MiMoModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::MiMo(m)
        }
        ModelType::ExaOne => {
            let args: models::exaone::ExaOneConfig = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::ExaOneModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::ExaOne(m)
        }
        ModelType::ExaOne4 => {
            let args: models::exaone4::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::ExaOne4Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::ExaOne4(models::ExaOne4Wrapper::new(m))
        }
        ModelType::ExaOneMoe => {
            let args: models::exaone_moe::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::ExaoneMoeModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::ExaOneMoe(m)
        }
        ModelType::Olmo => {
            let args: models::olmo::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::OlmoModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Olmo(m)
        }
        ModelType::Olmo2 => {
            let args: models::olmo2::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::OLMo2Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Olmo2(m)
        }
        ModelType::Olmo3 => {
            let args: models::olmo3::OLMo3Config = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::OLMo3Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Olmo3(m)
        }
        ModelType::StarCoder2 => {
            let args: models::starcoder2::StarCoder2Config = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::StarCoder2Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::StarCoder2(m)
        }
        ModelType::MiniCPM => {
            let args: models::minicpm::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::MiniCPMModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::MiniCPM(m)
        }
        ModelType::MiniCPM3 => {
            let args: models::minicpm3::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::MiniCPM3Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::MiniCPM3(m)
        }
        ModelType::StableLM => {
            let args: models::stablelm::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::StableLMModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::StableLM(m)
        }
        ModelType::SmolLM3 => {
            let args: models::smollm3::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::SmolLM3Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::SmolLM3(m)
        }
        ModelType::Ministral3 => {
            let args: models::ministral3::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Ministral3Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Ministral3(models::Ministral3Wrapper::new(m))
        }
        ModelType::Nemotron => {
            let args: models::nemotron::ModelArgs = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::NemotronModel::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Nemotron(m)
        }
        // SSM/Hybrid models that take ownership of weights
        ModelType::Mamba => {
            let args: models::mamba::MambaConfig = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let owned: WeightMap = weights
                .iter()
                .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
                .collect();
            let m = models::MambaModel::from_weights(args, owned)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Mamba(m)
        }
        ModelType::Mamba2 => {
            let args: models::mamba2::Mamba2Config = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let owned: WeightMap = weights
                .iter()
                .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
                .collect();
            let m = models::Mamba2Model::from_weights(args, owned)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Mamba2(m)
        }
        ModelType::Jamba => {
            let args: models::jamba::JambaConfig = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let owned: WeightMap = weights
                .iter()
                .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
                .collect();
            let m = models::JambaModel::from_weights(args, owned)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Jamba(m)
        }
        ModelType::NemotronH => {
            let args: models::nemotron_h::NemotronHConfig = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let block_types: Vec<models::nemotron_h::BlockType> = args
                .hybrid_override_pattern
                .iter()
                .map(|s| models::nemotron_h::BlockType::from_str(s))
                .collect();
            let owned: WeightMap = weights
                .iter()
                .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
                .collect();
            let owned = models::NemotronHModel::sanitize_weights(owned, &args);
            let m = models::NemotronHModel::from_weights(args, owned, block_types)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::NemotronH(m)
        }
        ModelType::NemotronNAS => {
            let args: models::nemotron_nas::NemotronNASConfig =
                serde_json::from_str(&config_str)
                    .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let owned: WeightMap = weights
                .iter()
                .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
                .collect();
            let m = models::NemotronNASModel::from_weights(args, owned)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::NemotronNAS(m)
        }
        ModelType::Step3p5 => {
            let args: models::step3p5::Step3p5Config = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m = models::Step3p5Model::from_weights(weights, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Step3p5(m)
        }
        ModelType::KimiLinear => {
            let args: models::kimi_linear::KimiLinearConfig = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let mut owned: WeightMap = weights
                .iter()
                .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
                .collect();
            owned = models::KimiLinearModel::sanitize_weights(owned, &args);
            let m = models::KimiLinearModel::from_weights(&owned, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::KimiLinear(m)
        }
        ModelType::LongcatFlash | ModelType::LongcatFlashNgram => {
            let args: models::longcat_flash_ngram::LongcatFlashNgramConfig =
                serde_json::from_str(&config_str)
                    .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let mut owned: WeightMap = weights
                .iter()
                .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
                .collect();
            owned = models::longcat_flash_ngram::sanitize_weights(owned, &args);
            let m = models::LongcatFlashNgramModel::from_weights(&owned, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            if model_type == ModelType::LongcatFlashNgram {
                LoadedModel::LongcatFlashNgram(m)
            } else {
                LoadedModel::LongcatFlash(m)
            }
        }
        ModelType::Rwkv7 => {
            let args: models::rwkv7::Rwkv7Config = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let m =
                models::Rwkv7::from_weights(weights, args).map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Rwkv7(m)
        }
        ModelType::RecurrentGemma => {
            let args: models::recurrent_gemma::GriffinConfig = serde_json::from_str(&config_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
            let owned: WeightMap = weights
                .iter()
                .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
                .collect();
            let m = models::GriffinModel::from_weights(args, owned)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::RecurrentGemma(m)
        }
    };

    Ok(model)
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;

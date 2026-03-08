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

fn copy_weight_map(weights: &WeightMap) -> WeightMap {
    weights
        .iter()
        .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
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

macro_rules! load_model_from_config {
    ($config_str:expr, $weights:expr, $args_ty:ty, $builder:path, $wrap:expr) => {{
        let args: $args_ty = parse_model_config($config_str)?;
        let model = $builder($weights, &args).map_err(|e| anyhow::anyhow!("{}", e))?;
        ($wrap)(model)
    }};
}

macro_rules! load_owned_model_from_config {
    ($config_str:expr, $weights:expr, $args_ty:ty, $builder:path, $wrap:expr) => {{
        let args: $args_ty = parse_model_config($config_str)?;
        let owned = copy_weight_map($weights);
        let model = $builder(args, owned).map_err(|e| anyhow::anyhow!("{}", e))?;
        ($wrap)(model)
    }};
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
        ModelType::Qwen35VLM | ModelType::Qwen35MoeVLM => {
            let loaded = match qwen35_vlm_kind(model_type).unwrap() {
                Qwen35VlmKind::Dense => load_qwen3_5_vlm(model_path)?,
                Qwen35VlmKind::Moe => load_qwen3_5_moe_vlm(model_path)?,
            };
            return Ok((loaded, tokenizer::load_tokenizer(model_path)?));
        }
        ModelType::Qwen35Moe => {
            LoadedModel::Qwen35Moe(load_pair_from_dir(path_str, models::Qwen35Model::load)?)
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
        }
        ModelType::Llama4 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::llama4::TextArgs,
                models::Llama4CxxModel::from_weights,
                |m| LoadedModel::Llama4(models::Llama4Wrapper::new(m))
            )
        }
        ModelType::Llama4VLM => {
            return Err(anyhow::anyhow!(
                "Llama4 VLM cannot be loaded with LoRA adapters yet"
            ));
        }
        ModelType::Qwen2 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::llama3::ModelArgs,
                models::Qwen2Model::from_weights,
                LoadedModel::Qwen2
            )
        }
        ModelType::Qwen3 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::qwen3::ModelArgs,
                models::Qwen3Model::from_weights,
                LoadedModel::Qwen3
            )
        }
        ModelType::Qwen3Moe => {
            load_model_from_config!(
                &config_str,
                weights,
                models::qwen3_moe::ModelArgs,
                models::Qwen3MoeModel::from_weights,
                LoadedModel::Qwen3Moe
            )
        }
        ModelType::Qwen3Next => {
            load_model_from_config!(
                &config_str,
                weights,
                models::qwen3_next::Qwen3NextConfig,
                models::Qwen3NextModel::from_weights,
                LoadedModel::Qwen3Next
            )
        }
        ModelType::Qwen35VLM | ModelType::Qwen35MoeVLM => {
            return Err(anyhow::anyhow!(
                "Qwen3.5 VLM does not support adapter loading"
            ));
        }
        ModelType::Qwen35 | ModelType::Qwen35Moe => {
            let v: serde_json::Value = parse_model_config(&config_str)?;
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
        ModelType::Qwen2Moe => {
            load_model_from_config!(
                &config_str,
                weights,
                models::qwen2_moe::ModelArgs,
                models::Qwen2MoeModel::from_weights,
                LoadedModel::Qwen2Moe
            )
        }
        ModelType::Gemma => {
            load_model_from_config!(
                &config_str,
                weights,
                models::gemma::ModelArgs,
                models::GemmaModel::from_weights,
                LoadedModel::Gemma
            )
        }
        ModelType::Gemma2 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::gemma2::ModelArgs,
                models::Gemma2Model::from_weights,
                LoadedModel::Gemma2
            )
        }
        ModelType::Gemma3 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::gemma3::ModelArgs,
                models::Gemma3Model::from_weights,
                |m| LoadedModel::Gemma3(models::Gemma3Wrapper::new(m))
            )
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
            let top_args: models::gemma3n::ModelArgs = parse_model_config(&config_str)?;
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
            load_model_from_config!(
                &config_str,
                weights,
                models::phi::ModelArgs,
                models::PhiModel::from_weights,
                LoadedModel::Phi
            )
        }
        ModelType::Phi3 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::phi3::ModelArgs,
                models::Phi3Model::from_weights,
                LoadedModel::Phi3
            )
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
            load_model_from_config!(
                &config_str,
                weights,
                models::phi3small::ModelArgs,
                models::Phi3SmallModel::from_weights,
                LoadedModel::Phi3Small
            )
        }
        ModelType::PhiMoe => {
            load_model_from_config!(
                &config_str,
                weights,
                models::phimoe::ModelArgs,
                models::PhiMoeModel::from_weights,
                LoadedModel::PhiMoe
            )
        }
        ModelType::Mixtral => {
            load_model_from_config!(
                &config_str,
                weights,
                models::mixtral::ModelArgs,
                models::MixtralModel::from_weights,
                LoadedModel::Mixtral
            )
        }
        ModelType::OLMoE => {
            load_model_from_config!(
                &config_str,
                weights,
                models::olmoe::ModelArgs,
                models::OlmoeModel::from_weights,
                LoadedModel::OLMoE
            )
        }
        ModelType::DeepSeek => {
            load_model_from_config!(
                &config_str,
                weights,
                models::deepseek::ModelArgs,
                models::DeepSeekModel::from_weights,
                LoadedModel::DeepSeek
            )
        }
        ModelType::DeepSeekV2 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::deepseek_v2::ModelArgs,
                models::DeepSeekV2Model::from_weights,
                LoadedModel::DeepSeekV2
            )
        }
        ModelType::DeepSeekV3 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::deepseek_v3::DeepSeekV3Config,
                models::DeepSeekV3Model::from_weights,
                LoadedModel::DeepSeekV3
            )
        }
        ModelType::DeepSeekV32 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::deepseek_v32::ModelArgs,
                models::DeepSeekV32Model::from_weights,
                LoadedModel::DeepSeekV32
            )
        }
        ModelType::Cohere => {
            load_model_from_config!(
                &config_str,
                weights,
                models::cohere::ModelArgs,
                models::CohereModel::from_weights,
                LoadedModel::Cohere
            )
        }
        ModelType::Cohere2 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::cohere2::Cohere2Config,
                models::Cohere2Model::from_weights,
                LoadedModel::Cohere2
            )
        }
        ModelType::InternLM2 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::internlm2::ModelArgs,
                models::InternLM2Model::from_weights,
                LoadedModel::InternLM2
            )
        }
        ModelType::InternLM3 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::internlm3::ModelArgs,
                models::InternLM3Model::from_weights,
                LoadedModel::InternLM3
            )
        }
        ModelType::Baichuan => {
            load_model_from_config!(
                &config_str,
                weights,
                models::baichuan::BaichuanConfig,
                models::BaichuanModel::from_weights,
                LoadedModel::Baichuan
            )
        }
        ModelType::Glm4 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::glm4::ModelArgs,
                models::Glm4Model::from_weights,
                LoadedModel::Glm4
            )
        }
        ModelType::Glm4Moe => {
            load_model_from_config!(
                &config_str,
                weights,
                models::glm4_moe::ModelArgs,
                models::Glm4MoeModel::from_weights,
                LoadedModel::Glm4Moe
            )
        }
        ModelType::Glm4MoeLite => {
            load_model_from_config!(
                &config_str,
                weights,
                models::glm4_moe_lite::ModelArgs,
                models::Glm4MoeLiteModel::from_weights,
                LoadedModel::Glm4MoeLite
            )
        }
        ModelType::GlmMoeDsa => {
            load_model_from_config!(
                &config_str,
                weights,
                models::glm_moe_dsa::ModelArgs,
                models::GlmMoeDsaModel::from_weights,
                LoadedModel::GlmMoeDsa
            )
        }
        ModelType::Ernie45 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::ernie4_5::ModelArgs,
                models::Ernie45Model::from_weights,
                LoadedModel::Ernie45
            )
        }
        ModelType::Ernie45Moe => {
            load_model_from_config!(
                &config_str,
                weights,
                models::ernie4_5_moe::ModelArgs,
                models::Ernie45MoeModel::from_weights,
                LoadedModel::Ernie45Moe
            )
        }
        ModelType::HunyuanMoe => {
            load_model_from_config!(
                &config_str,
                weights,
                models::hunyuan_moe::ModelArgs,
                models::HunyuanMoeModel::from_weights,
                LoadedModel::HunyuanMoe
            )
        }
        ModelType::HunyuanV1Dense => {
            load_model_from_config!(
                &config_str,
                weights,
                models::hunyuan_v1_dense::ModelArgs,
                models::HunyuanV1DenseModel::from_weights,
                LoadedModel::HunyuanV1Dense
            )
        }
        ModelType::MiMo => {
            load_model_from_config!(
                &config_str,
                weights,
                models::mimo::ModelArgs,
                models::MiMoModel::from_weights,
                LoadedModel::MiMo
            )
        }
        ModelType::ExaOne => {
            load_model_from_config!(
                &config_str,
                weights,
                models::exaone::ExaOneConfig,
                models::ExaOneModel::from_weights,
                LoadedModel::ExaOne
            )
        }
        ModelType::ExaOne4 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::exaone4::ModelArgs,
                models::ExaOne4Model::from_weights,
                |m| LoadedModel::ExaOne4(models::ExaOne4Wrapper::new(m))
            )
        }
        ModelType::ExaOneMoe => {
            load_model_from_config!(
                &config_str,
                weights,
                models::exaone_moe::ModelArgs,
                models::ExaoneMoeModel::from_weights,
                LoadedModel::ExaOneMoe
            )
        }
        ModelType::Olmo => {
            load_model_from_config!(
                &config_str,
                weights,
                models::olmo::ModelArgs,
                models::OlmoModel::from_weights,
                LoadedModel::Olmo
            )
        }
        ModelType::Olmo2 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::olmo2::ModelArgs,
                models::OLMo2Model::from_weights,
                LoadedModel::Olmo2
            )
        }
        ModelType::Olmo3 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::olmo3::OLMo3Config,
                models::OLMo3Model::from_weights,
                LoadedModel::Olmo3
            )
        }
        ModelType::StarCoder2 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::starcoder2::StarCoder2Config,
                models::StarCoder2Model::from_weights,
                LoadedModel::StarCoder2
            )
        }
        ModelType::MiniCPM => {
            load_model_from_config!(
                &config_str,
                weights,
                models::minicpm::ModelArgs,
                models::MiniCPMModel::from_weights,
                LoadedModel::MiniCPM
            )
        }
        ModelType::MiniCPM3 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::minicpm3::ModelArgs,
                models::MiniCPM3Model::from_weights,
                LoadedModel::MiniCPM3
            )
        }
        ModelType::StableLM => {
            load_model_from_config!(
                &config_str,
                weights,
                models::stablelm::ModelArgs,
                models::StableLMModel::from_weights,
                LoadedModel::StableLM
            )
        }
        ModelType::SmolLM3 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::smollm3::ModelArgs,
                models::SmolLM3Model::from_weights,
                LoadedModel::SmolLM3
            )
        }
        ModelType::Ministral3 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::ministral3::ModelArgs,
                models::Ministral3Model::from_weights,
                |m| LoadedModel::Ministral3(models::Ministral3Wrapper::new(m))
            )
        }
        ModelType::Nemotron => {
            load_model_from_config!(
                &config_str,
                weights,
                models::nemotron::ModelArgs,
                models::NemotronModel::from_weights,
                LoadedModel::Nemotron
            )
        }
        // SSM/Hybrid models that take ownership of weights
        ModelType::Mamba => {
            load_owned_model_from_config!(
                &config_str,
                weights,
                models::mamba::MambaConfig,
                models::MambaModel::from_weights,
                LoadedModel::Mamba
            )
        }
        ModelType::Mamba2 => {
            load_owned_model_from_config!(
                &config_str,
                weights,
                models::mamba2::Mamba2Config,
                models::Mamba2Model::from_weights,
                LoadedModel::Mamba2
            )
        }
        ModelType::Jamba => {
            load_owned_model_from_config!(
                &config_str,
                weights,
                models::jamba::JambaConfig,
                models::JambaModel::from_weights,
                LoadedModel::Jamba
            )
        }
        ModelType::NemotronH => {
            let args: models::nemotron_h::NemotronHConfig = parse_model_config(&config_str)?;
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
        ModelType::NemotronNAS => {
            load_owned_model_from_config!(
                &config_str,
                weights,
                models::nemotron_nas::NemotronNASConfig,
                models::NemotronNASModel::from_weights,
                LoadedModel::NemotronNAS
            )
        }
        ModelType::Step3p5 => {
            load_model_from_config!(
                &config_str,
                weights,
                models::step3p5::Step3p5Config,
                models::Step3p5Model::from_weights,
                LoadedModel::Step3p5
            )
        }
        ModelType::KimiLinear => {
            let args: models::kimi_linear::KimiLinearConfig = parse_model_config(&config_str)?;
            let mut owned = copy_weight_map(weights);
            owned = models::KimiLinearModel::sanitize_weights(owned, &args);
            let m = models::KimiLinearModel::from_weights(&owned, &args)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::KimiLinear(m)
        }
        ModelType::LongcatFlash | ModelType::LongcatFlashNgram => {
            let args: models::longcat_flash_ngram::LongcatFlashNgramConfig =
                parse_model_config(&config_str)?;
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
        ModelType::Rwkv7 => {
            let args: models::rwkv7::Rwkv7Config = parse_model_config(&config_str)?;
            let m =
                models::Rwkv7::from_weights(weights, args).map_err(|e| anyhow::anyhow!("{}", e))?;
            LoadedModel::Rwkv7(m)
        }
        ModelType::RecurrentGemma => {
            load_owned_model_from_config!(
                &config_str,
                weights,
                models::recurrent_gemma::GriffinConfig,
                models::GriffinModel::from_weights,
                LoadedModel::RecurrentGemma
            )
        }
    };

    Ok(model)
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;

use anyhow::Result;
use serde::de::DeserializeOwned;
use std::fmt::Display;
use std::path::{Path, PathBuf};

use crate::LoadedModel;
use crate::lora;
use crate::models::{self, ModelType, get_model_type, sanitize_config_json};
use crate::tokenizer::{self, MlxcelTokenizer};
use mlxcel_core::weights::WeightMap;

mod vlm;

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

fn try_load_config_backed_model_from_dir(
    model_type: ModelType,
    path_str: &str,
) -> Result<Option<LoadedModel>> {
    Ok(match model_type {
        ModelType::Llama => {
            Some(load_pair_from_dir(path_str, models::Llama3Model::load).map(LoadedModel::Llama)?)
        }
        ModelType::Llama4 => Some(
            load_pair_from_dir(path_str, models::Llama4CxxModel::load)
                .map(|m| LoadedModel::Llama4(models::Llama4Wrapper::new(m)))?,
        ),
        ModelType::Qwen2 => {
            Some(load_pair_from_dir(path_str, models::Qwen2Model::load).map(LoadedModel::Qwen2)?)
        }
        ModelType::Qwen3 => {
            Some(load_pair_from_dir(path_str, models::Qwen3Model::load).map(LoadedModel::Qwen3)?)
        }
        ModelType::Qwen3Moe => Some(
            load_pair_from_dir(path_str, models::Qwen3MoeModel::load).map(LoadedModel::Qwen3Moe)?,
        ),
        ModelType::Qwen3Next => Some(
            load_pair_from_dir(path_str, models::Qwen3NextModel::load)
                .map(LoadedModel::Qwen3Next)?,
        ),
        ModelType::Qwen2Moe => Some(
            load_pair_from_dir(path_str, models::Qwen2MoeModel::load).map(LoadedModel::Qwen2Moe)?,
        ),
        ModelType::Gemma => {
            Some(load_pair_from_dir(path_str, models::GemmaModel::load).map(LoadedModel::Gemma)?)
        }
        ModelType::Gemma2 => {
            Some(load_pair_from_dir(path_str, models::Gemma2Model::load).map(LoadedModel::Gemma2)?)
        }
        ModelType::Gemma3 => Some(
            load_pair_from_dir(path_str, models::Gemma3Model::load)
                .map(|m| LoadedModel::Gemma3(models::Gemma3Wrapper::new(m)))?,
        ),
        ModelType::Phi => {
            Some(load_pair_from_dir(path_str, models::PhiModel::load).map(LoadedModel::Phi)?)
        }
        ModelType::Phi3 => {
            Some(load_pair_from_dir(path_str, models::Phi3Model::load).map(LoadedModel::Phi3)?)
        }
        ModelType::Phi3Small => Some(
            load_pair_from_dir(path_str, models::Phi3SmallModel::load)
                .map(LoadedModel::Phi3Small)?,
        ),
        ModelType::PhiMoe => {
            Some(load_pair_from_dir(path_str, models::PhiMoeModel::load).map(LoadedModel::PhiMoe)?)
        }
        ModelType::Mixtral => Some(
            load_pair_from_dir(path_str, models::MixtralModel::load).map(LoadedModel::Mixtral)?,
        ),
        ModelType::OLMoE => {
            Some(load_pair_from_dir(path_str, models::OlmoeModel::load).map(LoadedModel::OLMoE)?)
        }
        ModelType::DeepSeek => Some(
            load_pair_from_dir(path_str, models::DeepSeekModel::load).map(LoadedModel::DeepSeek)?,
        ),
        ModelType::DeepSeekV2 => Some(
            load_pair_from_dir(path_str, models::DeepSeekV2Model::load)
                .map(LoadedModel::DeepSeekV2)?,
        ),
        ModelType::DeepSeekV3 => Some(
            load_pair_from_dir(path_str, models::DeepSeekV3Model::load)
                .map(LoadedModel::DeepSeekV3)?,
        ),
        ModelType::DeepSeekV32 => Some(
            load_pair_from_dir(path_str, models::DeepSeekV32Model::load)
                .map(LoadedModel::DeepSeekV32)?,
        ),
        ModelType::Cohere => {
            Some(load_pair_from_dir(path_str, models::CohereModel::load).map(LoadedModel::Cohere)?)
        }
        ModelType::Cohere2 => Some(
            load_pair_from_dir(path_str, models::Cohere2Model::load).map(LoadedModel::Cohere2)?,
        ),
        ModelType::InternLM2 => Some(
            load_pair_from_dir(path_str, models::InternLM2Model::load)
                .map(LoadedModel::InternLM2)?,
        ),
        ModelType::InternLM3 => Some(
            load_pair_from_dir(path_str, models::InternLM3Model::load)
                .map(LoadedModel::InternLM3)?,
        ),
        ModelType::Baichuan => Some(
            load_pair_from_dir(path_str, models::BaichuanModel::load).map(LoadedModel::Baichuan)?,
        ),
        ModelType::Glm4 => {
            Some(load_pair_from_dir(path_str, models::Glm4Model::load).map(LoadedModel::Glm4)?)
        }
        ModelType::Glm4Moe => Some(
            load_pair_from_dir(path_str, models::Glm4MoeModel::load).map(LoadedModel::Glm4Moe)?,
        ),
        ModelType::Glm4MoeLite => Some(
            load_pair_from_dir(path_str, models::Glm4MoeLiteModel::load)
                .map(LoadedModel::Glm4MoeLite)?,
        ),
        ModelType::GlmMoeDsa => Some(
            load_pair_from_dir(path_str, models::GlmMoeDsaModel::load)
                .map(LoadedModel::GlmMoeDsa)?,
        ),
        ModelType::Ernie45 => Some(
            load_pair_from_dir(path_str, models::Ernie45Model::load).map(LoadedModel::Ernie45)?,
        ),
        ModelType::Ernie45Moe => Some(
            load_pair_from_dir(path_str, models::Ernie45MoeModel::load)
                .map(LoadedModel::Ernie45Moe)?,
        ),
        ModelType::HunyuanMoe => Some(
            load_pair_from_dir(path_str, models::HunyuanMoeModel::load)
                .map(LoadedModel::HunyuanMoe)?,
        ),
        ModelType::HunyuanV1Dense => Some(
            load_pair_from_dir(path_str, models::HunyuanV1DenseModel::load)
                .map(LoadedModel::HunyuanV1Dense)?,
        ),
        ModelType::MiMo => {
            Some(load_pair_from_dir(path_str, models::MiMoModel::load).map(LoadedModel::MiMo)?)
        }
        ModelType::ExaOne => {
            Some(load_pair_from_dir(path_str, models::ExaOneModel::load).map(LoadedModel::ExaOne)?)
        }
        ModelType::ExaOne4 => Some(
            load_pair_from_dir(path_str, models::ExaOne4Model::load)
                .map(|m| LoadedModel::ExaOne4(models::ExaOne4Wrapper::new(m)))?,
        ),
        ModelType::ExaOneMoe => Some(
            load_pair_from_dir(path_str, models::ExaoneMoeModel::load)
                .map(LoadedModel::ExaOneMoe)?,
        ),
        ModelType::Olmo => {
            Some(load_pair_from_dir(path_str, models::OlmoModel::load).map(LoadedModel::Olmo)?)
        }
        ModelType::Olmo2 => {
            Some(load_pair_from_dir(path_str, models::OLMo2Model::load).map(LoadedModel::Olmo2)?)
        }
        ModelType::Olmo3 => {
            Some(load_pair_from_dir(path_str, models::OLMo3Model::load).map(LoadedModel::Olmo3)?)
        }
        ModelType::StarCoder2 => Some(
            load_pair_from_dir(path_str, models::StarCoder2Model::load)
                .map(LoadedModel::StarCoder2)?,
        ),
        ModelType::MiniCPM => Some(
            load_pair_from_dir(path_str, models::MiniCPMModel::load).map(LoadedModel::MiniCPM)?,
        ),
        ModelType::MiniCPM3 => Some(
            load_pair_from_dir(path_str, models::MiniCPM3Model::load).map(LoadedModel::MiniCPM3)?,
        ),
        ModelType::StableLM => Some(
            load_pair_from_dir(path_str, models::StableLMModel::load).map(LoadedModel::StableLM)?,
        ),
        ModelType::SmolLM3 => Some(
            load_pair_from_dir(path_str, models::SmolLM3Model::load).map(LoadedModel::SmolLM3)?,
        ),
        ModelType::Ministral3 => Some(
            load_pair_from_dir(path_str, models::Ministral3Model::load)
                .map(|m| LoadedModel::Ministral3(models::Ministral3Wrapper::new(m)))?,
        ),
        ModelType::Nemotron => Some(
            load_pair_from_dir(path_str, models::NemotronModel::load).map(LoadedModel::Nemotron)?,
        ),
        ModelType::Step3p5 => Some(
            load_pair_from_dir(path_str, models::Step3p5Model::load).map(LoadedModel::Step3p5)?,
        ),
        _ => None,
    })
}

fn try_load_config_backed_model_from_weights(
    model_type: ModelType,
    config_str: &str,
    weights: &mut WeightMap,
) -> Result<Option<LoadedModel>> {
    Ok(match model_type {
        ModelType::Llama => Some(load_model_from_config!(
            &config_str,
            weights,
            models::llama3::ModelArgs,
            models::Llama3Model::from_weights,
            LoadedModel::Llama
        )),
        ModelType::Llama4 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::llama4::TextArgs,
            models::Llama4CxxModel::from_weights,
            |m| LoadedModel::Llama4(models::Llama4Wrapper::new(m))
        )),
        ModelType::Qwen2 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::llama3::ModelArgs,
            models::Qwen2Model::from_weights,
            LoadedModel::Qwen2
        )),
        ModelType::Qwen3 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::qwen3::ModelArgs,
            models::Qwen3Model::from_weights,
            LoadedModel::Qwen3
        )),
        ModelType::Qwen3Moe => Some(load_model_from_config!(
            &config_str,
            weights,
            models::qwen3_moe::ModelArgs,
            models::Qwen3MoeModel::from_weights,
            LoadedModel::Qwen3Moe
        )),
        ModelType::Qwen3Next => Some(load_model_from_config!(
            &config_str,
            weights,
            models::qwen3_next::Qwen3NextConfig,
            models::Qwen3NextModel::from_weights,
            LoadedModel::Qwen3Next
        )),
        ModelType::Qwen2Moe => Some(load_model_from_config!(
            &config_str,
            weights,
            models::qwen2_moe::ModelArgs,
            models::Qwen2MoeModel::from_weights,
            LoadedModel::Qwen2Moe
        )),
        ModelType::Gemma => Some(load_model_from_config!(
            &config_str,
            weights,
            models::gemma::ModelArgs,
            models::GemmaModel::from_weights,
            LoadedModel::Gemma
        )),
        ModelType::Gemma2 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::gemma2::ModelArgs,
            models::Gemma2Model::from_weights,
            LoadedModel::Gemma2
        )),
        ModelType::Gemma3 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::gemma3::ModelArgs,
            models::Gemma3Model::from_weights,
            |m| LoadedModel::Gemma3(models::Gemma3Wrapper::new(m))
        )),
        ModelType::Phi => Some(load_model_from_config!(
            &config_str,
            weights,
            models::phi::ModelArgs,
            models::PhiModel::from_weights,
            LoadedModel::Phi
        )),
        ModelType::Phi3 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::phi3::ModelArgs,
            models::Phi3Model::from_weights,
            LoadedModel::Phi3
        )),
        ModelType::Phi3Small => Some(load_model_from_config!(
            &config_str,
            weights,
            models::phi3small::ModelArgs,
            models::Phi3SmallModel::from_weights,
            LoadedModel::Phi3Small
        )),
        ModelType::PhiMoe => Some(load_model_from_config!(
            &config_str,
            weights,
            models::phimoe::ModelArgs,
            models::PhiMoeModel::from_weights,
            LoadedModel::PhiMoe
        )),
        ModelType::Mixtral => Some(load_model_from_config!(
            &config_str,
            weights,
            models::mixtral::ModelArgs,
            models::MixtralModel::from_weights,
            LoadedModel::Mixtral
        )),
        ModelType::OLMoE => Some(load_model_from_config!(
            &config_str,
            weights,
            models::olmoe::ModelArgs,
            models::OlmoeModel::from_weights,
            LoadedModel::OLMoE
        )),
        ModelType::DeepSeek => Some(load_model_from_config!(
            &config_str,
            weights,
            models::deepseek::ModelArgs,
            models::DeepSeekModel::from_weights,
            LoadedModel::DeepSeek
        )),
        ModelType::DeepSeekV2 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::deepseek_v2::ModelArgs,
            models::DeepSeekV2Model::from_weights,
            LoadedModel::DeepSeekV2
        )),
        ModelType::DeepSeekV3 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::deepseek_v3::DeepSeekV3Config,
            models::DeepSeekV3Model::from_weights,
            LoadedModel::DeepSeekV3
        )),
        ModelType::DeepSeekV32 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::deepseek_v32::ModelArgs,
            models::DeepSeekV32Model::from_weights,
            LoadedModel::DeepSeekV32
        )),
        ModelType::Cohere => Some(load_model_from_config!(
            &config_str,
            weights,
            models::cohere::ModelArgs,
            models::CohereModel::from_weights,
            LoadedModel::Cohere
        )),
        ModelType::Cohere2 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::cohere2::Cohere2Config,
            models::Cohere2Model::from_weights,
            LoadedModel::Cohere2
        )),
        ModelType::InternLM2 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::internlm2::ModelArgs,
            models::InternLM2Model::from_weights,
            LoadedModel::InternLM2
        )),
        ModelType::InternLM3 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::internlm3::ModelArgs,
            models::InternLM3Model::from_weights,
            LoadedModel::InternLM3
        )),
        ModelType::Baichuan => Some(load_model_from_config!(
            &config_str,
            weights,
            models::baichuan::BaichuanConfig,
            models::BaichuanModel::from_weights,
            LoadedModel::Baichuan
        )),
        ModelType::Glm4 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::glm4::ModelArgs,
            models::Glm4Model::from_weights,
            LoadedModel::Glm4
        )),
        ModelType::Glm4Moe => Some(load_model_from_config!(
            &config_str,
            weights,
            models::glm4_moe::ModelArgs,
            models::Glm4MoeModel::from_weights,
            LoadedModel::Glm4Moe
        )),
        ModelType::Glm4MoeLite => Some(load_model_from_config!(
            &config_str,
            weights,
            models::glm4_moe_lite::ModelArgs,
            models::Glm4MoeLiteModel::from_weights,
            LoadedModel::Glm4MoeLite
        )),
        ModelType::GlmMoeDsa => Some(load_model_from_config!(
            &config_str,
            weights,
            models::glm_moe_dsa::ModelArgs,
            models::GlmMoeDsaModel::from_weights,
            LoadedModel::GlmMoeDsa
        )),
        ModelType::Ernie45 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::ernie4_5::ModelArgs,
            models::Ernie45Model::from_weights,
            LoadedModel::Ernie45
        )),
        ModelType::Ernie45Moe => Some(load_model_from_config!(
            &config_str,
            weights,
            models::ernie4_5_moe::ModelArgs,
            models::Ernie45MoeModel::from_weights,
            LoadedModel::Ernie45Moe
        )),
        ModelType::HunyuanMoe => Some(load_model_from_config!(
            &config_str,
            weights,
            models::hunyuan_moe::ModelArgs,
            models::HunyuanMoeModel::from_weights,
            LoadedModel::HunyuanMoe
        )),
        ModelType::HunyuanV1Dense => Some(load_model_from_config!(
            &config_str,
            weights,
            models::hunyuan_v1_dense::ModelArgs,
            models::HunyuanV1DenseModel::from_weights,
            LoadedModel::HunyuanV1Dense
        )),
        ModelType::MiMo => Some(load_model_from_config!(
            &config_str,
            weights,
            models::mimo::ModelArgs,
            models::MiMoModel::from_weights,
            LoadedModel::MiMo
        )),
        ModelType::ExaOne => Some(load_model_from_config!(
            &config_str,
            weights,
            models::exaone::ExaOneConfig,
            models::ExaOneModel::from_weights,
            LoadedModel::ExaOne
        )),
        ModelType::ExaOne4 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::exaone4::ModelArgs,
            models::ExaOne4Model::from_weights,
            |m| LoadedModel::ExaOne4(models::ExaOne4Wrapper::new(m))
        )),
        ModelType::ExaOneMoe => Some(load_model_from_config!(
            &config_str,
            weights,
            models::exaone_moe::ModelArgs,
            models::ExaoneMoeModel::from_weights,
            LoadedModel::ExaOneMoe
        )),
        ModelType::Olmo => Some(load_model_from_config!(
            &config_str,
            weights,
            models::olmo::ModelArgs,
            models::OlmoModel::from_weights,
            LoadedModel::Olmo
        )),
        ModelType::Olmo2 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::olmo2::ModelArgs,
            models::OLMo2Model::from_weights,
            LoadedModel::Olmo2
        )),
        ModelType::Olmo3 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::olmo3::OLMo3Config,
            models::OLMo3Model::from_weights,
            LoadedModel::Olmo3
        )),
        ModelType::StarCoder2 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::starcoder2::StarCoder2Config,
            models::StarCoder2Model::from_weights,
            LoadedModel::StarCoder2
        )),
        ModelType::MiniCPM => Some(load_model_from_config!(
            &config_str,
            weights,
            models::minicpm::ModelArgs,
            models::MiniCPMModel::from_weights,
            LoadedModel::MiniCPM
        )),
        ModelType::MiniCPM3 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::minicpm3::ModelArgs,
            models::MiniCPM3Model::from_weights,
            LoadedModel::MiniCPM3
        )),
        ModelType::StableLM => Some(load_model_from_config!(
            &config_str,
            weights,
            models::stablelm::ModelArgs,
            models::StableLMModel::from_weights,
            LoadedModel::StableLM
        )),
        ModelType::SmolLM3 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::smollm3::ModelArgs,
            models::SmolLM3Model::from_weights,
            LoadedModel::SmolLM3
        )),
        ModelType::Ministral3 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::ministral3::ModelArgs,
            models::Ministral3Model::from_weights,
            |m| LoadedModel::Ministral3(models::Ministral3Wrapper::new(m))
        )),
        ModelType::Nemotron => Some(load_model_from_config!(
            &config_str,
            weights,
            models::nemotron::ModelArgs,
            models::NemotronModel::from_weights,
            LoadedModel::Nemotron
        )),
        ModelType::Step3p5 => Some(load_model_from_config!(
            &config_str,
            weights,
            models::step3p5::Step3p5Config,
            models::Step3p5Model::from_weights,
            LoadedModel::Step3p5
        )),
        _ => None,
    })
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

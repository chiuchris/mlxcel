//! Registry for standard config-backed text models.
//!
//! These model families follow the common `config.json` + standard text-weight
//! loading path, so keeping them in one module makes new architecture ports
//! easier to compare and extend.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;

use crate::LoadedModel;
use crate::models::{self, ModelType};

macro_rules! for_each_config_backed_model {
    ($macro:ident) => {
        $macro! {
            Llama, models::Llama3Model::load, models::llama3::ModelArgs, models::Llama3Model::from_weights, LoadedModel::Llama;
            Llama4, models::Llama4CxxModel::load, models::llama4::TextArgs, models::Llama4CxxModel::from_weights, |m| LoadedModel::Llama4(models::Llama4Wrapper::new(m));
            Qwen2, models::Qwen2Model::load, models::llama3::ModelArgs, models::Qwen2Model::from_weights, LoadedModel::Qwen2;
            Qwen3, models::Qwen3Model::load, models::qwen3::ModelArgs, models::Qwen3Model::from_weights, LoadedModel::Qwen3;
            Qwen3Moe, models::Qwen3MoeModel::load, models::qwen3_moe::ModelArgs, models::Qwen3MoeModel::from_weights, LoadedModel::Qwen3Moe;
            Qwen3Next, models::Qwen3NextModel::load, models::qwen3_next::Qwen3NextConfig, models::Qwen3NextModel::from_weights, LoadedModel::Qwen3Next;
            Qwen2Moe, models::Qwen2MoeModel::load, models::qwen2_moe::ModelArgs, models::Qwen2MoeModel::from_weights, LoadedModel::Qwen2Moe;
            Gemma, models::GemmaModel::load, models::gemma::ModelArgs, models::GemmaModel::from_weights, LoadedModel::Gemma;
            Gemma2, models::Gemma2Model::load, models::gemma2::ModelArgs, models::Gemma2Model::from_weights, LoadedModel::Gemma2;
            Gemma3, models::Gemma3Model::load, models::gemma3::ModelArgs, models::Gemma3Model::from_weights, |m| LoadedModel::Gemma3(models::Gemma3Wrapper::new(m));
            Phi, models::PhiModel::load, models::phi::ModelArgs, models::PhiModel::from_weights, LoadedModel::Phi;
            Phi3, models::Phi3Model::load, models::phi3::ModelArgs, models::Phi3Model::from_weights, LoadedModel::Phi3;
            Phi3Small, models::Phi3SmallModel::load, models::phi3small::ModelArgs, models::Phi3SmallModel::from_weights, LoadedModel::Phi3Small;
            PhiMoe, models::PhiMoeModel::load, models::phimoe::ModelArgs, models::PhiMoeModel::from_weights, LoadedModel::PhiMoe;
            MiniMax, models::MiniMaxModel::load, models::minimax::ModelArgs, models::MiniMaxModel::from_weights, LoadedModel::MiniMax;
            Mixtral, models::MixtralModel::load, models::mixtral::ModelArgs, models::MixtralModel::from_weights, LoadedModel::Mixtral;
            OLMoE, models::OlmoeModel::load, models::olmoe::ModelArgs, models::OlmoeModel::from_weights, LoadedModel::OLMoE;
            DeepSeek, models::DeepSeekModel::load, models::deepseek::ModelArgs, models::DeepSeekModel::from_weights, LoadedModel::DeepSeek;
            DeepSeekV2, models::DeepSeekV2Model::load, models::deepseek_v2::ModelArgs, models::DeepSeekV2Model::from_weights, LoadedModel::DeepSeekV2;
            DeepSeekV3, models::DeepSeekV3Model::load, models::deepseek_v3::DeepSeekV3Config, models::DeepSeekV3Model::from_weights, LoadedModel::DeepSeekV3;
            DeepSeekV32, models::DeepSeekV32Model::load, models::deepseek_v32::ModelArgs, models::DeepSeekV32Model::from_weights, LoadedModel::DeepSeekV32;
            Cohere, models::CohereModel::load, models::cohere::ModelArgs, models::CohereModel::from_weights, LoadedModel::Cohere;
            Cohere2, models::Cohere2Model::load, models::cohere2::Cohere2Config, models::Cohere2Model::from_weights, LoadedModel::Cohere2;
            InternLM2, models::InternLM2Model::load, models::internlm2::ModelArgs, models::InternLM2Model::from_weights, LoadedModel::InternLM2;
            InternLM3, models::InternLM3Model::load, models::internlm3::ModelArgs, models::InternLM3Model::from_weights, LoadedModel::InternLM3;
            Baichuan, models::BaichuanModel::load, models::baichuan::BaichuanConfig, models::BaichuanModel::from_weights, LoadedModel::Baichuan;
            Glm4, models::Glm4Model::load, models::glm4::ModelArgs, models::Glm4Model::from_weights, LoadedModel::Glm4;
            Glm4Moe, models::Glm4MoeModel::load, models::glm4_moe::ModelArgs, models::Glm4MoeModel::from_weights, LoadedModel::Glm4Moe;
            Glm4MoeLite, models::Glm4MoeLiteModel::load, models::glm4_moe_lite::ModelArgs, models::Glm4MoeLiteModel::from_weights, LoadedModel::Glm4MoeLite;
            GlmMoeDsa, models::GlmMoeDsaModel::load, models::glm_moe_dsa::ModelArgs, models::GlmMoeDsaModel::from_weights, LoadedModel::GlmMoeDsa;
            Ernie45, models::Ernie45Model::load, models::ernie4_5::ModelArgs, models::Ernie45Model::from_weights, LoadedModel::Ernie45;
            Ernie45Moe, models::Ernie45MoeModel::load, models::ernie4_5_moe::ModelArgs, models::Ernie45MoeModel::from_weights, LoadedModel::Ernie45Moe;
            HunyuanMoe, models::HunyuanMoeModel::load, models::hunyuan_moe::ModelArgs, models::HunyuanMoeModel::from_weights, LoadedModel::HunyuanMoe;
            HunyuanV1Dense, models::HunyuanV1DenseModel::load, models::hunyuan_v1_dense::ModelArgs, models::HunyuanV1DenseModel::from_weights, LoadedModel::HunyuanV1Dense;
            MiMo, models::MiMoModel::load, models::mimo::ModelArgs, models::MiMoModel::from_weights, LoadedModel::MiMo;
            ExaOne, models::ExaOneModel::load, models::exaone::ExaOneConfig, models::ExaOneModel::from_weights, LoadedModel::ExaOne;
            ExaOne4, models::ExaOne4Model::load, models::exaone4::ModelArgs, models::ExaOne4Model::from_weights, |m| LoadedModel::ExaOne4(models::ExaOne4Wrapper::new(m));
            ExaOneMoe, models::ExaoneMoeModel::load, models::exaone_moe::ModelArgs, models::ExaoneMoeModel::from_weights, LoadedModel::ExaOneMoe;
            SolarOpen, models::SolarOpenModel::load, models::solar_open::ModelArgs, models::SolarOpenModel::from_weights, LoadedModel::SolarOpen;
            Olmo, models::OlmoModel::load, models::olmo::ModelArgs, models::OlmoModel::from_weights, LoadedModel::Olmo;
            Olmo2, models::OLMo2Model::load, models::olmo2::ModelArgs, models::OLMo2Model::from_weights, LoadedModel::Olmo2;
            Olmo3, models::OLMo3Model::load, models::olmo3::OLMo3Config, models::OLMo3Model::from_weights, LoadedModel::Olmo3;
            StarCoder2, models::StarCoder2Model::load, models::starcoder2::StarCoder2Config, models::StarCoder2Model::from_weights, LoadedModel::StarCoder2;
            MiniCPM, models::MiniCPMModel::load, models::minicpm::ModelArgs, models::MiniCPMModel::from_weights, LoadedModel::MiniCPM;
            MiniCPM3, models::MiniCPM3Model::load, models::minicpm3::ModelArgs, models::MiniCPM3Model::from_weights, LoadedModel::MiniCPM3;
            StableLM, models::StableLMModel::load, models::stablelm::ModelArgs, models::StableLMModel::from_weights, LoadedModel::StableLM;
            SmolLM3, models::SmolLM3Model::load, models::smollm3::ModelArgs, models::SmolLM3Model::from_weights, LoadedModel::SmolLM3;
            Ministral3, models::Ministral3Model::load, models::ministral3::ModelArgs, models::Ministral3Model::from_weights, |m| LoadedModel::Ministral3(models::Ministral3Wrapper::new(m));
            Nemotron, models::NemotronModel::load, models::nemotron::ModelArgs, models::NemotronModel::from_weights, LoadedModel::Nemotron;
            Step3p5, models::Step3p5Model::load, models::step3p5::Step3p5Config, models::Step3p5Model::from_weights, LoadedModel::Step3p5;
        }
    };
}

macro_rules! match_config_backed_dir_loader {
    ($( $variant:ident, $dir_loader:path, $args_ty:ty, $weight_builder:path, $wrap:expr; )*) => {
        pub(crate) fn try_load_config_backed_model_from_dir(
            model_type: ModelType,
            path_str: &str,
        ) -> Result<Option<LoadedModel>> {
            Ok(match model_type {
                $(
                    ModelType::$variant => {
                        Some(super::load_pair_from_dir(path_str, $dir_loader).map($wrap)?)
                    }
                )*
                _ => None,
            })
        }
    };
}

macro_rules! match_config_backed_weight_loader {
    ($( $variant:ident, $dir_loader:path, $args_ty:ty, $weight_builder:path, $wrap:expr; )*) => {
        pub(crate) fn try_load_config_backed_model_from_weights(
            model_type: ModelType,
            config_str: &str,
            weights: &mut WeightMap,
        ) -> Result<Option<LoadedModel>> {
            Ok(match model_type {
                $(
                    ModelType::$variant => {
                        let args: $args_ty = super::parse_model_config(config_str)?;
                        let model = $weight_builder(weights, &args)
                            .map_err(|err| anyhow::anyhow!("{}", err))?;
                        Some(($wrap)(model))
                    }
                )*
                _ => None,
            })
        }
    };
}

macro_rules! match_config_backed_support {
    ($( $variant:ident, $dir_loader:path, $args_ty:ty, $weight_builder:path, $wrap:expr; )*) => {
        #[cfg_attr(not(test), allow(dead_code))]
        pub(crate) fn is_config_backed_model_type(model_type: ModelType) -> bool {
            matches!(model_type, $(ModelType::$variant)|*)
        }
    };
}

for_each_config_backed_model!(match_config_backed_dir_loader);
for_each_config_backed_model!(match_config_backed_weight_loader);
for_each_config_backed_model!(match_config_backed_support);

#[cfg(test)]
#[path = "config_backed_tests.rs"]
mod tests;

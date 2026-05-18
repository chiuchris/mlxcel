use crate::{models, vision};
use mlxcel_core::UniquePtr;
use mlxcel_core::generate::LanguageModel;

/// Model wrapper that holds any model implementing LanguageModel
///
/// For sliding window models (Gemma3, ExaOne4, Ministral3), we use wrapper types
/// that manage their internal cache state since they require mixed cache types.
pub enum LoadedModel {
    Llama(models::Llama3Model),
    Llama4(models::Llama4Wrapper),
    Qwen2(models::Qwen2Model),
    Qwen3(models::Qwen3Model),
    Qwen3Moe(models::Qwen3MoeModel),
    Qwen3Next(models::Qwen3NextModel),
    Qwen35(models::Qwen35Model),
    Qwen35VLM(vision::Qwen35VLModel),
    Qwen35Moe(models::Qwen35Model),
    Qwen35MoeVLM(vision::Qwen35VLModel),
    Qwen2Moe(models::Qwen2MoeModel),
    Gemma(models::GemmaModel),
    Gemma2(models::Gemma2Model),
    // Sliding window models use wrappers that implement LanguageModel
    Gemma3(models::Gemma3Wrapper),
    // Vision-language models
    Gemma3VLM(vision::VisionLanguageModel),
    Llama4VLM(vision::VisionLanguageModel),
    LlavaVLM(vision::VisionLanguageModel),
    Qwen2VL(vision::Qwen2VLModel),
    Qwen25VL(vision::Qwen25VLModel),
    Qwen3VL(vision::Qwen3VLModel),
    Qwen3VLMoe(vision::Qwen3VLMoeModel),
    Gemma3n(models::Gemma3nModel),
    Gemma3nVLM(vision::Gemma3nVLModel),
    Phi(models::PhiModel),
    Phi3(models::Phi3Model),
    Phi3VLM(vision::Phi3VLModel),
    Molmo2VLM(vision::Molmo2VLModel),
    Phi3Small(models::Phi3SmallModel),
    PhiMoe(models::PhiMoeModel),
    Mixtral(models::MixtralModel),
    OLMoE(models::OlmoeModel),
    DeepSeek(models::DeepSeekModel),
    DeepSeekV2(models::DeepSeekV2Model),
    DeepSeekV3(models::DeepSeekV3Model),
    DeepSeekV32(models::DeepSeekV32Model),
    Cohere(models::CohereModel),
    Cohere2(models::Cohere2Model),
    InternLM2(models::InternLM2Model),
    InternLM3(models::InternLM3Model),
    Baichuan(models::BaichuanModel),
    Glm4(models::Glm4Model),
    Glm4Moe(models::Glm4MoeModel),
    Glm4MoeLite(models::Glm4MoeLiteModel),
    GlmMoeDsa(models::GlmMoeDsaModel),
    Ernie45(models::Ernie45Model),
    Ernie45Moe(models::Ernie45MoeModel),
    HunyuanMoe(models::HunyuanMoeModel),
    HunyuanV1Dense(models::HunyuanV1DenseModel),
    MiMo(models::MiMoModel),
    ExaOne(models::ExaOneModel),
    // Sliding window model uses wrapper
    ExaOne4(models::ExaOne4Wrapper),
    ExaOneMoe(models::ExaoneMoeModel),
    Olmo(models::OlmoModel),
    Olmo2(models::OLMo2Model),
    Olmo3(models::OLMo3Model),
    StarCoder2(models::StarCoder2Model),
    MiniCPM(models::MiniCPMModel),
    MiniCPM3(models::MiniCPM3Model),
    StableLM(models::StableLMModel),
    SmolLM3(models::SmolLM3Model),
    // Sliding window model uses wrapper
    Ministral3(models::Ministral3Wrapper),
    Nemotron(models::NemotronModel),
    Mamba(models::MambaModel),
    Mamba2(models::Mamba2Model),
    Jamba(models::JambaModel),
    NemotronH(models::NemotronHModel),
    NemotronNAS(models::NemotronNASModel),
    Step3p5(models::Step3p5Model),
    KimiLinear(models::KimiLinearModel),
    LongcatFlash(models::LongcatFlashNgramModel),
    LongcatFlashNgram(models::LongcatFlashNgramModel),
    Rwkv7(models::Rwkv7),
    RecurrentGemma(models::GriffinModel),
}

impl LoadedModel {
    /// Check if this model is a vision-language model
    pub fn is_vlm(&self) -> bool {
        matches!(
            self,
            Self::Gemma3VLM(_)
                | Self::Llama4VLM(_)
                | Self::LlavaVLM(_)
                | Self::Qwen2VL(_)
                | Self::Qwen25VL(_)
                | Self::Qwen3VL(_)
                | Self::Qwen3VLMoe(_)
                | Self::Qwen35VLM(_)
                | Self::Qwen35MoeVLM(_)
                | Self::Gemma3nVLM(_)
                | Self::Phi3VLM(_)
                | Self::Molmo2VLM(_)
        )
    }

    /// Get the vision module if this is a VLM (Gemma3/LLaVA-style)
    pub fn vision_module(&self) -> Option<&vision::VisionModule> {
        match self {
            Self::Gemma3VLM(vlm) => Some(&vlm.vision),
            Self::Llama4VLM(vlm) => Some(&vlm.vision),
            Self::LlavaVLM(vlm) => Some(&vlm.vision),
            _ => None,
        }
    }

    /// Get the Qwen2-VL model if this is a Qwen2-VL VLM
    pub fn qwen2_vl_model(&self) -> Option<&vision::Qwen2VLModel> {
        match self {
            Self::Qwen2VL(m) => Some(m),
            _ => None,
        }
    }

    /// Get the Qwen2.5-VL model if this is a Qwen2.5-VL VLM
    pub fn qwen2_5_vl_model(&self) -> Option<&vision::Qwen25VLModel> {
        match self {
            Self::Qwen25VL(m) => Some(m),
            _ => None,
        }
    }

    /// Get the Qwen3-VL model if this is a Qwen3-VL VLM
    pub fn qwen3_vl_model(&self) -> Option<&vision::Qwen3VLModel> {
        match self {
            Self::Qwen3VL(m) => Some(m),
            _ => None,
        }
    }

    /// Get the Qwen3-VL-MoE model if this is a Qwen3-VL-MoE VLM
    pub fn qwen3_vl_moe_model(&self) -> Option<&vision::Qwen3VLMoeModel> {
        match self {
            Self::Qwen3VLMoe(m) => Some(m),
            _ => None,
        }
    }

    /// Get the Qwen3.5 VLM model
    pub fn qwen3_5_vl_model(&self) -> Option<&vision::Qwen35VLModel> {
        match self {
            Self::Qwen35VLM(m) | Self::Qwen35MoeVLM(m) => Some(m),
            _ => None,
        }
    }

    /// Get the Gemma3n-VL model if this is a Gemma3n VLM
    pub fn gemma3n_vl_model(&self) -> Option<&vision::Gemma3nVLModel> {
        match self {
            Self::Gemma3nVLM(m) => Some(m),
            _ => None,
        }
    }

    /// Get the Phi3-V model if this is a Phi3 VLM
    pub fn phi3_vl_model(&self) -> Option<&vision::Phi3VLModel> {
        match self {
            Self::Phi3VLM(m) => Some(m),
            _ => None,
        }
    }

    /// Get the Molmo2 VLM model if this is a Molmo2 VLM
    pub fn molmo2_vl_model(&self) -> Option<&vision::Molmo2VLModel> {
        match self {
            Self::Molmo2VLM(m) => Some(m),
            _ => None,
        }
    }
}

impl LanguageModel for LoadedModel {
    fn num_layers(&self) -> usize {
        match self {
            Self::Llama(m) => LanguageModel::num_layers(m),
            Self::Llama4(m) => LanguageModel::num_layers(m),
            Self::Qwen2(m) => LanguageModel::num_layers(m),
            Self::Qwen3(m) => LanguageModel::num_layers(m),
            Self::Qwen3Moe(m) => LanguageModel::num_layers(m),
            Self::Qwen3Next(m) => LanguageModel::num_layers(m),
            Self::Qwen35(m) => LanguageModel::num_layers(m),
            Self::Qwen35VLM(m) => LanguageModel::num_layers(m),
            Self::Qwen35Moe(m) => LanguageModel::num_layers(m),
            Self::Qwen35MoeVLM(m) => LanguageModel::num_layers(m),
            Self::Qwen2Moe(m) => LanguageModel::num_layers(m),
            Self::Gemma(m) => LanguageModel::num_layers(m),
            Self::Gemma2(m) => LanguageModel::num_layers(m),
            Self::Gemma3(m) => LanguageModel::num_layers(m),
            Self::Gemma3VLM(m) => LanguageModel::num_layers(m),
            Self::Llama4VLM(m) => LanguageModel::num_layers(m),
            Self::LlavaVLM(m) => LanguageModel::num_layers(m),
            Self::Qwen2VL(m) => LanguageModel::num_layers(m),
            Self::Qwen25VL(m) => LanguageModel::num_layers(m),
            Self::Qwen3VL(m) => LanguageModel::num_layers(m),
            Self::Qwen3VLMoe(m) => LanguageModel::num_layers(m),
            Self::Gemma3n(m) => LanguageModel::num_layers(m),
            Self::Gemma3nVLM(m) => LanguageModel::num_layers(m),
            Self::Phi(m) => LanguageModel::num_layers(m),
            Self::Phi3(m) => LanguageModel::num_layers(m),
            Self::Phi3VLM(m) => LanguageModel::num_layers(m),
            Self::Molmo2VLM(m) => LanguageModel::num_layers(m),
            Self::Phi3Small(m) => LanguageModel::num_layers(m),
            Self::PhiMoe(m) => LanguageModel::num_layers(m),
            Self::Mixtral(m) => LanguageModel::num_layers(m),
            Self::OLMoE(m) => LanguageModel::num_layers(m),
            Self::DeepSeek(m) => LanguageModel::num_layers(m),
            Self::DeepSeekV2(m) => LanguageModel::num_layers(m),
            Self::DeepSeekV3(m) => LanguageModel::num_layers(m),
            Self::DeepSeekV32(m) => LanguageModel::num_layers(m),
            Self::Cohere(m) => LanguageModel::num_layers(m),
            Self::Cohere2(m) => LanguageModel::num_layers(m),
            Self::InternLM2(m) => LanguageModel::num_layers(m),
            Self::InternLM3(m) => LanguageModel::num_layers(m),
            Self::Baichuan(m) => LanguageModel::num_layers(m),
            Self::Glm4(m) => LanguageModel::num_layers(m),
            Self::Glm4Moe(m) => LanguageModel::num_layers(m),
            Self::Glm4MoeLite(m) => LanguageModel::num_layers(m),
            Self::GlmMoeDsa(m) => LanguageModel::num_layers(m),
            Self::Ernie45(m) => LanguageModel::num_layers(m),
            Self::Ernie45Moe(m) => LanguageModel::num_layers(m),
            Self::HunyuanMoe(m) => LanguageModel::num_layers(m),
            Self::HunyuanV1Dense(m) => LanguageModel::num_layers(m),
            Self::MiMo(m) => LanguageModel::num_layers(m),
            Self::ExaOne(m) => LanguageModel::num_layers(m),
            Self::ExaOne4(m) => LanguageModel::num_layers(m),
            Self::ExaOneMoe(m) => LanguageModel::num_layers(m),
            Self::Olmo(m) => LanguageModel::num_layers(m),
            Self::Olmo2(m) => LanguageModel::num_layers(m),
            Self::Olmo3(m) => LanguageModel::num_layers(m),
            Self::StarCoder2(m) => LanguageModel::num_layers(m),
            Self::MiniCPM(m) => LanguageModel::num_layers(m),
            Self::MiniCPM3(m) => LanguageModel::num_layers(m),
            Self::StableLM(m) => LanguageModel::num_layers(m),
            Self::SmolLM3(m) => LanguageModel::num_layers(m),
            Self::Ministral3(m) => LanguageModel::num_layers(m),
            Self::Nemotron(m) => LanguageModel::num_layers(m),
            Self::Mamba(m) => LanguageModel::num_layers(m),
            Self::Mamba2(m) => LanguageModel::num_layers(m),
            Self::Jamba(m) => LanguageModel::num_layers(m),
            Self::NemotronH(m) => LanguageModel::num_layers(m),
            Self::NemotronNAS(m) => LanguageModel::num_layers(m),
            Self::Step3p5(m) => LanguageModel::num_layers(m),
            Self::KimiLinear(m) => LanguageModel::num_layers(m),
            Self::LongcatFlash(m) => LanguageModel::num_layers(m),
            Self::LongcatFlashNgram(m) => LanguageModel::num_layers(m),
            Self::Rwkv7(m) => LanguageModel::num_layers(m),
            Self::RecurrentGemma(m) => LanguageModel::num_layers(m),
        }
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        match self {
            Self::Llama(m) => LanguageModel::eos_token_ids(m),
            Self::Llama4(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen2(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen3(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen3Moe(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen3Next(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen35(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen35VLM(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen35Moe(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen35MoeVLM(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen2Moe(m) => LanguageModel::eos_token_ids(m),
            Self::Gemma(m) => LanguageModel::eos_token_ids(m),
            Self::Gemma2(m) => LanguageModel::eos_token_ids(m),
            Self::Gemma3(m) => LanguageModel::eos_token_ids(m),
            Self::Gemma3VLM(m) => LanguageModel::eos_token_ids(m),
            Self::Llama4VLM(m) => LanguageModel::eos_token_ids(m),
            Self::LlavaVLM(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen2VL(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen25VL(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen3VL(m) => LanguageModel::eos_token_ids(m),
            Self::Qwen3VLMoe(m) => LanguageModel::eos_token_ids(m),
            Self::Gemma3n(m) => LanguageModel::eos_token_ids(m),
            Self::Gemma3nVLM(m) => LanguageModel::eos_token_ids(m),
            Self::Phi(m) => LanguageModel::eos_token_ids(m),
            Self::Phi3(m) => LanguageModel::eos_token_ids(m),
            Self::Phi3VLM(m) => LanguageModel::eos_token_ids(m),
            Self::Molmo2VLM(m) => LanguageModel::eos_token_ids(m),
            Self::Phi3Small(m) => LanguageModel::eos_token_ids(m),
            Self::PhiMoe(m) => LanguageModel::eos_token_ids(m),
            Self::Mixtral(m) => LanguageModel::eos_token_ids(m),
            Self::OLMoE(m) => LanguageModel::eos_token_ids(m),
            Self::DeepSeek(m) => LanguageModel::eos_token_ids(m),
            Self::DeepSeekV2(m) => LanguageModel::eos_token_ids(m),
            Self::DeepSeekV3(m) => LanguageModel::eos_token_ids(m),
            Self::DeepSeekV32(m) => LanguageModel::eos_token_ids(m),
            Self::Cohere(m) => LanguageModel::eos_token_ids(m),
            Self::Cohere2(m) => LanguageModel::eos_token_ids(m),
            Self::InternLM2(m) => LanguageModel::eos_token_ids(m),
            Self::InternLM3(m) => LanguageModel::eos_token_ids(m),
            Self::Baichuan(m) => LanguageModel::eos_token_ids(m),
            Self::Glm4(m) => LanguageModel::eos_token_ids(m),
            Self::Glm4Moe(m) => LanguageModel::eos_token_ids(m),
            Self::Glm4MoeLite(m) => LanguageModel::eos_token_ids(m),
            Self::GlmMoeDsa(m) => LanguageModel::eos_token_ids(m),
            Self::Ernie45(m) => LanguageModel::eos_token_ids(m),
            Self::Ernie45Moe(m) => LanguageModel::eos_token_ids(m),
            Self::HunyuanMoe(m) => LanguageModel::eos_token_ids(m),
            Self::HunyuanV1Dense(m) => LanguageModel::eos_token_ids(m),
            Self::MiMo(m) => LanguageModel::eos_token_ids(m),
            Self::ExaOne(m) => LanguageModel::eos_token_ids(m),
            Self::ExaOne4(m) => LanguageModel::eos_token_ids(m),
            Self::ExaOneMoe(m) => LanguageModel::eos_token_ids(m),
            Self::Olmo(m) => LanguageModel::eos_token_ids(m),
            Self::Olmo2(m) => LanguageModel::eos_token_ids(m),
            Self::Olmo3(m) => LanguageModel::eos_token_ids(m),
            Self::StarCoder2(m) => LanguageModel::eos_token_ids(m),
            Self::MiniCPM(m) => LanguageModel::eos_token_ids(m),
            Self::MiniCPM3(m) => LanguageModel::eos_token_ids(m),
            Self::StableLM(m) => LanguageModel::eos_token_ids(m),
            Self::SmolLM3(m) => LanguageModel::eos_token_ids(m),
            Self::Ministral3(m) => LanguageModel::eos_token_ids(m),
            Self::Nemotron(m) => LanguageModel::eos_token_ids(m),
            Self::Mamba(m) => LanguageModel::eos_token_ids(m),
            Self::Mamba2(m) => LanguageModel::eos_token_ids(m),
            Self::Jamba(m) => LanguageModel::eos_token_ids(m),
            Self::NemotronH(m) => LanguageModel::eos_token_ids(m),
            Self::NemotronNAS(m) => LanguageModel::eos_token_ids(m),
            Self::Step3p5(m) => LanguageModel::eos_token_ids(m),
            Self::KimiLinear(m) => LanguageModel::eos_token_ids(m),
            Self::LongcatFlash(m) => LanguageModel::eos_token_ids(m),
            Self::LongcatFlashNgram(m) => LanguageModel::eos_token_ids(m),
            Self::Rwkv7(m) => LanguageModel::eos_token_ids(m),
            Self::RecurrentGemma(m) => LanguageModel::eos_token_ids(m),
        }
    }

    fn make_caches(&self) -> Vec<mlxcel_core::layers::KVCache> {
        // Use fully qualified syntax to ensure we call the trait method, not inherent methods
        match self {
            Self::Llama(m) => LanguageModel::make_caches(m),
            Self::Llama4(m) => LanguageModel::make_caches(m),
            Self::Qwen2(m) => LanguageModel::make_caches(m),
            Self::Qwen3(m) => LanguageModel::make_caches(m),
            Self::Qwen3Moe(m) => LanguageModel::make_caches(m),
            Self::Qwen3Next(m) => LanguageModel::make_caches(m),
            Self::Qwen35(m) => LanguageModel::make_caches(m),
            Self::Qwen35VLM(m) => LanguageModel::make_caches(m),
            Self::Qwen35Moe(m) => LanguageModel::make_caches(m),
            Self::Qwen35MoeVLM(m) => LanguageModel::make_caches(m),
            Self::Qwen2Moe(m) => LanguageModel::make_caches(m),
            Self::Gemma(m) => LanguageModel::make_caches(m),
            Self::Gemma2(m) => LanguageModel::make_caches(m),
            Self::Gemma3(m) => LanguageModel::make_caches(m),
            Self::Gemma3VLM(m) => LanguageModel::make_caches(m),
            Self::Llama4VLM(m) => LanguageModel::make_caches(m),
            Self::LlavaVLM(m) => LanguageModel::make_caches(m),
            Self::Qwen2VL(m) => LanguageModel::make_caches(m),
            Self::Qwen25VL(m) => LanguageModel::make_caches(m),
            Self::Qwen3VL(m) => LanguageModel::make_caches(m),
            Self::Qwen3VLMoe(m) => LanguageModel::make_caches(m),
            Self::Gemma3n(m) => LanguageModel::make_caches(m),
            Self::Gemma3nVLM(m) => LanguageModel::make_caches(m),
            Self::Phi(m) => LanguageModel::make_caches(m),
            Self::Phi3(m) => LanguageModel::make_caches(m),
            Self::Phi3VLM(m) => LanguageModel::make_caches(m),
            Self::Molmo2VLM(m) => LanguageModel::make_caches(m),
            Self::Phi3Small(m) => LanguageModel::make_caches(m),
            Self::PhiMoe(m) => LanguageModel::make_caches(m),
            Self::Mixtral(m) => LanguageModel::make_caches(m),
            Self::OLMoE(m) => LanguageModel::make_caches(m),
            Self::DeepSeek(m) => LanguageModel::make_caches(m),
            Self::DeepSeekV2(m) => LanguageModel::make_caches(m),
            Self::DeepSeekV3(m) => LanguageModel::make_caches(m),
            Self::DeepSeekV32(m) => LanguageModel::make_caches(m),
            Self::Cohere(m) => LanguageModel::make_caches(m),
            Self::Cohere2(m) => LanguageModel::make_caches(m),
            Self::InternLM2(m) => LanguageModel::make_caches(m),
            Self::InternLM3(m) => LanguageModel::make_caches(m),
            Self::Baichuan(m) => LanguageModel::make_caches(m),
            Self::Glm4(m) => LanguageModel::make_caches(m),
            Self::Glm4Moe(m) => LanguageModel::make_caches(m),
            Self::Glm4MoeLite(m) => LanguageModel::make_caches(m),
            Self::GlmMoeDsa(m) => LanguageModel::make_caches(m),
            Self::Ernie45(m) => LanguageModel::make_caches(m),
            Self::Ernie45Moe(m) => LanguageModel::make_caches(m),
            Self::HunyuanMoe(m) => LanguageModel::make_caches(m),
            Self::HunyuanV1Dense(m) => LanguageModel::make_caches(m),
            Self::MiMo(m) => LanguageModel::make_caches(m),
            Self::ExaOne(m) => LanguageModel::make_caches(m),
            Self::ExaOne4(m) => LanguageModel::make_caches(m),
            Self::ExaOneMoe(m) => LanguageModel::make_caches(m),
            Self::Olmo(m) => LanguageModel::make_caches(m),
            Self::Olmo2(m) => LanguageModel::make_caches(m),
            Self::Olmo3(m) => LanguageModel::make_caches(m),
            Self::StarCoder2(m) => LanguageModel::make_caches(m),
            Self::MiniCPM(m) => LanguageModel::make_caches(m),
            Self::MiniCPM3(m) => LanguageModel::make_caches(m),
            Self::StableLM(m) => LanguageModel::make_caches(m),
            Self::SmolLM3(m) => LanguageModel::make_caches(m),
            Self::Ministral3(m) => LanguageModel::make_caches(m),
            Self::Nemotron(m) => LanguageModel::make_caches(m),
            Self::Mamba(m) => LanguageModel::make_caches(m),
            Self::Mamba2(m) => LanguageModel::make_caches(m),
            Self::Jamba(m) => LanguageModel::make_caches(m),
            Self::NemotronH(m) => LanguageModel::make_caches(m),
            Self::NemotronNAS(m) => LanguageModel::make_caches(m),
            Self::Step3p5(m) => LanguageModel::make_caches(m),
            Self::KimiLinear(m) => LanguageModel::make_caches(m),
            Self::LongcatFlash(m) => LanguageModel::make_caches(m),
            Self::LongcatFlashNgram(m) => LanguageModel::make_caches(m),
            Self::Rwkv7(m) => LanguageModel::make_caches(m),
            Self::RecurrentGemma(m) => LanguageModel::make_caches(m),
        }
    }

    fn forward(
        &self,
        input_ids: &mlxcel_core::MlxArray,
        caches: &mut [mlxcel_core::layers::KVCache],
        mask: Option<&mlxcel_core::MlxArray>,
    ) -> UniquePtr<mlxcel_core::MlxArray> {
        match self {
            Self::Llama(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Llama4(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen2(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen3(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen3Moe(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen3Next(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen35(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen35VLM(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen35Moe(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen35MoeVLM(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen2Moe(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Gemma(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Gemma2(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Gemma3(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Gemma3VLM(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Llama4VLM(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::LlavaVLM(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen2VL(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen25VL(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen3VL(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Qwen3VLMoe(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Gemma3n(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Gemma3nVLM(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Phi(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Phi3(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Phi3VLM(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Molmo2VLM(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Phi3Small(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::PhiMoe(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Mixtral(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::OLMoE(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::DeepSeek(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::DeepSeekV2(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::DeepSeekV3(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::DeepSeekV32(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Cohere(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Cohere2(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::InternLM2(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::InternLM3(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Baichuan(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Glm4(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Glm4Moe(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Glm4MoeLite(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::GlmMoeDsa(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Ernie45(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Ernie45Moe(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::HunyuanMoe(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::HunyuanV1Dense(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::MiMo(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::ExaOne(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::ExaOne4(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::ExaOneMoe(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Olmo(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Olmo2(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Olmo3(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::StarCoder2(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::MiniCPM(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::MiniCPM3(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::StableLM(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::SmolLM3(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Ministral3(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Nemotron(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Mamba(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Mamba2(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Jamba(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::NemotronH(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::NemotronNAS(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Step3p5(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::KimiLinear(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::LongcatFlash(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::LongcatFlashNgram(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::Rwkv7(m) => LanguageModel::forward(m, input_ids, caches, mask),
            Self::RecurrentGemma(m) => LanguageModel::forward(m, input_ids, caches, mask),
        }
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &mlxcel_core::MlxArray,
        input_embeddings: Option<&mlxcel_core::MlxArray>,
        caches: &mut [mlxcel_core::layers::KVCache],
        mask: Option<&mlxcel_core::MlxArray>,
    ) -> UniquePtr<mlxcel_core::MlxArray> {
        // Only VLM-capable models override this
        match self {
            Self::Gemma3(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Gemma3VLM(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Llama(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Qwen2(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Llama4VLM(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::LlavaVLM(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Llama4(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Qwen2VL(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Qwen25VL(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Qwen3VL(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Qwen3VLMoe(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Qwen35VLM(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Qwen35MoeVLM(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Gemma3nVLM(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Phi3VLM(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Molmo2VLM(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Cohere2(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Gemma(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            Self::Gemma2(m) => {
                LanguageModel::forward_with_embeddings(m, input_ids, input_embeddings, caches, mask)
            }
            // All other models use the default (ignores embeddings, calls forward)
            _ => self.forward(input_ids, caches, mask),
        }
    }

    fn embed_tokens(
        &self,
        input_ids: &mlxcel_core::MlxArray,
    ) -> Option<UniquePtr<mlxcel_core::MlxArray>> {
        // VLM-capable models override this
        match self {
            Self::Gemma3(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Gemma3VLM(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Llama(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Qwen2(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Llama4(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Llama4VLM(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::LlavaVLM(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Qwen2VL(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Qwen25VL(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Qwen3VL(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Qwen3VLMoe(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Qwen35(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Qwen35VLM(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Qwen35Moe(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Qwen35MoeVLM(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Gemma3n(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Gemma3nVLM(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Phi3VLM(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Molmo2VLM(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Cohere2(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Gemma(m) => LanguageModel::embed_tokens(m, input_ids),
            Self::Gemma2(m) => LanguageModel::embed_tokens(m, input_ids),
            _ => None,
        }
    }
}

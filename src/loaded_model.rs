use crate::{models, qwen_vl, vision, vlm_prompt};
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

// Keep the full variant list in one place so LanguageModel delegation stays
// consistent as new models are added.
macro_rules! delegate_language_model {
    ($self:expr, $method:ident ( $($arg:expr),* $(,)? )) => {
        match $self {
            LoadedModel::Llama(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Llama4(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen2(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen3(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen3Moe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen3Next(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen35(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen35VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen35Moe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen35MoeVLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen2Moe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma2(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma3(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma3VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Llama4VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::LlavaVLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen2VL(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen25VL(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen3VL(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen3VLMoe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma3n(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma3nVLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Phi(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Phi3(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Phi3VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Molmo2VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Phi3Small(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::PhiMoe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Mixtral(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::OLMoE(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::DeepSeek(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::DeepSeekV2(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::DeepSeekV3(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::DeepSeekV32(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Cohere(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Cohere2(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::InternLM2(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::InternLM3(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Baichuan(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Glm4(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Glm4Moe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Glm4MoeLite(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::GlmMoeDsa(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Ernie45(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Ernie45Moe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::HunyuanMoe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::HunyuanV1Dense(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::MiMo(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::ExaOne(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::ExaOne4(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::ExaOneMoe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Olmo(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Olmo2(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Olmo3(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::StarCoder2(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::MiniCPM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::MiniCPM3(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::StableLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::SmolLM3(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Ministral3(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Nemotron(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Mamba(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Mamba2(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Jamba(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::NemotronH(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::NemotronNAS(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Step3p5(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::KimiLinear(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::LongcatFlash(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::LongcatFlashNgram(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Rwkv7(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::RecurrentGemma(inner) => LanguageModel::$method(inner, $($arg),*),
        }
    };
}

// Keep the embedding-aware subset in one place as well. These variants
// implement custom token-embedding or embedding-prefill behavior that VLM
// flows depend on.
macro_rules! delegate_embedding_language_model {
    ($self:expr, $method:ident ( $($arg:expr),* $(,)? ); fallback = $fallback:expr) => {
        match $self {
            LoadedModel::Gemma3(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma3VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Llama(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen2(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Llama4(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Llama4VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::LlavaVLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen2VL(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen25VL(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen3VL(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen3VLMoe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen35(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen35VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen35Moe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Qwen35MoeVLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma3n(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma3nVLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Phi3VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Molmo2VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Cohere2(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma2(inner) => LanguageModel::$method(inner, $($arg),*),
            _ => $fallback,
        }
    };
}

macro_rules! with_qwen_vl_model {
    ($self:expr, $model:ident => $expr:expr, else $fallback:expr) => {
        match $self {
            LoadedModel::Qwen2VL($model) => $expr,
            LoadedModel::Qwen25VL($model) => $expr,
            LoadedModel::Qwen3VL($model) => $expr,
            LoadedModel::Qwen3VLMoe($model) => $expr,
            LoadedModel::Qwen35VLM($model) | LoadedModel::Qwen35MoeVLM($model) => $expr,
            _ => $fallback,
        }
    };
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

    pub fn qwen_vl_prompt_info(&self) -> Option<qwen_vl::QwenVlmPromptInfo<'_>> {
        with_qwen_vl_model!(
            self,
            model => Some(qwen_vl::QwenVlmPromptInfo {
                processor: &model.processor,
                spatial_merge_size: model.spatial_merge_size,
                vision_start_token_id: model.vision_start_token_id,
                image_token_id: model.image_token_id,
            }),
            else None
        )
    }

    pub fn qwen_vl_input_embeddings(
        &self,
        input_ids: &mlxcel_core::MlxArray,
        pixel_values: &mlxcel_core::MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> Option<vision::merge::InputEmbeddings> {
        with_qwen_vl_model!(
            self,
            model => Some(model.get_input_embeddings(input_ids, pixel_values, grid_thw)),
            else None
        )
    }

    pub fn image_token_block_info(&self) -> Option<vlm_prompt::ImageTokenBlockInfo> {
        if let Some(g3n) = self.gemma3n_vl_model() {
            Some(vlm_prompt::ImageTokenBlockInfo {
                use_boi_eoi: true,
                image_token_id: g3n.image_token_id,
                mm_tokens_per_image: 256,
                boi_token_id: g3n.boi_token_id,
                eoi_token_id: g3n.eoi_token_id,
            })
        } else {
            self.vision_module()
                .map(|vm| vlm_prompt::ImageTokenBlockInfo {
                    use_boi_eoi: vm.boi_token_id != 0,
                    image_token_id: vm.image_token_id,
                    mm_tokens_per_image: vm.mm_tokens_per_image,
                    boi_token_id: vm.boi_token_id,
                    eoi_token_id: vm.eoi_token_id,
                })
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
        delegate_language_model!(self, num_layers())
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        delegate_language_model!(self, eos_token_ids())
    }

    fn make_caches(&self) -> Vec<mlxcel_core::layers::KVCache> {
        // Use fully qualified syntax to ensure we call the trait method, not inherent methods
        delegate_language_model!(self, make_caches())
    }

    fn forward(
        &self,
        input_ids: &mlxcel_core::MlxArray,
        caches: &mut [mlxcel_core::layers::KVCache],
        mask: Option<&mlxcel_core::MlxArray>,
    ) -> UniquePtr<mlxcel_core::MlxArray> {
        delegate_language_model!(self, forward(input_ids, caches, mask))
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &mlxcel_core::MlxArray,
        input_embeddings: Option<&mlxcel_core::MlxArray>,
        caches: &mut [mlxcel_core::layers::KVCache],
        mask: Option<&mlxcel_core::MlxArray>,
    ) -> UniquePtr<mlxcel_core::MlxArray> {
        delegate_embedding_language_model!(
            self,
            forward_with_embeddings(input_ids, input_embeddings, caches, mask);
            fallback = self.forward(input_ids, caches, mask)
        )
    }

    fn embed_tokens(
        &self,
        input_ids: &mlxcel_core::MlxArray,
    ) -> Option<UniquePtr<mlxcel_core::MlxArray>> {
        delegate_embedding_language_model!(self, embed_tokens(input_ids); fallback = None)
    }
}

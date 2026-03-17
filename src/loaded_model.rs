// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Unified loaded-model storage and capability dispatch.
//!
//! `LoadedModel` is the control-plane bridge between model loading and runtime
//! execution. New model variants are stored here once, then exposed through a
//! small set of dispatch/capability helpers instead of ad hoc matches in CLI,
//! server, or multimodal code.
//!
//! Responsibilities:
//! - store every loaded text/VLM family in one enum
//! - forward `LanguageModel` methods through centralized macros
//! - expose capability-oriented helpers used by multimodal paths
//!
//! Rationale:
//! - keep model-family wiring explicit and exhaustively matchable
//! - expose stable control-plane capabilities without erasing family identity
//! - centralize the retest surface for new model variants
//!
//! When adding a new variant, update the dispatch macros below before adding
//! any one-off match blocks elsewhere.

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
    MiniCPMOVLM(vision::MiniCPMOVLModel),
    Moondream3VLM(vision::Moondream3VLModel),
    Gemma3n(models::Gemma3nModel),
    Gemma3nVLM(vision::Gemma3nVLModel),
    Phi(models::PhiModel),
    Phi3(models::Phi3Model),
    Phi4MMVLM(vision::Phi4MMVLModel),
    Phi4SigLipVLM(vision::Phi4SigLipVLModel),
    Phi3VLM(vision::Phi3VLModel),
    Molmo2VLM(vision::Molmo2VLModel),
    Phi3Small(models::Phi3SmallModel),
    PhiMoe(models::PhiMoeModel),
    GptOss(models::GptOssWrapper),
    MiniMax(models::MiniMaxModel),
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
    SolarOpen(models::SolarOpenModel),
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
            LoadedModel::MiniCPMOVLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Moondream3VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma3n(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Gemma3nVLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Phi(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Phi3(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Phi4MMVLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Phi4SigLipVLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Phi3VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Molmo2VLM(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::Phi3Small(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::PhiMoe(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::GptOss(inner) => LanguageModel::$method(inner, $($arg),*),
            LoadedModel::MiniMax(inner) => LanguageModel::$method(inner, $($arg),*),
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
            LoadedModel::SolarOpen(inner) => LanguageModel::$method(inner, $($arg),*),
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
        delegate_language_model!(
            self,
            forward_with_embeddings(input_ids, input_embeddings, caches, mask)
        )
    }

    fn embed_tokens(
        &self,
        input_ids: &mlxcel_core::MlxArray,
    ) -> Option<UniquePtr<mlxcel_core::MlxArray>> {
        delegate_language_model!(self, embed_tokens(input_ids))
    }

    fn after_prefill(&self) {
        delegate_language_model!(self, after_prefill())
    }

    fn supports_batching(&self) -> bool {
        delegate_language_model!(self, supports_batching())
    }

    fn forward_batched(
        &self,
        input_ids: &mlxcel_core::MlxArray,
        batch_caches: &mut [&mut [mlxcel_core::layers::KVCache]],
        mask: Option<&mlxcel_core::MlxArray>,
    ) -> UniquePtr<mlxcel_core::MlxArray> {
        delegate_language_model!(self, forward_batched(input_ids, batch_caches, mask))
    }
}

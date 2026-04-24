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

//! Model implementations for mlxcel
//!
//! All implementations use mlxcel-core for direct MLX C++ bindings.

mod detection;
mod gemma3n_helpers;
mod llama4_helpers;
mod model_owned;
mod sanitize;

// Shared modules
pub mod gated_delta;
pub mod switch_layers;

// Model implementations (mlxcel-core based)
pub mod baichuan;
pub mod cohere;
pub mod cohere2;
pub mod deepseek;
pub mod deepseek_v2;
pub mod deepseek_v3;
pub mod deepseek_v32;
pub mod ernie4_5;
pub mod ernie4_5_moe;
pub mod exaone;
pub mod exaone4;
pub mod exaone_moe;
pub mod gemma;
pub mod gemma2;
pub mod gemma3;
pub mod gemma3n;
pub mod gemma4;
pub mod glm4;
pub mod glm4_moe;
pub mod glm4_moe_lite;
pub mod glm_moe_dsa;
pub mod gpt_oss;
pub mod hunyuan_moe;
pub mod hunyuan_v1_dense;
pub mod internlm2;
pub mod internlm3;
pub mod jamba;
pub mod kimi_linear;
pub mod llama3;
pub mod llama4;
pub mod longcat_flash_ngram;
pub mod mamba;
pub mod mamba2;
pub mod mimo;
pub mod minicpm;
pub mod minicpm3;
pub mod minimax;
pub mod ministral3;
pub mod mistral4;
pub mod mixtral;
pub mod molmo2;
pub mod molmo_point;
pub mod moondream3;
pub mod nemotron;
pub mod nemotron_h;
pub mod nemotron_nas;
pub mod olmo;
pub mod olmo2;
pub mod olmo3;
pub mod olmoe;
pub mod phi;
pub mod phi3;
pub mod phi3small;
pub mod phi4mm;
pub mod phimoe;
pub mod qwen2;
pub mod qwen2_moe;
pub mod qwen2_vl;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_moe;
pub mod qwen3_next;
pub mod qwen3_vl;
pub mod qwen3_vl_moe;
pub mod recurrent_gemma;
pub mod rwkv7;
pub mod smollm3;
pub mod solar_open;
pub mod stablelm;
pub mod starcoder2;
pub mod step3p5;

// Re-export model types
pub use baichuan::BaichuanModel;
pub use cohere::CohereModel;
pub use cohere2::Cohere2Model;
pub use deepseek::DeepSeekModel;
pub use deepseek_v2::DeepSeekV2Model;
pub use deepseek_v3::DeepSeekV3Model;
pub use deepseek_v32::DeepSeekV32Model;
pub use detection::get_model_type;
pub use ernie4_5::Ernie45Model;
pub use ernie4_5_moe::Ernie45MoeModel;
pub use exaone::ExaOneModel;
pub use exaone_moe::ExaoneMoeModel;
pub use exaone4::{ExaOne4Model, ExaOne4Wrapper};
pub use gemma::GemmaModel;
pub use gemma2::Gemma2Model;
pub use gemma3::{Gemma3Model, Gemma3Wrapper};
pub use gemma3n::Gemma3nModel;
pub use gemma4::{Gemma4Model, Gemma4Wrapper};
pub use glm_moe_dsa::GlmMoeDsaModel;
pub use glm4::Glm4Model;
pub use glm4_moe::Glm4MoeModel;
pub use glm4_moe_lite::Glm4MoeLiteModel;
pub use gpt_oss::{GptOssModel, GptOssWrapper};
pub use hunyuan_moe::HunyuanMoeModel;
pub use hunyuan_v1_dense::HunyuanV1DenseModel;
pub use internlm2::InternLM2Model;
pub use internlm3::InternLM3Model;
pub use jamba::JambaModel;
pub use kimi_linear::KimiLinearModel;
pub use llama3::Llama3Model;
pub use llama4::{Llama4CxxModel, Llama4Wrapper};
pub use longcat_flash_ngram::LongcatFlashNgramModel;
pub use mamba::MambaModel;
pub use mamba2::Mamba2Model;
pub use mimo::MiMoModel;
pub use minicpm::MiniCPMModel;
pub use minicpm3::MiniCPM3Model;
pub use minimax::MiniMaxModel;
pub use ministral3::{Ministral3Model, Ministral3Wrapper};
pub use mistral4::Mistral4Model;
pub use mixtral::MixtralModel;
pub use molmo2::Molmo2Model;
pub use moondream3::Moondream3Model;
pub use nemotron::NemotronModel;
pub use nemotron_h::NemotronHModel;
pub use nemotron_nas::NemotronNASModel;
pub use olmo::OlmoModel;
pub use olmo2::OLMo2Model;
pub use olmo3::OLMo3Model;
pub use olmoe::OlmoeModel;
pub use phi::PhiModel;
pub use phi3::Phi3Model;
pub use phi3small::Phi3SmallModel;
pub use phi4mm::Phi4MMModel;
pub use phimoe::PhiMoeModel;
pub use qwen2::Qwen2Model;
pub use qwen2_moe::Qwen2MoeModel;
pub use qwen2_vl::Qwen2VLModel;
pub use qwen3::Qwen3Model;
pub use qwen3_5::Qwen35Model;
pub use qwen3_moe::Qwen3MoeModel;
pub use qwen3_next::Qwen3NextModel;
pub use qwen3_vl::Qwen3VLModel;
pub use qwen3_vl_moe::Qwen3VLMoeModel;
pub use recurrent_gemma::GriffinModel;
pub use rwkv7::Rwkv7;
pub use sanitize::{
    convert_bf16_weights, load_and_sanitize_weights, sanitize_config_json,
    sanitize_tied_embeddings, warn_bf16_precision,
};
pub(crate) use sanitize::{
    Gemma4WeightBacking, load_gemma4_text_weights_with_backing,
    load_gemma4_vlm_weights_with_backing,
};
pub use smollm3::SmolLM3Model;
pub use solar_open::SolarOpenModel;
pub use stablelm::StableLMModel;
pub use starcoder2::StarCoder2Model;
pub use step3p5::Step3p5Model;

/// Supported model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    // Standard Transformer models
    Llama,         // Llama 1/2/3, Mistral
    Llama4,        // Llama 4 (MoE)
    Llama4VLM,     // Llama 4 VLM (vision-language)
    Qwen2,         // Qwen 2/2.5
    Qwen3,         // Qwen 3
    Qwen3Moe,      // Qwen 3 MoE
    Qwen3Next,     // Qwen 3 with GatedDeltaNet
    Qwen35,        // Qwen 3.5 Hybrid (Transformer + GatedDeltaNet)
    Qwen35VLM,     // Qwen 3.5 VLM (Qwen3-VL vision + Qwen3.5 hybrid text)
    Qwen35Moe,     // Qwen 3.5 MoE Hybrid
    Qwen35MoeVLM,  // Qwen 3.5 MoE VLM
    Gemma,         // Gemma 1
    Gemma2,        // Gemma 2
    Gemma3,        // Gemma 3 (text-only)
    Gemma4,        // Gemma 4 text-only route
    Gemma3VLM,     // Gemma 3 VLM (vision-language)
    Gemma4VLM,     // Gemma 4 VLM (vision-language)
    LlavaVLM,      // LLaVA (CLIP/SigLIP + Llama/Qwen2)
    LlavaBunnyVLM, // LLaVA-Bunny (SigLIP + Qwen2)
    AyaVisionVLM,  // Aya Vision (SigLIP + Cohere2)
    PaliGemmaVLM,  // PaliGemma (SigLIP + Gemma)
    PixtralVLM,    // Pixtral (ViT w/ 2D RoPE + Mistral)
    Mistral3VLM,   // Mistral 3 VLM (Pixtral ViT + PatchMerger + Mistral)
    Qwen2VL,       // Qwen2-VL (custom ViT + Qwen2 w/ MRoPE)
    Qwen25VL,      // Qwen2.5-VL (windowed ViT + Qwen2 w/ MRoPE)
    Qwen3VL,       // Qwen3-VL (ViT + interleaved MRoPE + DeepStack)
    Qwen3VLMoe,    // Qwen3-VL-MoE (Qwen3-VL + MoE text backbone)
    MiniCPMOVLM,   // MiniCPM-o (dynamic SigLIP + resampler + Qwen3-VL text)
    Moondream3VLM, // Moondream3 (custom ViT + custom text decoder, query/caption image path)
    Gemma3n,       // Gemma 3n (text-only)
    Gemma3nVLM,    // Gemma 3n VLM (MobileNetV5 + Gemma3n)
    Phi,           // Phi 1/2
    Phi3,          // Phi 3
    Phi4MMVLM,     // Phi-4 Multimodal (SigLIP2 NaFlex + Phi4 text, image path only)
    Phi4SigLipVLM, // Phi-4 reasoning vision (SigLIP2 NaFlex + Phi3-style text)
    Phi3VLM,       // Phi 3.5 Vision (CLIP + Phi3)
    Molmo2VLM,     // Molmo2 (custom ViT + attention pooling + Molmo2 text)
    MolmoPointVLM, // Molmo-Point (custom ViT + point prediction + Molmo2 text)
    Phi3Small,     // Phi 3 Small
    PhiMoe,        // Phi MoE

    // MoE models
    GptOss,
    MiniMax,
    Mixtral,
    Qwen2Moe,
    OLMoE,

    // DeepSeek family
    DeepSeek,
    DeepSeekV2,
    DeepSeekV3,
    DeepSeekV32,

    // Cohere family
    Cohere,
    Cohere2,

    // Chinese/Asian models
    InternLM2,
    InternLM3,
    Baichuan,
    Glm4,
    Glm4Moe,
    Glm4MoeLite,
    GlmMoeDsa,
    Ernie45,
    Ernie45Moe,
    HunyuanMoe,
    HunyuanV1Dense,
    MiMo,

    // Korean models
    ExaOne,
    ExaOne4,
    ExaOneMoe,
    SolarOpen,

    // OLMo family
    Olmo,
    Olmo2,
    Olmo3,

    // Code models
    StarCoder2,

    // Other Transformer models
    MiniCPM,
    MiniCPM3,
    StableLM,
    SmolLM3,
    Ministral3,
    Mistral3,
    Mistral4,
    Nemotron,

    // SSM/Mamba models
    Mamba,
    Mamba2,
    Jamba,
    NemotronH,
    NemotronNAS,

    // Kimi models
    KimiLinear,

    // Longcat models
    LongcatFlash,
    LongcatFlashNgram,

    // Step models
    Step3p5,

    // RNN models
    Rwkv7,
    RecurrentGemma,
}
#[cfg(test)]
#[path = "detection_tests.rs"]
mod detection_tests;

#[cfg(test)]
#[path = "gemma3n_helpers_tests.rs"]
mod gemma3n_helpers_tests;

#[cfg(test)]
#[path = "gemma4_tests.rs"]
mod gemma4_tests;

#[cfg(test)]
#[path = "llama4_helpers_tests.rs"]
mod llama4_helpers_tests;

#[cfg(test)]
#[path = "sanitize_tests.rs"]
mod sanitize_tests;

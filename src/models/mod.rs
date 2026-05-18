//! Model implementations for mlxcel
//!
//! All implementations use mlxcel-core for direct MLX C++ bindings.

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
pub mod glm4;
pub mod glm4_moe;
pub mod glm4_moe_lite;
pub mod glm_moe_dsa;
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
pub mod ministral3;
pub mod mixtral;
pub mod molmo2;
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
pub use ernie4_5::Ernie45Model;
pub use ernie4_5_moe::Ernie45MoeModel;
pub use exaone::ExaOneModel;
pub use exaone_moe::ExaoneMoeModel;
pub use exaone4::{ExaOne4Model, ExaOne4Wrapper};
pub use gemma::GemmaModel;
pub use gemma2::Gemma2Model;
pub use gemma3::{Gemma3Model, Gemma3Wrapper};
pub use gemma3n::Gemma3nModel;
pub use glm_moe_dsa::GlmMoeDsaModel;
pub use glm4::Glm4Model;
pub use glm4_moe::Glm4MoeModel;
pub use glm4_moe_lite::Glm4MoeLiteModel;
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
pub use ministral3::{Ministral3Model, Ministral3Wrapper};
pub use mixtral::MixtralModel;
pub use molmo2::Molmo2Model;
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
pub use smollm3::SmolLM3Model;
pub use stablelm::StableLMModel;
pub use starcoder2::StarCoder2Model;
pub use step3p5::Step3p5Model;

use anyhow::Result;
use std::path::Path;

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
    Gemma3VLM,     // Gemma 3 VLM (vision-language)
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
    Gemma3n,       // Gemma 3n (text-only)
    Gemma3nVLM,    // Gemma 3n VLM (MobileNetV5 + Gemma3n)
    Phi,           // Phi 1/2
    Phi3,          // Phi 3
    Phi3VLM,       // Phi 3.5 Vision (CLIP + Phi3)
    Molmo2VLM,     // Molmo2 (custom ViT + attention pooling + Molmo2 text)
    Phi3Small,     // Phi 3 Small
    PhiMoe,        // Phi MoE

    // MoE models
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

/// Ensure lm_head weights exist for models with tied embeddings.
///
/// Many models share embedding weights for the output projection (lm_head).
/// When tie_word_embeddings is true (or omitted), lm_head.weight may not be
/// saved in safetensors. This function auto-detects the missing weight and
/// copies model.embed_tokens.* → lm_head.* so model loaders work uniformly.
///
/// Auto-detection: if tie_word_embeddings is explicitly false, do nothing.
/// Otherwise (true or absent), copy if lm_head.weight is missing.
///
/// Used by: all VLM loaders, load_model_from_weights, load_and_sanitize_weights
pub fn sanitize_tied_embeddings(
    weights: &mut mlxcel_core::weights::WeightMap,
    config: &serde_json::Value,
) {
    // If tie_word_embeddings is explicitly false, skip
    let tie = config
        .get("tie_word_embeddings")
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("tie_word_embeddings"))
        })
        .and_then(|v| v.as_bool());

    if tie == Some(false) {
        return;
    }

    // Pattern 1: standard keys (after language_model. prefix stripping)
    // model.embed_tokens.* → lm_head.*
    if !weights.contains_key("lm_head.weight") {
        for suffix in &["weight", "scales", "biases"] {
            let src = format!("model.embed_tokens.{}", suffix);
            let dst = format!("lm_head.{}", suffix);
            if let Some(w) = weights.get(&src) {
                weights.insert(dst, mlxcel_core::copy(w));
            }
        }
    }

    // Pattern 2: VLM keys with language_model. prefix (not stripped)
    // language_model.model.embed_tokens.* → language_model.lm_head.*
    if !weights.contains_key("language_model.lm_head.weight") {
        for suffix in &["weight", "scales", "biases"] {
            let src = format!("language_model.model.embed_tokens.{}", suffix);
            let dst = format!("language_model.lm_head.{}", suffix);
            if let Some(w) = weights.get(&src) {
                weights.insert(dst, mlxcel_core::copy(w));
            }
        }
    }
}

/// Load weights from a model directory with automatic tied-embedding sanitization.
///
/// This is the common weight loading entry point for text model `load()` functions.
/// It reads safetensors, parses config.json, and ensures lm_head weights exist.
pub fn load_and_sanitize_weights<P: AsRef<std::path::Path>>(
    model_dir: P,
) -> Result<mlxcel_core::weights::WeightMap, String> {
    let model_dir = model_dir.as_ref();
    let mut weights = mlxcel_core::weights::load_weights_from_dir(model_dir)?;

    let config_path = model_dir.join("config.json");
    if let Ok(config_str) = std::fs::read_to_string(&config_path) {
        let config_str = sanitize_config_json(&config_str);
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
            sanitize_tied_embeddings(&mut weights, &config);
        }
    }

    Ok(weights)
}

/// Sanitize config JSON string by replacing non-standard JSON values
pub fn sanitize_config_json(config_str: &str) -> String {
    config_str
        .replace("Infinity", "1e38")
        .replace("-Infinity", "-1e38")
        .replace("NaN", "0.0")
}

/// Detect model type from config.json
pub fn get_model_type(model_path: &Path) -> Result<ModelType> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(config_path)?;
    let config_str = sanitize_config_json(&config_str);
    let v: serde_json::Value = serde_json::from_str(&config_str)?;

    let model_type = v["model_type"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("model_type not found"))?;

    match model_type {
        "llama" | "mistral" => Ok(ModelType::Llama),
        "llama4" => {
            // Detect VLM: has vision_config in config.json
            if v.get("vision_config").is_some() {
                Ok(ModelType::Llama4VLM)
            } else {
                Ok(ModelType::Llama4)
            }
        }
        "qwen2" => Ok(ModelType::Qwen2),
        "qwen3" => Ok(ModelType::Qwen3),
        "qwen3_moe" => Ok(ModelType::Qwen3Moe),
        "qwen3_next" | "qwen3next" => Ok(ModelType::Qwen3Next),
        "qwen3_5" => {
            // Detect VLM: has vision_config in config.json
            if v.get("vision_config").is_some() {
                Ok(ModelType::Qwen35VLM)
            } else {
                Ok(ModelType::Qwen35)
            }
        }
        "qwen3_5_moe" => {
            if v.get("vision_config").is_some() {
                Ok(ModelType::Qwen35MoeVLM)
            } else {
                Ok(ModelType::Qwen35Moe)
            }
        }
        "qwen2_moe" => Ok(ModelType::Qwen2Moe),
        "gemma" => Ok(ModelType::Gemma),
        "gemma2" => Ok(ModelType::Gemma2),
        "gemma3" | "gemma3_text" => {
            // Detect VLM: has vision_config in config.json
            if v.get("vision_config").is_some() {
                Ok(ModelType::Gemma3VLM)
            } else {
                Ok(ModelType::Gemma3)
            }
        }
        "gemma3n" | "gemma3n_text" => {
            // Detect VLM: has vision_config in config.json
            if v.get("vision_config").is_some() {
                Ok(ModelType::Gemma3nVLM)
            } else {
                Ok(ModelType::Gemma3n)
            }
        }
        "phi" | "phi-msft" => Ok(ModelType::Phi),
        "phi3" => Ok(ModelType::Phi3),
        "phi3_v" => Ok(ModelType::Phi3VLM),
        "phi3small" => Ok(ModelType::Phi3Small),
        "phimoe" => Ok(ModelType::PhiMoe),
        "mixtral" => Ok(ModelType::Mixtral),
        "olmoe" => Ok(ModelType::OLMoE),
        "deepseek" => Ok(ModelType::DeepSeek),
        "deepseek_v2" => Ok(ModelType::DeepSeekV2),
        "deepseek_v3" => Ok(ModelType::DeepSeekV3),
        "deepseek_v32" | "deepseek_v3.2" => Ok(ModelType::DeepSeekV32),
        "cohere" => Ok(ModelType::Cohere),
        "cohere2" => Ok(ModelType::Cohere2),
        "internlm2" => Ok(ModelType::InternLM2),
        "internlm3" => Ok(ModelType::InternLM3),
        "baichuan_m1" => Ok(ModelType::Baichuan),
        "glm4" => Ok(ModelType::Glm4),
        "glm4_moe" => Ok(ModelType::Glm4Moe),
        "glm4_moe_lite" => Ok(ModelType::Glm4MoeLite),
        "glm_moe_dsa" => Ok(ModelType::GlmMoeDsa),
        "ernie4_5" | "ernie4.5" => Ok(ModelType::Ernie45),
        "ernie4_5_moe" | "ernie4.5_moe" => Ok(ModelType::Ernie45Moe),
        "hunyuan_v1_dense" | "hunyuan_dense" => Ok(ModelType::HunyuanV1Dense),
        "hunyuan" => {
            // Detect MoE vs Dense based on num_experts
            let num_experts = v["num_experts"].as_i64().unwrap_or(1);
            if num_experts > 1 {
                Ok(ModelType::HunyuanMoe)
            } else {
                Ok(ModelType::HunyuanV1Dense)
            }
        }
        "mimo" => Ok(ModelType::MiMo),
        "exaone" => Ok(ModelType::ExaOne),
        "exaone4" => Ok(ModelType::ExaOne4),
        "exaone_moe" => Ok(ModelType::ExaOneMoe),
        "olmo" => Ok(ModelType::Olmo),
        "olmo2" => Ok(ModelType::Olmo2),
        "olmo3" => Ok(ModelType::Olmo3),
        "starcoder2" => Ok(ModelType::StarCoder2),
        "minicpm" => Ok(ModelType::MiniCPM),
        "minicpm3" => Ok(ModelType::MiniCPM3),
        "stablelm" => Ok(ModelType::StableLM),
        "smollm3" => Ok(ModelType::SmolLM3),
        "ministral3" => Ok(ModelType::Ministral3),
        "mistral3" => {
            if v.get("vision_config").is_some() {
                Ok(ModelType::Mistral3VLM)
            } else {
                Ok(ModelType::Mistral3)
            }
        }
        "nemotron" => Ok(ModelType::Nemotron),
        "mamba" | "falcon_mamba" => Ok(ModelType::Mamba),
        "mamba2" => Ok(ModelType::Mamba2),
        "jamba" => Ok(ModelType::Jamba),
        "nemotron_h" => Ok(ModelType::NemotronH),
        "nemotron-nas" => Ok(ModelType::NemotronNAS),
        "rwkv7" => Ok(ModelType::Rwkv7),
        "kimi_linear" => Ok(ModelType::KimiLinear),
        "longcat_flash" => Ok(ModelType::LongcatFlash),
        "longcat_flash_ngram" => Ok(ModelType::LongcatFlashNgram),
        "step3p5" => Ok(ModelType::Step3p5),
        "recurrent_gemma" | "griffin" => Ok(ModelType::RecurrentGemma),
        "qwen2_vl" => Ok(ModelType::Qwen2VL),
        "qwen2_5_vl" => Ok(ModelType::Qwen25VL),
        "qwen3_vl" => Ok(ModelType::Qwen3VL),
        "qwen3_vl_moe" => Ok(ModelType::Qwen3VLMoe),
        "llava" | "llava_next" => Ok(ModelType::LlavaVLM),
        "llava_bunny" | "bunny-llama" | "llava-qwen2" => Ok(ModelType::LlavaBunnyVLM),
        "aya_vision" => Ok(ModelType::AyaVisionVLM),
        "paligemma" => Ok(ModelType::PaliGemmaVLM),
        "pixtral" => Ok(ModelType::PixtralVLM),
        "molmo2" => Ok(ModelType::Molmo2VLM),
        _ => Err(anyhow::anyhow!("Unsupported model type: {}", model_type)),
    }
}

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use models::sanitize_config_json;

#[path = "loader_vlm_llava.rs"]
mod llava;
#[path = "loader_vlm_qwen.rs"]
mod qwen;

use llava::infer_llama_config_from_weights;
pub(crate) use llava::{load_llava_bunny_vlm, load_llava_vlm};
pub(crate) use qwen::{
    load_qwen2_5_vl, load_qwen2_vl, load_qwen3_5_moe_vlm, load_qwen3_5_vlm, load_qwen3_vl,
    load_qwen3_vl_moe,
};

fn read_sanitized_vlm_config(model_path: &Path) -> Result<(String, Value)> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let full_config: Value = parse_vlm_config(&config_str, "config")?;
    Ok((config_str, full_config))
}

fn parse_vlm_config<T: DeserializeOwned>(config_str: &str, label: &str) -> Result<T> {
    serde_json::from_str(config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", label, e))
}

fn parse_required_vlm_subconfig<T: DeserializeOwned>(
    full_config: &Value,
    key: &str,
    label: &str,
) -> Result<T> {
    let value = full_config
        .get(key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing {} in config.json", key))?;
    serde_json::from_value(value).map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", label, e))
}

fn load_vlm_weights(model_path: &Path) -> Result<WeightMap> {
    mlxcel_core::weights::load_weights_from_dir(model_path).map_err(|e| anyhow::anyhow!("{}", e))
}

fn strip_language_model_prefix(raw_weights: WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("language_model.") {
            stripped.to_string()
        } else {
            key
        };
        weights.insert(new_key, value);
    }
    weights
}

trait QwenVisionConfigExt {
    fn quant_group_size(&self) -> i32;
    fn quant_bits(&self) -> i32;
    fn set_quant_group_size(&mut self, value: i32);
    fn set_quant_bits(&mut self, value: i32);
    fn patch_size(&self) -> usize;
    fn temporal_patch_size(&self) -> usize;
    fn spatial_merge_size(&self) -> usize;
}

impl QwenVisionConfigExt for vision::encoders::qwen2_vl::Qwen2VLVisionConfig {
    fn quant_group_size(&self) -> i32 {
        self.quant_group_size
    }

    fn quant_bits(&self) -> i32 {
        self.quant_bits
    }

    fn set_quant_group_size(&mut self, value: i32) {
        self.quant_group_size = value;
    }

    fn set_quant_bits(&mut self, value: i32) {
        self.quant_bits = value;
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }
}

impl QwenVisionConfigExt for vision::encoders::qwen2_5_vl::Qwen25VLVisionConfig {
    fn quant_group_size(&self) -> i32 {
        self.quant_group_size
    }

    fn quant_bits(&self) -> i32 {
        self.quant_bits
    }

    fn set_quant_group_size(&mut self, value: i32) {
        self.quant_group_size = value;
    }

    fn set_quant_bits(&mut self, value: i32) {
        self.quant_bits = value;
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }
}

impl QwenVisionConfigExt for vision::encoders::qwen3_vl::Qwen3VLVisionConfig {
    fn quant_group_size(&self) -> i32 {
        self.quant_group_size
    }

    fn quant_bits(&self) -> i32 {
        self.quant_bits
    }

    fn set_quant_group_size(&mut self, value: i32) {
        self.quant_group_size = value;
    }

    fn set_quant_bits(&mut self, value: i32) {
        self.quant_bits = value;
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }
}

fn inherit_qwen_vision_quantization<T: QwenVisionConfigExt>(
    vision_config: &mut T,
    full_config: &Value,
) {
    if (vision_config.quant_group_size() == 0 || vision_config.quant_bits() == 0)
        && let Some(q) = full_config.get("quantization")
    {
        vision_config.set_quant_group_size(
            q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32,
        );
        vision_config.set_quant_bits(q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32);
    }
}

trait QwenTextQuantizationExt {
    fn has_quantization(&self) -> bool;
    fn set_quantization_from_value(&mut self, value: &Value);
}

impl QwenTextQuantizationExt for models::qwen3_vl::Qwen3VLConfig {
    fn has_quantization(&self) -> bool {
        self.quantization.is_some()
    }

    fn set_quantization_from_value(&mut self, value: &Value) {
        self.quantization = serde_json::from_value(value.clone()).ok();
    }
}

impl QwenTextQuantizationExt for models::qwen3_vl_moe::Qwen3VLMoeConfig {
    fn has_quantization(&self) -> bool {
        self.quantization.is_some()
    }

    fn set_quantization_from_value(&mut self, value: &Value) {
        self.quantization = serde_json::from_value(value.clone()).ok();
    }
}

fn inherit_qwen_text_quantization<T: QwenTextQuantizationExt>(
    text_config: &mut T,
    full_config: &Value,
) {
    if !text_config.has_quantization()
        && let Some(q) = full_config.get("quantization")
    {
        text_config.set_quantization_from_value(q);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QwenVisionTokenIds {
    image_token_id: i32,
    video_token_id: i32,
    vision_start_token_id: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35VlmVariant {
    Dense,
    Moe,
}

fn qwen_vl_token_ids(full_config: &Value, defaults: QwenVisionTokenIds) -> QwenVisionTokenIds {
    QwenVisionTokenIds {
        image_token_id: full_config
            .get("image_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(defaults.image_token_id as i64) as i32,
        video_token_id: full_config
            .get("video_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(defaults.video_token_id as i64) as i32,
        vision_start_token_id: full_config
            .get("vision_start_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(defaults.vision_start_token_id as i64) as i32,
    }
}

fn qwen35_vlm_token_defaults() -> QwenVisionTokenIds {
    QwenVisionTokenIds {
        image_token_id: 248056,
        video_token_id: 248057,
        vision_start_token_id: 248045,
    }
}

fn wrap_qwen35_vlm(vlm: vision::Qwen35VLModel, variant: Qwen35VlmVariant) -> LoadedModel {
    match variant {
        Qwen35VlmVariant::Dense => LoadedModel::Qwen35VLM(vlm),
        Qwen35VlmVariant::Moe => LoadedModel::Qwen35MoeVLM(vlm),
    }
}

fn qwen_vl_processor<T: QwenVisionConfigExt>(
    vision_config: &T,
) -> vision::processors::qwen2_vl::Qwen2VLProcessor {
    vision::processors::qwen2_vl::Qwen2VLProcessor::new(
        vision_config.patch_size(),
        vision_config.temporal_patch_size(),
        vision_config.spatial_merge_size(),
    )
}

fn qwen_vl_processor_with_norm<T: QwenVisionConfigExt>(
    vision_config: &T,
    mean: [f32; 3],
    std: [f32; 3],
) -> vision::processors::qwen2_vl::Qwen2VLProcessor {
    vision::processors::qwen2_vl::Qwen2VLProcessor::new_with_norm(
        vision_config.patch_size(),
        vision_config.temporal_patch_size(),
        vision_config.spatial_merge_size(),
        mean,
        std,
    )
}

fn rewrite_qwen3_vl_weight_key(key: String, moe_experts: bool) -> String {
    let new_key = if key.starts_with("model.language_model.") {
        key.replacen("model.language_model.", "model.", 1)
    } else if key.starts_with("model.visual.") {
        key.replacen("model.visual.", "vision_tower.", 1)
    } else if key.starts_with("language_model.") {
        key.replacen("language_model.", "", 1)
    } else {
        key
    };

    if moe_experts && new_key.contains(".mlp.experts.") {
        let after_experts = new_key.rsplit_once(".mlp.experts.").unwrap().1;
        if !after_experts.contains('.') {
            format!(
                "{}.weight",
                new_key.replace(".mlp.experts.", ".mlp.switch_mlp.")
            )
        } else {
            new_key.replace(".mlp.experts.", ".mlp.switch_mlp.")
        }
    } else {
        new_key
    }
}

fn remap_qwen3_vl_weights(raw_weights: WeightMap, moe_experts: bool) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        weights.insert(rewrite_qwen3_vl_weight_key(key, moe_experts), value);
    }
    weights
}

/// Load a Gemma3 VLM model (text + vision tower + projector)
pub(crate) fn load_gemma3_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::config::VLMConfig;
    use vision::connectors::avg_pool::AvgPoolProjector;
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::processors::siglip::SigLipProcessor;

    // Parse full VLM config
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let vlm_config: VLMConfig = serde_json::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse VLM config: {}", e))?;

    // Parse text config as Gemma3 ModelArgs
    let text_config: models::gemma3::ModelArgs =
        serde_json::from_value(vlm_config.text_config.clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse text_config: {}", e))?;

    // Load all weights
    let raw_weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Remap VLM weight keys: strip "language_model." prefix for text model weights
    // VLM weights: "language_model.model.layers.X..." -> "model.layers.X..."
    // VLM weights: "language_model.lm_head..." -> "lm_head..."
    // Vision/projector weights keep their original prefix
    let mut weights = mlxcel_core::weights::WeightMap::new();
    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("language_model.") {
            stripped.to_string()
        } else {
            key
        };
        weights.insert(new_key, value);
    }

    // Sanitize tied embeddings: copy model.embed_tokens.* → lm_head.* if needed
    let config_value: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse config as Value: {}", e))?;
    models::sanitize_tied_embeddings(&mut weights, &config_value);

    // Build text model from weights with "model." prefix
    let text_model = models::Gemma3Model::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;
    let text_wrapper = models::Gemma3Wrapper::new(text_model);

    // Build vision encoder (SigLIP)
    let vision_encoder = SigLipVisionModel::from_weights(
        &weights,
        &vlm_config.vision_config,
        "vision_tower.vision_model",
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?;

    // Build connector (AvgPool projector)
    let mm_tokens_per_image = vlm_config.get_mm_tokens_per_image();
    let connector = AvgPoolProjector::from_weights(
        &weights,
        "multi_modal_projector",
        vlm_config.vision_config.hidden_size,
        vlm_config.vision_config.image_size,
        vlm_config.vision_config.patch_size,
        mm_tokens_per_image,
        vlm_config.vision_config.layer_norm_eps,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load projector: {}", e))?;

    // Build image processor
    let processor = SigLipProcessor::new(vlm_config.vision_config.image_size);

    // Assemble VisionModule
    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: vlm_config.image_token_index,
        pad_token_id: vlm_config.pad_token_id,
        hidden_size: if vlm_config.hidden_size > 0 {
            vlm_config.hidden_size
        } else {
            text_config.hidden_size
        },
        boi_token_id: vlm_config.boi_token_index,
        eoi_token_id: vlm_config.eoi_token_index,
        mm_tokens_per_image,
        merge_strategy: vision::MergeStrategy::Gemma3,
    };

    // Assemble VisionLanguageModel
    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(LoadedModel::Gemma3(text_wrapper)),
        vision: vision_module,
    };

    Ok(LoadedModel::Gemma3VLM(vlm))
}

pub(crate) fn load_gemma3n_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::gemma3n::load_gemma3n_vision;
    use vision::processors::siglip::SigLipProcessor;

    // Parse full config
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let config: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

    // Parse text config
    let top_args: models::gemma3n::ModelArgs = serde_json::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
    let text_config = top_args.text_args();

    // Vision config values
    let vision_hidden_size = config
        .get("vision_config")
        .and_then(|vc| vc.get("hidden_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;
    // Gemma3n MobileNetV5: 256x256 produces 16x16=256 patches matching vision_soft_tokens_per_image
    let image_size = config
        .get("vision_config")
        .and_then(|vc| vc.get("image_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as usize;
    let image_token_id = config
        .get("image_token_id")
        .or_else(|| config.get("image_token_index"))
        .and_then(|v| v.as_i64())
        .unwrap_or(262_145) as i32;
    let boi_token_id = config
        .get("boi_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(255_999) as i32;
    let eoi_token_id = config
        .get("eoi_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(262_144) as i32;
    let vision_rms_eps = config
        .get("vision_config")
        .and_then(|vc| vc.get("rms_norm_eps"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;

    // Load all weights
    let raw_weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Sanitize weights: strip "model." prefix and transpose conv2d weights
    let mut weights = mlxcel_core::weights::WeightMap::new();
    // Check if conv weights need transposing (key has "model." prefix in raw_weights)
    let needs_transpose = raw_weights
        .get("model.vision_tower.timm_model.blocks.0.0.conv_exp.weight")
        .map(|w| {
            let shape = mlxcel_core::array_shape(w);
            shape.len() == 4 && shape[1] > shape[2]
        })
        .unwrap_or(false);

    for (key, value) in raw_weights {
        // Strip "model." prefix if present (VLM weight layout)
        let new_key = if let Some(stripped) = key.strip_prefix("model.") {
            stripped.to_string()
        } else {
            key
        };

        // Transpose conv2d weights from PyTorch [O,I,H,W] to MLX [O,H,W,I]
        let value = if needs_transpose {
            let shape = mlxcel_core::array_shape(&value);
            if shape.len() == 4 {
                mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1])
            } else {
                mlxcel_core::copy(&value)
            }
        } else {
            mlxcel_core::copy(&value)
        };
        weights.insert(new_key, value);
    }

    // Sanitize tied embeddings
    models::sanitize_tied_embeddings(&mut weights, &config);

    // Auto-detect language model prefix:
    // Non-quantized: "language_model.embed_tokens.weight" → prefix "language_model"
    // Quantized (4-bit): "language_model.model.embed_tokens.weight" → prefix "language_model.model"
    let lm_prefix = if weights.contains_key("language_model.model.embed_tokens.weight") {
        "language_model.model"
    } else {
        "language_model"
    };
    let language_model =
        models::gemma3n::Gemma3nLanguageModel::from_weights(&weights, &text_config, lm_prefix)
            .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;

    let text_model = models::Gemma3nModel {
        language_model,
        config: text_config.clone(),
    };

    // Load vision tower (weights already sanitized)
    let vision_tower = load_gemma3n_vision(&weights, "vision_tower.timm_model")
        .map_err(|e| anyhow::anyhow!("Failed to load vision tower: {}", e))?;

    // Load multimodal embedder
    let group_size = text_config
        .quantization
        .as_ref()
        .map(|q| q.group_size as i32)
        .unwrap_or(64);
    let bits = text_config
        .quantization
        .as_ref()
        .map(|q| q.bits as i32)
        .unwrap_or(4);

    let embed_vision = models::gemma3n::Gemma3nMultimodalEmbedder::from_weights(
        &weights,
        "embed_vision",
        vision_hidden_size,
        text_config.hidden_size,
        vision_rms_eps,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load multimodal embedder: {}", e))?;

    // Image processor: Gemma3n uses rescale-only (no normalization)
    let processor = SigLipProcessor::new_rescale_only(image_size);

    // Assemble Gemma3nVLModel
    let vlm = vision::Gemma3nVLModel::new(
        text_model,
        vision_tower,
        embed_vision,
        processor,
        image_token_id,
        boi_token_id,
        eoi_token_id,
        vision_hidden_size,
    );

    Ok(LoadedModel::Gemma3nVLM(vlm))
}

pub(crate) fn load_phi3_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::config::VisionConfig;
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::processors::phi3_v::Phi3VProcessor;

    // Parse config.json
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let config: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

    // Parse text config as Phi3 ModelArgs
    let text_args: models::phi3::ModelArgs = serde_json::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse text config: {}", e))?;

    // Vision config (CLIP-ViT-Large-Patch14-336 defaults)
    let vision_config = VisionConfig {
        model_type: "phi3_v".to_string(),
        hidden_size: 1024,
        image_size: 336,
        intermediate_size: 4096,
        num_attention_heads: 16,
        num_hidden_layers: 24,
        num_channels: 3,
        patch_size: 14,
        layer_norm_eps: 1e-5,
    };

    let image_dim_out = vision_config.hidden_size; // 1024

    // Load weights
    let raw_weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Remap weights
    let mut weights = mlxcel_core::weights::WeightMap::new();
    for (key, value) in &raw_weights {
        // Skip position_ids
        if key.contains("position_ids") {
            continue;
        }

        let new_key = if let Some(rest) =
            key.strip_prefix("model.vision_embed_tokens.img_processor.")
        {
            // model.vision_embed_tokens.img_processor.vision_model.* → vision_tower.vision_model.*
            format!("vision_tower.{}", rest)
        } else if let Some(rest) = key.strip_prefix("model.vision_embed_tokens.img_projection.0.") {
            format!("img_projection.0.{}", rest)
        } else if let Some(rest) = key.strip_prefix("model.vision_embed_tokens.img_projection.2.") {
            format!("img_projection.2.{}", rest)
        } else if key == "model.vision_embed_tokens.glb_GN" {
            "glb_GN".to_string()
        } else if key == "model.vision_embed_tokens.sub_GN" {
            "sub_GN".to_string()
        } else {
            key.clone()
        };

        // Transpose patch_embedding.weight if needed
        let mapped_value = if new_key.contains("patch_embedding.weight") {
            let shape = mlxcel_core::array_shape(value);
            if shape.len() == 4 {
                let (out_ch, dim1, dim2, _dim3) = (shape[0], shape[1], shape[2], shape[3]);
                if !(out_ch >= dim1 && out_ch >= dim2 && dim1 == dim2) {
                    mlxcel_core::transpose_axes(value, &[0, 2, 3, 1])
                } else {
                    mlxcel_core::copy(value)
                }
            } else {
                mlxcel_core::copy(value)
            }
        } else {
            mlxcel_core::copy(value)
        };

        weights.insert(new_key, mapped_value);
    }

    // Sanitize tied embeddings
    models::sanitize_tied_embeddings(&mut weights, &config);

    // Load text model
    let text_model = models::Phi3Model::from_weights(&weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;

    // Load vision encoder (CLIP with penultimate layer, CLS dropped)
    let vision_encoder =
        SigLipVisionModel::from_weights(&weights, &vision_config, "vision_tower.vision_model")
            .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?
            .with_feature_selection(-2, "default".to_string());

    // Load GN tensors
    let glb_gn = weights
        .get("glb_GN")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("glb_GN weight not found"))?;
    let sub_gn = weights
        .get("sub_GN")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("sub_GN weight not found"))?;

    // Load projection MLP
    let group_size = text_args.group_size();
    let bits = text_args.bits();
    let img_proj_linear1 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "img_projection.0",
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load img_projection.0: {}", e))?;
    let img_proj_linear2 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "img_projection.2",
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load img_projection.2: {}", e))?;

    // Load num_crops from preprocessor_config.json
    let num_crops = {
        let pp_path = model_path.join("preprocessor_config.json");
        if pp_path.exists() {
            let pp_str = std::fs::read_to_string(&pp_path).unwrap_or_default();
            let pp: serde_json::Value = serde_json::from_str(&pp_str).unwrap_or_default();
            pp.get("num_crops").and_then(|v| v.as_u64()).unwrap_or(4) as usize
        } else {
            // Phi-3.5V default: num_crops=16
            config
                .get("vision_config")
                .and_then(|vc| vc.get("num_crops"))
                .and_then(|v| v.as_u64())
                .unwrap_or(16) as usize
        }
    };

    let processor = Phi3VProcessor::new(num_crops);

    let vlm = vision::Phi3VLModel {
        text_model,
        vision_encoder,
        glb_gn,
        sub_gn,
        img_proj_linear1,
        img_proj_linear2,
        processor,
        image_dim_out,
    };

    Ok(LoadedModel::Phi3VLM(vlm))
}

pub(crate) fn load_molmo2_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::molmo2::Molmo2VisionModel;
    use vision::processors::molmo2::Molmo2Processor;

    // Parse config.json
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let config: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

    // Parse text config
    let text_config_val = config.get("text_config").unwrap_or(&config);
    let text_config: models::molmo2::Molmo2TextConfig =
        serde_json::from_value(text_config_val.clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse text config: {}", e))?;

    // Parse vision config (nested: vision_config.vit_config, vision_config.adapter_config)
    let vision_config = config.get("vision_config").unwrap_or(&config);
    let vit_config = vision_config.get("vit_config").unwrap_or(vision_config);
    let adapter_config = vision_config.get("adapter_config").unwrap_or(vision_config);

    // ViT config
    let vit_num_layers = vit_config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(25) as usize;
    // Workaround: HF config may say 27 but weights only have 25
    let vit_num_layers = vit_num_layers.min(25);
    let vit_hidden_size = vit_config
        .get("hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(1152) as i32;
    let vit_intermediate_size = vit_config
        .get("intermediate_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(4304) as i32;
    let vit_num_heads = vit_config
        .get("num_attention_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let vit_num_kv_heads = vit_config
        .get("num_key_value_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let vit_head_dim = vit_config
        .get("head_dim")
        .and_then(|v| v.as_i64())
        .unwrap_or(72) as i32;
    let vit_image_num_pos = vit_config
        .get("image_num_pos")
        .and_then(|v| v.as_u64())
        .unwrap_or(729) as usize;
    let vit_layer_norm_eps = vit_config
        .get("layer_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let vit_float32_attention = vit_config
        .get("float32_attention")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // Adapter config
    let adapter_hidden_size = adapter_config
        .get("hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(1152) as i32;
    let adapter_intermediate_size = adapter_config
        .get("intermediate_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(9728) as i32;
    let adapter_text_hidden_size = adapter_config
        .get("text_hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(text_config.hidden_size as i64) as i32;
    let adapter_num_heads = adapter_config
        .get("num_attention_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let adapter_num_kv_heads = adapter_config
        .get("num_key_value_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let adapter_head_dim = adapter_config
        .get("head_dim")
        .and_then(|v| v.as_i64())
        .unwrap_or(72) as i32;
    let adapter_float32_attention = adapter_config
        .get("float32_attention")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let pooling_attention_mask = adapter_config
        .get("pooling_attention_mask")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // ViT layers to select
    let vit_layers: Vec<i32> = adapter_config
        .get("vit_layers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect()
        })
        .unwrap_or_else(|| vec![-3, -9]);

    // Token IDs
    let image_patch_id = config
        .get("image_patch_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151938) as i32;
    let image_end_token_id = config
        .get("image_end_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151937) as i32;

    // Load weights
    let raw_weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Remap weights
    let mut weights = mlxcel_core::weights::WeightMap::new();
    for (key, value) in &raw_weights {
        let mut new_key = key.clone();

        // model.transformer.* → language_model.model.*
        if new_key.starts_with("model.transformer.") {
            new_key = new_key.replacen("model.transformer.", "language_model.model.", 1);
        }
        // model.vision_backbone.* → vision_tower.*
        if new_key.starts_with("model.vision_backbone.") {
            new_key = new_key.replacen("model.vision_backbone.", "vision_tower.", 1);
        }
        // lm_head.* → language_model.lm_head.*
        if new_key.starts_with("lm_head.") {
            new_key = new_key.replacen("lm_head.", "language_model.lm_head.", 1);
        }
        // .transformer.resblocks. → .transformer.
        new_key = new_key.replace(".transformer.resblocks.", ".transformer.");

        weights.insert(new_key, mlxcel_core::copy(value));
    }

    // Sanitize tied embeddings
    models::sanitize_tied_embeddings(&mut weights, &config);

    // Load text model
    let text_model =
        models::Molmo2Model::from_weights(&weights, &text_config, "language_model.model")
            .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;

    // Load vision model
    let vision_tower = Molmo2VisionModel::from_weights(
        &weights,
        "vision_tower",
        vit_num_layers,
        vit_hidden_size,
        vit_intermediate_size,
        vit_num_heads,
        vit_num_kv_heads,
        vit_head_dim,
        vit_image_num_pos,
        vit_layer_norm_eps,
        vit_float32_attention,
        adapter_hidden_size,
        adapter_intermediate_size,
        adapter_text_hidden_size,
        adapter_num_heads,
        adapter_num_kv_heads,
        adapter_head_dim,
        adapter_float32_attention,
        &vit_layers,
        pooling_attention_mask,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision model: {}", e))?;

    // Load preprocessor config
    let max_crops = {
        let pp_path = model_path.join("preprocessor_config.json");
        if pp_path.exists() {
            let pp_str = std::fs::read_to_string(&pp_path).unwrap_or_default();
            let pp: serde_json::Value = serde_json::from_str(&pp_str).unwrap_or_default();
            pp.get("max_crops").and_then(|v| v.as_u64()).unwrap_or(8) as usize
        } else {
            8
        }
    };

    let processor = Molmo2Processor::new(max_crops, None, None, None, None);

    let vlm = vision::Molmo2VLModel {
        text_model,
        vision_tower,
        processor,
        image_patch_id,
        image_end_token_id,
    };

    Ok(LoadedModel::Molmo2VLM(vlm))
}

/// Load an Aya Vision VLM model (SigLIP + SwiGLU projector + Cohere2 text)
pub(crate) fn load_aya_vision_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::aya_vision::AyaVisionProjector;
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::processors::siglip::SigLipProcessor;

    // Parse config
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let full_config: serde_json::Value = serde_json::from_str(&config_str)?;

    // Load all weights
    let raw_weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Remap weight keys: strip "language_model." prefix and sanitize Aya-specific keys
    let mut weights = mlxcel_core::weights::WeightMap::new();
    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("language_model.") {
            stripped.to_string()
        } else {
            key
        };
        weights.insert(new_key, value);
    }

    // Sanitize Aya Vision weight keys:
    // - model.vision_tower.* → vision_tower.*
    // - model.multi_modal_projector.* → multi_modal_projector.*
    // - model.language_model.* → model.* (text model)
    // - lm_head.* → language_model.lm_head.* (if not already prefixed)
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in keys {
        let new_key = if let Some(rest) = key.strip_prefix("model.vision_tower.") {
            Some(format!("vision_tower.{}", rest))
        } else if let Some(rest) = key.strip_prefix("model.multi_modal_projector.") {
            Some(format!("multi_modal_projector.{}", rest))
        } else {
            key.strip_prefix("model.language_model.")
                .map(|rest| format!("model.{}", rest))
        };
        if let Some(new_key) = new_key
            && let Some(value) = weights.remove(&key)
        {
            weights.insert(new_key, value);
        }
    }

    // Sanitize tied embeddings
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    // Parse vision config
    let vision_config_val = full_config
        .get("vision_config")
        .ok_or_else(|| anyhow::anyhow!("Missing vision_config"))?;
    let vision_config: vision::config::VisionConfig =
        serde_json::from_value(vision_config_val.clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse vision_config: {}", e))?;

    // Parse text config as Cohere2Config
    let text_config_val = full_config
        .get("text_config")
        .ok_or_else(|| anyhow::anyhow!("Missing text_config"))?;
    let mut text_config_obj = text_config_val.clone();

    // Inherit quantization from top-level config
    if text_config_obj.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        text_config_obj
            .as_object_mut()
            .unwrap()
            .insert("quantization".to_string(), q.clone());
    }

    // Inject defaults for fields that VLM text_config may omit
    {
        let obj = text_config_obj.as_object_mut().unwrap();
        if !obj.contains_key("vocab_size") {
            obj.insert("vocab_size".to_string(), serde_json::json!(256000));
        }
        if !obj.contains_key("layer_norm_eps") {
            obj.insert("layer_norm_eps".to_string(), serde_json::json!(1e-5));
        }
        if !obj.contains_key("head_dim") {
            let hidden = obj
                .get("hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096);
            let heads = obj
                .get("num_attention_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(32);
            obj.insert("head_dim".to_string(), serde_json::json!(hidden / heads));
        }
        if !obj.contains_key("sliding_window") {
            obj.insert("sliding_window".to_string(), serde_json::json!(4096));
        }
        // Auto-detect tied embeddings: if no lm_head.weight in weights, use tied
        if !obj.contains_key("tie_word_embeddings") && !weights.contains_key("lm_head.weight") {
            obj.insert("tie_word_embeddings".to_string(), serde_json::json!(true));
        }
    }

    let text_args: models::cohere2::Cohere2Config = serde_json::from_value(text_config_obj.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse text_config as Cohere2: {}", e))?;
    let text_model = models::Cohere2Model::from_weights(&weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load Cohere2 text model: {}", e))?;

    // Get quantization params
    let quant_group_size = full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(64) as i32;
    let quant_bits = full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;

    // Build vision encoder (SigLIP)
    let vision_feature_layer = full_config
        .get("vision_feature_layer")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1) as i32;
    let vision_feature_select_strategy = full_config
        .get("vision_feature_select_strategy")
        .and_then(|v| v.as_str())
        .unwrap_or("full")
        .to_string();

    let vision_encoder =
        SigLipVisionModel::from_weights(&weights, &vision_config, "vision_tower.vision_model")
            .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?
            .with_feature_selection(vision_feature_layer, vision_feature_select_strategy);

    // Build connector (Aya Vision SwiGLU projector with pixel shuffle)
    let downsample_factor = full_config
        .get("downsample_factor")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;
    let adapter_layer_norm_eps = full_config
        .get("adapter_layer_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;

    let connector = AyaVisionProjector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
        downsample_factor,
        adapter_layer_norm_eps,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Aya Vision projector: {}", e))?;

    // Build image processor
    let processor = SigLipProcessor::new(vision_config.image_size);

    // Compute mm_tokens_per_image (after pixel shuffle: patches / downsample_factor²)
    let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
    let mm_tokens_per_image = num_patches / downsample_factor.pow(2);

    let image_token_index = full_config
        .get("image_token_index")
        .or_else(|| full_config.get("image_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(255036) as i32;

    let text_hidden_size = text_args.hidden_size;

    // Assemble VisionModule
    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: image_token_index,
        pad_token_id: 0,
        hidden_size: text_hidden_size,
        boi_token_id: 0,
        eoi_token_id: 0,
        mm_tokens_per_image,
        merge_strategy: vision::MergeStrategy::LLaVA,
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(LoadedModel::Cohere2(text_model)),
        vision: vision_module,
    };

    Ok(LoadedModel::LlavaVLM(vlm))
}

/// Load a PaliGemma VLM model (SigLIP + Linear projector + Gemma text)
pub(crate) fn load_paligemma_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::linear::LinearProjector;
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::processors::siglip::SigLipProcessor;

    // Parse config
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let full_config: serde_json::Value = serde_json::from_str(&config_str)?;

    // Load all weights
    let raw_weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Remap weight keys: strip "language_model." prefix
    let mut weights = mlxcel_core::weights::WeightMap::new();
    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("language_model.") {
            stripped.to_string()
        } else {
            key
        };
        weights.insert(new_key, value);
    }

    // Sanitize PaliGemma weight keys:
    // - multi_modal_projector.linear.* → multi_modal_projector.linear_1.*
    //   (our LinearProjector expects linear_1 prefix)
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in keys {
        if let Some(rest) = key.strip_prefix("multi_modal_projector.linear.") {
            let new_key = format!("multi_modal_projector.linear_1.{}", rest);
            if let Some(value) = weights.remove(&key) {
                weights.insert(new_key, value);
            }
        }
    }

    // Sanitize tied embeddings
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    // Parse vision config
    let vision_config_val = full_config
        .get("vision_config")
        .ok_or_else(|| anyhow::anyhow!("Missing vision_config"))?;
    let vision_config: vision::config::VisionConfig =
        serde_json::from_value(vision_config_val.clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse vision_config: {}", e))?;

    // Parse text config as Gemma ModelArgs
    let text_config_val = full_config
        .get("text_config")
        .ok_or_else(|| anyhow::anyhow!("Missing text_config"))?;
    let mut text_config_obj = text_config_val.clone();

    // Inherit quantization from top-level config
    if text_config_obj.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        text_config_obj
            .as_object_mut()
            .unwrap()
            .insert("quantization".to_string(), q.clone());
    }

    // Inject defaults for fields that VLM text_config may omit
    {
        let obj = text_config_obj.as_object_mut().unwrap();
        if !obj.contains_key("rms_norm_eps") {
            obj.insert("rms_norm_eps".to_string(), serde_json::json!(1e-6));
        }
        if !obj.contains_key("head_dim") {
            // Gemma2 default head_dim is 256; use query_pre_attn_scalar as hint if available
            let default_head_dim = obj
                .get("query_pre_attn_scalar")
                .and_then(|v| v.as_u64())
                .unwrap_or(256);
            obj.insert("head_dim".to_string(), serde_json::json!(default_head_dim));
        }
    }

    // Determine text model type (gemma or gemma2)
    let text_model_type = text_config_obj
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("gemma");

    let text_model: LoadedModel = match text_model_type {
        "gemma" => {
            let text_args: models::gemma::ModelArgs =
                serde_json::from_value(text_config_obj.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to parse text_config as Gemma: {}", e))?;
            let m = models::GemmaModel::from_weights(&weights, &text_args)
                .map_err(|e| anyhow::anyhow!("Failed to load Gemma text model: {}", e))?;
            LoadedModel::Gemma(m)
        }
        "gemma2" => {
            let text_args: models::gemma2::ModelArgs =
                serde_json::from_value(text_config_obj.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to parse text_config as Gemma2: {}", e))?;
            let m = models::Gemma2Model::from_weights(&weights, &text_args)
                .map_err(|e| anyhow::anyhow!("Failed to load Gemma2 text model: {}", e))?;
            LoadedModel::Gemma2(m)
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported PaliGemma text backend: {}",
                text_model_type
            ));
        }
    };

    // Get quantization params
    let quant_group_size = full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(64) as i32;
    let quant_bits = full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;

    // Build vision encoder (SigLIP) — pass correct quantization params
    let vision_encoder = SigLipVisionModel::from_weights_with_quant(
        &weights,
        &vision_config,
        "vision_tower.vision_model",
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?;

    // Build connector (single Linear projector)
    let connector = LinearProjector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load linear projector: {}", e))?;

    // Build image processor (SigLIP)
    let processor = SigLipProcessor::new(vision_config.image_size);

    // Compute mm_tokens_per_image
    let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);

    let image_token_index = full_config
        .get("image_token_index")
        .or_else(|| full_config.get("image_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(257152) as i32;
    let pad_token_id = full_config
        .get("pad_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    let text_hidden_size = full_config
        .get("hidden_size")
        .or_else(|| text_config_val.get("hidden_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;

    // Assemble VisionModule — use Gemma3-style merge (mask-based with 4D attention mask)
    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: image_token_index,
        pad_token_id,
        hidden_size: text_hidden_size,
        boi_token_id: 0,
        eoi_token_id: 0,
        mm_tokens_per_image: num_patches,
        merge_strategy: vision::MergeStrategy::Gemma3,
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(text_model),
        vision: vision_module,
    };

    Ok(LoadedModel::Gemma3VLM(vlm))
}

/// Load a Pixtral VLM model (Pixtral ViT with 2D RoPE + Mistral text + MLP projector)
pub(crate) fn load_pixtral_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::mlp::MLPProjector;
    use vision::encoders::pixtral::{PixtralVisionConfig, PixtralVisionModel};
    use vision::processors::siglip::SigLipProcessor;

    // Parse full config as Value (Pixtral's vision_config may lack fields that VLMConfig requires)
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let full_config: serde_json::Value = serde_json::from_str(&config_str)?;

    // Load all weights
    let raw_weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Remap VLM weight keys: strip "language_model." prefix for text model weights
    let mut weights = mlxcel_core::weights::WeightMap::new();
    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("language_model.") {
            stripped.to_string()
        } else {
            key
        };
        weights.insert(new_key, value);
    }

    // Sanitize Pixtral-specific weight key naming conventions
    vision::encoders::pixtral::sanitize_pixtral_weights(&mut weights);

    // Sanitize tied embeddings
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    // Build text model (Mistral → Llama3Model)
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    infer_llama_config_from_weights(&mut text_config_value, &weights);

    // Fix num_attention_heads: infer from q_proj weight shape instead of hidden_size/128
    // (Pixtral's Mistral has hidden_size=5120 but num_attention_heads=32, not 40)
    if let Some(obj) = text_config_value.as_object_mut() {
        let head_dim = obj.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
        if let Some(w) = weights
            .get("model.layers.0.self_attn.q_proj.scales")
            .or_else(|| weights.get("model.layers.0.self_attn.q_proj.weight"))
        {
            let shape = mlxcel_core::array_shape(w);
            if !shape.is_empty() {
                let q_out = shape[0] as usize;
                let num_heads = q_out / head_dim;
                obj.insert(
                    "num_attention_heads".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(num_heads)),
                );
            }
        }
    }

    // Inherit quantization from top-level config if not in text_config
    if text_config_value.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        text_config_value
            .as_object_mut()
            .unwrap()
            .insert("quantization".to_string(), q.clone());
    }

    let text_args: models::llama3::ModelArgs = serde_json::from_value(text_config_value.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse text_config as Mistral: {}", e))?;
    let m = models::Llama3Model::from_weights(&weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;
    let text_model = LoadedModel::Llama(m);

    // Get quantization params
    let quant_group_size = full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(64) as i32;
    let quant_bits = full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;

    // Parse Pixtral vision config (handles missing fields with defaults)
    let vision_config_value = full_config
        .get("vision_config")
        .ok_or_else(|| anyhow::anyhow!("vision_config not found in config.json"))?;
    let pixtral_config = PixtralVisionConfig::from_json(vision_config_value);

    // Build Pixtral vision encoder
    let vision_feature_layer = full_config
        .get("vision_feature_layer")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1) as i32;
    let vision_encoder =
        PixtralVisionModel::from_weights(&weights, &pixtral_config, "vision_tower.vision_model")
            .map_err(|e| anyhow::anyhow!("Failed to load Pixtral vision encoder: {}", e))?
            .with_feature_layer(vision_feature_layer);

    // Build MLP projector (Linear → GELU → Linear, with bias)
    let connector = MLPProjector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load MLP projector: {}", e))?;

    // Build image processor (rescale only, no mean/std normalization for Pixtral)
    let processor = SigLipProcessor::new_rescale_only(pixtral_config.image_size);

    // Compute mm_tokens_per_image
    let num_patches = (pixtral_config.image_size / pixtral_config.patch_size).pow(2);
    let mm_tokens_per_image = full_config
        .get("mm_tokens_per_image")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_patches as u64) as usize;

    // Get text hidden size
    let text_hidden_size = text_config_value
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as usize;

    // Resolve image_token_id (config may use image_token_index or image_token_id)
    let image_token_id = full_config
        .get("image_token_index")
        .or_else(|| full_config.get("image_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(10) as i32;

    let pad_token_id = full_config
        .get("pad_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    let hidden_size = full_config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Assemble VisionModule
    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id,
        pad_token_id,
        hidden_size: if hidden_size > 0 {
            hidden_size
        } else {
            text_hidden_size
        },
        boi_token_id: 0,
        eoi_token_id: 0,
        mm_tokens_per_image,
        merge_strategy: vision::MergeStrategy::LLaVA,
    };

    // Assemble VisionLanguageModel (reuses LlavaVLM variant)
    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(text_model),
        vision: vision_module,
    };

    Ok(LoadedModel::LlavaVLM(vlm))
}

/// Load a Mistral 3 VLM model (Pixtral ViT + PatchMerger projector + Mistral text model)
pub(crate) fn load_mistral3_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::mistral3::Mistral3Projector;
    use vision::encoders::pixtral::{PixtralVisionConfig, PixtralVisionModel};
    use vision::processors::siglip::SigLipProcessor;

    // Parse full config
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let full_config: serde_json::Value = serde_json::from_str(&config_str)?;

    // Load all weights
    let raw_weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Remap VLM weight keys: strip "language_model." prefix for text model weights
    let mut weights = mlxcel_core::weights::WeightMap::new();
    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("language_model.") {
            stripped.to_string()
        } else {
            key
        };
        weights.insert(new_key, value);
    }

    // Sanitize Pixtral-specific weight key naming conventions
    vision::encoders::pixtral::sanitize_pixtral_weights(&mut weights);

    // Sanitize tied embeddings
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    // Build text model (Mistral → Llama3Model)
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    infer_llama_config_from_weights(&mut text_config_value, &weights);

    // Fix num_attention_heads: infer from q_proj weight shape
    if let Some(obj) = text_config_value.as_object_mut() {
        let head_dim = obj.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
        if let Some(w) = weights
            .get("model.layers.0.self_attn.q_proj.scales")
            .or_else(|| weights.get("model.layers.0.self_attn.q_proj.weight"))
        {
            let shape = mlxcel_core::array_shape(w);
            if !shape.is_empty() {
                let q_out = shape[0] as usize;
                let num_heads = q_out / head_dim;
                obj.insert(
                    "num_attention_heads".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(num_heads)),
                );
            }
        }
    }

    // Inherit quantization from top-level config if not in text_config
    if text_config_value.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        text_config_value
            .as_object_mut()
            .unwrap()
            .insert("quantization".to_string(), q.clone());
    }

    let text_args: models::llama3::ModelArgs = serde_json::from_value(text_config_value.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse text_config as Mistral: {}", e))?;
    let m = models::Llama3Model::from_weights(&weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;
    let text_model = LoadedModel::Llama(m);

    // Get quantization params
    let quant_group_size = full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(64) as i32;
    let quant_bits = full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;

    // Parse Pixtral vision config
    let vision_config_value = full_config
        .get("vision_config")
        .ok_or_else(|| anyhow::anyhow!("vision_config not found in config.json"))?;
    let pixtral_config = PixtralVisionConfig::from_json(vision_config_value);

    // Build Pixtral vision encoder
    let vision_feature_layer = full_config
        .get("vision_feature_layer")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1) as i32;
    let vision_encoder =
        PixtralVisionModel::from_weights(&weights, &pixtral_config, "vision_tower.vision_model")
            .map_err(|e| anyhow::anyhow!("Failed to load Pixtral vision encoder: {}", e))?
            .with_feature_layer(vision_feature_layer);

    // Compute patch grid size for PatchMerger
    let patch_h = (pixtral_config.image_size / pixtral_config.patch_size) as i32;
    let spatial_merge_size = full_config
        .get("spatial_merge_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(2) as i32;

    // Get RMS norm eps from text config
    let rms_norm_eps = text_config_value
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;

    // Build Mistral3 projector (RMSNorm + PatchMerger + MLP)
    let connector = Mistral3Projector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
        patch_h,
        spatial_merge_size,
        rms_norm_eps,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Mistral3 projector: {}", e))?;

    // Build image processor (rescale only, no mean/std normalization for Pixtral)
    let processor = SigLipProcessor::new_rescale_only(pixtral_config.image_size);

    // Compute mm_tokens_per_image (after spatial merge)
    let num_patches_per_side = pixtral_config.image_size / pixtral_config.patch_size;
    let merged_per_side = num_patches_per_side / spatial_merge_size as usize;
    let mm_tokens_per_image = full_config
        .get("mm_tokens_per_image")
        .and_then(|v| v.as_u64())
        .unwrap_or((merged_per_side * merged_per_side) as u64)
        as usize;

    // Get text hidden size
    let text_hidden_size = text_config_value
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as usize;

    // Resolve image_token_id
    let image_token_id = full_config
        .get("image_token_index")
        .or_else(|| full_config.get("image_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(10) as i32;

    let pad_token_id = full_config
        .get("pad_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    let hidden_size = full_config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Assemble VisionModule
    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id,
        pad_token_id,
        hidden_size: if hidden_size > 0 {
            hidden_size
        } else {
            text_hidden_size
        },
        boi_token_id: 0,
        eoi_token_id: 0,
        mm_tokens_per_image,
        merge_strategy: vision::MergeStrategy::LLaVA,
    };

    // Assemble VisionLanguageModel (reuses LlavaVLM variant)
    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(text_model),
        vision: vision_module,
    };

    Ok(LoadedModel::LlavaVLM(vlm))
}

/// Load a Llama 4 VLM model (Llama4 vision encoder + linear projector + Llama4 text model)
pub(crate) fn load_llama4_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::linear::LinearProjector;
    use vision::encoders::llama4::{Llama4VisionConfig, Llama4VisionModel};
    use vision::processors::siglip::SigLipProcessor;

    // Parse config
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let full_config: serde_json::Value = serde_json::from_str(&config_str)?;

    // Parse text config (at root level or under text_config)
    let text_config_value = if let Some(tc) = full_config.get("text_config") {
        let mut tc = tc.clone();
        // Inherit quantization from top level
        if tc.get("quantization").is_none()
            && let Some(q) = full_config.get("quantization")
        {
            tc.as_object_mut()
                .unwrap()
                .insert("quantization".to_string(), q.clone());
        }
        tc
    } else {
        full_config.clone()
    };

    let text_args: models::llama4::TextArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Llama4 text config: {}", e))?;

    // Parse vision config
    let vision_config: Llama4VisionConfig = serde_json::from_value(
        full_config
            .get("vision_config")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing vision_config in config.json"))?,
    )
    .map_err(|e| anyhow::anyhow!("Failed to parse vision_config: {}", e))?;

    // Load all weights (no prefix remapping needed - Llama4 VLM uses direct prefixes)
    let weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Build text model
    // Weight keys for text model use "language_model.model.*" prefix
    // Keep prefix intact — Llama4CxxModel::from_weights expects "language_model.*" keys
    let mut text_weights = mlxcel_core::weights::WeightMap::new();
    for (key, value) in &weights {
        if key.starts_with("language_model.") {
            text_weights.insert(key.clone(), mlxcel_core::copy(value));
        }
    }

    // Sanitize tied embeddings
    models::sanitize_tied_embeddings(&mut text_weights, &full_config);

    let text_model = models::Llama4CxxModel::from_weights(&text_weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load Llama4 text model: {}", e))?;
    let text_wrapper = models::Llama4Wrapper::new(text_model);

    // Build vision encoder — auto-detect prefix (vision_tower or vision_model)
    let vision_prefix = if weights.contains_key("vision_model.patch_embedding.linear.weight") {
        "vision_model"
    } else {
        "vision_tower"
    };
    let vision_encoder =
        Llama4VisionModel::from_weights(&weights, &vision_config, vision_prefix)
            .map_err(|e| anyhow::anyhow!("Failed to load Llama4 vision encoder: {}", e))?;

    // Build connector (single linear projector, no activation, no bias)
    let quant_group_size = full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(64) as i32;
    let quant_bits = full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;

    let connector = LinearProjector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Llama4 projector: {}", e))?;

    // Build image processor (SigLIP-style: resize to image_size, 0.5/0.5 norm)
    let processor = SigLipProcessor::new(vision_config.image_size);

    // Get token IDs from config
    let image_token_id = full_config
        .get("image_token_index")
        .or_else(|| full_config.get("image_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(200092) as i32;

    let pad_token_id = full_config
        .get("text_config")
        .and_then(|tc| tc.get("pad_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(200018) as i32;

    // mm_tokens_per_image: after pixel shuffle
    // num_patches = (image_size / patch_size)^2 = (1120/14)^2 = 6400
    // After pixel shuffle with ratio 0.5: 6400 * 0.5^2 = 1600
    let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
    let mm_tokens_per_image =
        (num_patches as f32 * vision_config.pixel_shuffle_ratio.powi(2)) as usize;

    // Assemble VisionModule
    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id,
        pad_token_id,
        hidden_size: text_args.hidden_size,
        boi_token_id: 0, // Llama4 VLM doesn't use BOI/EOI framing
        eoi_token_id: 0,
        mm_tokens_per_image,
        merge_strategy: vision::MergeStrategy::LLaVA,
    };

    // Assemble VisionLanguageModel
    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(LoadedModel::Llama4(text_wrapper)),
        vision: vision_module,
    };

    Ok(LoadedModel::Llama4VLM(vlm))
}

#[cfg(test)]
#[path = "loader_vlm_tests.rs"]
mod tests;

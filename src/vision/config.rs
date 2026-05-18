//! Vision model configuration types
//!
//! Deserialization for vision_config and projector config from model config.json

use serde::{Deserialize, Deserializer};

/// Deserialize null as default value (serde #[serde(default)] only handles missing fields)
fn null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Option::deserialize(deserializer).map(|opt| opt.unwrap_or_default())
}

/// Vision encoder configuration (SigLIP)
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub model_type: String,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
}

fn default_image_size() -> usize {
    224
}
fn default_num_channels() -> usize {
    3
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}

/// Full VLM model config (wraps text + vision configs)
#[derive(Debug, Clone, Deserialize)]
pub struct VLMConfig {
    pub model_type: String,
    pub text_config: serde_json::Value,
    pub vision_config: VisionConfig,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default, deserialize_with = "null_default")]
    pub image_token_index: i32,
    #[serde(default, deserialize_with = "null_default")]
    pub pad_token_id: i32,
    #[serde(default)]
    pub mm_tokens_per_image: Option<usize>,
    /// Begin-of-image token index
    #[serde(default = "default_boi_token")]
    pub boi_token_index: i32,
    /// End-of-image token index
    #[serde(default = "default_eoi_token")]
    pub eoi_token_index: i32,
    /// Which encoder layer to extract features from (LLaVA: -2 = second-to-last)
    #[serde(default = "default_vision_feature_layer")]
    pub vision_feature_layer: i32,
    /// Feature selection strategy: "default" strips CLS token, "full" keeps all
    #[serde(default = "default_vision_feature_select_strategy")]
    pub vision_feature_select_strategy: String,
}

fn default_boi_token() -> i32 {
    255999
}
fn default_eoi_token() -> i32 {
    256000
}

fn default_vocab_size() -> usize {
    257152
}
fn default_vision_feature_layer() -> i32 {
    -2
}
fn default_vision_feature_select_strategy() -> String {
    "default".to_string()
}

impl VLMConfig {
    /// Get mm_tokens_per_image, falling back to text_config value
    pub fn get_mm_tokens_per_image(&self) -> usize {
        self.mm_tokens_per_image
            .or_else(|| {
                self.text_config
                    .get("mm_tokens_per_image")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
            })
            .unwrap_or(256)
    }
}

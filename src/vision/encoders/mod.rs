//! Vision encoder implementations
//!
//! Provides the VisionEncoder trait and encoder implementations.

pub mod gemma3n;
pub mod llama4;
pub mod molmo2;
pub mod pixtral;
pub mod qwen2_5_vl;
pub mod qwen2_vl;
pub mod qwen3_vl;
pub mod siglip;

use mlxcel_core::{MlxArray, UniquePtr};

/// Output from a vision encoder
pub struct VisionEncoderOutput {
    pub hidden_states: UniquePtr<MlxArray>, // [batch, num_patches, hidden_dim]
}

/// Trait for vision encoders
pub trait VisionEncoder {
    fn forward(&self, pixel_values: &MlxArray) -> VisionEncoderOutput;
}

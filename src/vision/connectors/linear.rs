//! Linear Multi-Modal Projector (single linear layer, no activation, no bias)
//!
//! Port of references/mlx-vlm/mlx_vlm/models/llama4/vision.py:Llama4MultiModalProjector
//!
//! Architecture: Linear(vision_output_dim → text_hidden_size, bias=False)
//!
//! Used by: Llama 4 Vision

use super::MultiModalConnector;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct LinearProjector {
    linear_1: UnifiedLinear,
}

impl LinearProjector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let linear_1 = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_1", prefix),
            group_size,
            bits,
        )?;
        Ok(Self { linear_1 })
    }
}

impl MultiModalConnector for LinearProjector {
    fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        self.linear_1.forward(vision_features)
    }
}

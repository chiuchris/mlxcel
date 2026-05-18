//! LLaVA MLP Multi-Modal Projector
//!
//! Port of references/mlx-vlm/mlx_vlm/models/llava/llava.py:13-28
//!
//! Architecture: Linear(vision_hidden → text_hidden) → GELU → Linear(text_hidden → text_hidden)
//!
//! Used by: LLaVA

use super::MultiModalConnector;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct MLPProjector {
    linear_1: UnifiedLinear,
    linear_2: UnifiedLinear,
}

impl MLPProjector {
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
        let linear_2 = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_2", prefix),
            group_size,
            bits,
        )?;
        Ok(Self { linear_1, linear_2 })
    }
}

impl MultiModalConnector for MLPProjector {
    fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.linear_1.forward(vision_features);
        let x = mlxcel_core::gelu(&x);
        self.linear_2.forward(&x)
    }
}

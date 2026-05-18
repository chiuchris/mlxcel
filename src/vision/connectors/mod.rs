//! Multi-modal connectors (projectors)
//!
//! Provides the MultiModalConnector trait and connector implementations.

pub mod avg_pool;
pub mod aya_vision;
pub mod identity;
pub mod linear;
pub mod mistral3;
pub mod mlp;

use mlxcel_core::{MlxArray, UniquePtr};

/// Trait for multi-modal connectors that project vision features to text space
pub trait MultiModalConnector {
    fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray>;
}

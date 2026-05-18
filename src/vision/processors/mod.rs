//! Image processors for vision models
//!
//! Provides the ImageProcessor trait and processor implementations.

pub mod molmo2;
pub mod phi3_v;
pub mod qwen2_vl;
pub mod siglip;

use mlxcel_core::{MlxArray, UniquePtr};

/// Trait for image preprocessors
pub trait ImageProcessor {
    /// Preprocess images to tensor format ready for vision encoder
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray>;
}

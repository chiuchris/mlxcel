//! Identity connector (pass-through)
//!
//! Used when the vision encoder already projects to text hidden size
//! (e.g., Qwen2-VL PatchMerger handles projection internally).
//!
//! Used by: Qwen2-VL

use super::MultiModalConnector;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct IdentityConnector;

impl MultiModalConnector for IdentityConnector {
    fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::copy(vision_features)
    }
}

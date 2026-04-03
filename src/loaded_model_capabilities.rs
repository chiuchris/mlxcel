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

//! Capability-facing helpers for `LoadedModel`.
//!
//! `src/loaded_model.rs` keeps explicit model storage and `LanguageModel`
//! forwarding. This module owns the narrower multimodal surfaces consumed by
//! CLI/server control-plane code so those callers do not need to know concrete
//! model variant names.

use crate::loaded_model::LoadedModel;
use crate::{qwen_vl, vision, vlm_prompt};

/// Capability-oriented references used by CLI/server multimodal preparation.
///
/// The control plane should ask for the narrowest VLM runtime capability it
/// needs instead of depending on concrete `LoadedModel` variants. Add new
/// variants here only when a family truly needs distinct preparation logic.
pub enum VlmRuntimeRef<'a> {
    Qwen(&'a dyn qwen_vl::QwenVlRuntime),
    MiniCPMO(&'a vision::MiniCPMOVLModel),
    Moondream3(&'a vision::Moondream3VLModel),
    Gemma3n(&'a vision::Gemma3nVLModel),
    Gemma4(&'a vision::Gemma4VLModel),
    Phi4MM(&'a vision::Phi4MMVLModel),
    Phi4SigLip(&'a vision::Phi4SigLipVLModel),
    Phi3V(&'a vision::Phi3VLModel),
    Molmo2(&'a vision::Molmo2VLModel),
    MolmoPoint(&'a vision::MolmoPointVLModel),
    Standard(&'a vision::VisionModule),
}

pub(crate) fn standard_image_token_block_info(
    vm: &vision::VisionModule,
) -> vlm_prompt::ImageTokenBlockInfo {
    vlm_prompt::ImageTokenBlockInfo {
        use_boi_eoi: vm.boi_token_id != 0,
        image_token_id: vm.image_token_id,
        mm_tokens_per_image: vm.mm_tokens_per_image,
        boi_token_id: vm.boi_token_id,
        eoi_token_id: vm.eoi_token_id,
        has_bos: vm.has_bos,
        separator_token_id: vm.separator_token_id,
        suffix_tokens: vm.suffix_tokens.clone(),
        block_prefix_tokens: vm.block_prefix_tokens.clone(),
        block_suffix_tokens: vm.block_suffix_tokens.clone(),
    }
}

fn gemma3n_image_token_block_info(
    model: &vision::Gemma3nVLModel,
) -> vlm_prompt::ImageTokenBlockInfo {
    vlm_prompt::ImageTokenBlockInfo {
        use_boi_eoi: true,
        image_token_id: model.image_token_id,
        mm_tokens_per_image: 256,
        boi_token_id: model.boi_token_id,
        eoi_token_id: model.eoi_token_id,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    }
}

fn qwen_runtime(runtime: VlmRuntimeRef<'_>) -> Option<&dyn qwen_vl::QwenVlRuntime> {
    match runtime {
        VlmRuntimeRef::Qwen(runtime) => Some(runtime),
        _ => None,
    }
}

pub(crate) fn vision_module_from_runtime(
    runtime: VlmRuntimeRef<'_>,
) -> Option<&vision::VisionModule> {
    match runtime {
        VlmRuntimeRef::Standard(vision) => Some(vision),
        _ => None,
    }
}

pub(crate) fn image_token_block_info_from_runtime(
    runtime: VlmRuntimeRef<'_>,
) -> Option<vlm_prompt::ImageTokenBlockInfo> {
    match runtime {
        VlmRuntimeRef::Gemma3n(model) => Some(gemma3n_image_token_block_info(model)),
        VlmRuntimeRef::Standard(vision) => Some(standard_image_token_block_info(vision)),
        _ => None,
    }
}

impl LoadedModel {
    /// Check if this model is a vision-language model.
    pub fn is_vlm(&self) -> bool {
        self.vlm_runtime().is_some()
    }

    /// Get the vision module if this is a standard `VisionModule`-backed VLM.
    pub fn vision_module(&self) -> Option<&vision::VisionModule> {
        vision_module_from_runtime(self.vlm_runtime()?)
    }

    /// Return the multimodal runtime capability needed by prompt/image prep.
    ///
    /// Keep this as the single VLM switchboard so new variants do not require
    /// ad hoc getter methods throughout CLI or server code.
    pub fn vlm_runtime(&self) -> Option<VlmRuntimeRef<'_>> {
        match self {
            Self::Qwen2VL(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::Qwen25VL(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::Qwen3VL(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::Qwen3VLMoe(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::Qwen35VLM(model) | Self::Qwen35MoeVLM(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::MiniCPMOVLM(model) => Some(VlmRuntimeRef::MiniCPMO(model)),
            Self::Moondream3VLM(model) => Some(VlmRuntimeRef::Moondream3(model)),
            Self::Gemma3nVLM(model) => Some(VlmRuntimeRef::Gemma3n(model)),
            Self::Gemma4VLM(model) => Some(VlmRuntimeRef::Gemma4(model)),
            Self::Phi4MMVLM(model) => Some(VlmRuntimeRef::Phi4MM(model)),
            Self::Phi4SigLipVLM(model) => Some(VlmRuntimeRef::Phi4SigLip(model)),
            Self::Phi3VLM(model) => Some(VlmRuntimeRef::Phi3V(model)),
            Self::Molmo2VLM(model) => Some(VlmRuntimeRef::Molmo2(model)),
            Self::MolmoPointVLM(model) => Some(VlmRuntimeRef::MolmoPoint(model)),
            Self::Gemma3VLM(vlm) => Some(VlmRuntimeRef::Standard(&vlm.vision)),
            Self::Llama4VLM(vlm) => Some(VlmRuntimeRef::Standard(&vlm.vision)),
            Self::LlavaVLM(vlm) => Some(VlmRuntimeRef::Standard(&vlm.vision)),
            _ => None,
        }
    }

    pub fn qwen_vl_prompt_info(&self) -> Option<qwen_vl::QwenVlmPromptInfo<'_>> {
        Some(qwen_runtime(self.vlm_runtime()?)?.prompt_info())
    }

    pub fn qwen_vl_input_embeddings(
        &self,
        input_ids: &mlxcel_core::MlxArray,
        pixel_values: &mlxcel_core::MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> Option<vision::merge::InputEmbeddings> {
        Some(qwen_runtime(self.vlm_runtime()?)?.input_embeddings(input_ids, pixel_values, grid_thw))
    }

    pub fn image_token_block_info(&self) -> Option<vlm_prompt::ImageTokenBlockInfo> {
        image_token_block_info_from_runtime(self.vlm_runtime()?)
    }
}

#[cfg(test)]
#[path = "loaded_model_tests.rs"]
mod tests;

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

//! Vision model support for mlxcel
//!
//! Provides modular vision functionality that layers on top of existing text
//! models. The vision stack is intentionally split into:
//!
//! - `processors`: image normalization / tiling / resizing
//! - `encoders`: vision towers
//! - `connectors`: projection into the text hidden space
//! - `merge`: text/vision embedding composition
//!
//! `VisionModule` composes those pieces into one runtime unit that can be
//! attached to any `LanguageModel` through `VisionLanguageModel`.

use anyhow::Result;

pub mod config;
pub mod connectors;
pub mod encoders;
pub mod merge;
pub mod processors;

// VLM model implementations
pub mod gemma3n_vl;
pub mod minicpmo_vl;
pub mod molmo2_vl;
pub mod phi3_vl;
pub mod phi4_siglip_vl;
pub mod phi4mm_vl;
pub mod qwen2_5_vl;
pub mod qwen2_vl;
pub mod qwen3_5_vl;
pub mod qwen3_vl;
pub mod qwen3_vl_moe;

// Re-export VLM model types
pub use gemma3n_vl::Gemma3nVLModel;
pub use minicpmo_vl::MiniCPMOVLModel;
pub use molmo2_vl::Molmo2VLModel;
pub use phi3_vl::Phi3VLModel;
pub use phi4_siglip_vl::Phi4SigLipVLModel;
pub use phi4mm_vl::Phi4MMVLModel;
pub use qwen2_5_vl::Qwen25VLModel;
pub use qwen2_vl::Qwen2VLModel;
pub use qwen3_5_vl::Qwen35VLModel;
pub use qwen3_vl::Qwen3VLModel;
pub use qwen3_vl_moe::Qwen3VLMoeModel;

use crate::LanguageModel;
use connectors::MultiModalConnector;
use encoders::VisionEncoder;
use merge::InputEmbeddings;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};
use processors::ImageProcessor;

fn require_array_ref<'a>(array: &'a UniquePtr<MlxArray>, label: &str) -> Result<&'a MlxArray> {
    array
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Vision module produced a null {}", label))
}

/// Merge strategy for combining vision and text embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Used by: Gemma3, Gemma3n
    ///
    /// Gemma-style multimodal routing keeps a 4D attention mask and merges
    /// image features with a masked-scatter style operation.
    Gemma3,
    /// Used by: LLaVA, Aya Vision, PaliGemma, Pixtral, Mistral3, Qwen2/2.5/3-VL,
    /// Phi3V, Molmo2, Llama4
    ///
    /// LLaVA-style routing replaces image-token positions directly and then
    /// relies on standard causal masking.
    LLaVA,
}

/// Vision module: encodes images and merges with text embeddings
pub struct VisionModule {
    /// Vision tower that produces image hidden states.
    pub encoder: Box<dyn VisionEncoder>,
    /// Projection layer from vision hidden size into text hidden size.
    pub connector: Box<dyn MultiModalConnector>,
    /// Preprocessing policy for raw images before the vision tower.
    pub processor: Box<dyn ImageProcessor>,
    /// Token ID that marks image placeholder positions in the text prompt.
    pub image_token_id: i32,
    /// Padding token used by merge helpers when prompts include image padding.
    pub pad_token_id: i32,
    /// Hidden size expected by the text tower.
    pub hidden_size: usize,
    /// Begin-of-image token ID (0 = no BOI/EOI framing)
    pub boi_token_id: i32,
    /// End-of-image token ID
    pub eoi_token_id: i32,
    /// Number of image tokens per image
    pub mm_tokens_per_image: usize,
    /// Strategy for merging vision and text embeddings
    pub merge_strategy: MergeStrategy,
    /// Whether the tokenizer adds a BOS token before text.
    /// When false (e.g., PaliGemma), image tokens are prepended without BOS.
    /// Used by: PaliGemma (false), all others (true)
    pub has_bos: bool,
    /// Token to insert between image tokens and text when `has_bos` is false.
    /// PaliGemma: BOS(2) between images and text despite add_bos_token=false.
    pub separator_token_id: Option<i32>,
    /// Tokens to append after text.
    /// PaliGemma: newline(108) appended after text prompt.
    pub suffix_tokens: Vec<i32>,
}

impl VisionModule {
    /// Process images and merge with text embeddings.
    ///
    /// Used by: Gemma3, Gemma3n, LLaVA, Aya Vision, PaliGemma, Pixtral,
    /// Mistral3, Qwen2/2.5/3-VL, Phi3V, Molmo2, Llama4
    ///
    /// 1. Get text embeddings from the text model
    /// 2. Encode images through vision encoder
    /// 3. Project vision features to text space
    /// 4. Merge vision and text embeddings at image token positions
    pub fn get_input_embeddings(
        &self,
        text_model: &dyn LanguageModel,
        input_ids: &MlxArray,
        pixel_values: Option<&MlxArray>,
        attention_mask: &MlxArray,
    ) -> Result<InputEmbeddings> {
        // Get text embeddings
        let inputs_embeds = text_model
            .embed_tokens(input_ids)
            .ok_or_else(|| anyhow::anyhow!("Text model must support embed_tokens for VLM"))?;
        let _ = require_array_ref(&inputs_embeds, "text embedding buffer")?;

        let Some(pixel_values) = pixel_values else {
            return Ok(InputEmbeddings {
                inputs_embeds,
                attention_mask_4d: None,
            });
        };

        // Get dtype of text embeddings for casting
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);

        // Vision encoders use channels-last tensors. Most image processors emit
        // channels-first `[B, C, H, W]`, so we normalize the layout here once.
        let pv_shape = mlxcel_core::array_shape(pixel_values);
        let pv = if pv_shape.len() == 4 && pv_shape[1] <= 4 {
            let transposed = mlxcel_core::transpose_axes(pixel_values, &[0, 2, 3, 1]);
            mlxcel_core::astype(&transposed, embed_dtype)
        } else {
            mlxcel_core::astype(pixel_values, embed_dtype)
        };

        // Encode images
        let encoder_output = self.encoder.forward(&pv);

        // Project to text embedding space
        let image_features = self.connector.forward(&encoder_output.hidden_states);

        // Different VLM families make different masking assumptions, so merge
        // strategy selection stays explicit instead of being inferred.
        Ok(match self.merge_strategy {
            MergeStrategy::Gemma3 => merge::prepare_inputs_for_multimodal(
                self.hidden_size,
                self.pad_token_id,
                self.image_token_id,
                &image_features,
                &inputs_embeds,
                input_ids,
                attention_mask,
            ),
            MergeStrategy::LLaVA => merge::merge_llava(
                self.image_token_id,
                &image_features,
                &inputs_embeds,
                input_ids,
            ),
        })
    }
}

/// Vision-language model: wraps a text model with a vision module
pub struct VisionLanguageModel {
    /// Underlying text model, already loaded through `LoadedModel`.
    pub text_model: Box<crate::LoadedModel>,
    /// Vision runtime components used to turn images into text-space embeddings.
    pub vision: VisionModule,
}

impl LanguageModel for VisionLanguageModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        self.text_model.embed_tokens(input_ids)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.text_model.eos_token_ids()
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

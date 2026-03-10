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

//! Gemma 3n Vision-Language Model
//!
//! MobileNetV5 vision encoder + Gemma3n language model

use super::{encoders, merge, processors};
use crate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Gemma 3n VLM: MobileNetV5 vision encoder + Gemma3n language model
///
/// Unlike ViT-based VLMs, Gemma 3n uses a convolutional MobileNetV5 encoder
/// with Multi-Scale Fusion Adapter. The language model has a unique per_layer_inputs
/// mechanism that requires special handling (cached between get_input_embeddings
/// and forward_with_embeddings).
pub struct Gemma3nVLModel {
    pub text_model: crate::models::Gemma3nModel,
    pub vision_tower: encoders::gemma3n::Gemma3nVisionModel,
    pub embed_vision: crate::models::gemma3n::Gemma3nMultimodalEmbedder,
    pub processor: processors::siglip::SigLipProcessor, // 224x224
    pub image_token_id: i32,                            // 262_145 (<image_soft_token>)
    pub boi_token_id: i32,                              // 255_999 (<start_of_image>)
    pub eoi_token_id: i32,                              // 262_144 (<end_of_image>)
    pub vision_hidden_size: usize,                      // 2048
    /// Store per_layer_inputs between get_input_embeddings and forward_with_embeddings
    cached_per_layer_inputs: std::cell::RefCell<Option<UniquePtr<MlxArray>>>,
}

impl Gemma3nVLModel {
    pub fn new(
        text_model: crate::models::Gemma3nModel,
        vision_tower: encoders::gemma3n::Gemma3nVisionModel,
        embed_vision: crate::models::gemma3n::Gemma3nMultimodalEmbedder,
        processor: processors::siglip::SigLipProcessor,
        image_token_id: i32,
        boi_token_id: i32,
        eoi_token_id: i32,
        vision_hidden_size: usize,
    ) -> Self {
        Self {
            text_model,
            vision_tower,
            embed_vision,
            processor,
            image_token_id,
            boi_token_id,
            eoi_token_id,
            vision_hidden_size,
            cached_per_layer_inputs: std::cell::RefCell::new(None),
        }
    }

    /// Get input embeddings with vision features merged in
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
    ) -> merge::InputEmbeddings {
        // 1. Text embeddings
        let inputs_embeds = self.text_model.language_model.get_embed_tokens(input_ids);

        // 2. Per-layer inputs (image_token_id >= vocab_size_per_layer, auto-zeroed)
        let per_layer_inputs = self
            .text_model
            .language_model
            .get_per_layer_inputs(input_ids);
        let per_layer_inputs = self
            .text_model
            .language_model
            .project_per_layer_inputs(&inputs_embeds, &per_layer_inputs);

        // 3. Vision: pixel_values → VisionTower → [B, H, W, C] (NHWC)
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);
        let vision_out = self.vision_tower.forward(&pv);

        // Reshape NHWC → [B, num_patches, hidden_size]
        let vo = mlxcel_core::transpose_axes(&vision_out, &[0, 3, 1, 2]);
        let vo_shape = mlxcel_core::array_shape(&vo);
        let b = vo_shape[0];
        let c = vo_shape[1]; // hidden_size (2048)
        let num_patches = vo_shape[2] * vo_shape[3]; // H*W
        let vo = mlxcel_core::reshape(&vo, &[b, c, num_patches]);
        let vo = mlxcel_core::transpose_axes(&vo, &[0, 2, 1]); // [B, num_patches, hidden_size]

        // Scale by sqrt(vision_hidden_size)
        let scale = mlxcel_core::full_f32(
            &[1],
            (self.vision_hidden_size as f32).sqrt(),
            mlxcel_core::dtype::FLOAT32,
        );
        let vo = mlxcel_core::multiply(&vo, &scale);

        // 4. Multimodal embedder: → [B, num_patches, text_hidden]
        let image_features = self.embed_vision.forward_soft(&vo);

        // 5. masked_scatter merge
        let merged = merge::merge_llava(
            self.image_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        );

        // 6. Cache per_layer_inputs for use in forward_with_embeddings
        *self.cached_per_layer_inputs.borrow_mut() = Some(per_layer_inputs);

        merged
    }
}

impl LanguageModel for Gemma3nVLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.language_model.forward(input_ids, caches)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if let Some(embeds) = input_embeddings {
            // VLM prefill: use cached per_layer_inputs
            let pli = self.cached_per_layer_inputs.borrow_mut().take().unwrap();
            self.text_model
                .language_model
                .forward_with_inputs(embeds, &pli, caches)
        } else {
            self.text_model.language_model.forward(input_ids, caches)
        }
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.language_model.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        mlxcel_core::generate::LanguageModel::eos_token_ids(&self.text_model)
    }
}

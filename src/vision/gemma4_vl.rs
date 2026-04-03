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

//! Gemma4 Vision-Language Model.
//!
//! Used by: Gemma4 VLM

use super::{encoders, merge, processors};
use crate::LanguageModel;
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Gemma4MultimodalEmbedder {
    embedding_projection: UnifiedLinear,
    post_projection_norm: crate::models::gemma4::RMSNormNoScale,
}

impl Gemma4MultimodalEmbedder {
    pub fn from_weights(
        weights: &mlxcel_core::weights::WeightMap,
        prefix: &str,
        hidden_size: usize,
        eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            embedding_projection: UnifiedLinear::from_weights(
                weights,
                &format!("{}.embedding_projection", prefix),
                group_size,
                bits,
            )?,
            post_projection_norm: crate::models::gemma4::RMSNormNoScale::new(
                hidden_size as i32,
                eps,
            ),
        })
    }

    pub fn forward(&self, inputs_embeds: &MlxArray) -> UniquePtr<MlxArray> {
        let proj = self.embedding_projection.forward(inputs_embeds);
        self.post_projection_norm.forward(&proj)
    }
}

pub struct Gemma4VLModel {
    pub text_model: crate::models::Gemma4Wrapper,
    pub vision_tower: encoders::gemma4::Gemma4VisionModel,
    pub embed_vision: Gemma4MultimodalEmbedder,
    pub processor: processors::gemma4::Gemma4Processor,
    pub image_token_id: i32,
    pub boi_token_id: i32,
    pub eoi_token_id: i32,
    cached_per_layer_inputs: std::cell::RefCell<Option<UniquePtr<MlxArray>>>,
}

impl Gemma4VLModel {
    pub fn new(
        text_model: crate::models::Gemma4Wrapper,
        vision_tower: encoders::gemma4::Gemma4VisionModel,
        embed_vision: Gemma4MultimodalEmbedder,
        processor: processors::gemma4::Gemma4Processor,
        image_token_id: i32,
        boi_token_id: i32,
        eoi_token_id: i32,
    ) -> Self {
        Self {
            text_model,
            vision_tower,
            embed_vision,
            processor,
            image_token_id,
            boi_token_id,
            eoi_token_id,
            cached_per_layer_inputs: std::cell::RefCell::new(None),
        }
    }

    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        images: &[processors::gemma4::Gemma4ImageInput],
    ) -> merge::InputEmbeddings {
        let inputs_embeds = self.text_model.input_embeddings(input_ids);

        let image_token = mlxcel_core::from_slice_i32(&[self.image_token_id], &[1]);
        let is_image = mlxcel_core::equal(input_ids, &image_token);
        let zero_ids = mlxcel_core::zeros_like(input_ids);
        let per_layer_token_ids = mlxcel_core::where_cond(&is_image, &zero_ids, input_ids);
        let per_layer_inputs = self
            .text_model
            .get_per_layer_inputs(&per_layer_token_ids)
            .map(|per_layer| {
                self.text_model
                    .project_per_layer_inputs(&inputs_embeds, Some(per_layer.as_ref().unwrap()))
                    .expect("Gemma4 projected per-layer inputs")
            });

        let mut image_features = Vec::with_capacity(images.len());
        for image in images {
            let features = self
                .vision_tower
                .forward(image.pixel_values.as_ref().unwrap(), image.patch_grid);
            let features = self.embed_vision.forward(&features);
            image_features.push(features);
        }

        let image_features = if image_features.len() == 1 {
            mlxcel_core::astype(
                image_features[0].as_ref().unwrap(),
                mlxcel_core::array_dtype(&inputs_embeds),
            )
        } else {
            let merged = crate::vision::encoders::qwen2_vl::concat_many(&image_features, 1);
            mlxcel_core::astype(&merged, mlxcel_core::array_dtype(&inputs_embeds))
        };

        *self.cached_per_layer_inputs.borrow_mut() = per_layer_inputs;
        merge::merge_llava(
            self.image_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        )
    }
}

impl LanguageModel for Gemma4VLModel {
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
        if let Some(embeds) = input_embeddings {
            let per_layer_inputs = self.cached_per_layer_inputs.borrow_mut().take();
            self.text_model.forward_with_inputs(
                input_ids,
                Some(embeds),
                per_layer_inputs.as_ref().and_then(|arr| arr.as_ref()),
                mask,
            )
        } else {
            self.text_model.forward(input_ids, caches, mask)
        }
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.input_embeddings(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers_value()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.text_model.eos_token_ids_value()
    }

    fn supports_padded_prefill(&self) -> bool {
        false
    }

    fn supports_batching(&self) -> bool {
        false
    }
}

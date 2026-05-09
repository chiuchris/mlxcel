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

//! Nemotron H Nano Omni vision-language model wrapper (issue #554).
//!
//! Faithful Rust port of the vision path in
//! `references/mlx-vlm/mlx_vlm/models/nemotron_h_nano_omni/nemotron_h_nano_omni.py`.
//! Composes:
//! - the existing Nemotron-H text backbone
//!   ([`crate::models::NemotronHModel`])
//! - the RADIO vision tower
//!   ([`crate::vision::encoders::nemotron_h_nano_omni::NemotronHNanoOmniVisionModel`])
//! - the multimodal projector (`mlp1`): `RMSNorm -> Linear -> ReLU² ->
//!   Linear` with the upstream "pixel shuffle" downsample applied to
//!   the patch grid before projection
//!
//! Audio is intentionally out of scope for this port — see issue #554
//! (PR vision-only scope decision). This wrapper raises an error when
//! the loader supplies an audio config, rather than partially wiring
//! up audio embeddings.
//!
//! Used by: Nemotron H Nano Omni VLM

use crate::LanguageModel;
use crate::models::NemotronHModel;
use crate::vision::encoders::nemotron_h_nano_omni::NemotronHNanoOmniVisionModel;
use crate::vision::merge::InputEmbeddings;
use crate::vision::processors::nemotron_h_nano_omni::{
    NemotronHNanoOmniImageInput, NemotronHNanoOmniImageProcessor,
};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Multimodal projector configuration.
///
/// Mirrors the upstream `ModelConfig` fields that drive `VisionProjection`
/// and the `pixel_shuffle` downsample. The defaults match the released
/// 30B-A3B Nano Omni checkpoint.
#[derive(Debug, Clone)]
pub struct NemotronHNanoOmniVlConfig {
    pub vit_hidden_size: usize,
    pub projector_hidden_size: usize,
    pub text_hidden_size: usize,
    pub downsample_ratio: f32,
    pub ps_version: String,
    /// Image placeholder token ID expanded into the prompt before
    /// `get_input_embeddings`. Mirrors upstream `img_context_token_id`.
    pub img_context_token_id: i32,
    /// Image-start / image-end framing token IDs. Both default to `0`
    /// when the upstream config does not surface them — `0` means "no
    /// framing tokens emitted in the prompt expansion".
    pub image_start_token_id: i32,
    pub image_end_token_id: i32,
    /// EOS token IDs from the released checkpoint's `generation_config.json`.
    pub eos_token_ids: Vec<i32>,
}

/// Inverse of `downsample_ratio` rounded to the nearest integer.
fn downsample_factor(ratio: f32) -> i32 {
    let raw = (1.0 / ratio.max(f32::EPSILON)).round() as i32;
    raw.max(1)
}

/// Multimodal projector (`mlp1`).
///
/// Faithful port of upstream `VisionProjection`:
/// `RMSNorm(in_features) -> Linear(in_features, proj_hidden) ->
/// SquaredReLU -> Linear(proj_hidden, text_hidden)`.
/// The upstream `sanitize` rewrites `mlp1.0.* -> mlp1.layers.0.*` etc.,
/// so the loader applies the same rewrite and we read from
/// `mlp1.layers.{0,1,3}`.
pub struct NemotronHNanoOmniProjector {
    norm: RMSNorm,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl NemotronHNanoOmniProjector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let norm_weight = weights
            .get(&format!("{prefix}.layers.0.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| {
                format!("Missing multimodal projector norm weight ({prefix}.layers.0.weight)")
            })?;
        let norm = RMSNorm::new(norm_weight, 1e-5);
        let fc1 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.layers.1"), group_size, bits)?;
        let fc2 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.layers.3"), group_size, bits)?;
        Ok(Self { norm, fc1, fc2 })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.norm.forward(x);
        let h = self.fc1.forward(&h);
        let h = mlxcel_core::utils::relu_squared(&h);
        self.fc2.forward(&h)
    }
}

/// Top-level Nemotron H Nano Omni VLM (vision-only scope, issue #554).
pub struct NemotronHNanoOmniVlModel {
    pub text_model: NemotronHModel,
    pub vision_tower: NemotronHNanoOmniVisionModel,
    pub projector: NemotronHNanoOmniProjector,
    pub processor: NemotronHNanoOmniImageProcessor,
    pub config: NemotronHNanoOmniVlConfig,
}

impl NemotronHNanoOmniVlModel {
    pub fn new(
        text_model: NemotronHModel,
        vision_tower: NemotronHNanoOmniVisionModel,
        projector: NemotronHNanoOmniProjector,
        processor: NemotronHNanoOmniImageProcessor,
        config: NemotronHNanoOmniVlConfig,
    ) -> Self {
        Self {
            text_model,
            vision_tower,
            projector,
            processor,
            config,
        }
    }

    /// Run the vision tower + projector for a single preprocessed image.
    /// Returns `[1, num_tokens, text_hidden]` ready to scatter into the
    /// text embedding stream.
    fn extract_image_features_single(
        &self,
        image: &NemotronHNanoOmniImageInput,
    ) -> UniquePtr<MlxArray> {
        let pixel_values = image.pixel_values.as_ref().unwrap();
        let radio = self.vision_tower.forward(pixel_values, false);
        let features = radio.features;

        let in_shape = mlxcel_core::array_shape(pixel_values);
        let height = in_shape[2];
        let width = in_shape[3];
        let patch_size = self.vision_tower.patch_size() as i32;
        let patch_h = height / patch_size;
        let patch_w = width / patch_size;

        let feat_shape = mlxcel_core::array_shape(&features);
        let batch = feat_shape[0];
        let channels = feat_shape[2];
        let reshaped = mlxcel_core::reshape(&features, &[batch, patch_h, patch_w, channels]);
        let shuffled = self.pixel_shuffle(&reshaped);

        let post_shape = mlxcel_core::array_shape(&shuffled);
        let post_channels = post_shape[3];
        let flattened = mlxcel_core::reshape(&shuffled, &[batch, -1, post_channels]);
        self.projector.forward(&flattened)
    }

    /// Run the vision pipeline across all images and concatenate the
    /// per-image features along the token axis. Mirrors upstream
    /// `extract_feature` behaviour for a list of pixel tensors.
    pub fn extract_image_features(
        &self,
        images: &[NemotronHNanoOmniImageInput],
    ) -> Option<UniquePtr<MlxArray>> {
        if images.is_empty() {
            return None;
        }
        let mut iter = images.iter();
        let first = self.extract_image_features_single(iter.next()?);
        let mut acc = first;
        for image in iter {
            let features = self.extract_image_features_single(image);
            acc = mlxcel_core::concatenate(&acc, &features, 0);
        }
        Some(acc)
    }

    /// Compose vision features with text embeddings at image-token slots.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        images: &[NemotronHNanoOmniImageInput],
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.input_embeddings(input_ids);

        let Some(features) = self.extract_image_features(images) else {
            return InputEmbeddings {
                inputs_embeds,
                attention_mask_4d: None,
            };
        };

        // Cast features to text-embedding dtype before scatter so the
        // merge helper sees a uniform dtype across vision and text.
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let features = mlxcel_core::astype(&features, embed_dtype);

        crate::vision::merge::merge_llava(
            self.config.img_context_token_id,
            &features,
            &inputs_embeds,
            input_ids,
        )
    }

    fn pixel_shuffle(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let scale = self.config.downsample_ratio;
        if scale == 1.0 {
            return mlxcel_core::copy(x);
        }
        let factor = downsample_factor(scale);

        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let width = shape[1];
        let height = shape[2];
        let channels = shape[3];

        // Step 1: reshape (B, W, H, C) -> (B, W, H/factor, C*factor)
        let new_h = height / factor;
        let new_c1 = channels * factor;
        let h1 = mlxcel_core::reshape(x, &[batch, width, new_h, new_c1]);

        // Step 2: transpose (0, 2, 1, 3) -> (B, H/factor, W, C*factor)
        let h2 = mlxcel_core::transpose_axes(&h1, &[0, 2, 1, 3]);

        // Step 3: reshape -> (B, H/factor, W/factor, C*factor*factor)
        let new_w = width / factor;
        let new_c2 = new_c1 * factor;
        let h3 = mlxcel_core::reshape(&h2, &[batch, new_h, new_w, new_c2]);

        // Step 4: optional final transpose for `ps_version != "v1"`.
        if self.config.ps_version != "v1" {
            mlxcel_core::transpose_axes(&h3, &[0, 2, 1, 3])
        } else {
            h3
        }
    }

    /// Hidden size advertised to the runtime — same as the text backbone.
    pub fn hidden_size(&self) -> usize {
        self.text_model.hidden_size()
    }
}

impl LanguageModel for NemotronHNanoOmniVlModel {
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
            self.text_model.forward_with_inputs_embeds(embeds)
        } else {
            self.text_model.forward(input_ids, caches, mask)
        }
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.input_embeddings(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // NemotronH owns its mixed-cache state internally and exposes
        // empty `KVCache` placeholders to satisfy the trait — the VLM
        // wrapper passes those through unchanged via the trait method.
        <NemotronHModel as LanguageModel>::make_caches(&self.text_model)
    }

    fn num_layers(&self) -> usize {
        <NemotronHModel as LanguageModel>::num_layers(&self.text_model)
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        if !self.config.eos_token_ids.is_empty() {
            self.config.eos_token_ids.clone()
        } else {
            <NemotronHModel as LanguageModel>::eos_token_ids(&self.text_model)
        }
    }

    fn supports_padded_prefill(&self) -> bool {
        // Nemotron-H is a hybrid Mamba+Attention model: padding tokens
        // corrupt Mamba recurrent state. The text path declares this
        // explicitly; the VLM wrapper inherits the same constraint.
        false
    }

    fn supports_batching(&self) -> bool {
        // Internal cache state of the text backbone is not per-sequence
        // isolated, so multiple sequences cannot share one model
        // instance.
        false
    }

    fn trim_internal_caches(&self, excess: i32) {
        self.text_model.trim_internal_caches(excess);
    }
}

#[cfg(test)]
#[path = "nemotron_h_nano_omni_vl_tests.rs"]
mod tests;

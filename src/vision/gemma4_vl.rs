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

//! Gemma4 Vision-Language Model (with optional audio support).
//!
//! Used by: Gemma4 VLM

use super::feature_cache::{CacheKey, SingleArrayFeatures, VisionFeatureCache};
use super::{encoders, merge, processors};
use crate::LanguageModel;
use crate::audio;
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Gemma4MultimodalEmbedder {
    embedding_projection: UnifiedLinear,
    pre_projection_norm: crate::models::gemma4::RMSNormNoScale,
}

impl Gemma4MultimodalEmbedder {
    /// Construct the Gemma 4 multimodal embedder.
    ///
    /// `embedding_dim` is the input feature dimension (the vision or audio
    /// encoder output size) — i.e. the column count of the projection weight
    /// `embedding_projection.weight`. The RMS norm is applied on this dim
    /// BEFORE the projection, matching upstream mlx-vlm after the reordering
    /// that renamed `embedding_post_projection_norm` to
    /// `embedding_pre_projection_norm`.
    pub fn from_weights(
        weights: &mlxcel_core::weights::WeightMap,
        prefix: &str,
        embedding_dim: usize,
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
            pre_projection_norm: crate::models::gemma4::RMSNormNoScale::new(
                embedding_dim as i32,
                eps,
            ),
        })
    }

    pub fn forward(&self, inputs_embeds: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.pre_projection_norm.forward(inputs_embeds);
        self.embedding_projection.forward(&normed)
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
    pub audio_tower: Option<audio::AudioEncoder>,
    pub embed_audio: Option<Gemma4MultimodalEmbedder>,
    pub audio_token_id: i32,
    pub boa_token_id: i32,
    pub eoa_token_id: i32,
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
            audio_tower: None,
            embed_audio: None,
            audio_token_id: 258_881,
            boa_token_id: 256_000,
            eoa_token_id: 258_883,
            cached_per_layer_inputs: std::cell::RefCell::new(None),
        }
    }

    /// Set audio tower and embedder for audio-capable models.
    pub fn set_audio(
        &mut self,
        audio_tower: audio::AudioEncoder,
        embed_audio: Gemma4MultimodalEmbedder,
        audio_token_id: i32,
        boa_token_id: i32,
        eoa_token_id: i32,
    ) {
        self.audio_tower = Some(audio_tower);
        self.embed_audio = Some(embed_audio);
        self.audio_token_id = audio_token_id;
        self.boa_token_id = boa_token_id;
        self.eoa_token_id = eoa_token_id;
    }

    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        images: &[processors::gemma4::Gemma4ImageInput],
    ) -> merge::InputEmbeddings {
        self.get_input_embeddings_with_audio_and_cache(input_ids, images, None, None, None, None)
    }

    /// Compute input embeddings with both vision and audio features.
    pub fn get_input_embeddings_with_audio(
        &self,
        input_ids: &MlxArray,
        images: &[processors::gemma4::Gemma4ImageInput],
        audio_features: Option<&MlxArray>,
        audio_mask: Option<&MlxArray>,
    ) -> merge::InputEmbeddings {
        self.get_input_embeddings_with_audio_and_cache(
            input_ids,
            images,
            audio_features,
            audio_mask,
            None,
            None,
        )
    }

    /// Compute input embeddings with an optional per-image vision feature cache.
    ///
    /// `image_keys` — when `Some(..)`, must have the same length as `images`.
    /// Each entry is the [`CacheKey`] to look up for that image. `None` entries
    /// skip the cache and always run the vision tower.
    ///
    /// `vision_cache` — when `Some(..)`, provides shared storage for post-
    /// projection image features. The cache is a no-op when disabled (its
    /// `max_size == 0`); see [`VisionFeatureCache`] for LRU semantics.
    ///
    /// Cache hits short-circuit both `self.vision_tower.forward(...)` and
    /// `self.embed_vision.forward(...)`. On miss, the freshly-computed features
    /// are inserted before merging.
    pub fn get_input_embeddings_with_audio_and_cache(
        &self,
        input_ids: &MlxArray,
        images: &[processors::gemma4::Gemma4ImageInput],
        audio_features: Option<&MlxArray>,
        audio_mask: Option<&MlxArray>,
        image_keys: Option<&[Option<CacheKey>]>,
        vision_cache: Option<&std::sync::Mutex<VisionFeatureCache<SingleArrayFeatures>>>,
    ) -> merge::InputEmbeddings {
        // Apply the `sqrt(hidden_size)` embed scale to the text embeddings
        // once, up front. Vision features (produced by `self.embed_vision`)
        // and audio features (produced by `self.embed_audio`) are already in
        // the language-model embedding space, so they must NOT be scaled
        // again. `Gemma4TextModel::forward` detects that we are passing
        // `input_embeddings` and skips its own embed scale to avoid
        // double-scaling the text tokens. See issue #317.
        let inputs_embeds = self.text_model.input_embeddings(input_ids);
        let inputs_embeds = mlxcel_core::multiply_scalar(
            &inputs_embeds,
            (self.text_model.hidden_size() as f32).sqrt(),
        );

        // Build per-layer inputs, masking out both image and audio tokens
        let image_token = mlxcel_core::from_slice_i32(&[self.image_token_id], &[1]);
        let is_image = mlxcel_core::equal(input_ids, &image_token);

        let is_multimodal = if self.audio_tower.is_some() {
            let audio_token = mlxcel_core::from_slice_i32(&[self.audio_token_id], &[1]);
            let is_audio = mlxcel_core::equal(input_ids, &audio_token);
            mlxcel_core::logical_or(&is_image, &is_audio)
        } else {
            is_image
        };

        let zero_ids = mlxcel_core::zeros_like(input_ids);
        let per_layer_token_ids = mlxcel_core::where_cond(&is_multimodal, &zero_ids, input_ids);
        let per_layer_inputs = self
            .text_model
            .get_per_layer_inputs(&per_layer_token_ids)
            .map(|per_layer| {
                self.text_model
                    .project_per_layer_inputs(&inputs_embeds, Some(per_layer.as_ref().unwrap()))
                    .expect("Gemma4 projected per-layer inputs")
            });

        // Vision features — consult the per-image cache first when enabled.
        //
        // Cache semantics: `image_keys[i] == Some(key)` plus a non-None
        // `vision_cache` triggers a lookup; on hit we skip the vision tower
        // and the multimodal embedder. On miss we run both, then insert the
        // result so the next turn that ships the same image can reuse it.
        let mut image_features = Vec::with_capacity(images.len());
        for (idx, image) in images.iter().enumerate() {
            let cache_key = image_keys
                .and_then(|keys| keys.get(idx))
                .and_then(|k| k.as_ref());

            // Try cache hit when both a key and a cache were provided.
            if let (Some(key), Some(cache)) = (cache_key, vision_cache)
                && let Ok(mut guard) = cache.lock()
                && let Some(cached) = guard.get(key)
            {
                image_features.push(cached.features);
                continue;
            }

            let features = self
                .vision_tower
                .forward(image.pixel_values.as_ref().unwrap(), image.patch_grid);
            let features = self.embed_vision.forward(&features);

            // Populate the cache on miss. We store a deep copy so the
            // returned `features` remains free for the merge path below.
            if let (Some(key), Some(cache)) = (cache_key, vision_cache) {
                // Materialize before hashing/storing; the cache must hold a
                // stable tensor rather than a deferred graph node.
                mlxcel_core::eval(&features);
                let snapshot =
                    SingleArrayFeatures::new(mlxcel_core::copy(features.as_ref().unwrap()));
                if let Ok(mut guard) = cache.lock() {
                    guard.put(key.clone(), &snapshot);
                }
            }

            image_features.push(features);
        }

        let mut result_embeds = if !image_features.is_empty() {
            let image_features_merged = if image_features.len() == 1 {
                mlxcel_core::astype(
                    image_features[0].as_ref().unwrap(),
                    mlxcel_core::array_dtype(&inputs_embeds),
                )
            } else {
                let merged = crate::vision::encoders::qwen2_vl::concat_many(&image_features, 1);
                mlxcel_core::astype(&merged, mlxcel_core::array_dtype(&inputs_embeds))
            };

            merge::merge_llava(
                self.image_token_id,
                &image_features_merged,
                &inputs_embeds,
                input_ids,
            )
        } else {
            merge::InputEmbeddings {
                inputs_embeds: mlxcel_core::copy(&inputs_embeds),
                attention_mask_4d: None,
            }
        };

        // Audio features
        if let (Some(audio_tower), Some(embed_audio), Some(audio_feat)) =
            (&self.audio_tower, &self.embed_audio, audio_features)
        {
            let audio_mel_mask = audio_mask.map_or_else(
                || {
                    let shape = mlxcel_core::array_shape(audio_feat);
                    mlxcel_core::zeros(&[shape[0], shape[1]], mlxcel_core::dtype::BOOL)
                },
                mlxcel_core::copy,
            );

            let (audio_encodings, _) = audio_tower.forward(audio_feat, &audio_mel_mask);
            let audio_encodings = embed_audio.forward(&audio_encodings);

            let current_embeds = &result_embeds.inputs_embeds;
            let audio_encodings =
                mlxcel_core::astype(&audio_encodings, mlxcel_core::array_dtype(current_embeds));

            // masked_scatter: replace audio token positions with audio encodings
            let audio_token_arr = mlxcel_core::from_slice_i32(&[self.audio_token_id], &[1]);
            let audio_token_mask = mlxcel_core::equal(input_ids, &audio_token_arr);
            let audio_mask_expanded = mlxcel_core::expand_dims(&audio_token_mask, -1);
            let audio_mask_expanded = mlxcel_core::broadcast_to(
                &audio_mask_expanded,
                &mlxcel_core::array_shape(current_embeds),
            );

            let scattered = masked_scatter(current_embeds, &audio_mask_expanded, &audio_encodings);
            result_embeds.inputs_embeds = scattered;
        }

        *self.cached_per_layer_inputs.borrow_mut() = per_layer_inputs;
        result_embeds
    }
}

/// Scatter source values into input at positions where mask is true.
///
/// Equivalent to Python's `masked_scatter`: flattens mask, computes cumulative
/// indices, aligns source, and uses where_cond.
fn masked_scatter(input: &MlxArray, mask: &MlxArray, source: &MlxArray) -> UniquePtr<MlxArray> {
    let input_shape = mlxcel_core::array_shape(input);
    let total_size: i32 = input_shape.iter().product();

    let mask_flat = mlxcel_core::reshape(mask, &[total_size]);
    let mask_i32 = mlxcel_core::astype(&mask_flat, mlxcel_core::dtype::INT32);
    let indices = mlxcel_core::cumsum(&mask_i32, 0, false, true);
    let ones = mlxcel_core::ones(&[1], mlxcel_core::dtype::INT32);
    let indices = mlxcel_core::subtract(&indices, &ones);

    let source_flat_size: i32 = mlxcel_core::array_shape(source).iter().product();
    let source_size_arr = mlxcel_core::from_slice_i32(&[source_flat_size], &[1]);
    let safe_indices = mlxcel_core::remainder(&indices, &source_size_arr);

    let source_flat = mlxcel_core::reshape(source, &[source_flat_size]);
    let aligned = mlxcel_core::take(&source_flat, &safe_indices, 0);

    let input_flat = mlxcel_core::reshape(input, &[total_size]);
    let result = mlxcel_core::where_cond(&mask_flat, &aligned, &input_flat);
    mlxcel_core::reshape(&result, &input_shape)
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

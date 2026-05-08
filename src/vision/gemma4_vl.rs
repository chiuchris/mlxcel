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
use super::gemma4_multimodal_embedder::{Gemma4MultimodalEmbedder, masked_scatter};
use super::gemma4_per_layer_inputs_state::Gemma4PerLayerInputsState;
use super::{encoders, merge, processors};
use crate::LanguageModel;
use crate::audio;
use crate::multimodal::batched_dispatch::forward_batched_with_seq_ids_dispatch;
use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::generate::DecodeBatchContext;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

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
    /// Per-`SequenceId` storage for projected `per_layer_inputs`
    /// (issue #543). Mirrors `MRopeState` (#540).
    per_layer_inputs_state: Gemma4PerLayerInputsState,
    _weight_backing: crate::models::Gemma4WeightBacking,
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
            per_layer_inputs_state: Gemma4PerLayerInputsState::new(),
            _weight_backing: crate::models::Gemma4WeightBacking::default(),
        }
    }

    pub(crate) fn set_weight_backing(
        &mut self,
        weight_backing: crate::models::Gemma4WeightBacking,
    ) {
        self._weight_backing = weight_backing;
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

        // Issue #543: park the freshly projected tensor in the
        // container's fallback slot. The scheduler binds it to a
        // `SequenceId` right after `prepare_request_vlm_embeddings`
        // returns; legacy CLI/single-row callers consume it via
        // `take_fallback` on the next prefill.
        self.per_layer_inputs_state.set_fallback(per_layer_inputs);
        result_embeds
    }

    // -- Per-sequence per_layer_inputs (issue #543) --------------------
    //
    // These thin wrappers route through `Gemma4PerLayerInputsState`. The
    // scheduler calls them via `LoadedModel::*` capability helpers (see
    // `loaded_model_capabilities.rs`) right after
    // `prepare_request_vlm_embeddings` so a burst of Gemma 4 VLM
    // requests cannot have one row's prefill consume another row's
    // tensor. Mirrors the Qwen MRoPE binding flow from issue #540.

    /// Drain the container's fallback slot into the per-`SequenceId`
    /// map under `seq_id`. No-op when the slot is empty (E1B variant
    /// or text-only request).
    pub fn bind_per_layer_inputs_to_sequence(&self, seq_id: SequenceId) {
        self.per_layer_inputs_state
            .bind_fallback_to_sequence(seq_id);
    }

    /// Drop a sequence's stored `per_layer_inputs`. Called from
    /// [`Self::release_sequence_state_by_id`] so the map drains in
    /// step with the scheduler's per-sequence cache cleanup.
    pub fn release_per_layer_inputs(&self, seq_id: SequenceId) {
        self.per_layer_inputs_state.release_sequence(seq_id);
    }

    /// Take a sequence's tensor out of the map without dropping the
    /// `UniquePtr`. Used by the scheduler's preemption path so the
    /// tensor can be carried across the eviction (which releases the
    /// old sequence id) and reinstalled under the freshly allocated
    /// id with [`Self::install_per_layer_inputs_for_sequence`].
    pub fn take_per_layer_inputs_for_sequence(
        &self,
        seq_id: SequenceId,
    ) -> Option<UniquePtr<MlxArray>> {
        self.per_layer_inputs_state.take_for_sequence(seq_id)
    }

    /// Re-install a previously taken tensor under `seq_id`. No-op
    /// when the snapshot is `None`.
    pub fn install_per_layer_inputs_for_sequence(
        &self,
        seq_id: SequenceId,
        snapshot: Option<UniquePtr<MlxArray>>,
    ) {
        if let Some(value) = snapshot {
            self.per_layer_inputs_state.bind_for_sequence(seq_id, value);
        }
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
        // No `seq_id` plumbed through (legacy CLI / direct VLM call).
        // Route to the text model's fallback `internal` cache slot.
        self.forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            None,
            caches,
            mask,
        )
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_with_sequence_id(input_ids, seq_id, caches, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if let Some(embeds) = input_embeddings {
            // Issue #543: prefer the per-`SequenceId` slot so each
            // row of a burst-enqueued batch sees its own projection.
            // Fall back to the legacy fallback slot when there is no
            // `seq_id` (CLI/single-row) or the bind step did not run
            // before this consumer.
            let per_layer_inputs = match seq_id {
                Some(id) => self
                    .per_layer_inputs_state
                    .take_for_sequence(id)
                    .or_else(|| self.per_layer_inputs_state.take_fallback()),
                None => self.per_layer_inputs_state.take_fallback(),
            };
            self.text_model.forward_with_inputs_and_sequence_id(
                input_ids,
                Some(embeds),
                per_layer_inputs.as_ref().and_then(|arr| arr.as_ref()),
                mask,
                seq_id,
            )
        } else {
            self.text_model
                .forward_with_sequence_id(input_ids, seq_id, caches, mask)
        }
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.input_embeddings(input_ids))
    }

    /// Empty external caches: the text wrapper owns all cache state
    /// internally and resolves it per `SequenceId`. The matching layout
    /// descriptor is [`SequenceStateLayout::model_owned`] returned by
    /// [`Self::sequence_state_layout`].
    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        self.text_model.sequence_state_layout()
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.text_model.prepare_sequence_state(seq_id);
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        // Issue #543: drop the per-sequence `per_layer_inputs`
        // alongside the text model's per-sequence cache release so
        // the map cannot grow without bound across long-running
        // server sessions.
        self.per_layer_inputs_state.release_sequence(seq_id);
        self.text_model.release_sequence_state_by_id(seq_id);
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

    /// Issue #542: Gemma 4 supports batched decode now that the inner
    /// [`crate::models::Gemma4Wrapper`] uses per-`SequenceId` cache
    /// isolation via `ModelOwnedSequenceState<Cache>`. The
    /// `forward_batched_with_context_and_ids` override below routes each
    /// row through `forward_with_sequence_id` so per-sequence cache state
    /// resolves correctly even with mixed prompt lengths.
    fn supports_batching(&self) -> bool {
        true
    }

    /// Issue #542: per-row batched dispatch with seq_ids so each row of a
    /// mixed-length batch reaches the text model's seq-aware forward path
    /// independently. Mirrors the Qwen VL fix in PR #558 and shares the
    /// same helper (`multimodal::batched_dispatch`).
    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        forward_batched_with_seq_ids_dispatch(
            &self.text_model,
            input_ids,
            seq_ids,
            batch_caches,
            mask,
            context,
        )
    }
}

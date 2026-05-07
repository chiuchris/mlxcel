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

//! Qwen-VL prompt token insertion rules.
//!
//! Qwen2/2.5/3/3.5-VL families reserve image-token blocks based on the image
//! grid and spatial merge size. This module keeps that token arithmetic out of
//! CLI/server callers so Qwen-VL prompt preparation stays consistent.

use crate::models::qwen_mrope_state::MRopeEntry;
use crate::vision;
use crate::vision::feature_cache::{CacheKey, ModelVisionCaches};
use mlxcel_core::MlxArray;
use mlxcel_core::cache::SequenceId;

#[derive(Clone, Copy)]
pub struct QwenVlmPromptInfo<'a> {
    pub processor: &'a vision::processors::qwen2_vl::Qwen2VLProcessor,
    pub spatial_merge_size: usize,
    pub vision_start_token_id: i32,
    pub image_token_id: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedQwenVlmTokens {
    pub image_blocks: usize,
    pub total_image_tokens: i32,
}

/// Opaque container for an MRoPE entry that has been removed from the
/// per-sequence map. Used so callers (e.g. the server preemption path)
/// can carry the entry across operations that release the original
/// sequence id without leaking the underlying `MRopeEntry` type.
pub struct QwenVlMRopeSnapshot(pub(crate) Option<MRopeEntry>);

impl QwenVlMRopeSnapshot {
    /// True when the snapshot holds no entry (text-only or already-released).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }
}

pub trait QwenVlRuntime {
    fn prompt_info(&self) -> QwenVlmPromptInfo<'_>;
    fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> vision::merge::InputEmbeddings;

    /// Variant of [`input_embeddings`] that consults a shared vision feature
    /// cache. Implementors that do not support caching (e.g. older Qwen-VL
    /// variants not yet wired for the cache) should fall through to the plain
    /// [`input_embeddings`] path. The default implementation here matches that
    /// pass-through behavior so trait users always get *something* compiled.
    ///
    /// `caches` is shared per model instance. Runtimes whose vision output
    /// shape matches [`super::feature_cache::SingleArrayFeatures`] use
    /// `caches.single`; Qwen3-VL uses `caches.deepstack`.
    fn input_embeddings_with_cache(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
        _cache_key: Option<&CacheKey>,
        _caches: Option<&ModelVisionCaches>,
    ) -> vision::merge::InputEmbeddings {
        self.input_embeddings(input_ids, pixel_values, grid_thw)
    }

    /// Bind the MRoPE state computed during embedding preparation to a
    /// specific `SequenceId` so the per-row delta cannot leak into other
    /// requests' decode steps (issue #540 / mlx-vlm PR #1095).
    ///
    /// The default implementation is a no-op for runtimes that do not use
    /// MRoPE (in mlxcel today this trait only covers Qwen VL families that
    /// always use MRoPE, but the default keeps the trait additive).
    fn bind_mrope_state_to_sequence(&self, _seq_id: SequenceId) {}

    /// Take the per-sequence MRoPE entry under `seq_id` out of the
    /// language model's per-sequence map. Used by the server preemption
    /// path so the entry survives the eviction (which releases the old
    /// sequence id) and can be reinstalled under the freshly allocated id.
    ///
    /// The default returns an empty snapshot so non-Qwen runtimes are
    /// unaffected.
    fn take_mrope_entry_for_sequence(&self, _seq_id: SequenceId) -> QwenVlMRopeSnapshot {
        QwenVlMRopeSnapshot(None)
    }

    /// Re-install a previously taken MRoPE entry under `seq_id`. The
    /// default is a no-op so non-Qwen runtimes are unaffected.
    fn install_mrope_entry_for_sequence(
        &self,
        _seq_id: SequenceId,
        _snapshot: QwenVlMRopeSnapshot,
    ) {
    }

    // NOTE: per-row batched dispatch lives directly on each
    // `vision::Qwen*VLModel`'s `LanguageModel::forward_batched_with_context_and_ids`
    // override, not on this trait. Most wrappers delegate to the free
    // helper [`forward_batched_with_seq_ids_dispatch`]; Qwen3.5 forwards
    // straight to its text model's batched-with-ids method.
}

/// Per-row batched dispatch helper. Re-exported for backwards
/// compatibility with the Qwen VL wrappers that imported this symbol
/// when the helper lived in this module (PR #558). The implementation
/// now lives in [`super::batched_dispatch`] so Gemma 4 (issue #542) and
/// the Qwen VL families share a single source of truth — see the
/// duplication report flagged on PR #560 (M-2).
///
/// Used by: Qwen2VLModel, Qwen25VLModel, Qwen3VLModel, Qwen3VLMoeModel.
pub use super::batched_dispatch::forward_batched_with_seq_ids_dispatch;

/// Per-row dispatch shared by every Qwen VL wrapper whose text model
/// uses the default `forward_batched_with_context_and_ids` trait impl
/// (i.e. all of them except Qwen3.5). Calls the shared helper.
macro_rules! impl_qwen_vl_runtime_loop_dispatch {
    ($ty:ty) => {
        impl QwenVlRuntime for $ty {
            fn prompt_info(&self) -> QwenVlmPromptInfo<'_> {
                QwenVlmPromptInfo {
                    processor: &self.processor,
                    spatial_merge_size: self.spatial_merge_size,
                    vision_start_token_id: self.vision_start_token_id,
                    image_token_id: self.image_token_id,
                }
            }

            fn input_embeddings(
                &self,
                input_ids: &MlxArray,
                pixel_values: &MlxArray,
                grid_thw: &[(i32, i32, i32)],
            ) -> vision::merge::InputEmbeddings {
                self.get_input_embeddings(input_ids, pixel_values, grid_thw)
            }

            fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
                self.text_model.bind_mrope_state_to_sequence(seq_id);
            }

            fn take_mrope_entry_for_sequence(&self, seq_id: SequenceId) -> QwenVlMRopeSnapshot {
                QwenVlMRopeSnapshot(self.text_model.take_mrope_entry(seq_id))
            }

            fn install_mrope_entry_for_sequence(
                &self,
                seq_id: SequenceId,
                snapshot: QwenVlMRopeSnapshot,
            ) {
                if let Some(entry) = snapshot.0 {
                    self.text_model.install_mrope_entry(seq_id, entry);
                }
            }
        }
    };
}

// Runtimes without cache wiring (yet) — they fall back to the default
// trait method which just routes through `input_embeddings`.
impl_qwen_vl_runtime_loop_dispatch!(vision::Qwen2VLModel);
impl_qwen_vl_runtime_loop_dispatch!(vision::Qwen3VLMoeModel);

// Qwen2.5-VL: single-array cache path.
impl QwenVlRuntime for vision::Qwen25VLModel {
    fn prompt_info(&self) -> QwenVlmPromptInfo<'_> {
        QwenVlmPromptInfo {
            processor: &self.processor,
            spatial_merge_size: self.spatial_merge_size,
            vision_start_token_id: self.vision_start_token_id,
            image_token_id: self.image_token_id,
        }
    }

    fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> vision::merge::InputEmbeddings {
        self.get_input_embeddings(input_ids, pixel_values, grid_thw)
    }

    fn input_embeddings_with_cache(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
        cache_key: Option<&CacheKey>,
        caches: Option<&ModelVisionCaches>,
    ) -> vision::merge::InputEmbeddings {
        self.get_input_embeddings_with_cache(
            input_ids,
            pixel_values,
            grid_thw,
            cache_key,
            caches.map(|c| &c.single),
        )
    }

    fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.text_model.bind_mrope_state_to_sequence(seq_id);
    }

    fn take_mrope_entry_for_sequence(&self, seq_id: SequenceId) -> QwenVlMRopeSnapshot {
        QwenVlMRopeSnapshot(self.text_model.take_mrope_entry(seq_id))
    }

    fn install_mrope_entry_for_sequence(&self, seq_id: SequenceId, snapshot: QwenVlMRopeSnapshot) {
        if let Some(entry) = snapshot.0 {
            self.text_model.install_mrope_entry(seq_id, entry);
        }
    }
}

// Qwen3-VL: DeepStack-shaped cache path.
impl QwenVlRuntime for vision::Qwen3VLModel {
    fn prompt_info(&self) -> QwenVlmPromptInfo<'_> {
        QwenVlmPromptInfo {
            processor: &self.processor,
            spatial_merge_size: self.spatial_merge_size,
            vision_start_token_id: self.vision_start_token_id,
            image_token_id: self.image_token_id,
        }
    }

    fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> vision::merge::InputEmbeddings {
        self.get_input_embeddings(input_ids, pixel_values, grid_thw)
    }

    fn input_embeddings_with_cache(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
        cache_key: Option<&CacheKey>,
        caches: Option<&ModelVisionCaches>,
    ) -> vision::merge::InputEmbeddings {
        self.get_input_embeddings_with_cache(
            input_ids,
            pixel_values,
            grid_thw,
            cache_key,
            caches.map(|c| &c.deepstack),
        )
    }

    fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.text_model.bind_mrope_state_to_sequence(seq_id);
    }

    fn take_mrope_entry_for_sequence(&self, seq_id: SequenceId) -> QwenVlMRopeSnapshot {
        QwenVlMRopeSnapshot(self.text_model.take_mrope_entry(seq_id))
    }

    fn install_mrope_entry_for_sequence(&self, seq_id: SequenceId, snapshot: QwenVlMRopeSnapshot) {
        if let Some(entry) = snapshot.0 {
            self.text_model.install_mrope_entry(seq_id, entry);
        }
    }
}

// Qwen3.5-VL: text model already implements
// `forward_batched_with_context_and_ids` natively (per-row dispatch and
// batched-prefill fast path), so the wrapper forwards directly to it.
impl QwenVlRuntime for vision::Qwen35VLModel {
    fn prompt_info(&self) -> QwenVlmPromptInfo<'_> {
        QwenVlmPromptInfo {
            processor: &self.processor,
            spatial_merge_size: self.spatial_merge_size,
            vision_start_token_id: self.vision_start_token_id,
            image_token_id: self.image_token_id,
        }
    }

    fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> vision::merge::InputEmbeddings {
        self.get_input_embeddings(input_ids, pixel_values, grid_thw)
    }

    fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.text_model.bind_mrope_state_to_sequence(seq_id);
    }

    fn take_mrope_entry_for_sequence(&self, seq_id: SequenceId) -> QwenVlMRopeSnapshot {
        QwenVlMRopeSnapshot(self.text_model.take_mrope_entry(seq_id))
    }

    fn install_mrope_entry_for_sequence(&self, seq_id: SequenceId, snapshot: QwenVlMRopeSnapshot) {
        if let Some(entry) = snapshot.0 {
            self.text_model.install_mrope_entry(seq_id, entry);
        }
    }
}

pub fn insert_qwen_vl_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    grid_thw: &[(i32, i32, i32)],
    spatial_merge_size: usize,
    vision_start_token_id: i32,
    image_token_id: i32,
) -> Option<InsertedQwenVlmTokens> {
    if prompt_tokens.is_empty()
        || grid_thw.is_empty()
        || prompt_tokens.contains(&image_token_id)
        || spatial_merge_size == 0
    {
        return None;
    }

    let merge = spatial_merge_size as i32;
    let vision_end_token_id = vision_start_token_id + 1;
    let mut image_tokens = Vec::new();
    let mut total_image_tokens = 0;

    for &(t, h, w) in grid_thw {
        let tokens_per_image = t * (h / merge) * (w / merge);
        total_image_tokens += tokens_per_image;
        image_tokens.push(vision_start_token_id);
        for _ in 0..tokens_per_image {
            image_tokens.push(image_token_id);
        }
        image_tokens.push(vision_end_token_id);
    }

    let bos = prompt_tokens[0];
    let rest = prompt_tokens[1..].to_vec();
    *prompt_tokens = vec![bos];
    prompt_tokens.extend(image_tokens);
    prompt_tokens.extend(rest);

    Some(InsertedQwenVlmTokens {
        image_blocks: grid_thw.len(),
        total_image_tokens,
    })
}

#[cfg(test)]
#[path = "qwen_vl_tests.rs"]
mod tests;

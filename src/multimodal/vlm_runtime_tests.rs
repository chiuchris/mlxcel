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

use super::{VlmPreparationSummary, prepared_embedding_refs, should_prepare_vlm_embeddings};
use crate::vision::merge::InputEmbeddings;
use crate::vlm_prompt::{ImageTokenBlockAction, ImageTokenBlockStats};
use mlxcel_core::{self, UniquePtr, dtype};

#[test]
fn should_prepare_vlm_embeddings_rejects_non_vlm_image_requests() {
    let err = should_prepare_vlm_embeddings(1, false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Images provided but model is not a vision-language model"));
}

#[test]
fn should_prepare_vlm_embeddings_accepts_vlm_image_requests() {
    assert_eq!(should_prepare_vlm_embeddings(2, true).unwrap(), true);
    assert_eq!(should_prepare_vlm_embeddings(0, true).unwrap(), false);
}

#[test]
fn image_block_summary_preserves_stats_shape() {
    let summary = VlmPreparationSummary::ImageBlocks(ImageTokenBlockStats {
        action: ImageTokenBlockAction::Inserted { image_blocks: 2 },
        tokens_per_image: 256,
    });

    assert_eq!(
        summary,
        VlmPreparationSummary::ImageBlocks(ImageTokenBlockStats {
            action: ImageTokenBlockAction::Inserted { image_blocks: 2 },
            tokens_per_image: 256,
        })
    );
}

#[test]
fn prepared_embedding_refs_requires_input_embeddings() {
    let embeddings = InputEmbeddings {
        inputs_embeds: UniquePtr::null(),
        attention_mask_4d: None,
    };

    let err = match prepared_embedding_refs(&embeddings) {
        Ok(_) => panic!("expected missing input embeddings to fail"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("missing input embeddings"));
}

#[test]
fn prepared_embedding_refs_rejects_null_attention_masks() {
    let embeddings = InputEmbeddings {
        inputs_embeds: mlxcel_core::ones(&[1, 2], dtype::FLOAT32),
        attention_mask_4d: Some(UniquePtr::null()),
    };

    let err = match prepared_embedding_refs(&embeddings) {
        Ok(_) => panic!("expected null attention mask to fail"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("null 4D attention mask"));
}

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

use anyhow::Result;
use std::path::PathBuf;

use mlxcel::LoadedModel;
use mlxcel::vision::merge::InputEmbeddings;
use mlxcel::vlm_prompt::ImageTokenBlockAction;
use mlxcel::vlm_runtime::{VlmPreparationSummary, prepare_and_compute_vlm_embeddings};

use crate::MlxcelTokenizer;

fn print_preparation_summary(summary: VlmPreparationSummary) {
    match summary {
        VlmPreparationSummary::QwenVlm {
            image_blocks,
            total_image_tokens,
        } => {
            println!(
                "Inserted {} Qwen VL image token blocks ({} total image tokens)",
                image_blocks, total_image_tokens
            );
        }
        VlmPreparationSummary::MiniCPMO {
            image_slots,
            total_tokens,
        } => {
            println!(
                "MiniCPM-o: tokenized with {} image slots ({} total tokens)",
                image_slots, total_tokens
            );
        }
        VlmPreparationSummary::Phi4MM {
            image_slots,
            total_tokens,
        } => {
            println!(
                "Phi4MM: tokenized with {} image slots ({} total tokens)",
                image_slots, total_tokens
            );
        }
        VlmPreparationSummary::Molmo2 { total_tokens } => {
            println!(
                "Molmo2: expanded prompt with image tokens ({} total tokens)",
                total_tokens
            );
        }
        VlmPreparationSummary::Phi3V {
            image_slots,
            total_tokens,
        } => {
            println!(
                "Phi3V: tokenized with {} image slots ({} total tokens)",
                image_slots, total_tokens
            );
        }
        VlmPreparationSummary::Phi4SigLip {
            image_slots,
            total_tokens,
        } => {
            println!(
                "Phi4-SigLIP: tokenized with {} image slots ({} total tokens)",
                image_slots, total_tokens
            );
        }
        VlmPreparationSummary::ImageBlocks(stats) => match stats.action {
            ImageTokenBlockAction::Expanded {
                existing_image_count,
            } => {
                println!(
                    "Expanded {} <image> token(s) to {} tokens each",
                    existing_image_count, stats.tokens_per_image
                );
            }
            ImageTokenBlockAction::Inserted { image_blocks } => {
                println!(
                    "Inserted {} image token blocks ({} tokens each)",
                    image_blocks, stats.tokens_per_image
                );
            }
        },
    }
}

pub(crate) fn compute_vlm_embeddings(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    prompt: &str,
    image_paths: &[PathBuf],
    tokenizer: &MlxcelTokenizer,
) -> Result<Option<InputEmbeddings>> {
    if image_paths.is_empty() {
        return Ok(None);
    }

    let images: Vec<image::DynamicImage> = image_paths
        .iter()
        .map(|path| {
            image::open(path).map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
        })
        .collect::<Result<Vec<_>>>()?;
    println!("Loaded {} image(s).", images.len());

    let prepared = prepare_and_compute_vlm_embeddings(
        model,
        prompt_tokens,
        prompt,
        &images,
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .iter()
                .map(|&t| t as i32)
                .collect()
        },
    )?;

    if let Some(prepared) = prepared {
        if let Some(summary) = prepared.preparation {
            print_preparation_summary(summary);
        }
        Ok(Some(prepared.embeddings))
    } else {
        Ok(None)
    }
}

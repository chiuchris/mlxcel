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

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
use std::path::{Path, PathBuf};

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
        VlmPreparationSummary::Moondream3 {
            mode,
            total_tokens,
            prefix_tokens,
        } => {
            println!(
                "Moondream3: prepared {:?} prompt ({} text tokens, {} image-prefix tokens)",
                mode, total_tokens, prefix_tokens
            );
        }
        VlmPreparationSummary::Gemma4 {
            image_slots,
            total_tokens,
        } => {
            println!(
                "Gemma4: expanded {} image slot(s) into dynamic soft tokens ({} total tokens)",
                image_slots, total_tokens
            );
        }
        VlmPreparationSummary::Gemma4Audio {
            audio_tokens,
            total_tokens,
        } => {
            println!(
                "Gemma4: expanded audio into {} soft tokens ({} total tokens)",
                audio_tokens, total_tokens
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
        VlmPreparationSummary::MolmoPoint { total_tokens } => {
            println!(
                "Molmo-Point: expanded prompt with image tokens ({} total tokens)",
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
    audio_path: Option<&Path>,
    tokenizer: &MlxcelTokenizer,
) -> Result<Option<InputEmbeddings>> {
    // Handle audio-only mode for Gemma4
    if image_paths.is_empty()
        && let Some(audio) = audio_path
        && let LoadedModel::Gemma4VLM(gemma4_vl) = model
    {
        return compute_gemma4_audio_embeddings(gemma4_vl, prompt_tokens, audio);
    }

    // Handle combined image + audio for Gemma4
    if !image_paths.is_empty()
        && let Some(audio) = audio_path
        && let LoadedModel::Gemma4VLM(gemma4_vl) = model
    {
        return compute_gemma4_multimodal_embeddings(gemma4_vl, prompt_tokens, image_paths, audio);
    }

    if image_paths.is_empty() {
        // Moondream3 needs special prompt formatting even for text-only
        if matches!(model, LoadedModel::Moondream3VLM(_)) {
            let prepared = mlxcel::moondream3_prompt::prepare_moondream3_prompt_tokens(
                prompt,
                0,
                |text, add_special| {
                    tokenizer
                        .encode(text, add_special)
                        .unwrap_or_default()
                        .iter()
                        .map(|&t| t as i32)
                        .collect()
                },
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
            *prompt_tokens = prepared.tokens;
        }
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

/// Compute audio-only embeddings for Gemma4 VLM.
fn compute_gemma4_audio_embeddings(
    gemma4_vl: &mlxcel::vision::Gemma4VLModel,
    prompt_tokens: &mut Vec<i32>,
    audio_path: &Path,
) -> Result<Option<InputEmbeddings>> {
    use mlxcel::audio;

    if gemma4_vl.audio_tower.is_none() {
        return Err(anyhow::anyhow!(
            "This model does not have an audio encoder. Audio input is not supported."
        ));
    }

    // Load and process audio
    let (samples, sample_rate) =
        audio::load_wav_file(audio_path).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!(
        "Loaded audio: {} samples at {} Hz ({:.1}s)",
        samples.len(),
        sample_rate,
        samples.len() as f64 / sample_rate as f64
    );

    // Compute number of audio tokens
    let num_audio_tokens = audio::compute_audio_num_tokens(
        samples.len(),
        sample_rate,
        40,  // ms_per_token
        750, // max_tokens
    );

    // Expand audio token placeholder in prompt: BOA + AUDIO*N + EOA
    expand_gemma4_audio_tokens(
        prompt_tokens,
        gemma4_vl.audio_token_id,
        gemma4_vl.boa_token_id,
        gemma4_vl.eoa_token_id,
        num_audio_tokens,
    );

    // Extract mel spectrogram
    let extractor =
        audio::AudioFeatureExtractor::new(audio::AudioFeatureExtractorConfig::default());
    let (features, mask) = extractor.extract(&samples, None);
    let num_frames = mask.len();

    // Convert to MlxArray: [1, T, 128]
    let audio_features = mlxcel_core::from_slice_f32(
        &features,
        &[1, num_frames as i32, extractor.feature_size() as i32],
    );
    // Convert mask to MlxArray: [1, T] (true = invalid)
    let mask_i32: Vec<i32> = mask.iter().map(|&b| if b { 1 } else { 0 }).collect();
    let audio_mask = mlxcel_core::from_slice_i32(&mask_i32, &[1, num_frames as i32]);
    let audio_mask = mlxcel_core::astype(&audio_mask, mlxcel_core::dtype::BOOL);

    // Compute embeddings
    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings = gemma4_vl.get_input_embeddings_with_audio(
        &input_ids_arr,
        &[], // no images
        Some(&audio_features),
        Some(&audio_mask),
    );

    print_preparation_summary(VlmPreparationSummary::Gemma4Audio {
        audio_tokens: num_audio_tokens,
        total_tokens: prompt_tokens.len(),
    });

    Ok(Some(embeddings))
}

/// Compute combined image + audio embeddings for Gemma4 VLM.
fn compute_gemma4_multimodal_embeddings(
    gemma4_vl: &mlxcel::vision::Gemma4VLModel,
    prompt_tokens: &mut Vec<i32>,
    image_paths: &[PathBuf],
    audio_path: &Path,
) -> Result<Option<InputEmbeddings>> {
    use mlxcel::audio;

    if gemma4_vl.audio_tower.is_none() {
        return Err(anyhow::anyhow!(
            "This model does not have an audio encoder. Audio input is not supported."
        ));
    }

    // Process images
    let images: Vec<image::DynamicImage> = image_paths
        .iter()
        .map(|path| {
            image::open(path).map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
        })
        .collect::<Result<Vec<_>>>()?;
    println!("Loaded {} image(s).", images.len());

    let processed_images = gemma4_vl.processor.preprocess(&images);
    let num_soft_tokens: Vec<usize> = processed_images.iter().map(|i| i.num_soft_tokens).collect();

    // Expand image tokens
    mlxcel::vlm_runtime::expand_gemma4_image_tokens_pub(
        prompt_tokens,
        gemma4_vl.image_token_id,
        gemma4_vl.boi_token_id,
        gemma4_vl.eoi_token_id,
        &num_soft_tokens,
    )?;

    // Process audio
    let (samples, sample_rate) =
        audio::load_wav_file(audio_path).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!(
        "Loaded audio: {} samples at {} Hz ({:.1}s)",
        samples.len(),
        sample_rate,
        samples.len() as f64 / sample_rate as f64
    );

    let num_audio_tokens = audio::compute_audio_num_tokens(samples.len(), sample_rate, 40, 750);
    expand_gemma4_audio_tokens(
        prompt_tokens,
        gemma4_vl.audio_token_id,
        gemma4_vl.boa_token_id,
        gemma4_vl.eoa_token_id,
        num_audio_tokens,
    );

    // Extract mel spectrogram
    let extractor =
        audio::AudioFeatureExtractor::new(audio::AudioFeatureExtractorConfig::default());
    let (features, mask) = extractor.extract(&samples, None);
    let num_frames = mask.len();
    let audio_features = mlxcel_core::from_slice_f32(
        &features,
        &[1, num_frames as i32, extractor.feature_size() as i32],
    );
    let mask_i32: Vec<i32> = mask.iter().map(|&b| if b { 1 } else { 0 }).collect();
    let audio_mask = mlxcel_core::from_slice_i32(&mask_i32, &[1, num_frames as i32]);
    let audio_mask = mlxcel_core::astype(&audio_mask, mlxcel_core::dtype::BOOL);

    // Compute embeddings with both images and audio
    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings = gemma4_vl.get_input_embeddings_with_audio(
        &input_ids_arr,
        &processed_images,
        Some(&audio_features),
        Some(&audio_mask),
    );

    Ok(Some(embeddings))
}

/// Expand audio token placeholder: single audio_token -> BOA + AUDIO*N + EOA
fn expand_gemma4_audio_tokens(
    prompt_tokens: &mut Vec<i32>,
    audio_token_id: i32,
    boa_token_id: i32,
    eoa_token_id: i32,
    num_audio_tokens: usize,
) {
    let mut expanded = Vec::with_capacity(prompt_tokens.len() + num_audio_tokens + 2);
    let mut found = false;
    for &token in prompt_tokens.iter() {
        if token == audio_token_id && !found {
            found = true;
            expanded.push(boa_token_id);
            expanded.extend(std::iter::repeat_n(audio_token_id, num_audio_tokens));
            expanded.push(eoa_token_id);
        } else {
            expanded.push(token);
        }
    }
    // If no audio placeholder found, insert BOA + AUDIO*N + EOA before last token
    if !found && !prompt_tokens.is_empty() {
        let last = expanded.pop().unwrap();
        expanded.push(boa_token_id);
        expanded.extend(std::iter::repeat_n(audio_token_id, num_audio_tokens));
        expanded.push(eoa_token_id);
        expanded.push(last);
    }
    *prompt_tokens = expanded;
}

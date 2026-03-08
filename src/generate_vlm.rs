use anyhow::Result;
use std::path::PathBuf;

use mlxcel::phi3v_prompt::prepare_phi3v_prompt_tokens;
use mlxcel::qwen_vl::insert_qwen_vl_image_tokens;
use mlxcel::vision::processors::ImageProcessor;
use mlxcel::vlm_prompt::{ImageTokenBlockAction, apply_image_token_blocks};
use mlxcel::{LoadedModel, vision::merge::InputEmbeddings};

use crate::MlxcelTokenizer;

/// Prepare VLM image tokens in the prompt token sequence.
///
/// This handles model-specific image token insertion/expansion for all VLM types:
/// - Qwen2-VL/2.5-VL/3-VL/3.5-VL: grid-based token insertion
/// - Molmo2: multi-scale token expansion
/// - Phi3-V: negative ID image token replacement
/// - Gemma3/LLaVA and others: BOI/EOI-framed image token blocks
pub(crate) fn prepare_vlm_tokens(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    prompt: &str,
    image_paths: &[PathBuf],
    tokenizer: &MlxcelTokenizer,
) -> Result<()> {
    if let Some(info) = model.qwen_vl_prompt_info() {
        // Qwen2-VL/2.5-VL: load images first to compute grid_thw and token counts
        let images: Vec<image::DynamicImage> = image_paths
            .iter()
            .map(|path| {
                image::open(path)
                    .map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
            })
            .collect::<Result<Vec<_>>>()?;

        let grid_thw = info.processor.compute_grid_thw(&images);
        if let Some(stats) = insert_qwen_vl_image_tokens(
            prompt_tokens,
            &grid_thw,
            info.spatial_merge_size,
            info.vision_start_token_id,
            info.image_token_id,
        ) {
            println!(
                "Inserted {} Qwen VL image token blocks ({} total image tokens)",
                stats.image_blocks, stats.total_image_tokens
            );
        }
    } else if let Some(molmo2) = model.molmo2_vl_model() {
        // Molmo2: multi-scale preprocessing -> build image token string -> expand prompt
        let images_for_tokens: Vec<image::DynamicImage> = image_paths
            .iter()
            .map(|path| {
                image::open(path)
                    .map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
            })
            .collect::<Result<Vec<_>>>()?;

        if let Some(img) = images_for_tokens.first() {
            // Preprocess image
            let proc_out = molmo2.processor.preprocess_image(img);

            // Generate image token string from grid
            let image_token_str = molmo2.processor.get_image_tokens(&proc_out.image_grid);

            // Replace <|image|> in prompt (or prepend)
            let mut text = prompt.to_string();
            if text.contains("<|image|>") {
                text = text.replace("<|image|>", &image_token_str);
            } else {
                // Insert before the user message content
                text = format!("{}{}", image_token_str, text);
            }

            // Re-tokenize with expanded tokens
            *prompt_tokens = tokenizer
                .encode(&text, true)
                .unwrap_or_default()
                .iter()
                .map(|&t| t as i32)
                .collect();
            println!(
                "Molmo2: expanded prompt with image tokens ({} total tokens)",
                prompt_tokens.len()
            );
        }
    } else if let Some(phi3v) = model.phi3_vl_model() {
        let images_for_tokens: Vec<image::DynamicImage> = image_paths
            .iter()
            .map(|path| {
                image::open(path)
                    .map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(prepared) = prepare_phi3v_prompt_tokens(
            prompt,
            images_for_tokens.len(),
            |text, add_special| {
                tokenizer
                    .encode(text, add_special)
                    .unwrap_or_default()
                    .iter()
                    .map(|&t| t as i32)
                    .collect()
            },
            |image_num| {
                let image = &images_for_tokens[image_num - 1];
                phi3v
                    .processor
                    .calc_num_image_tokens(image.width(), image.height())
            },
        ) {
            *prompt_tokens = prepared.tokens;
            println!(
                "Phi3V: tokenized with {} image slots ({} total tokens)",
                prepared.image_slots,
                prompt_tokens.len()
            );
        }
    } else if let Some(info) = model.image_token_block_info() {
        if let Some(stats) = apply_image_token_blocks(prompt_tokens, info, image_paths.len()) {
            match stats.action {
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
            }
        }
    }

    Ok(())
}

/// Compute VLM embeddings from images and prompt tokens.
///
/// Returns `Some(InputEmbeddings)` if the model is a VLM and images are provided,
/// `None` otherwise. Returns an error if images are provided but the model is not a VLM.
pub(crate) fn compute_vlm_embeddings(
    model: &LoadedModel,
    prompt_tokens: &[i32],
    image_paths: &[PathBuf],
    has_images: bool,
) -> Result<Option<InputEmbeddings>> {
    if has_images && model.is_vlm() {
        // Load images
        let images: Vec<image::DynamicImage> = image_paths
            .iter()
            .map(|path| {
                image::open(path)
                    .map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
            })
            .collect::<Result<Vec<_>>>()?;
        println!("Loaded {} image(s).", images.len());

        if let Some(info) = model.qwen_vl_prompt_info() {
            let (pixel_values, grid_thw) = info.processor.preprocess_with_grid(&images);
            let input_ids_arr =
                mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
            let merged = model
                .qwen_vl_input_embeddings(&input_ids_arr, &pixel_values, &grid_thw)
                .ok_or_else(|| anyhow::anyhow!("Qwen-VL prompt info without matching model"))?;
            Ok(Some(merged))
        } else if let Some(gemma3n_vl) = model.gemma3n_vl_model() {
            // Gemma3n VLM: MobileNetV5 + per_layer_inputs
            let pixel_values = gemma3n_vl.processor.preprocess(&images);
            let input_ids_arr =
                mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
            let merged = gemma3n_vl.get_input_embeddings(&input_ids_arr, &pixel_values);
            Ok(Some(merged))
        } else if let Some(phi3v) = model.phi3_vl_model() {
            // Phi-3V: HD tiling + negative token ID replacement
            let (pixel_values, image_sizes) = phi3v.processor.preprocess(&images);
            let input_ids_arr =
                mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
            let merged = phi3v.get_input_embeddings(&input_ids_arr, &pixel_values, &image_sizes);
            Ok(Some(merged))
        } else if let Some(molmo2) = model.molmo2_vl_model() {
            // Molmo2: additive merge of vision features at image_patch positions
            let proc_out = molmo2.processor.preprocess_image(&images[0]);
            let pixel_values =
                mlxcel_core::from_slice_f32(&proc_out.pixel_values, &proc_out.pixel_values_shape);
            let image_token_pooling = mlxcel_core::from_slice_i32(
                &proc_out.image_token_pooling,
                &proc_out.image_token_pooling_shape,
            );
            let image_grids = mlxcel_core::from_slice_i32(&proc_out.image_grid, &[4]);
            let image_num_crops = mlxcel_core::from_slice_i32(&[proc_out.image_num_crops], &[1]);
            let input_ids_arr =
                mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
            let merged = molmo2.get_input_embeddings(
                &input_ids_arr,
                &pixel_values,
                &image_token_pooling,
                &image_grids,
                &image_num_crops,
            );
            Ok(Some(merged))
        } else {
            // Gemma3/LLaVA: use VisionModule pipeline
            let vision_module = model.vision_module().unwrap();

            // Preprocess images
            let pixel_values = vision_module.processor.preprocess(&images);

            // Create attention mask (all ones for non-padded input)
            let mask =
                mlxcel_core::ones(&[1, prompt_tokens.len() as i32], mlxcel_core::dtype::INT32);

            // Get merged embeddings
            let input_ids_arr =
                mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
            let merged = vision_module.get_input_embeddings(
                model,
                &input_ids_arr,
                Some(&pixel_values),
                &mask,
            );
            Ok(Some(merged))
        }
    } else if has_images {
        Err(anyhow::anyhow!(
            "Images provided but model is not a vision-language model"
        ))
    } else {
        Ok(None)
    }
}

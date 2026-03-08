use anyhow::Result;
use std::path::PathBuf;

use mlxcel::qwen_vl::insert_qwen_vl_image_tokens;
use mlxcel::vision::processors::ImageProcessor;
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
        // Phi-3V: split text around <|image_N|> tags, tokenize chunks,
        // and interleave with negative IDs (matching Python Phi3VProcessor)
        let images_for_tokens: Vec<image::DynamicImage> = image_paths
            .iter()
            .map(|path| {
                image::open(path)
                    .map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
            })
            .collect::<Result<Vec<_>>>()?;
        let num_images = images_for_tokens.len();

        // Ensure <|image_N|> tags are in the prompt text
        let mut text = prompt.to_string();
        let has_image_tags = (1..=num_images).any(|i| text.contains(&format!("<|image_{}|>", i)));

        if !has_image_tags && num_images > 0 {
            // Insert <|image_N|> tags after <|user|>\n or at the beginning
            let image_tags: String = (1..=num_images)
                .map(|i| format!("<|image_{}|>\n", i))
                .collect();
            if let Some(pos) = text.find("<|user|>\n") {
                text.insert_str(pos + "<|user|>\n".len(), &image_tags);
            } else {
                text = format!("{}{}", image_tags, text);
            }
        }

        // Collect image tag positions and IDs, sorted by position
        let mut tag_positions: Vec<(usize, usize, usize)> = Vec::new(); // (start, end, image_num)
        for n in 1..=num_images {
            let tag = format!("<|image_{}|>", n);
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find(&tag) {
                let abs_pos = search_from + pos;
                tag_positions.push((abs_pos, abs_pos + tag.len(), n));
                search_from = abs_pos + tag.len();
            }
        }
        tag_positions.sort_by_key(|&(start, _, _)| start);

        if !tag_positions.is_empty() {
            // Split text around image tags and tokenize each chunk
            let mut new_tokens: Vec<i32> = Vec::new();
            let mut last_end = 0;

            for (chunk_idx, &(tag_start, tag_end, image_num)) in tag_positions.iter().enumerate() {
                // Tokenize text before this image tag
                let before = &text[last_end..tag_start];
                if !before.is_empty() {
                    let add_special = chunk_idx == 0 && last_end == 0;
                    let tokens = tokenizer.encode(before, add_special).unwrap_or_default();
                    new_tokens.extend(tokens.iter().map(|&t| t as i32));
                }

                // Insert negative IDs for this image
                if image_num <= num_images {
                    let (w, h) = (
                        images_for_tokens[image_num - 1].width(),
                        images_for_tokens[image_num - 1].height(),
                    );
                    let num_img_tokens = phi3v.processor.calc_num_image_tokens(w, h);
                    let neg_id = -(image_num as i32);
                    for _ in 0..num_img_tokens {
                        new_tokens.push(neg_id);
                    }
                }

                last_end = tag_end;
            }

            // Tokenize remaining text after the last image tag
            let after = &text[last_end..];
            if !after.is_empty() {
                let tokens = tokenizer.encode(after, false).unwrap_or_default();
                new_tokens.extend(tokens.iter().map(|&t| t as i32));
            }

            *prompt_tokens = new_tokens;
            println!(
                "Phi3V: tokenized with {} image slots ({} total tokens)",
                tag_positions.len(),
                prompt_tokens.len()
            );
        }
    } else if model.gemma3n_vl_model().is_some() || model.vision_module().is_some() {
        // Extract VLM token parameters from either Gemma3nVLM or generic VisionModule
        let (use_boi_eoi, image_token_id, mm_tokens_per_image, boi_token_id, eoi_token_id) =
            if let Some(g3n) = model.gemma3n_vl_model() {
                (
                    true,
                    g3n.image_token_id,
                    256usize,
                    g3n.boi_token_id,
                    g3n.eoi_token_id,
                )
            } else {
                let vm = model.vision_module().unwrap();
                (
                    vm.boi_token_id != 0,
                    vm.image_token_id,
                    vm.mm_tokens_per_image,
                    vm.boi_token_id,
                    vm.eoi_token_id,
                )
            };
        let num_images = image_paths.len();

        // Check if the tokenized prompt already contains image tokens
        let existing_image_count = prompt_tokens
            .iter()
            .filter(|&&t| t == image_token_id)
            .count();

        if existing_image_count > 0 {
            // Expand existing image tokens
            let mut expanded = Vec::with_capacity(
                prompt_tokens.len() + (mm_tokens_per_image - 1) * existing_image_count,
            );
            for &tok in prompt_tokens.iter() {
                if tok == image_token_id {
                    if use_boi_eoi {
                        expanded.push(boi_token_id);
                    }
                    for _ in 0..mm_tokens_per_image {
                        expanded.push(image_token_id);
                    }
                    if use_boi_eoi {
                        expanded.push(eoi_token_id);
                    }
                } else {
                    expanded.push(tok);
                }
            }
            *prompt_tokens = expanded;
            println!(
                "Expanded {} <image> token(s) to {} tokens each",
                existing_image_count, mm_tokens_per_image
            );
        } else {
            // No image tokens in prompt -- insert after BOS
            let mut image_tokens = Vec::new();
            for _ in 0..num_images {
                if use_boi_eoi {
                    image_tokens.push(boi_token_id);
                }
                for _ in 0..mm_tokens_per_image {
                    image_tokens.push(image_token_id);
                }
                if use_boi_eoi {
                    image_tokens.push(eoi_token_id);
                }
            }
            if !prompt_tokens.is_empty() {
                let bos = prompt_tokens[0];
                let rest = prompt_tokens[1..].to_vec();
                *prompt_tokens = vec![bos];
                prompt_tokens.extend(image_tokens);
                prompt_tokens.extend(rest);
            }
            println!(
                "Inserted {} image token blocks ({} tokens each)",
                num_images, mm_tokens_per_image
            );
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

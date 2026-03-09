use anyhow::Result;
use image::DynamicImage;
use mlxcel_core::MlxArray;

use crate::LoadedModel;
use crate::phi3v_prompt::prepare_phi3v_prompt_tokens;
use crate::qwen_vl::insert_qwen_vl_image_tokens;
use crate::vision::merge::InputEmbeddings;
use crate::vision::processors::ImageProcessor;
use crate::vlm_prompt::{ImageTokenBlockStats, apply_image_token_blocks};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VlmPreparationSummary {
    QwenVlm {
        image_blocks: usize,
        total_image_tokens: i32,
    },
    Molmo2 {
        total_tokens: usize,
    },
    Phi3V {
        image_slots: usize,
        total_tokens: usize,
    },
    ImageBlocks(ImageTokenBlockStats),
}

pub struct PreparedVlmEmbeddings {
    pub embeddings: InputEmbeddings,
    pub preparation: Option<VlmPreparationSummary>,
}

pub fn prepared_embedding_refs(
    embeddings: &InputEmbeddings,
) -> Result<(&MlxArray, Option<&MlxArray>)> {
    let input_embeds = embeddings
        .inputs_embeds
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Prepared VLM embeddings are missing input embeddings"))?;
    let attention_mask = match embeddings.attention_mask_4d.as_ref() {
        Some(mask) => Some(mask.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Prepared VLM embeddings contain a null 4D attention mask")
        })?),
        None => None,
    };
    Ok((input_embeds, attention_mask))
}

fn should_prepare_vlm_embeddings(image_count: usize, is_vlm: bool) -> Result<bool> {
    if image_count == 0 {
        Ok(false)
    } else if is_vlm {
        Ok(true)
    } else {
        Err(anyhow::anyhow!(
            "Images provided but model is not a vision-language model"
        ))
    }
}

fn prompt_ids_array(prompt_tokens: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32])
}

pub fn prepare_and_compute_vlm_embeddings<E>(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    prompt: &str,
    images: &[DynamicImage],
    mut encode: E,
) -> Result<Option<PreparedVlmEmbeddings>>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    if !should_prepare_vlm_embeddings(images.len(), model.is_vlm())? {
        return Ok(None);
    }

    if let Some(info) = model.qwen_vl_prompt_info() {
        let (pixel_values, grid_thw) = info.processor.preprocess_with_grid(images);
        let preparation = insert_qwen_vl_image_tokens(
            prompt_tokens,
            &grid_thw,
            info.spatial_merge_size,
            info.vision_start_token_id,
            info.image_token_id,
        )
        .map(|stats| VlmPreparationSummary::QwenVlm {
            image_blocks: stats.image_blocks,
            total_image_tokens: stats.total_image_tokens,
        });

        let input_ids_arr = prompt_ids_array(prompt_tokens);
        let embeddings = model
            .qwen_vl_input_embeddings(&input_ids_arr, &pixel_values, &grid_thw)
            .ok_or_else(|| anyhow::anyhow!("Qwen-VL prompt info without matching model"))?;

        Ok(Some(PreparedVlmEmbeddings {
            embeddings,
            preparation,
        }))
    } else if let Some(gemma3n_vl) = model.gemma3n_vl_model() {
        let preparation = model
            .image_token_block_info()
            .and_then(|info| apply_image_token_blocks(prompt_tokens, info, images.len()))
            .map(VlmPreparationSummary::ImageBlocks);

        let pixel_values = gemma3n_vl.processor.preprocess(images);
        let input_ids_arr = prompt_ids_array(prompt_tokens);
        let embeddings = gemma3n_vl.get_input_embeddings(&input_ids_arr, &pixel_values);

        Ok(Some(PreparedVlmEmbeddings {
            embeddings,
            preparation,
        }))
    } else if let Some(molmo2) = model.molmo2_vl_model() {
        let proc_out = molmo2.processor.preprocess_image(&images[0]);
        let image_token_str = molmo2.processor.get_image_tokens(&proc_out.image_grid);
        let text = if prompt.contains("<|image|>") {
            prompt.replace("<|image|>", &image_token_str)
        } else {
            format!("{}{}", image_token_str, prompt)
        };
        *prompt_tokens = encode(&text, true);

        let pixel_values =
            mlxcel_core::from_slice_f32(&proc_out.pixel_values, &proc_out.pixel_values_shape);
        let image_token_pooling = mlxcel_core::from_slice_i32(
            &proc_out.image_token_pooling,
            &proc_out.image_token_pooling_shape,
        );
        let image_grids = mlxcel_core::from_slice_i32(&proc_out.image_grid, &[4]);
        let image_num_crops = mlxcel_core::from_slice_i32(&[proc_out.image_num_crops], &[1]);
        let input_ids_arr = prompt_ids_array(prompt_tokens);
        let embeddings = molmo2.get_input_embeddings(
            &input_ids_arr,
            &pixel_values,
            &image_token_pooling,
            &image_grids,
            &image_num_crops,
        );

        Ok(Some(PreparedVlmEmbeddings {
            embeddings,
            preparation: Some(VlmPreparationSummary::Molmo2 {
                total_tokens: prompt_tokens.len(),
            }),
        }))
    } else if let Some(phi3v) = model.phi3_vl_model() {
        let preparation = if let Some(prepared) =
            prepare_phi3v_prompt_tokens(prompt, images.len(), &mut encode, |image_num| {
                let image = &images[image_num - 1];
                phi3v
                    .processor
                    .calc_num_image_tokens(image.width(), image.height())
            }) {
            *prompt_tokens = prepared.tokens;
            Some(VlmPreparationSummary::Phi3V {
                image_slots: prepared.image_slots,
                total_tokens: prompt_tokens.len(),
            })
        } else {
            None
        };

        let (pixel_values, image_sizes) = phi3v.processor.preprocess(images);
        let input_ids_arr = prompt_ids_array(prompt_tokens);
        let embeddings = phi3v.get_input_embeddings(&input_ids_arr, &pixel_values, &image_sizes);

        Ok(Some(PreparedVlmEmbeddings {
            embeddings,
            preparation,
        }))
    } else {
        let vision_module = model
            .vision_module()
            .ok_or_else(|| anyhow::anyhow!("VLM model is missing a standard vision module"))?;
        let preparation = model
            .image_token_block_info()
            .and_then(|info| apply_image_token_blocks(prompt_tokens, info, images.len()))
            .map(VlmPreparationSummary::ImageBlocks);

        let pixel_values = vision_module.processor.preprocess(images);
        let mask = mlxcel_core::ones(&[1, prompt_tokens.len() as i32], mlxcel_core::dtype::INT32);
        let input_ids_arr = prompt_ids_array(prompt_tokens);
        let embeddings = vision_module.get_input_embeddings(
            model,
            &input_ids_arr,
            Some(&pixel_values),
            &mask,
        )?;

        Ok(Some(PreparedVlmEmbeddings {
            embeddings,
            preparation,
        }))
    }
}

#[cfg(test)]
#[path = "vlm_runtime_tests.rs"]
mod tests;

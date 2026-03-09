//! Qwen-VL prompt token insertion rules.
//!
//! Qwen2/2.5/3/3.5-VL families reserve image-token blocks based on the image
//! grid and spatial merge size. This module keeps that token arithmetic out of
//! CLI/server callers so Qwen-VL prompt preparation stays consistent.

use crate::vision;
use mlxcel_core::MlxArray;

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

pub trait QwenVlRuntime {
    fn prompt_info(&self) -> QwenVlmPromptInfo<'_>;
    fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> vision::merge::InputEmbeddings;
}

macro_rules! impl_qwen_vl_runtime {
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
        }
    };
}

impl_qwen_vl_runtime!(vision::Qwen2VLModel);
impl_qwen_vl_runtime!(vision::Qwen25VLModel);
impl_qwen_vl_runtime!(vision::Qwen3VLModel);
impl_qwen_vl_runtime!(vision::Qwen3VLMoeModel);
impl_qwen_vl_runtime!(vision::Qwen35VLModel);

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

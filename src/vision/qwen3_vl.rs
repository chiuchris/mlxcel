//! Qwen3-VL Vision-Language Model
//!
//! Vision encoder with DeepStack + Qwen3 language model with interleaved MRoPE

use super::{encoders, merge, processors};
use crate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Qwen3-VL VLM: Vision encoder with DeepStack + Qwen3 language model with interleaved MRoPE
///
/// Key differences from Qwen2/2.5-VL:
/// - Learned position embeddings with bilinear interpolation
/// - DeepStack multi-layer visual injection
/// - Interleaved MRoPE (not chunked)
/// - q_norm/k_norm in attention
/// - No windowed attention
pub struct Qwen3VLModel {
    pub text_model: crate::models::Qwen3VLModel,
    pub vision_encoder: encoders::qwen3_vl::Qwen3VLVisionEncoder,
    pub processor: processors::qwen2_vl::Qwen2VLProcessor,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_token_id: i32,
    pub spatial_merge_size: usize,
}

impl Qwen3VLModel {
    /// Get input embeddings with vision features merged in
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> merge::InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        // Encode images through vision tower (returns hidden_states + deepstack features)
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);
        let vision_output = self.vision_encoder.forward_with_grid(&pv, grid_thw);
        let image_features = &vision_output.hidden_states;

        // Merge vision features at image token positions (LLaVA-style)
        let merged = merge::merge_llava(
            self.image_token_id,
            image_features,
            &inputs_embeds,
            input_ids,
        );

        // Compute visual_pos_masks: boolean mask of image/video token positions
        let visual_pos_masks = self.compute_visual_pos_masks(input_ids);
        mlxcel_core::eval(&visual_pos_masks);

        // Eval deepstack features before storing
        for feat in &vision_output.deepstack_features {
            mlxcel_core::eval(feat);
        }

        // Store deepstack state in language model
        if !vision_output.deepstack_features.is_empty() {
            self.text_model
                .set_deepstack_state(visual_pos_masks, vision_output.deepstack_features);
        }

        // Compute MRoPE position IDs (same algorithm as Qwen2-VL)
        let position_ids = self.compute_rope_index(input_ids, grid_thw);
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = ids_shape[1];

        mlxcel_core::eval(&position_ids);
        let max_pos = mlxcel_core::max_all(&position_ids);
        mlxcel_core::eval(&max_pos);
        let max_pos_val = mlxcel_core::item_i32(&max_pos);
        let rope_deltas = max_pos_val + 1 - seq_len;

        self.text_model.set_mrope_state(position_ids, rope_deltas);

        merged
    }

    /// Compute boolean mask of image/video token positions
    /// Returns: [batch, seq_len]
    fn compute_visual_pos_masks(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let img_token = mlxcel_core::from_slice_i32(&[self.image_token_id], &[1]);
        let vid_token = mlxcel_core::from_slice_i32(&[self.video_token_id], &[1]);

        let img_mask = mlxcel_core::equal(input_ids, &img_token);
        let vid_mask = mlxcel_core::equal(input_ids, &vid_token);

        // OR the two masks
        mlxcel_core::logical_or(&img_mask, &vid_mask)
    }

    /// Compute 3D position IDs [T, H, W] for mixed text+image sequences
    /// (Same algorithm as Qwen2-VL/2.5-VL)
    fn compute_rope_index(
        &self,
        input_ids: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::eval(input_ids);
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = ids_shape[1] as usize;

        let mut tokens = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let tok = mlxcel_core::slice(input_ids, &[0, i as i32], &[1, i as i32 + 1]);
            mlxcel_core::eval(&tok);
            tokens.push(mlxcel_core::item_i32(&tok));
        }

        let merge = self.spatial_merge_size as i32;
        let mut pos_ids: Vec<Vec<i32>> = vec![Vec::new(); 3];
        let mut image_idx = 0usize;
        let mut st = 0usize;
        let mut current_pos = 0i32;

        let mut i = 0;
        while i < seq_len {
            if tokens[i] == self.image_token_id || tokens[i] == self.video_token_id {
                let vision_start = i;

                while i < seq_len
                    && (tokens[i] == self.image_token_id || tokens[i] == self.video_token_id)
                {
                    i += 1;
                }

                if vision_start > st {
                    let text_len = vision_start - st;
                    for p in current_pos..current_pos + text_len as i32 {
                        pos_ids[0].push(p);
                        pos_ids[1].push(p);
                        pos_ids[2].push(p);
                    }
                    current_pos += text_len as i32;
                }

                if image_idx < grid_thw.len() {
                    let (t, h, w) = grid_thw[image_idx];
                    let llm_h = h / merge;
                    let llm_w = w / merge;
                    let llm_t = t;

                    for ti in 0..llm_t {
                        for hi in 0..llm_h {
                            for wi in 0..llm_w {
                                pos_ids[0].push(current_pos + ti);
                                pos_ids[1].push(current_pos + hi);
                                pos_ids[2].push(current_pos + wi);
                            }
                        }
                    }
                    current_pos += llm_t.max(llm_h).max(llm_w);
                    image_idx += 1;
                }

                st = i;
                continue;
            }
            i += 1;
        }

        if st < seq_len {
            let text_len = seq_len - st;
            for p in current_pos..current_pos + text_len as i32 {
                pos_ids[0].push(p);
                pos_ids[1].push(p);
                pos_ids[2].push(p);
            }
        }

        let total_len = pos_ids[0].len() as i32;
        let t_arr = mlxcel_core::from_slice_i32(&pos_ids[0], &[1, 1, total_len]);
        let h_arr = mlxcel_core::from_slice_i32(&pos_ids[1], &[1, 1, total_len]);
        let w_arr = mlxcel_core::from_slice_i32(&pos_ids[2], &[1, 1, total_len]);

        let th = mlxcel_core::concatenate(t_arr.as_ref().unwrap(), h_arr.as_ref().unwrap(), 0);
        mlxcel_core::concatenate(th.as_ref().unwrap(), w_arr.as_ref().unwrap(), 0)
    }
}

impl LanguageModel for Qwen3VLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_impl(input_ids, None, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        mlxcel_core::generate::LanguageModel::eos_token_ids(&self.text_model)
    }
}

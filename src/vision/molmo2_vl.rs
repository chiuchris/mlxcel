//! Molmo2 Vision-Language Model
//!
//! Custom ViT vision encoder + attention-based 2D pooling + SwiGLU projector + Molmo2 text model.
//! Uses additive merge: vision features are ADDED to text embeddings at image_patch_id positions.

use super::{encoders, merge, processors};
use crate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Molmo2 Vision-Language Model
///
/// Uses additive merge: vision features are ADDED to text embeddings at
/// image_patch_id positions (not replaced).
pub struct Molmo2VLModel {
    pub text_model: crate::models::molmo2::Molmo2Model,
    pub vision_tower: encoders::molmo2::Molmo2VisionModel,
    pub processor: processors::molmo2::Molmo2Processor,
    pub image_patch_id: i32,     // 151938
    pub image_end_token_id: i32, // 151937
}

impl Molmo2VLModel {
    /// Get input embeddings with vision features additively merged
    ///
    /// Unlike LLaVA-style models that REPLACE text embeddings at image positions,
    /// Molmo2 ADDS vision features to the text embeddings.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        image_token_pooling: &MlxArray,
        _image_grids: &MlxArray,
        _image_num_crops: &MlxArray,
    ) -> merge::InputEmbeddings {
        // 1. Embed all tokens
        let x = self.text_model.wte.forward(input_ids);
        let shape = mlxcel_core::array_shape(&x);

        // 2. Encode images through vision tower (includes pooling + projection)
        let (images, token_pooling) = self.build_batched_images(
            input_ids,
            pixel_values,
            image_token_pooling,
            _image_grids,
            _image_num_crops,
        );
        let image_features = self.vision_tower.forward(&images, &token_pooling);
        // image_features: [num_image_patches, hidden_dim]
        let image_features = mlxcel_core::astype(&image_features, mlxcel_core::array_dtype(&x));

        // 3. Find image_patch_id positions in input_ids
        let flat_ids = mlxcel_core::reshape(input_ids, &[-1]);
        let patch_token = mlxcel_core::from_slice_i32(&[self.image_patch_id], &[1]);
        let is_image_patch = mlxcel_core::equal(&flat_ids, &patch_token);
        mlxcel_core::eval(&is_image_patch);
        let total_tokens = mlxcel_core::array_shape(&flat_ids)[0];
        let mut positions: Vec<i32> = Vec::new();
        for i in 0..total_tokens {
            let idx = mlxcel_core::from_slice_i32(&[i], &[1]);
            let val = mlxcel_core::take(&is_image_patch, &idx, 0);
            mlxcel_core::eval(&val);
            if mlxcel_core::item_bool(&val) {
                positions.push(i);
            }
        }

        let x = if !positions.is_empty() {
            let x_shape = mlxcel_core::array_shape(&x);
            let h_dim = x_shape[x_shape.len() - 1];
            let flat_x = mlxcel_core::reshape(&x, &[-1, h_dim]);
            let num_pos = positions.len() as i32;

            // Additive scatter: flat_x += one_hot(positions) @ image_features
            let row_indices: Vec<i32> = (0..total_tokens).collect();
            let row_arr = mlxcel_core::from_slice_i32(&row_indices, &[total_tokens, 1]);
            let pos_arr = mlxcel_core::from_slice_i32(&positions, &[1, num_pos]);
            let one_hot = mlxcel_core::equal(&row_arr, &pos_arr);
            let one_hot_f =
                mlxcel_core::astype(&one_hot, mlxcel_core::array_dtype(&image_features));
            // spread: [total_tokens, num_pos] @ [num_pos, h_dim] = [total_tokens, h_dim]
            let spread = mlxcel_core::matmul(&one_hot_f, &image_features);
            let spread = mlxcel_core::astype(&spread, mlxcel_core::array_dtype(&flat_x));
            let merged = mlxcel_core::add(&flat_x, &spread);
            mlxcel_core::reshape(&merged, &shape)
        } else {
            x
        };

        merge::InputEmbeddings {
            inputs_embeds: x,
            attention_mask_4d: None,
        }
    }

    /// Build batched images from flattened inputs (batch=1 simplified)
    fn build_batched_images(
        &self,
        _input_ids: &MlxArray,
        pixel_values: &MlxArray,
        image_token_pooling: &MlxArray,
        _image_grids: &MlxArray,
        _image_num_crops: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // For batch_size=1 (typical inference), simplified path:
        // images = pixel_values[None, :, :, :]  (add batch dim)
        // token_pooling = image_token_pooling[None, :, :]  (add batch dim)
        let pv_shape = mlxcel_core::array_shape(pixel_values);
        let n_crops = pv_shape[0];
        let n_patches = pv_shape[1];
        let patch_dim = pv_shape[2];

        let images = mlxcel_core::reshape(pixel_values, &[1, n_crops, n_patches, patch_dim]);

        let tp_shape = mlxcel_core::array_shape(image_token_pooling);
        let total_pooled = tp_shape[0];
        let pool_size = tp_shape[1];

        let token_pooling =
            mlxcel_core::reshape(image_token_pooling, &[1, total_pooled, pool_size]);

        (images, token_pooling)
    }
}

impl LanguageModel for Molmo2VLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.wte.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.blocks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        mlxcel_core::generate::LanguageModel::eos_token_ids(&self.text_model)
    }
}

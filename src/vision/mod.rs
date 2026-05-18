//! Vision model support for mlxcel
//!
//! Provides modular vision functionality that layers on top of existing text models.
//! Uses the VisionModule pattern: vision encoder + connector + processor,
//! composable with any LanguageModel via the VisionLanguageModel wrapper.

pub mod config;
pub mod connectors;
pub mod encoders;
pub mod merge;
pub mod processors;

// VLM model implementations
pub mod gemma3n_vl;
pub mod molmo2_vl;
pub mod phi3_vl;
pub mod qwen2_5_vl;
pub mod qwen2_vl;
pub mod qwen3_5_vl;
pub mod qwen3_vl;
pub mod qwen3_vl_moe;

// Re-export VLM model types
pub use gemma3n_vl::Gemma3nVLModel;
pub use molmo2_vl::Molmo2VLModel;
pub use phi3_vl::Phi3VLModel;
pub use qwen2_5_vl::Qwen25VLModel;
pub use qwen2_vl::Qwen2VLModel;
pub use qwen3_5_vl::Qwen35VLModel;
pub use qwen3_vl::Qwen3VLModel;
pub use qwen3_vl_moe::Qwen3VLMoeModel;

use crate::LanguageModel;
use connectors::MultiModalConnector;
use encoders::VisionEncoder;
use merge::InputEmbeddings;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};
use processors::ImageProcessor;

/// Merge strategy for combining vision and text embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Gemma3-style: masked_scatter with 4D attention mask
    Gemma3,
    /// LLaVA-style: simple replacement at image token positions, standard causal masking
    LLaVA,
}

/// Vision module: encodes images and merges with text embeddings
pub struct VisionModule {
    pub encoder: Box<dyn VisionEncoder>,
    pub connector: Box<dyn MultiModalConnector>,
    pub processor: Box<dyn ImageProcessor>,
    pub image_token_id: i32,
    pub pad_token_id: i32,
    pub hidden_size: usize,
    /// Begin-of-image token ID (0 = no BOI/EOI framing)
    pub boi_token_id: i32,
    /// End-of-image token ID
    pub eoi_token_id: i32,
    /// Number of image tokens per image
    pub mm_tokens_per_image: usize,
    /// Strategy for merging vision and text embeddings
    pub merge_strategy: MergeStrategy,
}

impl VisionModule {
    /// Process images and merge with text embeddings
    ///
    /// 1. Get text embeddings from the text model
    /// 2. Encode images through vision encoder
    /// 3. Project vision features to text space
    /// 4. Merge vision and text embeddings at image token positions
    pub fn get_input_embeddings(
        &self,
        text_model: &dyn LanguageModel,
        input_ids: &MlxArray,
        pixel_values: Option<&MlxArray>,
        attention_mask: &MlxArray,
    ) -> InputEmbeddings {
        // Get text embeddings
        let inputs_embeds = text_model
            .embed_tokens(input_ids)
            .expect("Text model must support embed_tokens for VLM");

        if pixel_values.is_none() {
            return InputEmbeddings {
                inputs_embeds,
                attention_mask_4d: None,
            };
        }

        let pixel_values = pixel_values.unwrap();

        // Get dtype of text embeddings for casting
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);

        // Vision encoder expects [B, H, W, C] (channels-last)
        // Input pixel_values may be [B, C, H, W], need to transpose
        let pv_shape = mlxcel_core::array_shape(pixel_values);
        let pv = if pv_shape.len() == 4 && pv_shape[1] <= 4 {
            // [B, C, H, W] -> [B, H, W, C]
            let transposed = mlxcel_core::transpose_axes(pixel_values, &[0, 2, 3, 1]);
            mlxcel_core::astype(&transposed, embed_dtype)
        } else {
            mlxcel_core::astype(pixel_values, embed_dtype)
        };

        // Encode images
        let encoder_output = self.encoder.forward(&pv);

        // Project to text embedding space
        let image_features = self.connector.forward(&encoder_output.hidden_states);

        // Merge text and vision embeddings using the appropriate strategy
        match self.merge_strategy {
            MergeStrategy::Gemma3 => merge::prepare_inputs_for_multimodal(
                self.hidden_size,
                self.pad_token_id,
                self.image_token_id,
                &image_features,
                &inputs_embeds,
                input_ids,
                attention_mask,
            ),
            MergeStrategy::LLaVA => merge::merge_llava(
                self.image_token_id,
                &image_features,
                &inputs_embeds,
                input_ids,
            ),
        }
    }
}

/// Vision-language model: wraps a text model with a vision module
pub struct VisionLanguageModel {
    pub text_model: Box<crate::LoadedModel>,
    pub vision: VisionModule,
}

impl LanguageModel for VisionLanguageModel {
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
        self.text_model.embed_tokens(input_ids)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.text_model.eos_token_ids()
    }
}

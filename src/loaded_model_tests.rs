use super::standard_image_token_block_info;
use crate::vision::connectors::MultiModalConnector;
use crate::vision::encoders::{VisionEncoder, VisionEncoderOutput};
use crate::vision::processors::ImageProcessor;
use crate::vision::{MergeStrategy, VisionModule};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

struct DummyEncoder;

impl VisionEncoder for DummyEncoder {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        VisionEncoderOutput {
            hidden_states: mlxcel_core::ones(&[1, 1, 1], dtype::FLOAT32),
        }
    }
}

struct DummyConnector;

impl MultiModalConnector for DummyConnector {
    fn forward(&self, _vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::ones(&[1, 1, 1], dtype::FLOAT32)
    }
}

struct DummyProcessor;

impl ImageProcessor for DummyProcessor {
    fn preprocess(&self, _images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        mlxcel_core::ones(&[1, 3, 1, 1], dtype::FLOAT32)
    }
}

fn test_vision_module() -> VisionModule {
    VisionModule {
        encoder: Box::new(DummyEncoder),
        connector: Box::new(DummyConnector),
        processor: Box::new(DummyProcessor),
        image_token_id: 99,
        pad_token_id: 0,
        hidden_size: 64,
        boi_token_id: 10,
        eoi_token_id: 11,
        mm_tokens_per_image: 256,
        merge_strategy: MergeStrategy::LLaVA,
        has_bos: true,
        separator_token_id: Some(12),
        suffix_tokens: vec![13, 14],
    }
}

#[test]
fn standard_image_token_block_info_preserves_vision_module_fields() {
    let info = standard_image_token_block_info(&test_vision_module());

    assert!(info.use_boi_eoi);
    assert_eq!(info.image_token_id, 99);
    assert_eq!(info.mm_tokens_per_image, 256);
    assert_eq!(info.boi_token_id, 10);
    assert_eq!(info.eoi_token_id, 11);
    assert!(info.has_bos);
    assert_eq!(info.separator_token_id, Some(12));
    assert_eq!(info.suffix_tokens, vec![13, 14]);
}

#[test]
fn standard_image_token_block_info_disables_boi_eoi_when_tokens_are_absent() {
    let mut module = test_vision_module();
    module.boi_token_id = 0;
    module.eoi_token_id = 0;

    let info = standard_image_token_block_info(&module);

    assert!(!info.use_boi_eoi);
    assert_eq!(info.boi_token_id, 0);
    assert_eq!(info.eoi_token_id, 0);
}

use super::{VlmPreparationSummary, should_prepare_vlm_embeddings};
use crate::vlm_prompt::{ImageTokenBlockAction, ImageTokenBlockStats};

#[test]
fn should_prepare_vlm_embeddings_rejects_non_vlm_image_requests() {
    let err = should_prepare_vlm_embeddings(1, false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Images provided but model is not a vision-language model"));
}

#[test]
fn should_prepare_vlm_embeddings_accepts_vlm_image_requests() {
    assert_eq!(should_prepare_vlm_embeddings(2, true).unwrap(), true);
    assert_eq!(should_prepare_vlm_embeddings(0, true).unwrap(), false);
}

#[test]
fn image_block_summary_preserves_stats_shape() {
    let summary = VlmPreparationSummary::ImageBlocks(ImageTokenBlockStats {
        action: ImageTokenBlockAction::Inserted { image_blocks: 2 },
        tokens_per_image: 256,
    });

    assert_eq!(
        summary,
        VlmPreparationSummary::ImageBlocks(ImageTokenBlockStats {
            action: ImageTokenBlockAction::Inserted { image_blocks: 2 },
            tokens_per_image: 256,
        })
    );
}

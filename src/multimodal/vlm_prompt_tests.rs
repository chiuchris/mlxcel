use super::{
    ImageTokenBlockAction, ImageTokenBlockInfo, ImageTokenBlockStats, apply_image_token_blocks,
};

#[test]
fn apply_image_token_blocks_expands_existing_tokens_with_boi_eoi() {
    let info = ImageTokenBlockInfo {
        use_boi_eoi: true,
        image_token_id: 99,
        mm_tokens_per_image: 3,
        boi_token_id: 10,
        eoi_token_id: 11,
    };
    let mut prompt_tokens = vec![1, 99, 2];

    let stats = apply_image_token_blocks(&mut prompt_tokens, info, 1);

    assert_eq!(
        stats,
        Some(ImageTokenBlockStats {
            action: ImageTokenBlockAction::Expanded {
                existing_image_count: 1,
            },
            tokens_per_image: 3,
        })
    );
    assert_eq!(prompt_tokens, vec![1, 10, 99, 99, 99, 11, 2]);
}

#[test]
fn apply_image_token_blocks_inserts_blocks_after_bos() {
    let info = ImageTokenBlockInfo {
        use_boi_eoi: false,
        image_token_id: 77,
        mm_tokens_per_image: 2,
        boi_token_id: 0,
        eoi_token_id: 0,
    };
    let mut prompt_tokens = vec![1, 2, 3];

    let stats = apply_image_token_blocks(&mut prompt_tokens, info, 2);

    assert_eq!(
        stats,
        Some(ImageTokenBlockStats {
            action: ImageTokenBlockAction::Inserted { image_blocks: 2 },
            tokens_per_image: 2,
        })
    );
    assert_eq!(prompt_tokens, vec![1, 77, 77, 77, 77, 2, 3]);
}

#[test]
fn apply_image_token_blocks_is_noop_without_prompt_or_images() {
    let info = ImageTokenBlockInfo {
        use_boi_eoi: true,
        image_token_id: 5,
        mm_tokens_per_image: 4,
        boi_token_id: 6,
        eoi_token_id: 7,
    };

    let mut empty_prompt = Vec::new();
    assert_eq!(apply_image_token_blocks(&mut empty_prompt, info, 1), None);

    let mut prompt_tokens = vec![1, 2, 3];
    assert_eq!(apply_image_token_blocks(&mut prompt_tokens, info, 0), None);
    assert_eq!(prompt_tokens, vec![1, 2, 3]);
}

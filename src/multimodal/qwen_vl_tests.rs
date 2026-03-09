use super::{InsertedQwenVlmTokens, insert_qwen_vl_image_tokens};

#[test]
fn insert_qwen_vl_image_tokens_inserts_blocks_after_bos() {
    let mut prompt_tokens = vec![1, 42, 43];
    let stats = insert_qwen_vl_image_tokens(&mut prompt_tokens, &[(1, 4, 4)], 2, 100, 103);

    assert_eq!(
        stats,
        Some(InsertedQwenVlmTokens {
            image_blocks: 1,
            total_image_tokens: 4,
        })
    );
    assert_eq!(prompt_tokens, vec![1, 100, 103, 103, 103, 103, 101, 42, 43]);
}

#[test]
fn insert_qwen_vl_image_tokens_is_noop_when_image_tokens_already_exist() {
    let mut prompt_tokens = vec![1, 103, 42, 43];
    let original = prompt_tokens.clone();

    let stats = insert_qwen_vl_image_tokens(&mut prompt_tokens, &[(1, 4, 4)], 2, 100, 103);

    assert_eq!(stats, None);
    assert_eq!(prompt_tokens, original);
}

#[test]
fn insert_qwen_vl_image_tokens_supports_multiple_images() {
    let mut prompt_tokens = vec![1, 7];
    let stats =
        insert_qwen_vl_image_tokens(&mut prompt_tokens, &[(1, 4, 4), (2, 2, 2)], 2, 200, 203);

    assert_eq!(
        stats,
        Some(InsertedQwenVlmTokens {
            image_blocks: 2,
            total_image_tokens: 6,
        })
    );
    assert_eq!(
        prompt_tokens,
        vec![1, 200, 203, 203, 203, 203, 201, 200, 203, 203, 201, 7]
    );
}

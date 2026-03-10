// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

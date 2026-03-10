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

use super::{
    MINICPMO_IMAGE_END_TOKEN, MINICPMO_IMAGE_START_TOKEN, MINICPMO_UNK_TOKEN,
    compute_minicpmo_image_bounds, ensure_minicpmo_image_placeholders, minicpmo_image_placeholder,
    prepare_minicpmo_prompt_tokens,
};

fn fake_encode(text: &str, add_special: bool) -> Vec<i32> {
    let mut tokens = Vec::new();
    if add_special {
        tokens.push(1);
    }

    let mut remaining = text;
    while !remaining.is_empty() {
        if let Some(rest) = remaining.strip_prefix(MINICPMO_IMAGE_START_TOKEN) {
            tokens.push(10);
            remaining = rest;
        } else if let Some(rest) = remaining.strip_prefix(MINICPMO_IMAGE_END_TOKEN) {
            tokens.push(11);
            remaining = rest;
        } else if let Some(rest) = remaining.strip_prefix(MINICPMO_UNK_TOKEN) {
            tokens.push(12);
            remaining = rest;
        } else {
            let ch = remaining.chars().next().unwrap();
            if !ch.is_whitespace() {
                tokens.push(100 + ch as i32);
            }
            remaining = &remaining[ch.len_utf8()..];
        }
    }

    tokens
}

#[test]
fn minicpmo_placeholder_has_wrapping_tokens_and_repeated_unk() {
    let placeholder = minicpmo_image_placeholder(4);
    assert_eq!(placeholder, "<image><unk><unk><unk><unk></image>");
}

#[test]
fn ensure_minicpmo_placeholders_replaces_existing_markers() {
    let prompt = "<|im_start|>user\n<image>hello<|im_end|>";
    let text = ensure_minicpmo_image_placeholders(prompt, 1, 3).unwrap();
    assert!(text.contains("<image><unk><unk><unk></image>"));
    assert!(!text.contains("<image>hello"));
}

#[test]
fn ensure_minicpmo_placeholders_inserts_after_user_tag_when_missing() {
    let prompt = "<|im_start|>user\nDescribe this.<|im_end|>";
    let text = ensure_minicpmo_image_placeholders(prompt, 1, 2).unwrap();
    assert!(text.starts_with("<|im_start|>user\n<image><unk><unk></image>"));
}

#[test]
fn compute_minicpmo_image_bounds_finds_placeholder_spans() {
    let bounds =
        compute_minicpmo_image_bounds(&[1, 10, 12, 12, 11, 20, 10, 12, 11], 10, 11).unwrap();
    assert_eq!(bounds, vec![(2, 4), (7, 8)]);
}

#[test]
fn prepare_minicpmo_prompt_tokens_encodes_prompt_and_bounds() {
    let prepared = prepare_minicpmo_prompt_tokens("<image>abc", 1, 3, fake_encode).unwrap();

    assert_eq!(prepared.image_slots, 1);
    assert_eq!(prepared.image_bounds, vec![(2, 5)]);
    assert_eq!(&prepared.tokens[..6], &[1, 10, 12, 12, 12, 11]);
}

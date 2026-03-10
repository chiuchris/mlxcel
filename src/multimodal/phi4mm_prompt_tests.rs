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

use super::prepare_phi4mm_prompt_tokens;
use crate::phi4_siglip_prompt::PHI4_SIGLIP_IMAGE_TOKEN_INDEX;

#[test]
fn prepare_phi4mm_prompt_tokens_normalizes_numbered_image_tags() {
    let prepared = prepare_phi4mm_prompt_tokens(
        "<|user|>\n<|image_1|>\nDescribe it.",
        1,
        |text, add_special| {
            let mut out = if add_special { vec![101] } else { Vec::new() };
            out.extend(text.bytes().map(i32::from));
            out
        },
    )
    .unwrap();

    assert_eq!(prepared.image_slots, 1);
    assert!(prepared.tokens.contains(&PHI4_SIGLIP_IMAGE_TOKEN_INDEX));
}

#[test]
fn prepare_phi4mm_prompt_tokens_rejects_audio_placeholders() {
    let err =
        prepare_phi4mm_prompt_tokens("<|audio_1|>", 0, |_text, _add_special| vec![1]).unwrap_err();
    assert!(err.contains("audio prompts are not supported"));
}

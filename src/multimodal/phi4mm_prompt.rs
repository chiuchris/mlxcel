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

//! Phi4MM prompt normalization helpers.
//!
//! Phi4MM reuses the same `-200` image sentinel as Phi4-SigLIP, but the model
//! chat template emits `<|image_N|>` placeholders. Normalize that surface once
//! here so CLI and server VLM preparation stay aligned.

use crate::phi4_siglip_prompt::prepare_phi4_siglip_prompt_tokens;
use crate::phi4_siglip_prompt::{PHI4_SIGLIP_IMAGE_TOKEN, Phi4SigLipPromptTokens};

fn normalize_phi4mm_image_tags(prompt: &str, num_images: usize) -> String {
    let mut text = prompt.to_string();
    for image_num in 1..=num_images {
        text = text.replace(&format!("<|image_{}|>", image_num), PHI4_SIGLIP_IMAGE_TOKEN);
    }
    text
}

pub fn prepare_phi4mm_prompt_tokens<E>(
    prompt: &str,
    num_images: usize,
    encode: E,
) -> Result<Phi4SigLipPromptTokens, String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    if prompt.contains("<|audio_") {
        return Err("Phi4MM audio prompts are not supported yet".to_string());
    }

    prepare_phi4_siglip_prompt_tokens(
        &normalize_phi4mm_image_tags(prompt, num_images),
        num_images,
        encode,
    )
}

#[cfg(test)]
#[path = "phi4mm_prompt_tests.rs"]
mod tests;

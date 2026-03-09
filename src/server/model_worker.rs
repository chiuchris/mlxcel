//! Server-side generation helpers used by `ModelProvider`.
//!
//! These helpers keep the request loop focused on orchestration while the
//! image/VLM preparation and streaming decode state live in one place.

use std::time::Instant;

use anyhow::{Result, anyhow};
use image::DynamicImage;

use crate::LoadedModel;
use crate::SamplingConfig;
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::merge::InputEmbeddings;
use crate::vlm_runtime::prepare_and_compute_vlm_embeddings;

use super::GenerationResult;

pub(crate) fn merge_config_stop_tokens(
    mut sampling: SamplingConfig,
    config_eos: &[i32],
) -> SamplingConfig {
    for &id in config_eos {
        if !sampling.stop_token_ids.contains(&id) {
            sampling.stop_token_ids.push(id);
        }
    }
    sampling
}

pub(crate) fn decode_request_images(images: &[Vec<u8>]) -> Result<Vec<DynamicImage>> {
    let decoded_images: Vec<DynamicImage> = images
        .iter()
        .filter_map(|bytes| {
            image::load_from_memory(bytes)
                .map_err(|err| {
                    tracing::warn!("Failed to decode image: {}", err);
                    err
                })
                .ok()
        })
        .collect();

    if decoded_images.is_empty() {
        Err(anyhow!("Failed to decode any images"))
    } else {
        Ok(decoded_images)
    }
}

pub(crate) fn prepare_request_vlm_embeddings(
    model: &LoadedModel,
    tokenizer: &MlxcelTokenizer,
    prompt: &str,
    prompt_tokens: &mut Vec<i32>,
    images: &[Vec<u8>],
) -> Result<Option<InputEmbeddings>> {
    if images.is_empty() || !model.is_vlm() {
        return Ok(None);
    }

    let decoded_images = decode_request_images(images)?;
    let prepared = prepare_and_compute_vlm_embeddings(
        model,
        prompt_tokens,
        prompt,
        &decoded_images,
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .iter()
                .map(|&t| t as i32)
                .collect()
        },
    )?;

    Ok(prepared.map(|prepared| prepared.embeddings))
}

pub(crate) fn build_generation_result(
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    elapsed_ms: u64,
    prompt_eval_ms: u64,
    max_tokens: usize,
) -> GenerationResult {
    let finish_reason = if completion_tokens >= max_tokens {
        "length"
    } else {
        "stop"
    };

    GenerationResult {
        text,
        prompt_tokens,
        completion_tokens,
        generation_time_ms: elapsed_ms,
        prompt_eval_ms,
        generation_only_ms: elapsed_ms.saturating_sub(prompt_eval_ms),
        finish_reason: finish_reason.to_string(),
    }
}

pub(crate) struct StreamingDecodeState {
    all_ids: Vec<u32>,
    prev_decoded_len: usize,
    generated_text: String,
    completion_tokens: usize,
    first_token_time: Option<Instant>,
}

impl StreamingDecodeState {
    pub(crate) fn new(tokenizer: &MlxcelTokenizer, prompt_tokens: &[i32]) -> Self {
        let all_ids: Vec<u32> = prompt_tokens.iter().map(|&x| x as u32).collect();
        let prev_decoded_len = tokenizer.decode(&all_ids, false).unwrap_or_default().len();

        Self {
            all_ids,
            prev_decoded_len,
            generated_text: String::new(),
            completion_tokens: 0,
            first_token_time: None,
        }
    }

    pub(crate) fn on_token(
        &mut self,
        token_id: i32,
        tokenizer: &MlxcelTokenizer,
    ) -> Option<String> {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
        }
        self.completion_tokens += 1;
        self.all_ids.push(token_id as u32);

        let full_text = tokenizer.decode(&self.all_ids, false).unwrap_or_default();
        let new_text = &full_text[self.prev_decoded_len..];
        if new_text.is_empty() {
            return None;
        }

        self.generated_text.push_str(new_text);
        self.prev_decoded_len = full_text.len();
        Some(new_text.to_string())
    }

    pub(crate) fn finish(
        self,
        start: Instant,
        prompt_token_count: usize,
        max_tokens: usize,
    ) -> GenerationResult {
        let elapsed_ms = start.elapsed().as_millis() as u64;
        let prompt_eval_ms = self
            .first_token_time
            .map(|t| (t - start).as_millis() as u64)
            .unwrap_or(elapsed_ms);

        build_generation_result(
            self.generated_text,
            prompt_token_count,
            self.completion_tokens,
            elapsed_ms,
            prompt_eval_ms,
            max_tokens,
        )
    }
}

#[cfg(test)]
#[path = "model_worker_tests.rs"]
mod tests;

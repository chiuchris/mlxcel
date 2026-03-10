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

//! Server-side generation helpers and worker lifecycle for `ModelProvider`.
//!
//! `ModelProvider` owns the public channel API, while this module owns the
//! long-lived worker thread behavior plus the image/VLM preparation helpers.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use anyhow::{Result, anyhow};
use image::DynamicImage;

use crate::LoadedModel;
use crate::SamplingConfig;
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::merge::InputEmbeddings;
use crate::vlm_runtime::{prepare_and_compute_vlm_embeddings, prepared_embedding_refs};

use super::{GenerateEvent, GenerationResult, ModelRequest};

pub(crate) fn spawn_model_worker(
    model_path: PathBuf,
    adapter_path: Option<PathBuf>,
    request_rx: mpsc::Receiver<ModelRequest>,
    loaded: Arc<AtomicBool>,
    worker_model_id: String,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        tracing::info!("Model worker thread starting, loading model...");

        let result = if let Some(adapter) = adapter_path {
            tracing::info!("Loading LoRA adapter from {:?}", adapter);
            crate::load_model_with_adapter(&model_path, &adapter)
        } else {
            crate::load_model(&model_path)
        };

        let (model, tokenizer) = match result {
            Ok((model, tokenizer)) => {
                tracing::info!("Model {} loaded successfully", worker_model_id);
                loaded.store(true, Ordering::Release);
                (model, tokenizer)
            }
            Err(err) => {
                tracing::error!("Failed to load model: {}", err);
                return;
            }
        };

        let config_eos = crate::read_eos_token_ids(&model_path);
        if !config_eos.is_empty() {
            tracing::info!("EOS tokens from config: {:?}", config_eos);
        }

        let num_layers = crate::LanguageModel::num_layers(&model);
        let mut generator = crate::CxxGenerator::new(num_layers);

        loop {
            match request_rx.recv() {
                Ok(ModelRequest::Generate {
                    prompt,
                    options,
                    images,
                    response_tx,
                }) => handle_generate_request(
                    &model,
                    &tokenizer,
                    &config_eos,
                    &mut generator,
                    prompt,
                    options,
                    images,
                    response_tx,
                ),
                Ok(ModelRequest::Shutdown) => {
                    tracing::info!("Model worker thread shutting down");
                    break;
                }
                Err(_) => {
                    tracing::info!("Request channel closed, worker exiting");
                    break;
                }
            }
        }
    })
}

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

fn handle_generate_request(
    model: &LoadedModel,
    tokenizer: &MlxcelTokenizer,
    config_eos: &[i32],
    generator: &mut crate::CxxGenerator,
    prompt: String,
    options: crate::server::ServerGenerateOptions,
    images: Vec<Vec<u8>>,
    response_tx: mpsc::Sender<GenerateEvent>,
) {
    let start = std::time::Instant::now();

    let token_ids = match tokenizer.encode(prompt.as_str(), true) {
        Ok(ids) => ids,
        Err(err) => {
            let _ = response_tx.send(GenerateEvent::Error(format!("Tokenization error: {}", err)));
            return;
        }
    };
    let mut prompt_tokens: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
    let prompt_token_count = prompt_tokens.len();

    // Uses reset_with_model to also clear model-internal caches
    // such as sliding-window attention and SSM state.
    generator.reset_with_model(model);

    let max_tokens = options.max_tokens;
    let sampling = merge_config_stop_tokens(options.sampling.clone(), config_eos);

    let vlm_embeddings = match prepare_request_vlm_embeddings(
        model,
        tokenizer,
        &prompt,
        &mut prompt_tokens,
        &images,
    ) {
        Ok(prepared) => prepared,
        Err(err) => {
            let _ = response_tx.send(GenerateEvent::Error(err.to_string()));
            return;
        }
    };

    let mut decode_state = StreamingDecodeState::new(tokenizer, &prompt_tokens);
    let tx_clone = response_tx.clone();

    let on_token = |token_id: i32| {
        if let Some(new_text) = decode_state.on_token(token_id, tokenizer) {
            let _ = tx_clone.send(GenerateEvent::Token(new_text));
        }
        true
    };

    if let Some(ref embeddings) = vlm_embeddings {
        let (input_embeds, mask_ref) = match prepared_embedding_refs(embeddings) {
            Ok(refs) => refs,
            Err(err) => {
                let _ = response_tx.send(GenerateEvent::Error(err.to_string()));
                return;
            }
        };
        generator.generate_streaming_with_embeddings(
            model,
            &prompt_tokens,
            Some(input_embeds),
            mask_ref,
            max_tokens,
            &sampling,
            on_token,
        );
    } else {
        generator.generate_streaming(model, &prompt_tokens, max_tokens, &sampling, on_token);
    }

    // Flush any remaining text buffered due to incomplete UTF-8 sequences
    decode_state.flush(tokenizer);

    let result = decode_state.finish(start, prompt_token_count, max_tokens);
    let _ = response_tx.send(GenerateEvent::Done(result));
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

        // Find the safe emit boundary: skip trailing U+FFFD replacement characters.
        // Byte-level BPE tokenizers split multi-byte UTF-8 sequences across tokens.
        // Incomplete byte sequences decode as U+FFFD, but become valid characters
        // once the completing token arrives. Emitting FFFD prematurely corrupts
        // the output (e.g. "최솟값" → "최�값") because the byte offset shifts
        // when the replacement chars resolve into shorter real characters.
        let safe_len = safe_emit_boundary(&full_text);

        if safe_len <= self.prev_decoded_len {
            return None;
        }

        let new_text = &full_text[self.prev_decoded_len..safe_len];
        if new_text.is_empty() {
            return None;
        }

        self.generated_text.push_str(new_text);
        self.prev_decoded_len = safe_len;
        Some(new_text.to_string())
    }

    /// Flush any remaining buffered text (including unresolved replacement chars)
    /// at the end of generation.
    pub(crate) fn flush(&mut self, tokenizer: &MlxcelTokenizer) {
        let full_text = tokenizer.decode(&self.all_ids, false).unwrap_or_default();
        if full_text.len() > self.prev_decoded_len {
            let remaining = &full_text[self.prev_decoded_len..];
            self.generated_text.push_str(remaining);
            self.prev_decoded_len = full_text.len();
        }
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

/// Find the byte position after the last non-U+FFFD character.
/// Trailing replacement characters are buffered because they likely come from
/// incomplete multi-byte UTF-8 sequences that will be completed by the next token.
fn safe_emit_boundary(text: &str) -> usize {
    text.char_indices()
        .rev()
        .find(|(_, c)| *c != '\u{FFFD}')
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "model_worker_tests.rs"]
mod tests;

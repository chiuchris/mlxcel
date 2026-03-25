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
use crate::server::batch::BatchObservability;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::merge::InputEmbeddings;
use crate::vlm_runtime::prepare_and_compute_vlm_embeddings;

use super::{GenerationResult, ModelRequest};

/// Configuration for the scheduler, passed from `ModelProvider` to the
/// worker thread.
pub(crate) struct WorkerSchedulerConfig {
    pub max_batch_size: usize,
    pub max_queue_depth: usize,
    pub prefill_chunk_size: usize,
    pub enable_preemption: bool,
    pub preemption_policy: crate::server::config::PreemptionPolicy,
}

pub(crate) fn spawn_model_worker_with_batch_config(
    model_path: PathBuf,
    adapter_path: Option<PathBuf>,
    request_rx: mpsc::Receiver<ModelRequest>,
    loaded: Arc<AtomicBool>,
    worker_model_id: String,
    sched_config: WorkerSchedulerConfig,
    batch_metrics: Arc<BatchMetrics>,
    batch_observability: Arc<BatchObservability>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        tracing::info!("Model worker thread starting, loading model...");

        let load_start = Instant::now();
        let result = if let Some(adapter) = adapter_path {
            tracing::info!("Loading LoRA adapter from {:?}", adapter);
            crate::load_model_with_adapter(&model_path, &adapter)
        } else {
            crate::load_model(&model_path)
        };

        let (model, tokenizer) = match result {
            Ok((model, tokenizer)) => {
                let load_elapsed = load_start.elapsed();
                tracing::info!(
                    "Model {worker_model_id} loaded in {:.3}s",
                    load_elapsed.as_secs_f64()
                );
                loaded.store(true, Ordering::Release);
                (model, tokenizer)
            }
            Err(err) => {
                tracing::error!("Failed to load model: {err}");
                return;
            }
        };

        let config_eos = crate::read_eos_token_ids(&model_path);
        if !config_eos.is_empty() {
            tracing::info!("EOS tokens from config: {:?}", config_eos);
        }

        let chunk_info = if sched_config.prefill_chunk_size > 0 {
            format!(", prefill_chunk_size={}", sched_config.prefill_chunk_size)
        } else {
            String::new()
        };
        tracing::info!(
            "Starting BatchScheduler (max_batch_size={}, \
             max_queue_depth={}{chunk_info})",
            sched_config.max_batch_size,
            sched_config.max_queue_depth,
        );

        let mut scheduler = super::super::batch::BatchScheduler::with_config(
            model,
            tokenizer,
            config_eos,
            request_rx,
            sched_config.max_batch_size,
            sched_config.max_queue_depth,
            batch_metrics,
            batch_observability,
            sched_config.prefill_chunk_size,
            sched_config.enable_preemption,
            sched_config.preemption_policy,
        );
        scheduler.run();
    })
}

/// Spawn the legacy sequential model worker.
///
/// This worker processes one request at a time using the `BatchScheduler` with
/// `max_batch_size=1` and no chunked prefill, which is functionally equivalent
/// to the pre-scheduler sequential `recv()` loop. It is activated when
/// `--no-batch` is passed on the CLI.
///
/// Choosing this path explicitly guarantees:
/// - No batch scheduling data structures are allocated beyond size-1.
/// - No prefill chunking interleaving occurs.
/// - Log output clearly indicates the sequential execution mode.
///
/// The CLI `generate` command is unaffected and uses `CxxGenerator` directly.
pub(crate) fn spawn_legacy_model_worker(
    model_path: PathBuf,
    adapter_path: Option<PathBuf>,
    request_rx: mpsc::Receiver<ModelRequest>,
    loaded: Arc<AtomicBool>,
    worker_model_id: String,
    batch_metrics: Arc<BatchMetrics>,
    batch_observability: Arc<BatchObservability>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        tracing::info!(
            "Model worker thread starting (legacy sequential mode, --no-batch), loading model..."
        );

        let load_start = Instant::now();
        let result = if let Some(adapter) = adapter_path {
            tracing::info!("Loading LoRA adapter from {:?}", adapter);
            crate::load_model_with_adapter(&model_path, &adapter)
        } else {
            crate::load_model(&model_path)
        };

        let (model, tokenizer) = match result {
            Ok((model, tokenizer)) => {
                let load_elapsed = load_start.elapsed();
                tracing::info!(
                    "Model {worker_model_id} loaded in {:.3}s",
                    load_elapsed.as_secs_f64()
                );
                loaded.store(true, Ordering::Release);
                (model, tokenizer)
            }
            Err(err) => {
                tracing::error!("Failed to load model: {err}");
                return;
            }
        };

        let config_eos = crate::read_eos_token_ids(&model_path);
        if !config_eos.is_empty() {
            tracing::info!("EOS tokens from config: {:?}", config_eos);
        }

        tracing::info!(
            "Starting legacy sequential worker \
             (max_batch_size=1, prefill_chunk_size=disabled)"
        );

        // Reuse BatchScheduler with max_batch_size=1 and chunking disabled.
        // Per the scheduler docs, size-1 behavior is identical to the old
        // sequential recv() loop, with no extra overhead.
        let mut scheduler = super::super::batch::BatchScheduler::with_config(
            model,
            tokenizer,
            config_eos,
            request_rx,
            1,          // max_batch_size = 1 → sequential, no interleaving
            usize::MAX, // max_queue_depth: unbounded (one at a time anyway)
            batch_metrics,
            batch_observability,
            0,     // prefill_chunk_size = 0 → chunking disabled
            false, // enable_preemption = false
            crate::server::config::PreemptionPolicy::default(),
        );
        scheduler.run();
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
        // Moondream3 needs special prompt formatting even for text-only
        if images.is_empty() && matches!(model, LoadedModel::Moondream3VLM(_)) {
            let prepared = crate::moondream3_prompt::prepare_moondream3_prompt_tokens(
                prompt,
                0,
                |text, add_special| {
                    tokenizer
                        .encode(text, add_special)
                        .unwrap_or_default()
                        .iter()
                        .map(|&t| t as i32)
                        .collect()
                },
            )
            .map_err(|e| anyhow!("{}", e))?;
            *prompt_tokens = prepared.tokens;
        }
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

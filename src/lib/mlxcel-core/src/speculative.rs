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

//! Speculative decoding for accelerated inference
//!
//! Uses a small "draft" model to generate candidate tokens, then verifies
//! them in batch with the main model. Accepted tokens skip individual
//! forward passes, improving throughput when the draft model's predictions
//! match the main model's.
//!
//! Algorithm:
//! 1. Prefill prompt through both models
//! 2. Sample first token from main model
//! 3. Loop:
//!    a. Draft: generate `num_draft` tokens with draft model
//!    b. Verify: forward [current + draft tokens] through main model
//!    c. Accept matching prefix, rewind caches for rejected tokens
//!    d. Continue from the divergence point

use crate::ffi;
use crate::ffi::MlxStream;
use crate::generate::{GenerationStats, LanguageModel, SamplingConfig};
use crate::generation_policy::{initial_token_history, merged_eos_token_ids};
use crate::hardware;
use crate::layers::KVCache;
use crate::sampling::sample_token_optimized;
use crate::streams::{install_default_stream, new_generation_stream};
use crate::utils::{align_to_na_tile, create_padded_prefill_mask};
use cxx::UniquePtr;
use std::time::Instant;

/// Returns true when the current hardware is M5+ with a Neural Accelerator
/// and tile-aligned verification batching should be applied.
#[inline]
fn should_align_verification() -> bool {
    let hw = hardware::get_hardware();
    hw.has_neural_accelerator && hw.macos_supports_na
}

/// Speculative decoding generator
///
/// Uses a draft model to propose candidate tokens and a main model to verify them.
/// When the draft model's predictions match, multiple tokens are accepted per
/// main model forward pass, improving throughput.
pub struct SpeculativeGenerator {
    main_caches: Vec<KVCache>,
    draft_caches: Vec<KVCache>,
    generated_tokens: Vec<i32>,
    generation_stream: Option<UniquePtr<MlxStream>>,
}

impl SpeculativeGenerator {
    /// Create a new speculative generator
    pub fn new(main_num_layers: usize, draft_num_layers: usize) -> Self {
        Self {
            main_caches: (0..main_num_layers).map(|_| KVCache::new()).collect(),
            draft_caches: (0..draft_num_layers).map(|_| KVCache::new()).collect(),
            generated_tokens: Vec::new(),
            generation_stream: new_generation_stream(),
        }
    }

    /// Reset generator state
    pub fn reset(&mut self) {
        for cache in &mut self.main_caches {
            *cache = KVCache::new();
        }
        for cache in &mut self.draft_caches {
            *cache = KVCache::new();
        }
        self.generated_tokens.clear();
    }

    /// Get the generated tokens
    pub fn tokens(&self) -> &[i32] {
        &self.generated_tokens
    }

    /// Generate tokens using speculative decoding
    ///
    /// # Arguments
    /// * `main_model` - The main (target) model for verification
    /// * `draft_model` - The smaller draft model for candidate generation
    /// * `prompt_tokens` - Input prompt token IDs
    /// * `max_tokens` - Maximum number of tokens to generate
    /// * `num_draft` - Number of draft tokens to generate per speculation step
    /// * `sampling` - Sampling configuration
    pub fn generate<M: LanguageModel, D: LanguageModel>(
        &mut self,
        main_model: &M,
        draft_model: &D,
        prompt_tokens: &[i32],
        max_tokens: usize,
        num_draft: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        self.reset();

        // Set generation stream
        install_default_stream(self.generation_stream.as_ref());

        let eos_tokens = merged_eos_token_ids(main_model.eos_token_ids(), &sampling.stop_token_ids);
        let needs_history = sampling.needs_token_history();
        let mut token_history = initial_token_history(prompt_tokens, needs_history);

        // PREFILL PHASE.
        let prefill_start = Instant::now();

        let input = ffi::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);

        // Prefill both models
        let main_logits = main_model.forward(&input, &mut self.main_caches, None);
        let _draft_logits = draft_model.forward(&input, &mut self.draft_caches, None);

        // Sample first token from main model
        let (first_token_arr, _) = sample_token_optimized(&main_logits, sampling, &token_history);
        ffi::eval(&first_token_arr);
        let first_token = ffi::item_i32(&first_token_arr);
        let prefill_time = prefill_start.elapsed();

        if eos_tokens.contains(&first_token) || max_tokens <= 1 {
            let stats = Self::build_stats(
                prompt_tokens.len(),
                self.generated_tokens.len(),
                prefill_time,
                std::time::Duration::ZERO,
            );
            return (self.generated_tokens.clone(), stats);
        }

        self.generated_tokens.push(first_token);
        if needs_history {
            token_history.push(first_token);
        }

        // DECODE PHASE.
        let decode_start = Instant::now();
        let mut current_token = first_token;
        let mut done = false;

        while self.generated_tokens.len() < max_tokens && !done {
            // Step 1: Generate draft tokens
            let mut draft_tokens = Vec::with_capacity(num_draft);
            let mut draft_token = current_token;

            for _ in 0..num_draft {
                let draft_input = ffi::from_slice_i32(&[draft_token], &[1, 1]);
                let draft_logits = draft_model.forward(&draft_input, &mut self.draft_caches, None);
                let (tok_arr, _) = sample_token_optimized(&draft_logits, sampling, &token_history);
                ffi::eval(&tok_arr);
                draft_token = ffi::item_i32(&tok_arr);
                draft_tokens.push(draft_token);

                if eos_tokens.contains(&draft_token) {
                    break;
                }
            }

            if draft_tokens.is_empty() {
                break;
            }

            // Step 2: Verify draft tokens with main model in a single batched forward pass.
            // Input: [current_token, draft_token_0, ..., draft_token_n-1] shape [1, N+1]
            // Output: logits shape [1, N+1, vocab_size]
            //
            // This is structurally identical to a prefill pass, converting N memory-bound
            // GEMV decode operations into one compute-bound GEMM. On M5+ Neural Accelerator
            // hardware, this yields 3-4x speedup via tile-aligned GEMM dispatch.
            let mut verify_tokens = vec![current_token];
            verify_tokens.extend_from_slice(&draft_tokens);
            let actual_verify_len = verify_tokens.len();

            // On M5+ hardware with Neural Accelerator, pad the verification sequence
            // to a 32-token tile boundary for optimal GEMM throughput. On other
            // hardware, no padding is needed (batching is still beneficial but
            // tile alignment does not apply).
            let main_logits = if should_align_verification() && main_model.supports_padded_prefill()
            {
                let padded_len = align_to_na_tile(actual_verify_len);
                // Capture the current KV cache offset before the verification pass
                // so the attention mask correctly spans [offset, offset + padded_len).
                let kv_offset = self.main_caches.first().map(|c| c.offset).unwrap_or(0);

                if padded_len > actual_verify_len {
                    // Pad with zeros up to the tile boundary
                    let mut padded_tokens = verify_tokens.clone();
                    padded_tokens.resize(padded_len, 0);
                    let verify_input = ffi::from_slice_i32(&padded_tokens, &[1, padded_len as i32]);
                    // Create attention mask so padding positions cannot attend to
                    // anything and real tokens cannot attend to padding keys.
                    let mask = create_padded_prefill_mask(
                        actual_verify_len as i32,
                        padded_len as i32,
                        kv_offset,
                    );
                    let raw_logits = main_model.forward(
                        &verify_input,
                        &mut self.main_caches,
                        Some(mask.as_ref().unwrap()),
                    );
                    // Trim padding positions from KV caches so subsequent decode
                    // steps see the correct cache offset (actual_verify_len tokens,
                    // not padded_len tokens).
                    let excess = (padded_len - actual_verify_len) as i32;
                    for cache in self.main_caches.iter_mut() {
                        cache.trim(excess);
                    }
                    main_model.trim_internal_caches(excess);
                    // Return only the logits for the actual (non-padded) positions,
                    // sliced to shape [1, actual_verify_len, vocab].
                    let vocab = ffi::array_shape(&raw_logits)[2];
                    ffi::slice(
                        &raw_logits,
                        &[0, 0, 0],
                        &[1, actual_verify_len as i32, vocab],
                    )
                } else {
                    // Sequence already aligns to a tile boundary; no padding needed.
                    let verify_input =
                        ffi::from_slice_i32(&verify_tokens, &[1, actual_verify_len as i32]);
                    main_model.forward(&verify_input, &mut self.main_caches, None)
                }
            } else {
                // Non-NA hardware: plain batched forward pass, no tile alignment.
                let verify_input =
                    ffi::from_slice_i32(&verify_tokens, &[1, actual_verify_len as i32]);
                main_model.forward(&verify_input, &mut self.main_caches, None)
            };

            // The main model returns logits for each position:
            // - Position 0 (current_token): logits that would produce draft_tokens[0]
            // - Position i: logits that would produce draft_tokens[i]
            // - Last position: logits for the token after all draft tokens

            // Step 3: Compare draft tokens with main model's choices
            let main_shape = ffi::array_shape(&main_logits);
            let seq_len = main_shape[1]; // Number of logit positions (actual, not padded)
            let mut accepted = 0;

            for (i, draft_token) in draft_tokens.iter().copied().enumerate() {
                if (i as i32) >= seq_len {
                    break;
                }

                // Get main model's logits at position i
                let pos_logits = ffi::slice(
                    &main_logits,
                    &[0, i as i32, 0],
                    &[1, (i as i32) + 1, main_shape[2]],
                );
                // Reshape to [1, 1, vocab] for sample_token_optimized
                let pos_logits = ffi::reshape(&pos_logits, &[1, 1, main_shape[2]]);

                let (main_tok_arr, _) =
                    sample_token_optimized(&pos_logits, sampling, &token_history);
                ffi::eval(&main_tok_arr);
                let main_token = ffi::item_i32(&main_tok_arr);

                if main_token == draft_token {
                    // Accept draft token
                    accepted += 1;

                    if eos_tokens.contains(&draft_token) {
                        done = true;
                        break;
                    }

                    self.generated_tokens.push(draft_token);
                    if needs_history {
                        token_history.push(draft_token);
                    }

                    if self.generated_tokens.len() >= max_tokens {
                        done = true;
                        break;
                    }
                } else {
                    // Reject: use main model's token instead
                    if eos_tokens.contains(&main_token) {
                        done = true;
                    } else {
                        self.generated_tokens.push(main_token);
                        if needs_history {
                            token_history.push(main_token);
                        }
                    }
                    break;
                }
            }

            // If all draft tokens were accepted and we're not done,
            // sample one more token from the main model's last logit position
            if accepted == draft_tokens.len() && !done && self.generated_tokens.len() < max_tokens {
                let last_pos = seq_len - 1;
                let last_logits = ffi::slice(
                    &main_logits,
                    &[0, last_pos, 0],
                    &[1, last_pos + 1, main_shape[2]],
                );
                let last_logits = ffi::reshape(&last_logits, &[1, 1, main_shape[2]]);
                let (bonus_tok_arr, _) =
                    sample_token_optimized(&last_logits, sampling, &token_history);
                ffi::eval(&bonus_tok_arr);
                let bonus_token = ffi::item_i32(&bonus_tok_arr);

                if eos_tokens.contains(&bonus_token) {
                    done = true;
                } else {
                    self.generated_tokens.push(bonus_token);
                    if needs_history {
                        token_history.push(bonus_token);
                    }
                }

                current_token = bonus_token;
            } else if !done {
                current_token = *self.generated_tokens.last().unwrap();
            }

            // Step 4: Rewind caches for rejected tokens
            let rejected = draft_tokens.len() - accepted;
            if rejected > 0 {
                // Rewind main model caches: we forwarded all verify_tokens but
                // only accepted `accepted` draft tokens + 1 (the divergence token from main)
                // Main model cache has current_token + all draft tokens = verify_tokens.len() positions
                // We need to keep: accepted + 1 (for the token we're continuing from)
                // So trim: draft_tokens.len() - accepted
                let main_trim = rejected as i32;
                trim_caches(&mut self.main_caches, main_trim);

                // Rewind draft model caches: drafted all draft_tokens
                // Need to trim all rejected + 1 (because draft went past accepted)
                let draft_trim = (rejected + 1) as i32;
                trim_caches(
                    &mut self.draft_caches,
                    draft_trim.min(draft_tokens.len() as i32),
                );
            }

            // Periodic cache clearing
            if self.generated_tokens.len().is_multiple_of(256) {
                ffi::clear_memory_cache();
            }
        }

        let decode_time = decode_start.elapsed();

        let stats = Self::build_stats(
            prompt_tokens.len(),
            self.generated_tokens.len(),
            prefill_time,
            decode_time,
        );

        (self.generated_tokens.clone(), stats)
    }

    fn build_stats(
        prompt_count: usize,
        gen_count: usize,
        prefill_time: std::time::Duration,
        decode_time: std::time::Duration,
    ) -> GenerationStats {
        let prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        let decode_ms = decode_time.as_secs_f64() * 1000.0;

        GenerationStats {
            prompt_tokens: prompt_count,
            generated_tokens: gen_count,
            prefill_time_ms: prefill_ms,
            decode_time_ms: decode_ms,
            prefill_tok_per_sec: if prefill_ms > 0.0 {
                prompt_count as f64 / (prefill_ms / 1000.0)
            } else {
                0.0
            },
            decode_tok_per_sec: if decode_ms > 0.0 {
                gen_count as f64 / (decode_ms / 1000.0)
            } else {
                0.0
            },
        }
    }
}

/// Trim the last `n` entries from all caches in the slice
/// Returns the number of entries actually trimmed (from the first cache)
fn trim_caches(caches: &mut [KVCache], n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    let mut trimmed = 0;
    for cache in caches.iter_mut() {
        trimmed = cache.trim(n);
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;

    #[test]
    fn test_kv_cache_trim_basic() {
        let mut cache = KVCache::new();

        // Add some data: [batch=1, heads=2, seq_len=5, head_dim=4]
        let keys = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
        cache.update(keys, values);
        assert_eq!(cache.offset, 5);

        // Trim 2
        let trimmed = cache.trim(2);
        assert_eq!(trimmed, 2);
        assert_eq!(cache.offset, 3);

        // Verify shapes
        let k_shape = ffi::array_shape(cache.keys.as_ref().unwrap());
        assert_eq!(k_shape, vec![1, 2, 3, 4]);
        let v_shape = ffi::array_shape(cache.values.as_ref().unwrap());
        assert_eq!(v_shape, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_kv_cache_trim_all() {
        let mut cache = KVCache::new();
        let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        cache.update(keys, values);

        // Trim all
        let trimmed = cache.trim(3);
        assert_eq!(trimmed, 3);
        assert_eq!(cache.offset, 0);
        assert!(cache.keys.is_none());
        assert!(cache.values.is_none());
    }

    #[test]
    fn test_kv_cache_trim_zero() {
        let mut cache = KVCache::new();
        let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        cache.update(keys, values);

        let trimmed = cache.trim(0);
        assert_eq!(trimmed, 0);
        assert_eq!(cache.offset, 3);
    }

    #[test]
    fn test_kv_cache_trim_more_than_available() {
        let mut cache = KVCache::new();
        let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        cache.update(keys, values);

        // Trim more than available - should trim only what's available
        let trimmed = cache.trim(10);
        assert_eq!(trimmed, 3);
        assert_eq!(cache.offset, 0);
        assert!(cache.keys.is_none());
    }

    #[test]
    fn test_trim_caches_helper() {
        let mut caches = vec![KVCache::new(), KVCache::new()];
        for cache in caches.iter_mut() {
            let keys = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
            let values = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
            cache.update(keys, values);
        }

        let trimmed = trim_caches(&mut caches, 2);
        assert_eq!(trimmed, 2);
        for cache in &caches {
            assert_eq!(cache.offset, 3);
        }
    }
}

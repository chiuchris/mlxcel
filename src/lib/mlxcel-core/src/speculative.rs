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
use crate::layers::KVCache;
use crate::sampling::sample_token_optimized;
use cxx::UniquePtr;
use std::time::Instant;

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
        let generation_stream = if ffi::is_gpu_available() {
            Some(ffi::new_gpu_stream())
        } else {
            None
        };

        Self {
            main_caches: (0..main_num_layers).map(|_| KVCache::new()).collect(),
            draft_caches: (0..draft_num_layers).map(|_| KVCache::new()).collect(),
            generated_tokens: Vec::new(),
            generation_stream,
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
        if let Some(ref stream) = self.generation_stream {
            ffi::set_default_stream(stream);
        }

        let mut eos_tokens = main_model.eos_token_ids();
        for &id in &sampling.stop_token_ids {
            if !eos_tokens.contains(&id) {
                eos_tokens.push(id);
            }
        }
        let needs_history = sampling.needs_token_history();
        let mut token_history: Vec<i32> = if needs_history {
            prompt_tokens.to_vec()
        } else {
            Vec::new()
        };

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

            // Step 2: Verify draft tokens with main model
            // Forward [current_token, draft_token_0, ..., draft_token_n-1] through main model
            let mut verify_tokens = vec![current_token];
            verify_tokens.extend_from_slice(&draft_tokens);
            let verify_input =
                ffi::from_slice_i32(&verify_tokens, &[1, verify_tokens.len() as i32]);
            let main_logits = main_model.forward(&verify_input, &mut self.main_caches, None);

            // The main model returns logits for each position:
            // - Position 0 (current_token): logits that would produce draft_tokens[0]
            // - Position i: logits that would produce draft_tokens[i]
            // - Last position: logits for the token after all draft tokens

            // Step 3: Compare draft tokens with main model's choices
            let main_shape = ffi::array_shape(&main_logits);
            let seq_len = main_shape[1]; // Number of logit positions
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

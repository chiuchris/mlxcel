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

//! Token generation utilities for mlxcel-core models
//!
//! This module provides the generation loop and sampling functions
//! for text generation with mlxcel-core models.
//!
//! Key optimizations matching Python mlx-lm:
//! - Dedicated generation stream for pipelined execution
//! - Lookahead pipelining: compute token n+1 while returning token n
//! - Optimized decode loops for standard and embedding-prefill paths
//! - Shared sampling policy delegated to `crate::sampling`

use crate::ffi;
use crate::ffi::{MlxArray, MlxStream};
use crate::layers::KVCache;
use crate::sampling::sample_token_optimized;
use cxx::UniquePtr;

/// Trait for language models that can be used for generation
pub trait LanguageModel {
    /// Forward pass through the model
    /// Returns logits of shape [batch, seq_len, vocab_size]
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray>;

    /// Create KV caches for all layers
    fn make_caches(&self) -> Vec<KVCache>;

    /// Get the number of layers
    fn num_layers(&self) -> usize;

    /// Get the EOS token IDs for this model
    fn eos_token_ids(&self) -> Vec<i32>;

    /// Forward with pre-computed embeddings (for VLM prefill)
    /// Used by: VisionLanguageModel (Gemma3 VLM)
    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Default: ignore embeddings, use standard forward
        let _ = input_embeddings;
        self.forward(input_ids, caches, mask)
    }

    /// Get embeddings for token IDs (needed by VisionModule for merging)
    /// Used by: VisionModule::get_input_embeddings
    fn embed_tokens(&self, _input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        None // default: not supported
    }
}

/// Sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for sampling (1.0 = no change)
    pub temperature: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: i32,
    /// Top-p (nucleus) sampling (1.0 = disabled)
    pub top_p: f32,
    /// Min-p sampling threshold (0.0 = disabled)
    /// Removes tokens with probability < min_p * max_probability
    pub min_p: f32,
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
    /// Repetition penalty (1.0 = disabled)
    pub repetition_penalty: f32,
    /// DRY multiplier (0.0 = disabled)
    pub dry_multiplier: f32,
    /// DRY exponential base (default: 1.75)
    pub dry_base: f32,
    /// DRY minimum match length before penalty applies (default: 2)
    pub dry_allowed_length: usize,
    /// DRY lookback window (0 = all history)
    pub dry_penalty_last_n: usize,
    /// Token IDs that break DRY matching (e.g., newlines, punctuation)
    pub dry_sequence_breakers: Vec<i32>,
    /// OpenAI-style frequency penalty: subtract penalty * count(token) from logits (0.0 = disabled)
    pub frequency_penalty: f32,
    /// OpenAI-style presence penalty: subtract penalty if token appeared at all (0.0 = disabled)
    pub presence_penalty: f32,
    /// Additional stop token IDs (from generation_config.json or API request)
    /// Merged with model's built-in eos_token_ids during generation
    pub stop_token_ids: Vec<i32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: None,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 0,
            dry_sequence_breakers: Vec::new(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
        }
    }
}

impl SamplingConfig {
    /// Create greedy sampling config (temperature 0)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            seed: None,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 0,
            dry_sequence_breakers: Vec::new(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
        }
    }

    /// Create config with specific temperature
    pub fn with_temperature(temp: f32) -> Self {
        Self {
            temperature: temp,
            ..Default::default()
        }
    }

    /// Check if any penalty-based sampling is enabled
    pub fn needs_token_history(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.dry_multiplier > 0.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }
}

/// Generation statistics
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Number of prompt tokens processed
    pub prompt_tokens: usize,
    /// Number of tokens generated
    pub generated_tokens: usize,
    /// Time to process the prompt (prefill) in milliseconds
    pub prefill_time_ms: f64,
    /// Time to generate tokens (decode) in milliseconds
    pub decode_time_ms: f64,
    /// Prefill throughput: prompt tokens per second
    pub prefill_tok_per_sec: f64,
    /// Decode throughput: generated tokens per second
    pub decode_tok_per_sec: f64,
}

impl GenerationStats {
    /// Print formatted stats
    pub fn print(&self) {
        println!("  Prompt tokens:    {}", self.prompt_tokens);
        println!("  Generated tokens: {}", self.generated_tokens);
        println!(
            "  Prefill:          {:.2} ms ({:.2} tok/s)",
            self.prefill_time_ms, self.prefill_tok_per_sec
        );
        println!(
            "  Decode:           {:.2} ms ({:.2} tok/s)",
            self.decode_time_ms, self.decode_tok_per_sec
        );
    }
}

/// Generator state for managing generation
pub struct CxxGenerator {
    caches: Vec<KVCache>,
    generated_tokens: Vec<i32>,
    /// Dedicated generation stream for pipelining
    generation_stream: Option<UniquePtr<MlxStream>>,
}

impl CxxGenerator {
    /// Create a new generator
    pub fn new(num_layers: usize) -> Self {
        // Create dedicated generation stream like Python mlx-lm
        let generation_stream = if ffi::is_gpu_available() {
            Some(ffi::new_gpu_stream())
        } else {
            None
        };

        Self {
            caches: (0..num_layers).map(|_| KVCache::new()).collect(),
            generated_tokens: Vec::new(),
            generation_stream,
        }
    }

    /// Reset generator state
    ///
    /// Must call `reset_with_model` instead when the model uses internal caches
    /// (e.g. Gemma3, Jamba, Mamba, NemotronH, etc.) to ensure those are also reset.
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            *cache = KVCache::new();
        }
        self.generated_tokens.clear();
    }

    /// Reset generator state including model-internal caches.
    ///
    /// Models with internal RefCell caches (sliding window, SSM, hybrid) reset
    /// their own state inside `make_caches()`. This method ensures both the
    /// generator's cache vector and the model's internal caches are cleared.
    pub fn reset_with_model<M: LanguageModel + ?Sized>(&mut self, model: &M) {
        self.caches = model.make_caches();
        self.generated_tokens.clear();
    }

    /// Get mutable access to caches (used by speculative decoding)
    pub fn caches_mut(&mut self) -> &mut [KVCache] {
        &mut self.caches
    }

    /// Generate tokens from the model (original implementation)
    pub fn generate<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> Vec<i32> {
        self.generate_streaming(model, prompt_tokens, max_tokens, sampling, |_| true)
    }

    /// Streaming generation with per-token callback and lookahead pipelining.
    ///
    /// The callback receives each generated token ID and returns `true` to continue
    /// or `false` to abort early. Pipelining is preserved: next step computation
    /// starts before the current token is returned.
    ///
    /// Used by: CxxGenerator::generate, ModelProvider (server streaming)
    pub fn generate_streaming<M: LanguageModel, F: FnMut(i32) -> bool>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        mut on_token: F,
    ) -> Vec<i32> {
        // Reset state
        self.reset();

        // Set random seed if specified (for reproducibility)
        if let Some(seed) = sampling.seed {
            ffi::random_seed(seed);
        }

        // Ensure caches are initialized for this model
        if self.caches.len() != model.num_layers() {
            self.caches = model.make_caches();
        }

        // Set generation stream as default for better pipelining
        if let Some(ref stream) = self.generation_stream {
            ffi::set_default_stream(stream);
        }

        // Get EOS tokens for this model
        let mut eos_tokens = model.eos_token_ids();
        for &id in &sampling.stop_token_ids {
            if !eos_tokens.contains(&id) {
                eos_tokens.push(id);
            }
        }

        // Prefill: process all prompt tokens at once
        let input = ffi::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
        let logits = model.forward(&input, &mut self.caches, None);

        // Clear intermediate tensors from prefill to free memory
        ffi::clear_memory_cache();

        // Build token history from prompt for penalty-based sampling
        let needs_history = sampling.needs_token_history();
        let mut token_history: Vec<i32> = if needs_history {
            prompt_tokens.to_vec()
        } else {
            Vec::new()
        };

        // Sample first token
        let (mut y, mut _logprobs) = sample_token_optimized(&logits, sampling, &token_history);
        ffi::async_eval(&y);

        // Main generation loop - matches Python exactly:
        // 1. Start next step computation
        // 2. async_eval next step
        // 3. Extract current value (syncs current only)
        // 4. Yield/store current
        // 5. Move next to current
        let mut n = 0;
        loop {
            // Start next step (if not at max)
            let (next_y, next_logprobs) = if n + 1 < max_tokens {
                let next_input = ffi::reshape_token_for_forward(&y);
                let next_logits = model.forward(&next_input, &mut self.caches, None);
                let (next_tok, next_log) =
                    sample_token_optimized(&next_logits, sampling, &token_history);
                ffi::async_eval_pair(&next_tok, &next_log);
                (Some(next_tok), Some(next_log))
            } else {
                (None, None)
            };

            // First iteration: explicit eval
            if n == 0 {
                ffi::eval(&y);
            }

            // Check if we've reached max
            if n >= max_tokens {
                break;
            }

            // Extract current token value - this syncs y
            // (item_i32 implicitly evals if needed)
            let token_val = ffi::item_i32(&y);

            // Check EOS before sending to callback (avoid outputting stop tokens)
            if eos_tokens.contains(&token_val) {
                break;
            }

            self.generated_tokens.push(token_val);
            if needs_history {
                token_history.push(token_val);
            }

            // Invoke callback; abort if it returns false
            if !on_token(token_val) {
                break;
            }

            // Periodic cache clearing
            if n % 512 == 0 && n > 0 {
                ffi::clear_memory_cache();
            }

            // Move to next
            if let (Some(ny), Some(nl)) = (next_y, next_logprobs) {
                y = ny;
                _logprobs = nl;
            } else {
                break;
            }

            n += 1;
        }

        self.generated_tokens.clone()
    }

    /// Streaming generation with pre-computed embeddings for VLM prefill.
    ///
    /// The prefill step uses `model.forward_with_embeddings()` with provided
    /// embeddings and mask. Decode steps are identical to standard generation.
    ///
    /// Used by: VisionLanguageModel (Gemma3 VLM, etc.)
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_embeddings<M: LanguageModel, F: FnMut(i32) -> bool>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        input_embeddings: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        max_tokens: usize,
        sampling: &SamplingConfig,
        mut on_token: F,
    ) -> Vec<i32> {
        // Reset state
        self.reset();

        if let Some(seed) = sampling.seed {
            ffi::random_seed(seed);
        }

        if self.caches.len() != model.num_layers() {
            self.caches = model.make_caches();
        }

        if let Some(ref stream) = self.generation_stream {
            ffi::set_default_stream(stream);
        }

        let mut eos_tokens = model.eos_token_ids();
        for &id in &sampling.stop_token_ids {
            if !eos_tokens.contains(&id) {
                eos_tokens.push(id);
            }
        }

        // Prefill: use forward_with_embeddings for merged vision+text embeddings
        let input = ffi::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
        let logits =
            model.forward_with_embeddings(&input, input_embeddings, &mut self.caches, mask);

        ffi::clear_memory_cache();

        let needs_history = sampling.needs_token_history();
        let mut token_history: Vec<i32> = if needs_history {
            prompt_tokens.to_vec()
        } else {
            Vec::new()
        };

        let (mut y, mut _logprobs) = sample_token_optimized(&logits, sampling, &token_history);
        ffi::async_eval(&y);

        // Decode loop — identical to standard generation (no embeddings needed)
        let mut n = 0;
        loop {
            let (next_y, next_logprobs) = if n + 1 < max_tokens {
                let next_input = ffi::reshape_token_for_forward(&y);
                let next_logits = model.forward(&next_input, &mut self.caches, None);
                let (next_tok, next_log) =
                    sample_token_optimized(&next_logits, sampling, &token_history);
                ffi::async_eval_pair(&next_tok, &next_log);
                (Some(next_tok), Some(next_log))
            } else {
                (None, None)
            };

            if n == 0 {
                ffi::eval(&y);
            }

            if n >= max_tokens {
                break;
            }

            let token_val = ffi::item_i32(&y);

            // Check EOS before sending to callback (avoid outputting stop tokens)
            if eos_tokens.contains(&token_val) {
                break;
            }

            self.generated_tokens.push(token_val);
            if needs_history {
                token_history.push(token_val);
            }

            if !on_token(token_val) {
                break;
            }

            if n % 512 == 0 && n > 0 {
                ffi::clear_memory_cache();
            }

            if let (Some(ny), Some(nl)) = (next_y, next_logprobs) {
                y = ny;
                _logprobs = nl;
            } else {
                break;
            }

            n += 1;
        }

        self.generated_tokens.clone()
    }

    /// Generate with stats using pre-computed embeddings (VLM variant)
    /// Used by: CLI --image path, server VLM requests
    pub fn generate_with_stats_and_embeddings<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        input_embeddings: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        use std::time::Instant;

        self.reset();

        if let Some(seed) = sampling.seed {
            ffi::random_seed(seed);
        }
        if self.caches.len() != model.num_layers() {
            self.caches = model.make_caches();
        }
        if let Some(ref stream) = self.generation_stream {
            ffi::set_default_stream(stream);
        }

        let mut eos_tokens = model.eos_token_ids();
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

        // Prefill with embeddings
        let prefill_start = Instant::now();
        let input = ffi::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
        let logits =
            model.forward_with_embeddings(&input, input_embeddings, &mut self.caches, mask);
        let (mut y, mut _logprobs) = sample_token_optimized(&logits, sampling, &token_history);
        ffi::eval(&y);
        let prefill_time = prefill_start.elapsed();
        ffi::clear_memory_cache();

        // Decode
        let decode_start = Instant::now();
        let mut n = 0;
        loop {
            let next_y = if n + 1 < max_tokens {
                let next_input = ffi::reshape_token_for_forward(&y);
                let next_logits = model.forward(&next_input, &mut self.caches, None);
                let (next_tok, _next_log) =
                    sample_token_optimized(&next_logits, sampling, &token_history);
                ffi::async_eval(&next_tok);
                Some(next_tok)
            } else {
                None
            };

            if n == 0 {
                ffi::eval(&y);
            }
            if n >= max_tokens {
                break;
            }

            let token_val = ffi::item_i32(&y);
            if eos_tokens.contains(&token_val) {
                break;
            }
            self.generated_tokens.push(token_val);
            if needs_history {
                token_history.push(token_val);
            }
            if n % 512 == 0 && n > 0 {
                ffi::clear_memory_cache();
            }
            if let Some(ny) = next_y {
                y = ny;
            } else {
                break;
            }
            n += 1;
        }
        let decode_time = decode_start.elapsed();

        let prompt_count = prompt_tokens.len();
        let gen_count = self.generated_tokens.len();
        let prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        let decode_ms = decode_time.as_secs_f64() * 1000.0;

        let stats = GenerationStats {
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
        };

        (self.generated_tokens.clone(), stats)
    }

    /// Get the generated tokens
    pub fn tokens(&self) -> &[i32] {
        &self.generated_tokens
    }

    /// Generate tokens with detailed timing statistics
    /// Returns (generated_tokens, stats)
    pub fn generate_with_stats<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        use std::time::Instant;

        // Reset state
        self.reset();

        // Set random seed if specified (for reproducibility)
        if let Some(seed) = sampling.seed {
            ffi::random_seed(seed);
        }

        // Ensure caches are initialized for this model
        if self.caches.len() != model.num_layers() {
            self.caches = model.make_caches();
        }

        // Set generation stream as default for better pipelining
        if let Some(ref stream) = self.generation_stream {
            ffi::set_default_stream(stream);
        }

        // Get EOS tokens for this model
        let mut eos_tokens = model.eos_token_ids();
        for &id in &sampling.stop_token_ids {
            if !eos_tokens.contains(&id) {
                eos_tokens.push(id);
            }
        }

        // Build token history from prompt for penalty-based sampling
        let needs_history = sampling.needs_token_history();
        let mut token_history: Vec<i32> = if needs_history {
            prompt_tokens.to_vec()
        } else {
            Vec::new()
        };

        // PREFILL PHASE.
        let prefill_start = Instant::now();
        let input = ffi::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
        let logits = model.forward(&input, &mut self.caches, None);

        // Sample first token and force sync to measure prefill accurately
        let (mut y, mut _logprobs) = sample_token_optimized(&logits, sampling, &token_history);
        ffi::eval(&y);
        let prefill_time = prefill_start.elapsed();

        // Clear intermediate tensors from prefill to free memory
        ffi::clear_memory_cache();

        // DECODE PHASE (with lookahead pipelining).
        let decode_start = Instant::now();

        let mut n = 0;
        loop {
            // Start next step computation (if not at max)
            let next_y = if n + 1 < max_tokens {
                let next_input = ffi::reshape_token_for_forward(&y);
                let next_logits = model.forward(&next_input, &mut self.caches, None);
                let (next_tok, _next_log) =
                    sample_token_optimized(&next_logits, sampling, &token_history);
                ffi::async_eval(&next_tok);
                Some(next_tok)
            } else {
                None
            };

            // First iteration: explicit eval
            if n == 0 {
                ffi::eval(&y);
            }

            // Check if we've reached max
            if n >= max_tokens {
                break;
            }

            // Extract current token value (syncs y)
            let token_val = ffi::item_i32(&y);

            // Check EOS before storing (avoid including stop tokens in output)
            if eos_tokens.contains(&token_val) {
                break;
            }

            self.generated_tokens.push(token_val);
            if needs_history {
                token_history.push(token_val);
            }

            // Periodic cache clearing
            if n % 512 == 0 && n > 0 {
                ffi::clear_memory_cache();
            }

            // Move to next
            if let Some(ny) = next_y {
                y = ny;
            } else {
                break;
            }

            n += 1;
        }

        let decode_time = decode_start.elapsed();

        // Calculate stats
        let prompt_count = prompt_tokens.len();
        let gen_count = self.generated_tokens.len();
        let prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        let decode_ms = decode_time.as_secs_f64() * 1000.0;

        let stats = GenerationStats {
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
        };

        (self.generated_tokens.clone(), stats)
    }
}

/// Sample a token from logits (original version)
#[allow(dead_code)]
fn sample_token(logits: &MlxArray, config: &SamplingConfig) -> i32 {
    let (token_arr, _) = sample_token_optimized(logits, config, &[]);
    ffi::eval(&token_arr);
    ffi::item_i32(&token_arr)
}

/// Benchmark helper: measure throughput
pub struct BenchmarkResult {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_time_ms: f64,
    pub generation_time_ms: f64,
    pub tokens_per_second: f64,
}

impl BenchmarkResult {
    pub fn print(&self) {
        println!("Benchmark Results:");
        println!("  Prompt tokens: {}", self.prompt_tokens);
        println!("  Generated tokens: {}", self.generated_tokens);
        println!("  Prefill time: {:.2} ms", self.prefill_time_ms);
        println!("  Generation time: {:.2} ms", self.generation_time_ms);
        println!("  Throughput: {:.2} tok/s", self.tokens_per_second);
    }
}

/// Run a generation benchmark
pub fn run_benchmark<M: LanguageModel>(
    model: &M,
    prompt_tokens: &[i32],
    max_tokens: usize,
) -> BenchmarkResult {
    use std::time::Instant;

    let num_layers = model.num_layers();
    let mut generator = CxxGenerator::new(num_layers);

    // Warmup
    generator.generate(model, prompt_tokens, 5, &SamplingConfig::greedy());
    generator.reset();

    // Benchmark prefill
    let input = ffi::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);

    let start = Instant::now();
    let logits = model.forward(&input, &mut generator.caches, None);
    ffi::eval(&logits);
    let prefill_time = start.elapsed();

    // Benchmark generation
    generator.reset();
    let start = Instant::now();
    let tokens = generator.generate(model, prompt_tokens, max_tokens, &SamplingConfig::greedy());
    let total_time = start.elapsed();

    let generation_time = total_time.saturating_sub(prefill_time);
    let gen_tokens = tokens.len();

    BenchmarkResult {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: gen_tokens,
        prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
        generation_time_ms: generation_time.as_secs_f64() * 1000.0,
        tokens_per_second: if generation_time.as_secs_f64() > 0.0 {
            gen_tokens as f64 / generation_time.as_secs_f64()
        } else {
            0.0
        },
    }
}

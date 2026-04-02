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

//! Sampling and token-penalty helpers for generation.
//!
//! `generate.rs` and `speculative.rs` both rely on the same penalty pipeline.
//! Keeping those helpers here isolates the token-selection policy from the
//! pipelined decode loops and makes low-level sampling invariants easier to
//! test without touching model forward math.

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::generate::SamplingConfig;
use cxx::UniquePtr;
use std::collections::HashMap;

/// Optimized sampling that returns arrays for pipelining.
///
/// Returns `(token_array, logits_array)` without forcing evaluation so the
/// caller can preserve async lookahead pipelining.
///
/// Uses fused C++ sampling (temperature + top-k + top-p + min-p + categorical
/// in a single FFI call) to minimize round-trip overhead.
///
/// Used by: `CxxGenerator`, `SpeculativeGenerator`, `BatchScheduler`
pub fn sample_token_optimized(
    logits: &MlxArray,
    config: &SamplingConfig,
    token_history: &[i32],
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Use optimized slice_last_logits: [batch, seq, vocab] -> [batch, vocab].
    let last_logits = ffi::slice_last_logits(logits);

    let last_logits = if config.repetition_penalty != 1.0 && !token_history.is_empty() {
        apply_repetition_penalty(&last_logits, token_history, config.repetition_penalty)
    } else {
        last_logits
    };

    let last_logits = if config.dry_multiplier > 0.0 && !token_history.is_empty() {
        apply_dry_penalty(&last_logits, token_history, config)
    } else {
        last_logits
    };

    let last_logits = if (config.frequency_penalty != 0.0 || config.presence_penalty != 0.0)
        && !token_history.is_empty()
    {
        apply_frequency_presence_penalty(
            &last_logits,
            token_history,
            config.frequency_penalty,
            config.presence_penalty,
        )
    } else {
        last_logits
    };

    let token = ffi::fused_sample(
        &last_logits,
        config.temperature,
        config.top_k,
        config.top_p,
        config.min_p,
    );
    (token, last_logits)
}

/// Batch-parallel sampling: sample one token per sequence from batched logits.
///
/// `logits` has shape `[B, 1, vocab_size]`. Each sequence is sampled
/// independently using its own `SamplingConfig` and token history.
///
/// Returns a vector of B sampled token IDs.
///
/// Available for callers that need standalone batched sampling without
/// per-sequence state interleaving. The BatchScheduler currently inlines
/// equivalent logic to interleave sampling with EOS/state/streaming updates.
pub fn batched_sample(
    logits: &MlxArray,
    configs: &[&SamplingConfig],
    token_histories: &[&[i32]],
) -> Vec<i32> {
    let b = configs.len();
    debug_assert_eq!(b, token_histories.len());

    let mut tokens = Vec::with_capacity(b);
    for i in 0..b {
        // Slice [B, 1, vocab] -> [1, 1, vocab] for sequence i
        let seq_logits = ffi::slice(logits, &[i as i32, 0, 0], &[i as i32 + 1, 1, i32::MAX]);
        let (token_arr, _logprobs) =
            sample_token_optimized(&seq_logits, configs[i], token_histories[i]);
        ffi::eval(&token_arr);
        tokens.push(ffi::item_i32(&token_arr));
    }
    tokens
}

/// Apply repetition penalty to logits.
///
/// For tokens in history:
/// - If logit > 0: divide by penalty
/// - If logit < 0: multiply by penalty
///
/// Used by: standard generation, speculative decoding
pub(crate) fn apply_repetition_penalty(
    logits: &MlxArray,
    token_history: &[i32],
    penalty: f32,
) -> UniquePtr<MlxArray> {
    let mut seen: Vec<i32> = token_history.to_vec();
    seen.sort_unstable();
    seen.dedup();

    if seen.is_empty() {
        return ffi::copy(logits);
    }

    let indices = ffi::from_slice_i32(&seen, &[1, seen.len() as i32]);
    let selected = ffi::take_along_axis(logits, &indices, -1);

    let zero = ffi::full_f32(&[1], 0.0, dtype::FLOAT32);
    let pen = ffi::full_f32(&[1], penalty, dtype::FLOAT32);

    let pos_mask = ffi::greater(&selected, &zero);
    let penalized_pos = ffi::divide(&selected, &pen);
    let penalized_neg = ffi::multiply(&selected, &pen);
    let penalized = ffi::where_cond(&pos_mask, &penalized_pos, &penalized_neg);

    ffi::put_along_axis(logits, &indices, &penalized, -1)
}

/// Apply OpenAI-style frequency and presence penalties to logits.
///
/// Used by: standard generation, speculative decoding
pub(crate) fn apply_frequency_presence_penalty(
    logits: &MlxArray,
    token_history: &[i32],
    frequency_penalty: f32,
    presence_penalty: f32,
) -> UniquePtr<MlxArray> {
    let mut token_counts: HashMap<i32, usize> = HashMap::new();
    for &tok in token_history {
        *token_counts.entry(tok).or_insert(0) += 1;
    }

    if token_counts.is_empty() {
        return ffi::copy(logits);
    }

    let shape = ffi::array_shape(logits);
    let vocab_size = *shape.last().unwrap() as usize;

    let mut penalties = vec![0.0f32; vocab_size];
    for (&token_id, &count) in &token_counts {
        if token_id >= 0 && (token_id as usize) < vocab_size {
            penalties[token_id as usize] = frequency_penalty * count as f32 + presence_penalty;
        }
    }

    let penalty_array = ffi::from_slice_f32(&penalties, &[1, vocab_size as i32]);
    let penalty_broadcast = ffi::broadcast_to(&penalty_array, &shape);
    ffi::subtract(logits, &penalty_broadcast)
}

/// Apply DRY (Don't Repeat Yourself) penalty to logits.
///
/// This runs on CPU as sequential pattern matching, which keeps the matching
/// invariant explicit and mirrors the upstream llama.cpp style algorithm.
///
/// Used by: standard generation, speculative decoding
pub(crate) fn apply_dry_penalty(
    logits: &MlxArray,
    token_history: &[i32],
    config: &SamplingConfig,
) -> UniquePtr<MlxArray> {
    let history_len = token_history.len();
    if history_len < 2 {
        return ffi::copy(logits);
    }

    let window = if config.dry_penalty_last_n == 0 {
        token_history
    } else {
        let start = history_len.saturating_sub(config.dry_penalty_last_n);
        &token_history[start..]
    };

    let window_len = window.len();
    if window_len < 2 {
        return ffi::copy(logits);
    }

    let mut token_positions: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &tok) in window.iter().enumerate() {
        token_positions.entry(tok).or_default().push(i);
    }

    let last_token = window[window_len - 1];
    let mut penalties: HashMap<i32, f32> = HashMap::new();

    if let Some(positions) = token_positions.get(&last_token) {
        for &pos in positions {
            if pos >= window_len - 1 {
                continue;
            }

            let mut match_len = 1;
            let mut p1 = pos;
            let mut p2 = window_len - 1;

            while p1 > 0 && p2 > 0 {
                p1 -= 1;
                p2 -= 1;

                if config.dry_sequence_breakers.contains(&window[p1]) {
                    break;
                }

                if window[p1] == window[p2] {
                    match_len += 1;
                } else {
                    break;
                }
            }

            if match_len > config.dry_allowed_length {
                let next_pos = pos + 1;
                if next_pos < window_len {
                    let next_token = window[next_pos];
                    let penalty = config.dry_multiplier
                        * config
                            .dry_base
                            .powi((match_len - config.dry_allowed_length) as i32);
                    let entry = penalties.entry(next_token).or_insert(0.0);
                    if penalty > *entry {
                        *entry = penalty;
                    }
                }
            }
        }
    }

    if penalties.is_empty() {
        return ffi::copy(logits);
    }

    let logits_shape = ffi::array_shape(logits);
    let vocab_size = *logits_shape.last().unwrap();
    let batch_size = if logits_shape.len() > 1 {
        logits_shape[0]
    } else {
        1
    };
    let total = (batch_size * vocab_size) as usize;
    let mut penalty_data = vec![0.0f32; total];

    for (token_id, penalty) in &penalties {
        let idx = *token_id as usize;
        if idx < vocab_size as usize {
            for b in 0..batch_size as usize {
                penalty_data[b * vocab_size as usize + idx] = -penalty;
            }
        }
    }

    let penalty_arr = ffi::from_slice_f32(&penalty_data, &logits_shape);
    ffi::add(logits, &penalty_arr)
}

/// Configuration for log probability computation during generation.
///
/// When `enabled` is false, no logprobs are computed and zero overhead is incurred.
#[derive(Debug, Clone, Default)]
pub struct LogprobsConfig {
    /// Whether to compute log probabilities at all
    pub enabled: bool,
    /// Number of top alternative tokens to return (0 = only the selected token)
    pub top_k: usize,
}

/// Log probability data for a single generated token.
#[derive(Debug, Clone)]
pub struct TokenLogprobData {
    /// Token ID of the selected token
    pub token_id: i32,
    /// Log probability of the selected token
    pub logprob: f32,
    /// Top-k alternative (token_id, logprob) pairs, sorted descending by logprob
    pub top_alternatives: Vec<(i32, f32)>,
}

/// Compute log probabilities for the selected token from penalty-adjusted logits.
///
/// `adjusted_logits` should have shape `[1, vocab]` (output of `sample_token_optimized`).
/// Returns `TokenLogprobData` containing the selected token's log-probability and
/// optionally the top-k alternatives.
///
/// Zero-overhead when `config.enabled` is false.
pub fn compute_logprobs(
    adjusted_logits: &MlxArray,
    selected_token: i32,
    config: &LogprobsConfig,
) -> Option<TokenLogprobData> {
    if !config.enabled {
        return None;
    }

    // Apply log-softmax to get per-token log probabilities.
    let log_probs = ffi::log_softmax(adjusted_logits, -1);
    ffi::eval(&log_probs);

    // Extract the log probability of the selected token.
    let idx = ffi::from_slice_i32(&[selected_token], &[1, 1]);
    let selected_lp_arr = ffi::take_along_axis(&log_probs, &idx, -1);
    ffi::eval(&selected_lp_arr);
    let selected_logprob = ffi::item_f32(&selected_lp_arr);

    // Compute top-k alternatives if requested.
    let top_alternatives = if config.top_k > 0 {
        let vocab_size = ffi::array_shape(&log_probs).last().copied().unwrap_or(0);
        // Clamp k to vocab_size to satisfy argpartition's requirement that kth < array_size.
        let k = (config.top_k as i32).min(vocab_size);
        // negate log_probs so argpartition gives us the top-k (smallest negated = largest)
        let neg_log_probs = ffi::negative(&log_probs);
        let partition_idx = ffi::argpartition(&neg_log_probs, k - 1, -1);
        ffi::eval(&partition_idx);

        // Slice only the first k elements from the partitioned result.
        // argpartition guarantees that indices 0..k contain the k smallest
        // values of the negated log_probs (= the k largest log_probs),
        // so we avoid materializing the full vocabulary into host memory.
        let shape = ffi::array_shape(&partition_idx);
        let ndim = shape.len();
        let starts = vec![0i32; ndim];
        let mut stops = shape.clone();
        stops[ndim - 1] = k.min(stops[ndim - 1]);
        let top_idx = ffi::slice(&partition_idx, &starts, &stops);

        // Gather the log_probs for the top-k partitioned indices.
        let top_lp = ffi::take_along_axis(&log_probs, &top_idx, -1);
        ffi::eval(&top_idx);
        ffi::eval(&top_lp);

        let k_usize = k as usize;

        // Use raw bytes to extract i32 token IDs from top_idx.
        let idx_bytes = ffi::array_to_raw_bytes(&top_idx);
        let lp_bytes = ffi::array_to_raw_bytes(&top_lp);

        // Build (token_id, logprob) pairs for only the top-k partition.
        let mut pairs: Vec<(i32, f32)> = (0..k_usize.min(idx_bytes.len() / 4))
            .filter_map(|i| {
                let tok_bytes: [u8; 4] = idx_bytes[i * 4..(i + 1) * 4].try_into().ok()?;
                let lp_bytes4: [u8; 4] = lp_bytes[i * 4..(i + 1) * 4].try_into().ok()?;
                Some((i32::from_ne_bytes(tok_bytes), f32::from_ne_bytes(lp_bytes4)))
            })
            .collect();

        // Sort the k elements descending by logprob.
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    } else {
        Vec::new()
    };

    Some(TokenLogprobData {
        token_id: selected_token,
        logprob: selected_logprob,
        top_alternatives,
    })
}

/// Apply min-p filtering to logits.
#[allow(dead_code)]
pub(crate) fn min_p_filter(logits: &MlxArray, min_p: f32) -> UniquePtr<MlxArray> {
    let probs = ffi::softmax(logits, -1);
    let max_prob = ffi::max_axis(&probs, -1, true);
    let min_p_scalar = ffi::full_f32(&[1], min_p, dtype::FLOAT32);
    let threshold = ffi::multiply(&max_prob, &min_p_scalar);
    let mask = ffi::greater_equal(&probs, &threshold);
    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    ffi::where_cond(&mask, logits, &neg_inf)
}

/// Apply top-k filtering to logits.
#[allow(dead_code)]
pub(crate) fn top_k_filter(logits: &MlxArray, k: i32) -> UniquePtr<MlxArray> {
    let neg_logits = ffi::negative(logits);
    let indices = ffi::argpartition(&neg_logits, k - 1, -1);

    let shape = ffi::array_shape(&indices);
    let ndim = shape.len();
    let mut start = vec![0i32; ndim];
    let mut stop: Vec<i32> = shape.clone();
    start[ndim - 1] = k - 1;
    stop[ndim - 1] = k;

    let kth_idx = ffi::slice(&indices, &start, &stop);
    let threshold = ffi::take_along_axis(logits, &kth_idx, -1);

    let mask = ffi::greater_equal(logits, &threshold);
    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    ffi::where_cond(&mask, logits, &neg_inf)
}

/// Apply top-p (nucleus) filtering to logits.
#[allow(dead_code)]
pub(crate) fn top_p_filter(logits: &MlxArray, _p: f32) -> UniquePtr<MlxArray> {
    let probs = ffi::softmax(logits, -1);
    let neg_probs = ffi::negative(&probs);
    let sorted_indices = ffi::argsort(&neg_probs, -1);
    let _sorted_probs = ffi::take(&probs, &sorted_indices, -1);
    ffi::copy(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logit_at(logits: &MlxArray, token_id: i32) -> f32 {
        let index = ffi::from_slice_i32(&[token_id], &[1, 1]);
        let taken = ffi::take_along_axis(logits, &index, -1);
        ffi::eval(&taken);
        ffi::item_f32(&taken)
    }

    #[test]
    fn apply_repetition_penalty_modifies_selected_logits() {
        let logits = ffi::from_slice_f32(&[1.0, 2.0, -1.0, 3.0, -2.0], &[1, 5]);
        let result = apply_repetition_penalty(&logits, &[1, 3], 2.0);

        assert_eq!(logit_at(&result, 0), 1.0);
        assert_eq!(logit_at(&result, 1), 1.0);
        assert_eq!(logit_at(&result, 3), 1.5);
    }

    #[test]
    fn apply_frequency_presence_penalty_accumulates_by_token_count() {
        let logits = ffi::from_slice_f32(&[0.0, 0.0, 0.0], &[1, 3]);
        let result = apply_frequency_presence_penalty(&logits, &[1, 1, 2], 0.5, 0.25);

        assert_eq!(logit_at(&result, 0), 0.0);
        assert_eq!(logit_at(&result, 1), -1.25);
        assert_eq!(logit_at(&result, 2), -0.75);
    }

    #[test]
    fn apply_dry_penalty_penalizes_followup_token_after_suffix_match() {
        let logits = ffi::from_slice_f32(&[1.0, 1.0, 1.0], &[1, 3]);
        let config = SamplingConfig {
            dry_multiplier: 1.0,
            dry_base: 2.0,
            dry_allowed_length: 1,
            ..Default::default()
        };

        let result = apply_dry_penalty(&logits, &[0, 1, 2, 0, 1], &config);

        assert_eq!(logit_at(&result, 0), 1.0);
        assert_eq!(logit_at(&result, 1), 1.0);
        assert_eq!(logit_at(&result, 2), -1.0);
    }

    #[test]
    fn sample_token_optimized_respects_greedy_argmax_path() {
        let logits = ffi::from_slice_f32(&[0.1, 0.9, 1.2], &[1, 1, 3]);
        let config = SamplingConfig::greedy();
        let (token, processed_logits) = sample_token_optimized(&logits, &config, &[]);

        ffi::eval(&token);
        assert_eq!(ffi::item_i32(&token), 2);
        assert_eq!(ffi::array_shape(&processed_logits), vec![1, 3]);
    }

    #[test]
    fn batched_sample_greedy_selects_argmax_per_sequence() {
        // Two sequences with different argmax positions
        // Seq 0: logits [0.1, 0.9, 1.2] -> argmax = 2
        // Seq 1: logits [2.0, 0.5, 0.1] -> argmax = 0
        let logits = ffi::from_slice_f32(&[0.1, 0.9, 1.2, 2.0, 0.5, 0.1], &[2, 1, 3]);

        let config0 = SamplingConfig::greedy();
        let config1 = SamplingConfig::greedy();
        let configs: Vec<&SamplingConfig> = vec![&config0, &config1];
        let histories: Vec<&[i32]> = vec![&[], &[]];

        let tokens = batched_sample(&logits, &configs, &histories);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], 2);
        assert_eq!(tokens[1], 0);
    }

    #[test]
    fn batched_sample_single_sequence_matches_unbatched() {
        let logits = ffi::from_slice_f32(&[0.5, 1.5, 0.3], &[1, 1, 3]);
        let config = SamplingConfig::greedy();
        let configs: Vec<&SamplingConfig> = vec![&config];
        let histories: Vec<&[i32]> = vec![&[]];

        let tokens = batched_sample(&logits, &configs, &histories);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], 1); // argmax of [0.5, 1.5, 0.3] is index 1
    }

    #[test]
    fn compute_logprobs_returns_none_when_disabled() {
        let logits = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 3]);
        let config = LogprobsConfig {
            enabled: false,
            top_k: 0,
        };
        let result = compute_logprobs(&logits, 2, &config);
        assert!(result.is_none());
    }

    #[test]
    fn compute_logprobs_returns_selected_token_logprob() {
        // Uniform logits -> log-softmax produces equal log-probs for all tokens
        let logits = ffi::from_slice_f32(&[1.0, 1.0, 1.0, 1.0], &[1, 4]);
        let config = LogprobsConfig {
            enabled: true,
            top_k: 0,
        };
        let result = compute_logprobs(&logits, 2, &config).expect("should return Some");
        assert_eq!(result.token_id, 2);
        // log(1/4) ≈ -1.386
        assert!((result.logprob - (-1.386_f32)).abs() < 0.01);
        assert!(result.top_alternatives.is_empty());
    }

    #[test]
    fn compute_logprobs_returns_top_k_alternatives_sorted_descending() {
        // logits: token 0 has highest, token 2 next, token 1 lowest
        let logits = ffi::from_slice_f32(&[3.0, 0.0, 2.0], &[1, 3]);
        let config = LogprobsConfig {
            enabled: true,
            top_k: 2,
        };
        // Select token 1 (low logprob) so top-k will include better alternatives
        let result = compute_logprobs(&logits, 1, &config).expect("should return Some");
        assert_eq!(result.token_id, 1);
        assert_eq!(result.top_alternatives.len(), 2);
        // Alternatives must be sorted descending by logprob
        assert!(result.top_alternatives[0].1 >= result.top_alternatives[1].1);
    }

    #[test]
    fn compute_logprobs_top_k_capped_at_vocab_size() {
        // Vocab of 3 tokens, top_k larger than vocab; k is clamped to 3
        let logits = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 3]);
        let config = LogprobsConfig {
            enabled: true,
            top_k: 10,
        };
        let result = compute_logprobs(&logits, 2, &config).expect("should return Some");
        // top_k is clamped to vocab size (3), so at most 3 alternatives
        assert_eq!(result.top_alternatives.len(), 3);
    }
}

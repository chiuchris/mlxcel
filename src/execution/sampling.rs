//! Shared sampling-config assembly for CLI and server request surfaces.
//!
//! `SamplingConfig` lives in `mlxcel-core`, but the policy for how user-facing
//! request fields map onto greedy vs sampled generation belongs in this crate's
//! control plane so every entry point applies the same defaults.

use crate::SamplingConfig;

#[derive(Debug, Clone, PartialEq)]
/// Fully resolved sampling knobs before conversion into `SamplingConfig`.
///
/// The CLI and server each resolve their own defaults first, then pass the
/// merged values through this struct so the greedy/non-greedy branching remains
/// centralized in one place.
pub struct ResolvedSamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: Option<u64>,
    pub repetition_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_penalty_last_n: usize,
    pub dry_sequence_breakers: Vec<i32>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop_token_ids: Vec<i32>,
}

pub fn build_sampling_config(params: ResolvedSamplingParams) -> SamplingConfig {
    if params.temperature <= 0.0 {
        SamplingConfig {
            min_p: params.min_p,
            seed: params.seed,
            repetition_penalty: params.repetition_penalty,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            dry_multiplier: params.dry_multiplier,
            dry_base: params.dry_base,
            dry_allowed_length: params.dry_allowed_length,
            dry_penalty_last_n: params.dry_penalty_last_n,
            stop_token_ids: params.stop_token_ids,
            ..SamplingConfig::greedy()
        }
    } else {
        SamplingConfig {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            min_p: params.min_p,
            seed: params.seed,
            repetition_penalty: params.repetition_penalty,
            dry_multiplier: params.dry_multiplier,
            dry_base: params.dry_base,
            dry_allowed_length: params.dry_allowed_length,
            dry_penalty_last_n: params.dry_penalty_last_n,
            dry_sequence_breakers: params.dry_sequence_breakers,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            stop_token_ids: params.stop_token_ids,
        }
    }
}

#[cfg(test)]
#[path = "sampling_tests.rs"]
mod tests;

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

use super::{RequestOptionOverrides, build_server_generate_options};
use crate::server::ServerConfig;

#[test]
fn build_server_generate_options_uses_server_defaults() {
    let config = ServerConfig::default();

    let options = build_server_generate_options(&config, RequestOptionOverrides::default());

    assert_eq!(options.max_tokens, config.default_max_tokens);
    assert_eq!(options.sampling.temperature, config.default_temperature);
    assert_eq!(options.sampling.top_k, config.default_top_k);
    assert_eq!(options.sampling.top_p, config.default_top_p);
    assert_eq!(options.sampling.min_p, config.default_min_p);
    assert_eq!(
        options.sampling.repetition_penalty,
        config.default_repetition_penalty
    );
    assert_eq!(
        options.sampling.dry_multiplier,
        config.default_dry_multiplier
    );
    assert_eq!(options.stop_sequences, None);
}

#[test]
fn build_server_generate_options_applies_request_overrides() {
    let config = ServerConfig::default();
    let options = build_server_generate_options(
        &config,
        RequestOptionOverrides {
            max_tokens: Some(7),
            temperature: Some(0.0),
            top_k: Some(99),
            top_p: Some(0.5),
            min_p: Some(0.2),
            repetition_penalty: Some(1.3),
            seed: Some(42),
            frequency_penalty: Some(0.4),
            presence_penalty: Some(0.5),
            dry_multiplier: Some(0.9),
            dry_base: Some(2.2),
            dry_allowed_length: Some(5),
            dry_penalty_last_n: Some(17),
            dry_sequence_breakers: Some(vec![1, 2]),
            stop_sequences: Some(vec!["stop".to_string()]),
        },
    );

    assert_eq!(options.max_tokens, 7);
    assert_eq!(options.sampling.temperature, 0.0);
    assert_eq!(options.sampling.top_k, 1);
    assert_eq!(options.sampling.top_p, 1.0);
    assert_eq!(options.sampling.min_p, 0.2);
    assert_eq!(options.sampling.seed, Some(42));
    assert_eq!(options.sampling.repetition_penalty, 1.3);
    assert_eq!(options.sampling.frequency_penalty, 0.4);
    assert_eq!(options.sampling.presence_penalty, 0.5);
    assert_eq!(options.sampling.dry_multiplier, 0.9);
    assert_eq!(options.sampling.dry_base, 2.2);
    assert_eq!(options.sampling.dry_allowed_length, 5);
    assert_eq!(options.sampling.dry_penalty_last_n, 17);
    assert_eq!(options.sampling.dry_sequence_breakers, Vec::<i32>::new());
    assert_eq!(options.stop_sequences, Some(vec!["stop".to_string()]));
}

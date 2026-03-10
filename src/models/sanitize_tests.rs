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

use super::sanitize::{sanitize_config_json, sanitize_tied_embeddings};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{self, dtype};
use serde_json::json;

fn sample_weight_map(key: &str) -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert(key.to_string(), mlxcel_core::ones(&[2, 2], dtype::FLOAT32));
    weights
}

#[test]
fn sanitize_config_json_replaces_non_standard_values() {
    let sanitized = sanitize_config_json("{\"a\": Infinity, \"b\": -Infinity, \"c\": NaN}");
    assert_eq!(sanitized, "{\"a\": 1e38, \"b\": -1e38, \"c\": 0.0}");
}

#[test]
fn sanitize_tied_embeddings_copies_standard_embed_tokens_when_missing() {
    let mut weights = sample_weight_map("model.embed_tokens.weight");
    sanitize_tied_embeddings(&mut weights, &json!({}));

    assert!(weights.contains_key("lm_head.weight"));
}

#[test]
fn sanitize_tied_embeddings_copies_prefixed_language_model_keys() {
    let mut weights = sample_weight_map("language_model.model.embed_tokens.weight");
    sanitize_tied_embeddings(&mut weights, &json!({}));

    assert!(weights.contains_key("language_model.lm_head.weight"));
}

#[test]
fn sanitize_tied_embeddings_respects_explicit_untied_config() {
    let mut weights = sample_weight_map("model.embed_tokens.weight");
    sanitize_tied_embeddings(&mut weights, &json!({ "tie_word_embeddings": false }));

    assert!(!weights.contains_key("lm_head.weight"));
}

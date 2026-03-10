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

use super::{Cache, Step3p5Config, Step3p5Model};

fn parse_config(json: serde_json::Value) -> Step3p5Config {
    serde_json::from_value(json).expect("valid Step3p5Config")
}

fn assert_cache_kinds(caches: &[Cache], expected_rotating: &[bool]) {
    assert_eq!(caches.len(), expected_rotating.len());
    for (cache, expect_rotating) in caches.iter().zip(expected_rotating) {
        match (cache, expect_rotating) {
            (Cache::Rotating(_), true) | (Cache::Standard(_), false) => {}
            _ => panic!("unexpected cache kind"),
        }
    }
}

#[test]
fn build_layer_caches_uses_rotating_for_explicit_sliding_layers() {
    let config = parse_config(serde_json::json!({
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "vocab_size": 1024,
        "num_attention_heads": 4,
        "num_attention_groups": 2,
        "head_dim": 64,
        "intermediate_size": 512,
        "sliding_window": 16,
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ]
    }));

    let caches = Step3p5Model::build_layer_caches(&config);
    assert_cache_kinds(&caches, &[false, true, true, false]);
}

#[test]
fn build_layer_caches_uses_fallback_even_odd_pattern_without_layer_types() {
    let config = parse_config(serde_json::json!({
        "hidden_size": 256,
        "num_hidden_layers": 5,
        "vocab_size": 1024,
        "num_attention_heads": 4,
        "num_attention_groups": 2,
        "head_dim": 64,
        "intermediate_size": 512,
        "sliding_window": 32
    }));

    let caches = Step3p5Model::build_layer_caches(&config);
    assert_cache_kinds(&caches, &[true, false, true, false, true]);
}

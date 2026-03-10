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

use super::ModelArgs;

fn test_args() -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "phi3",
        "hidden_size": 3072,
        "num_hidden_layers": 32,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "vocab_size": 1000
    }))
    .unwrap()
}

#[test]
fn phi3_model_args_default_to_full_rotary_dims() {
    let args = test_args();
    assert_eq!(args.head_dim(), 128);
    assert_eq!(args.rope_dims(), 128);
}

#[test]
fn phi3_model_args_support_partial_rotary_factor() {
    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "phi4mm",
        "hidden_size": 3072,
        "num_hidden_layers": 32,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "vocab_size": 1000,
        "partial_rotary_factor": 0.75
    }))
    .unwrap();

    assert_eq!(args.head_dim(), 128);
    assert_eq!(args.rope_dims(), 96);
}

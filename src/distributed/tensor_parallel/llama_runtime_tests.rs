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

use std::collections::HashMap;
use std::fs;

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;

use super::{TensorParallelLlamaModel, logical_weight_name, validate_supported_runtime};
use crate::distributed::tensor_parallel::{ShardConfig, generate_shard_plan};
use crate::models::llama3::{Llama3Model, ModelArgs};

fn make_test_model_args() -> ModelArgs {
    ModelArgs {
        model_type: "llama".to_string(),
        hidden_size: 4,
        num_hidden_layers: 1,
        intermediate_size: 8,
        num_attention_heads: 2,
        rms_norm_eps: 1e-5,
        vocab_size: 8,
        head_dim: Some(2),
        num_key_value_heads: Some(2),
        attention_bias: false,
        mlp_bias: false,
        rope_theta: 10_000.0,
        rope_scaling: None,
        quantization: None,
        tie_word_embeddings: false,
    }
}

fn tensor(values: &[f32], shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_f32(values, shape)
}

fn insert_tensor(weights: &mut WeightMap, name: &str, values: &[f32], shape: &[i32]) {
    weights.insert(name.to_string(), tensor(values, shape));
}

fn make_test_weight_map() -> WeightMap {
    let mut weights = HashMap::new();
    insert_tensor(
        &mut weights,
        "model.embed_tokens.weight",
        &[
            0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30,
            1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70,
            2.80, 2.90, 3.00, 3.10,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.q_proj.weight",
        &[
            0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.k_proj.weight",
        &[
            0.7, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.2, 0.4, 0.3, 0.2, 0.1,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.v_proj.weight",
        &[
            0.05, 0.10, 0.15, 0.20, 0.10, 0.15, 0.20, 0.25, 0.15, 0.20, 0.25, 0.30, 0.20, 0.25,
            0.30, 0.35,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.o_proj.weight",
        &[
            0.20, 0.10, 0.30, 0.40, 0.10, 0.30, 0.20, 0.40, 0.40, 0.30, 0.10, 0.20, 0.30, 0.40,
            0.20, 0.10,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.gate_proj.weight",
        &[
            0.10, 0.20, 0.30, 0.40, 0.20, 0.30, 0.40, 0.50, 0.30, 0.40, 0.50, 0.60, 0.40, 0.50,
            0.60, 0.70, 0.50, 0.60, 0.70, 0.80, 0.60, 0.70, 0.80, 0.90, 0.70, 0.80, 0.90, 1.00,
            0.80, 0.90, 1.00, 1.10,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.up_proj.weight",
        &[
            0.15, 0.25, 0.35, 0.45, 0.25, 0.35, 0.45, 0.55, 0.35, 0.45, 0.55, 0.65, 0.45, 0.55,
            0.65, 0.75, 0.55, 0.65, 0.75, 0.85, 0.65, 0.75, 0.85, 0.95, 0.75, 0.85, 0.95, 1.05,
            0.85, 0.95, 1.05, 1.15,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.down_proj.weight",
        &[
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55,
        ],
        &[4, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.input_layernorm.weight",
        &[1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.post_attention_layernorm.weight",
        &[1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "model.norm.weight",
        &[1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "lm_head.weight",
        &[
            0.10, 0.20, 0.30, 0.40, 0.20, 0.30, 0.40, 0.50, 0.30, 0.40, 0.50, 0.60, 0.40, 0.50,
            0.60, 0.70, 0.50, 0.60, 0.70, 0.80, 0.60, 0.70, 0.80, 0.90, 0.70, 0.80, 0.90, 1.00,
            0.80, 0.90, 1.00, 1.10,
        ],
        &[8, 4],
    );
    weights
}

#[test]
fn logical_weight_name_maps_auxiliary_quantization_keys_to_weight_name() {
    assert_eq!(
        logical_weight_name("model.layers.0.self_attn.q_proj.scales"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        logical_weight_name("model.layers.0.self_attn.q_proj.biases"),
        "model.layers.0.self_attn.q_proj.weight"
    );
}

#[test]
fn validate_supported_runtime_accepts_llama_replicated_path() {
    let dir = std::env::temp_dir().join(format!("mlxcel-tp-llama-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 1
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tensor_parallel_llama_matches_full_model_logits() {
    let args = make_test_model_args();
    let weights = make_test_weight_map();
    let full = Llama3Model::from_weights(&weights, &args).unwrap();
    let plan = generate_shard_plan(
        "llama",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelLlamaModel::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

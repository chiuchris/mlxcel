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

use super::{
    TensorParallelErnie45Model, TensorParallelGemma3Model, TensorParallelHunyuanV1DenseModel,
    TensorParallelLlamaModel, TensorParallelQwen3Model, local_llama_args, logical_weight_name,
    validate_supported_runtime,
};
use crate::distributed::tensor_parallel::{ShardConfig, generate_shard_plan};
use crate::models::ernie4_5::{Ernie45Model, ModelArgs as Ernie45ModelArgs};
use crate::models::gemma3::{Gemma3Wrapper, ModelArgs as Gemma3ModelArgs};
use crate::models::hunyuan_v1_dense::{HunyuanV1DenseModel, ModelArgs as HunyuanV1DenseModelArgs};
use crate::models::llama3::{Llama3Model, ModelArgs as LlamaModelArgs};
use crate::models::qwen3::{ModelArgs as Qwen3ModelArgs, Qwen3Model};

fn make_test_model_args() -> LlamaModelArgs {
    LlamaModelArgs {
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

fn make_test_qwen3_args() -> Qwen3ModelArgs {
    Qwen3ModelArgs {
        model_type: "qwen3".to_string(),
        hidden_size: 4,
        num_hidden_layers: 1,
        intermediate_size: 8,
        num_attention_heads: 2,
        rms_norm_eps: 1e-5,
        vocab_size: 8,
        num_key_value_heads: 2,
        head_dim: 2,
        max_position_embeddings: None,
        rope_theta: 10_000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        quantization: None,
    }
}

fn make_test_ernie45_args() -> Ernie45ModelArgs {
    Ernie45ModelArgs {
        model_type: "ernie4_5".to_string(),
        hidden_size: 4,
        num_hidden_layers: 1,
        intermediate_size: 8,
        num_attention_heads: 2,
        rms_norm_eps: 1e-5,
        vocab_size: 8,
        head_dim: Some(2),
        num_key_value_heads: Some(2),
        use_bias: false,
        rope_theta: 10_000.0,
        max_position_embeddings: 4096,
        tie_word_embeddings: false,
        quantization: None,
    }
}

fn make_test_hunyuan_v1_dense_args() -> HunyuanV1DenseModelArgs {
    HunyuanV1DenseModelArgs {
        model_type: "hunyuan_v1_dense".to_string(),
        vocab_size: 8,
        hidden_size: 4,
        num_hidden_layers: 1,
        intermediate_size: 8,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 4096,
        attention_bias: false,
        use_qk_norm: true,
        rope_scaling: None,
        tie_word_embeddings: false,
        head_dim: Some(2),
        quantization: None,
    }
}

fn make_test_gemma3_args() -> Gemma3ModelArgs {
    Gemma3ModelArgs {
        model_type: "gemma3_text".to_string(),
        hidden_size: 4,
        num_hidden_layers: 1,
        intermediate_size: 8,
        num_attention_heads: 2,
        head_dim: 2,
        rms_norm_eps: 1e-6,
        vocab_size: 8,
        num_key_value_heads: 1,
        rope_theta: 10_000.0,
        rope_local_base_freq: 10_000.0,
        query_pre_attn_scalar: 2.0,
        sliding_window: 8,
        sliding_window_pattern: 1,
        max_position_embeddings: 4096,
        rope_scaling: None,
        quantization: None,
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

fn make_test_qwen3_weight_map() -> WeightMap {
    let mut weights = make_test_weight_map();
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.q_norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.k_norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    weights
}

fn make_test_hunyuan_v1_dense_weight_map() -> WeightMap {
    let mut weights = make_test_weight_map();
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.query_layernorm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.key_layernorm.weight",
        &[1.0, 1.0],
        &[2],
    );
    weights
}

fn make_test_gemma3_weight_map() -> WeightMap {
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
        &[0.7, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.3],
        &[2, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.v_proj.weight",
        &[0.05, 0.10, 0.15, 0.20, 0.10, 0.15, 0.20, 0.25],
        &[2, 4],
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
        "model.layers.0.self_attn.q_norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.k_norm.weight",
        &[1.0, 1.0],
        &[2],
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
    for norm_name in [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.pre_feedforward_layernorm.weight",
        "model.layers.0.post_feedforward_layernorm.weight",
        "model.norm.weight",
    ] {
        insert_tensor(&mut weights, norm_name, &[1.0, 1.0, 1.0, 1.0], &[4]);
    }
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
fn validate_supported_runtime_accepts_qwen3_replicated_path() {
    let dir = std::env::temp_dir().join(format!("mlxcel-tp-qwen3-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen3",
            "num_hidden_layers": 1
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_supported_runtime_accepts_ernie45_replicated_path() {
    let dir = std::env::temp_dir().join(format!("mlxcel-tp-ernie45-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "ernie4_5",
            "num_hidden_layers": 18
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_supported_runtime_accepts_hunyuan_v1_dense_replicated_path() {
    let dir = std::env::temp_dir().join(format!(
        "mlxcel-tp-hunyuan-v1-dense-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "hunyuan_v1_dense",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_supported_runtime_accepts_gemma3_replicated_path() {
    let dir = std::env::temp_dir().join(format!("mlxcel-tp-gemma3-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma3_text",
            "num_hidden_layers": 26
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

#[test]
fn local_llama_args_preserves_computed_head_dim_when_config_omits_it() {
    let args = LlamaModelArgs {
        model_type: "llama".to_string(),
        hidden_size: 896,
        num_hidden_layers: 1,
        intermediate_size: 4864,
        num_attention_heads: 14,
        rms_norm_eps: 1e-5,
        vocab_size: 1024,
        head_dim: None,
        num_key_value_heads: Some(2),
        attention_bias: false,
        mlp_bias: false,
        rope_theta: 1_000_000.0,
        rope_scaling: None,
        quantization: None,
        tie_word_embeddings: true,
    };
    let plan = generate_shard_plan("llama", 1, &ShardConfig::with_tp_size(2)).unwrap();

    let local = local_llama_args(&args, &plan).unwrap();

    assert_eq!(local.head_dim, Some(64));
    assert_eq!(local.num_attention_heads, 7);
}

#[test]
fn tensor_parallel_qwen3_matches_full_model_logits() {
    let args = make_test_qwen3_args();
    let weights = make_test_qwen3_weight_map();
    let full = Qwen3Model::from_weights(&weights, &args).unwrap();
    let plan = generate_shard_plan(
        "qwen3",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelQwen3Model::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
fn tensor_parallel_ernie45_matches_full_model_logits() {
    let args = make_test_ernie45_args();
    let weights = make_test_weight_map();
    let full = Ernie45Model::from_weights(&weights, &args).unwrap();
    let plan = generate_shard_plan(
        "ernie4_5",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelErnie45Model::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
fn tensor_parallel_hunyuan_v1_dense_matches_full_model_logits() {
    let args = make_test_hunyuan_v1_dense_args();
    let weights = make_test_hunyuan_v1_dense_weight_map();
    let full = HunyuanV1DenseModel::from_weights(&weights, &args).unwrap();
    let plan = generate_shard_plan(
        "hunyuan_v1_dense",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelHunyuanV1DenseModel::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
fn tensor_parallel_gemma3_matches_full_model_logits() {
    let args = make_test_gemma3_args();
    let weights = make_test_gemma3_weight_map();
    let full =
        Gemma3Wrapper::new(crate::models::Gemma3Model::from_weights(&weights, &args).unwrap());
    let plan = generate_shard_plan(
        "gemma3_text",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelGemma3Model::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

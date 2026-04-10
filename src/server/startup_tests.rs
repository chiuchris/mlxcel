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

use std::path::PathBuf;

use super::{
    ServerStartupConfig, build_server_config, resolve_api_key, resolve_chat_template,
    resolve_default_max_tokens, resolve_dry_penalty_last_n,
    resolve_tensor_parallel_runtime_support, validate_tensor_parallel_startup,
};
use crate::server::chat_template::ChatMessage;

fn temp_path(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mlxcel-{}-{}", name, uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn resolve_default_max_tokens_matches_server_policy() {
    assert_eq!(resolve_default_max_tokens(-1), 4096);
    assert_eq!(resolve_default_max_tokens(128), 128);
}

#[test]
fn resolve_dry_penalty_last_n_maps_negative_to_full_history_sentinel() {
    assert_eq!(resolve_dry_penalty_last_n(-1), 0);
    assert_eq!(resolve_dry_penalty_last_n(24), 24);
}

#[test]
fn resolve_api_key_prefers_explicit_value_and_reads_trimmed_file() {
    let dir = temp_path("api-key");
    let key_file = dir.join("key.txt");
    std::fs::write(&key_file, "  secret-key \n").unwrap();

    assert_eq!(
        resolve_api_key(Some("flag-key".to_string()), Some(&key_file)).unwrap(),
        Some("flag-key".to_string())
    );
    assert_eq!(
        resolve_api_key(None, Some(&key_file)).unwrap(),
        Some("secret-key".to_string())
    );

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_chat_template_respects_override_then_file_then_model_metadata() {
    let dir = temp_path("chat-template");
    let file_template = dir.join("override.jinja");
    std::fs::write(&file_template, "file={{ messages[0].content }}").unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"chat_template":"model={{ messages[0].content }}"}"#,
    )
    .unwrap();

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "hello".to_string(),
    }];

    let processor =
        resolve_chat_template(Some("inline={{ messages[0].content }}"), None, &dir).unwrap();
    assert_eq!(processor.apply(&messages, None).unwrap(), "inline=hello");

    let processor = resolve_chat_template(None, Some(&file_template), &dir).unwrap();
    assert_eq!(processor.apply(&messages, None).unwrap(), "file=hello");

    let processor = resolve_chat_template(None, None, &dir).unwrap();
    assert_eq!(processor.apply(&messages, None).unwrap(), "model=hello");

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn build_server_config_applies_normalized_startup_values() {
    let startup = ServerStartupConfig {
        model_alias: Some("alias".to_string()),
        timeout: 42,
        ctx_size: 2048,
        n_parallel: 3,
        enable_slots: false,
        enable_props: true,
        enable_metrics: true,
        temperature: 0.7,
        top_p: 0.95,
        top_k: 32,
        min_p: 0.05,
        repeat_penalty: 1.2,
        repeat_last_n: 96,
        n_predict: -1,
        seed: Some(7),
        presence_penalty: 0.4,
        frequency_penalty: 0.3,
        dry_multiplier: 0.6,
        dry_base: 2.0,
        dry_allowed_length: 4,
        dry_penalty_last_n: -1,
        draft_model_path: Some(PathBuf::from("draft")),
        draft_max: 5,
        ..ServerStartupConfig::default()
    };

    let config = build_server_config(&startup, Some("token".to_string()));
    assert_eq!(config.api_key, Some("token".to_string()));
    assert_eq!(config.timeout_seconds, 42);
    assert_eq!(config.model_alias.as_deref(), Some("alias"));
    assert_eq!(config.context_size, 2048);
    assert_eq!(config.n_parallel, 3);
    assert!(!config.enable_slots_endpoint);
    assert!(config.enable_props_endpoint);
    assert!(config.enable_metrics_endpoint);
    assert_eq!(config.default_temperature, 0.7);
    assert_eq!(config.default_top_p, 0.95);
    assert_eq!(config.default_top_k, 32);
    assert_eq!(config.default_min_p, 0.05);
    assert_eq!(config.default_repetition_penalty, 1.2);
    assert_eq!(config.default_repetition_context_size, 96);
    assert_eq!(config.default_max_tokens, 4096);
    assert_eq!(config.default_seed, Some(7));
    assert_eq!(config.default_presence_penalty, 0.4);
    assert_eq!(config.default_frequency_penalty, 0.3);
    assert_eq!(config.default_dry_multiplier, 0.6);
    assert_eq!(config.default_dry_base, 2.0);
    assert_eq!(config.default_dry_allowed_length, 4);
    assert_eq!(config.default_dry_penalty_last_n, 0);
    assert_eq!(config.draft_model_path, Some(PathBuf::from("draft")));
    assert_eq!(config.num_draft_tokens, 5);
    // max_batch_size derived from n_parallel (no explicit override);
    // max_queue_depth comes from the startup config default (32).
    assert_eq!(config.max_batch_size, 3);
    assert_eq!(config.max_queue_depth, 32);
}

#[test]
fn build_server_config_max_batch_size_is_at_least_one() {
    // n_parallel=0 is nonsensical but must not produce a zero batch size
    let startup = ServerStartupConfig {
        n_parallel: 0,
        ..ServerStartupConfig::default()
    };
    let config = build_server_config(&startup, None);
    assert_eq!(config.max_batch_size, 1);
}

#[test]
fn build_server_config_propagates_no_batch_flag() {
    let startup = ServerStartupConfig {
        no_batch: true,
        ..ServerStartupConfig::default()
    };
    let config = build_server_config(&startup, None);
    assert!(config.no_batch);
}

#[test]
fn build_server_config_preserves_batch_scheduler_for_tensor_parallel() {
    let startup = ServerStartupConfig {
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    let config = build_server_config(&startup, None);
    assert!(!config.no_batch);
    assert_eq!(config.tensor_parallel.tp_size, 2);
}

#[test]
fn resolve_tensor_parallel_runtime_support_allows_server_batching_for_llama() {
    let dir = temp_path("tp-llama-batching");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    let support = resolve_tensor_parallel_runtime_support(&startup).unwrap();
    assert!(!support.force_no_batch);

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_tensor_parallel_runtime_support_keeps_gemma3_on_sequential_worker() {
    let dir = temp_path("tp-gemma3-no-batch");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma3_text",
            "num_hidden_layers": 26
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    let support = resolve_tensor_parallel_runtime_support(&startup).unwrap();
    assert!(support.force_no_batch);

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_single_rank() {
    let dir = temp_path("tp-single");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_supported_multi_rank_runtime() {
    let dir = temp_path("tp-multi");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_qwen2_multi_rank_runtime() {
    let dir = temp_path("tp-qwen2");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen2",
            "num_hidden_layers": 24
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_qwen3_multi_rank_runtime() {
    let dir = temp_path("tp-qwen3");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen3",
            "num_hidden_layers": 28
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_ernie45_multi_rank_runtime() {
    let dir = temp_path("tp-ernie45");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "ernie4_5",
            "num_hidden_layers": 18
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_hunyuan_v1_dense_multi_rank_runtime() {
    let dir = temp_path("tp-hunyuan-v1-dense");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "hunyuan_v1_dense",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_gemma3_multi_rank_runtime() {
    let dir = temp_path("tp-gemma3");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma3_text",
            "num_hidden_layers": 26
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

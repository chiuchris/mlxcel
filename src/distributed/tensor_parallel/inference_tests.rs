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

use std::fs;

use super::{ensure_single_rank_runtime, resolve_model_shard_plan, shard_config_from_cli};

fn temp_model_dir(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("mlxcel-tp-{name}-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn shard_config_from_cli_parses_modes() {
    let config = shard_config_from_cli(2, "within_expert", "vocab_parallel", "replicated").unwrap();

    assert_eq!(config.tp_size, 2);
    assert_eq!(config.moe_mode.to_string(), "within_expert");
    assert_eq!(config.embedding_mode.to_string(), "vocab_parallel");
    assert_eq!(config.lm_head_mode.to_string(), "replicated");
}

#[test]
fn resolve_model_shard_plan_uses_text_config_layer_count() {
    let dir = temp_model_dir("text-config");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen3_vl",
            "text_config": {
                "model_type": "qwen3",
                "num_hidden_layers": 40
            }
        }"#,
    )
    .unwrap();

    let summary = resolve_model_shard_plan(
        &dir,
        shard_config_from_cli(1, "expert_parallel", "replicated", "replicated").unwrap(),
    )
    .unwrap();

    assert_eq!(summary.architecture, "qwen3");
    assert_eq!(summary.num_layers, 40);
    assert_eq!(summary.plan.num_layers, 40);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ensure_single_rank_runtime_rejects_multi_rank() {
    let dir = temp_model_dir("tp-reject");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let summary = resolve_model_shard_plan(
        &dir,
        shard_config_from_cli(2, "expert_parallel", "replicated", "replicated").unwrap(),
    )
    .unwrap();

    let error = ensure_single_rank_runtime(&summary, "mlxcel generate").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("multi-rank inference is not wired")
    );
    assert!(error.to_string().contains("tp_size=2"));

    fs::remove_dir_all(dir).unwrap();
}

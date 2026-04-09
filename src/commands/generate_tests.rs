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

use super::{
    apply_user_chat_template, generated_suffix, generation_stats_from_duration, resolve_cli_prompt,
    validate_tensor_parallel_args,
};
use mlxcel::server::chat_template::ChatTemplateProcessor;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

#[test]
fn generation_stats_from_duration_uses_elapsed_time_for_decode_rate() {
    let stats = generation_stats_from_duration(12, 6, Duration::from_secs(2));

    assert_eq!(stats.prompt_tokens, 12);
    assert_eq!(stats.generated_tokens, 6);
    assert_eq!(stats.decode_time_ms, 2000.0);
    assert_eq!(stats.decode_tok_per_sec, 3.0);
}

#[test]
fn generation_stats_from_duration_handles_zero_elapsed_time() {
    let stats = generation_stats_from_duration(4, 2, Duration::ZERO);

    assert_eq!(stats.decode_time_ms, 0.0);
    assert_eq!(stats.decode_tok_per_sec, 0.0);
}

#[test]
fn apply_user_chat_template_wraps_prompt_as_user_message() {
    let processor = ChatTemplateProcessor::with_template(
        "{{ messages[0].role }}: {{ messages[0].content }}".to_string(),
    );

    let rendered = apply_user_chat_template(&processor, "Hello");

    assert_eq!(rendered, "user: Hello");
}

#[test]
fn resolve_cli_prompt_skips_template_when_disabled() {
    let processor = ChatTemplateProcessor::with_template("wrapped".to_string());

    let prompt = resolve_cli_prompt("Hello", true, Some(&processor), 0);

    assert_eq!(prompt, "Hello");
}

#[test]
fn resolve_cli_prompt_falls_back_on_template_errors() {
    let processor = ChatTemplateProcessor::with_template("{% if %}".to_string());

    let prompt = resolve_cli_prompt("Hello", false, Some(&processor), 0);

    assert_eq!(prompt, "Hello");
}

#[test]
fn generated_suffix_strips_prompt_prefix() {
    assert_eq!(generated_suffix("Hello, world", "Hello"), ", world");
}

#[test]
fn generated_suffix_falls_back_when_prefix_is_missing() {
    assert_eq!(generated_suffix("world", "Hello"), "world");
}

fn temp_model_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "mlxcel-generate-test-{name}-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn sample_generate_args(model_path: PathBuf) -> crate::GenerateArgs {
    crate::GenerateArgs {
        model: crate::ModelOptions {
            model: model_path,
            adapter: None,
            draft_model: None,
            num_draft_tokens: 3,
        },
        generation: crate::GenerationOptions {
            prompt: "Hello".to_string(),
            image: Vec::new(),
            audio: None,
            max_tokens: 16,
            profile: false,
            no_chat_template: false,
            recommend_quant: false,
            kv_cache_mode: "fp16".to_string(),
        },
        sampling: crate::SamplingOptions {
            temp: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 0,
        },
        tensor_parallel: crate::TensorParallelOptions {
            tp_size: 1,
            tp_moe_mode: "expert_parallel".to_string(),
            tp_embedding_mode: "replicated".to_string(),
            tp_lm_head_mode: "replicated".to_string(),
        },
    }
}

#[test]
fn validate_tensor_parallel_args_accepts_single_rank() {
    let dir = temp_model_dir("tp1");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let args = sample_generate_args(dir.clone());
    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_args_accepts_supported_multi_rank_runtime() {
    let dir = temp_model_dir("tp2");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let mut args = sample_generate_args(dir.clone());
    args.tensor_parallel.tp_size = 2;

    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_args_accepts_qwen3_multi_rank_runtime() {
    let dir = temp_model_dir("tp-qwen3");
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen3",
            "num_hidden_layers": 28
        }"#,
    )
    .unwrap();

    let mut args = sample_generate_args(dir.clone());
    args.tensor_parallel.tp_size = 2;

    validate_tensor_parallel_args(&args).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

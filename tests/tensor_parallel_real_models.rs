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

use std::path::{Path, PathBuf};

use mlxcel::{
    CxxGenerator, LanguageModel, SamplingConfig, distributed::ShardConfig, initialize_runtime,
    load_model, load_model_with_tensor_parallel, tokenizer::MlxcelTokenizer,
};

fn repo_model_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name)
}

fn prompt_tokens(tokenizer: &MlxcelTokenizer, prompt: &str) -> Vec<i32> {
    let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
    tokenizer
        .encode(prompt, add_special)
        .unwrap()
        .into_iter()
        .map(|token| token as i32)
        .collect()
}

fn decode_tokens(tokenizer: &MlxcelTokenizer, tokens: &[i32]) -> String {
    let tokens: Vec<u32> = tokens.iter().map(|&token| token as u32).collect();
    tokenizer.decode(&tokens, true).unwrap_or_default()
}

fn assert_tp_matches_single_rank(
    model_dir: &Path,
    prompt: &str,
    max_tokens: usize,
    tp_size: usize,
) {
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let runtime = initialize_runtime();
    eprintln!(
        "Comparing TP parity on {} using {}",
        model_dir.display(),
        runtime.device
    );
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (tokenizer, single_rank_tokens, tensor_parallel_tokens) = {
        let (single_rank_model, tokenizer) = load_model(model_dir).unwrap();
        let (tensor_parallel_model, _) =
            load_model_with_tensor_parallel(model_dir, None, &ShardConfig::with_tp_size(tp_size))
                .unwrap();
        let prompt_tokens = prompt_tokens(&tokenizer, prompt);
        let sampling = SamplingConfig::greedy();

        let mut single_rank_generator = CxxGenerator::new(single_rank_model.num_layers());
        let single_rank_tokens = single_rank_generator.generate(
            &single_rank_model,
            &prompt_tokens,
            max_tokens,
            &sampling,
        );

        let mut tensor_parallel_generator = CxxGenerator::new(tensor_parallel_model.num_layers());
        let tensor_parallel_tokens = tensor_parallel_generator.generate(
            &tensor_parallel_model,
            &prompt_tokens,
            max_tokens,
            &sampling,
        );

        (tokenizer, single_rank_tokens, tensor_parallel_tokens)
    };
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    assert!(
        single_rank_tokens.len() >= 8,
        "expected a longer generation for {}, got {} tokens",
        model_dir.display(),
        single_rank_tokens.len()
    );
    assert_eq!(
        single_rank_tokens,
        tensor_parallel_tokens,
        "generated token mismatch for {}\nsingle-rank: {:?}\ntp={}: {:?}\nsingle-rank text: {}\ntp={} text: {}",
        model_dir.display(),
        single_rank_tokens,
        tp_size,
        tensor_parallel_tokens,
        decode_tokens(&tokenizer, &single_rank_tokens),
        tp_size,
        decode_tokens(&tokenizer, &tensor_parallel_tokens)
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn llama_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("llama-3.2-1b-4bit"),
        "Continue this sequence with more entries separated by commas: 1, 2, 3, 4, 5,",
        32,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn llama31_8b_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("llama-3.1-8b-4bit"),
        "Continue this sequence with more entries separated by commas: 1, 2, 3, 4, 5,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("qwen3-0.6b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        32,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("qwen3-0.6b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn qwen3_4b_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("qwen3-4b-4bit"),
        "Continue this sequence with more entries separated by commas: alpha, beta, gamma,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn ernie45_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("ernie-4.5-0.3b-4bit"),
        "Continue this sequence with more entries separated by commas: red, blue, green,",
        32,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn ernie45_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("ernie-4.5-0.3b-4bit"),
        "Continue this sequence with more entries separated by commas: red, blue, green,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn hunyuan_v1_dense_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("hunyuan-1.8b-4bit"),
        "Continue this sequence with more entries separated by commas: spring, summer, autumn,",
        32,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn hunyuan_v1_dense_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("hunyuan-1.8b-4bit"),
        "Continue this sequence with more entries separated by commas: spring, summer, autumn,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn gemma3_tp2_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("gemma3-1b-4bit"),
        "Continue this sequence with more entries separated by commas: north, south, east,",
        24,
        2,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn llama_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("llama-3.2-1b-4bit"),
        "Continue this sequence with more entries separated by commas: 1, 2, 3, 4, 5,",
        32,
        4,
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn gemma3_tp4_matches_single_rank_greedy_long_generation() {
    assert_tp_matches_single_rank(
        &repo_model_dir("gemma3-1b-4bit"),
        "Continue this sequence with more entries separated by commas: north, south, east,",
        24,
        4,
    );
}

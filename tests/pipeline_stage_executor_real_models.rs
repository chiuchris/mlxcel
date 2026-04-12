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

use mlxcel::distributed::pipeline::{LoadedStageExecutor, StageAssignment, StageExecutionInput};
use mlxcel::{LanguageModel, distributed::pipeline::StageExecutor};

fn repo_model_dir(name: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let primary = manifest_dir.join("models").join(name);
    if primary.exists() {
        return primary;
    }

    let shared_checkout = manifest_dir
        .parent()
        .map(|parent| parent.join("mlxcel-internal").join("models").join(name))
        .unwrap_or(primary.clone());
    if shared_checkout.exists() {
        return shared_checkout;
    }

    primary
}

fn two_stage_assignments(total_layers: usize) -> [StageAssignment; 2] {
    let split = total_layers / 2;
    [
        StageAssignment {
            stage_index: 0,
            device_id: "stage-0".to_string(),
            layer_range: 0..split,
            has_embedding: true,
            has_lm_head: false,
            estimated_memory_bytes: 0,
        },
        StageAssignment {
            stage_index: 1,
            device_id: "stage-1".to_string(),
            layer_range: split..total_layers,
            has_embedding: false,
            has_lm_head: true,
            estimated_memory_bytes: 0,
        },
    ]
}

fn assert_two_stage_llama_matches_full_model(model_dir: &Path, prompt: &[i32], decode_token: i32) {
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let (model, _) = mlxcel::load_model(model_dir).unwrap();
    let total_layers = model.num_layers();
    let assignments = two_stage_assignments(total_layers);
    let stage0 = LoadedStageExecutor::load(model_dir, &assignments[0]).unwrap();
    let stage1 = LoadedStageExecutor::load(model_dir, &assignments[1]).unwrap();

    let prompt_ids = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let decode_ids = mlxcel_core::from_slice_i32(&[decode_token], &[1, 1]);

    let mut full_caches = model.make_caches();
    let full_prefill = model.forward(&prompt_ids, &mut full_caches, None);
    let full_decode = model.forward(&decode_ids, &mut full_caches, None);

    let mut stage0_caches = stage0.make_caches();
    let mut stage1_caches = stage1.make_caches();
    let stage0_prefill = stage0
        .execute(
            StageExecutionInput::TokenIds(&prompt_ids),
            &mut stage0_caches,
            None,
        )
        .unwrap()
        .into_hidden_states()
        .unwrap();
    let stage_prefill = stage1
        .execute(
            StageExecutionInput::HiddenStates(stage0_prefill.as_ref().unwrap()),
            &mut stage1_caches,
            None,
        )
        .unwrap()
        .into_logits()
        .unwrap();
    let stage0_decode = stage0
        .execute(
            StageExecutionInput::TokenIds(&decode_ids),
            &mut stage0_caches,
            None,
        )
        .unwrap()
        .into_hidden_states()
        .unwrap();
    let stage_decode = stage1
        .execute(
            StageExecutionInput::HiddenStates(stage0_decode.as_ref().unwrap()),
            &mut stage1_caches,
            None,
        )
        .unwrap()
        .into_logits()
        .unwrap();

    let atol = 1e-4f64;
    let prefill_close = mlxcel_core::allclose(&full_prefill, &stage_prefill, atol, atol);
    let decode_close = mlxcel_core::allclose(&full_decode, &stage_decode, atol, atol);
    assert!(
        mlxcel_core::item_bool(&prefill_close),
        "prefill logits mismatch for {}",
        model_dir.display()
    );
    assert!(
        mlxcel_core::item_bool(&decode_close),
        "decode logits mismatch for {}",
        model_dir.display()
    );
}

#[test]
#[ignore = "requires local model weights and extended real-model generation"]
fn pipeline_stage_executor_llama_real_model_parity() {
    assert_two_stage_llama_matches_full_model(
        &repo_model_dir("llama-3.2-1b-4bit"),
        &[128000, 9906],
        13,
    );
}

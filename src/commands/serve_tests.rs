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

use super::build_startup_input;

fn sample_args() -> crate::ServeArgs {
    crate::ServeArgs {
        model: PathBuf::from("models/foo"),
        adapter: Some(PathBuf::from("adapters/bar")),
        alias: Some("alias".to_string()),
        host: "127.0.0.1".to_string(),
        port: 9000,
        api_key: Some("secret".to_string()),
        api_key_file: Some(PathBuf::from("api.key")),
        n_parallel: 3,
        ctx_size: 8192,
        n_predict: 128,
        draft_model: Some(PathBuf::from("models/draft")),
        draft_max: 4,
        max_batch_size: Some(4),
        max_queue_depth: 32,
        timeout: 30,
        chat_template: Some("{{ prompt }}".to_string()),
        chat_template_file: Some(PathBuf::from("chat.jinja")),
        slots: true,
        _no_slots: true,
        props: true,
        metrics: false,
        warmup: true,
        _no_warmup: true,
        temp: 0.7,
        top_k: 50,
        top_p: 0.95,
        min_p: 0.2,
        seed: -1,
        repeat_last_n: 32,
        repeat_penalty: 1.2,
        presence_penalty: 0.3,
        frequency_penalty: 0.4,
        dry_multiplier: 0.5,
        dry_base: 1.9,
        dry_allowed_length: 3,
        dry_penalty_last_n: -1,
        dry_sequence_breakers: vec!["\n".to_string(), "\t".to_string()],
        verbose: true,
        log_disable: false,
        log_file: Some(PathBuf::from("server.log")),
        _no_webui: false,
        _jinja: false,
        _n_gpu_layers: None,
        _mmproj: None,
        _flash_attn: false,
        _mlock: false,
        _no_mmap: false,
        _cont_batching: false,
    }
}

#[test]
fn build_startup_input_preserves_edge_flags_for_normalization() {
    let input = build_startup_input(sample_args());

    assert_eq!(input.model_path, PathBuf::from("models/foo"));
    assert_eq!(input.adapter_path, Some(PathBuf::from("adapters/bar")));
    assert_eq!(input.draft_model_path, Some(PathBuf::from("models/draft")));
    assert!(input.slots);
    assert!(input.no_slots);
    assert!(input.warmup);
    assert!(input.no_warmup);
    assert_eq!(input.seed, -1);
    assert_eq!(
        input.dry_sequence_breakers,
        vec!["\n".to_string(), "\t".to_string()]
    );
}

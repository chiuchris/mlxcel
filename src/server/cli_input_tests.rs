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

use super::{ServerStartupInput, resolve_compat_toggle, resolve_seed};

fn sample_input() -> ServerStartupInput {
    ServerStartupInput {
        model_path: PathBuf::from("models/foo"),
        adapter_path: Some(PathBuf::from("adapters/bar")),
        model_alias: Some("alias".to_string()),
        host: "127.0.0.1".to_string(),
        port: 8080,
        api_key: Some("secret".to_string()),
        api_key_file: Some(PathBuf::from("api.key")),
        n_parallel: 2,
        ctx_size: 4096,
        n_predict: 256,
        timeout: 600,
        draft_model_path: Some(PathBuf::from("models/draft")),
        draft_max: 8,
        chat_template: Some("{{ prompt }}".to_string()),
        chat_template_file: Some(PathBuf::from("chat.jinja")),
        slots: true,
        no_slots: false,
        props: true,
        metrics: true,
        warmup: true,
        no_warmup: false,
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.1,
        seed: 42,
        repeat_last_n: 64,
        repeat_penalty: 1.1,
        presence_penalty: 0.2,
        frequency_penalty: 0.3,
        dry_multiplier: 0.4,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_penalty_last_n: -1,
        dry_sequence_breakers: vec!["\n".to_string()],
        verbose: true,
        log_disable: false,
        log_file: Some(PathBuf::from("server.log")),
    }
}

#[test]
fn resolve_compat_toggle_honors_disable_override() {
    assert!(resolve_compat_toggle(true, false));
    assert!(!resolve_compat_toggle(true, true));
    assert!(!resolve_compat_toggle(false, false));
}

#[test]
fn resolve_seed_maps_negative_values_to_random_mode() {
    assert_eq!(resolve_seed(-1), None);
    assert_eq!(resolve_seed(7), Some(7));
}

#[test]
fn into_startup_config_normalizes_edge_only_flags() {
    let mut input = sample_input();
    input.no_slots = true;
    input.no_warmup = true;
    input.seed = -1;

    let startup = input.into_startup_config();

    assert!(!startup.enable_slots);
    assert!(!startup.warmup);
    assert_eq!(startup.seed, None);
    assert_eq!(startup.adapter_path, Some(PathBuf::from("adapters/bar")));
    assert_eq!(
        startup.draft_model_path,
        Some(PathBuf::from("models/draft"))
    );
    assert_eq!(startup.log_file, Some(PathBuf::from("server.log")));
}

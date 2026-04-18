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

use super::{ServerStartupInput, resolve_compat_toggle, resolve_prefill_chunk_size, resolve_seed};

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
        max_batch_size: Some(4),
        max_queue_depth: 32,
        prefill_chunk_size: 512,
        batch_size: None,
        ubatch_size: None,
        enable_preemption: false,
        preemption_policy: "longest-first".to_string(),
        no_batch: false,
        max_batch_prefill: 1,
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
        distributed_config: None,
        node_role: None,
        node_id: None,
        peers: Vec::new(),
        pp_layers: None,
        pp_micro_batch_size: 1,
        pp_auto: None,
        pp_peer: false,
        cluster_discovery: "static".to_string(),
        cluster_name: None,
        cluster_peers: Vec::new(),
        cluster_discovery_port: None,
        cluster_control_addr: None,
        cluster_config_out: None,
        dry_run: false,
        tp_size: 1,
        tp_moe_mode: "expert_parallel".to_string(),
        tp_embedding_mode: "replicated".to_string(),
        tp_lm_head_mode: "replicated".to_string(),
        vision_cache_size: 20,
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

#[test]
fn resolve_prefill_chunk_size_batch_size_alias_takes_effect() {
    let r = resolve_prefill_chunk_size(512, Some(1024), None);
    assert_eq!(r.prefill_chunk_size, 1024);
    assert!(!r.batch_size_conflict);
    assert!(!r.ubatch_size_provided);
}

#[test]
fn resolve_prefill_chunk_size_explicit_prefill_wins_with_conflict() {
    let r = resolve_prefill_chunk_size(256, Some(1024), None);
    assert_eq!(r.prefill_chunk_size, 256);
    assert!(r.batch_size_conflict);
}

#[test]
fn resolve_prefill_chunk_size_no_batch_size_returns_prefill() {
    let r = resolve_prefill_chunk_size(768, None, None);
    assert_eq!(r.prefill_chunk_size, 768);
    assert!(!r.batch_size_conflict);
}

#[test]
fn resolve_prefill_chunk_size_ubatch_sets_provided_flag() {
    let r = resolve_prefill_chunk_size(512, None, Some(256));
    assert!(r.ubatch_size_provided);
    assert_eq!(r.prefill_chunk_size, 512);
}

#[test]
fn resolve_prefill_chunk_size_both_same_value_no_conflict() {
    let r = resolve_prefill_chunk_size(1024, Some(1024), None);
    assert_eq!(r.prefill_chunk_size, 1024);
    assert!(!r.batch_size_conflict);
}

#[test]
fn into_startup_config_propagates_no_batch_flag() {
    let mut input = sample_input();
    input.no_batch = true;

    let startup = input.into_startup_config();
    assert!(startup.no_batch);

    let mut input2 = sample_input();
    input2.no_batch = false;
    let startup2 = input2.into_startup_config();
    assert!(!startup2.no_batch);
}

#[test]
fn into_startup_config_resolves_batch_size_alias() {
    let mut input = sample_input();
    input.batch_size = Some(1024);
    let startup = input.into_startup_config();
    assert_eq!(startup.prefill_chunk_size, 1024);
    assert!(!startup.batch_size_conflict);
    assert!(!startup.ubatch_size_provided);
}

#[test]
fn into_startup_config_detects_batch_size_conflict() {
    let mut input = sample_input();
    input.prefill_chunk_size = 256;
    input.batch_size = Some(1024);
    input.ubatch_size = Some(64);
    let startup = input.into_startup_config();
    assert_eq!(startup.prefill_chunk_size, 256);
    assert!(startup.batch_size_conflict);
    assert!(startup.ubatch_size_provided);
}

#[test]
fn into_startup_config_propagates_pp_layers() {
    let mut input = sample_input();
    input.pp_layers = Some("0-15,16-31".to_string());
    let startup = input.into_startup_config();
    assert_eq!(startup.pp_layers, Some("0-15,16-31".to_string()));
}

#[test]
fn into_startup_config_pp_layers_none_by_default() {
    let startup = sample_input().into_startup_config();
    assert_eq!(startup.pp_layers, None);
}

#[test]
fn into_startup_config_propagates_pp_micro_batch_size() {
    let mut input = sample_input();
    input.pp_micro_batch_size = 4;
    let startup = input.into_startup_config();
    assert_eq!(startup.pp_micro_batch_size, 4);
}

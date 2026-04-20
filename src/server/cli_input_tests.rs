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
use std::sync::{Mutex, OnceLock};

use super::{ServerStartupInput, resolve_compat_toggle, resolve_prefill_chunk_size, resolve_seed};
use crate::lang_bias::LangBiasCliArgs;

// Global mutex to serialize all tests that mutate env vars via `EnvGuard`.
// `std::env::set_var` / `remove_var` are not thread-safe under cargo's default
// parallel test runner.  Every test that calls `EnvGuard::set` must acquire
// this lock *before* constructing the guard so the lock outlives the guard's
// `Drop` (which calls `remove_var`).
static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

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
        enable_elastic_pp: false,
        elastic_pp_drain_timeout: 120,
        elastic_pp_pressure_fraction: 0.92,
        elastic_pp_cool_down: 30,
        metrics_port: None,
        debug_pp_trace: None,
        lang_bias_config: None,
        reasoning_budget: -1,
        chat_template_kwargs: None,
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

    let startup = input.into_startup_config().expect("valid startup input");

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

    let startup = input.into_startup_config().expect("valid startup input");
    assert!(startup.no_batch);

    let mut input2 = sample_input();
    input2.no_batch = false;
    let startup2 = input2.into_startup_config().expect("valid startup input");
    assert!(!startup2.no_batch);
}

#[test]
fn into_startup_config_resolves_batch_size_alias() {
    let mut input = sample_input();
    input.batch_size = Some(1024);
    let startup = input.into_startup_config().expect("valid startup input");
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
    let startup = input.into_startup_config().expect("valid startup input");
    assert_eq!(startup.prefill_chunk_size, 256);
    assert!(startup.batch_size_conflict);
    assert!(startup.ubatch_size_provided);
}

#[test]
fn into_startup_config_propagates_pp_layers() {
    let mut input = sample_input();
    input.pp_layers = Some("0-15,16-31".to_string());
    let startup = input.into_startup_config().expect("valid startup input");
    assert_eq!(startup.pp_layers, Some("0-15,16-31".to_string()));
}

#[test]
fn into_startup_config_pp_layers_none_by_default() {
    let startup = sample_input()
        .into_startup_config()
        .expect("valid startup input");
    assert_eq!(startup.pp_layers, None);
}

#[test]
fn into_startup_config_propagates_pp_micro_batch_size() {
    let mut input = sample_input();
    input.pp_micro_batch_size = 4;
    let startup = input.into_startup_config().expect("valid startup input");
    assert_eq!(startup.pp_micro_batch_size, 4);
}

// -------------------------------------------------------------------------
// Issue #410 — chat_template_kwargs normalization
// -------------------------------------------------------------------------

#[test]
fn into_startup_config_accepts_valid_chat_template_kwargs_json() {
    let mut input = sample_input();
    input.chat_template_kwargs = Some(r#"{"preserve_thinking": true}"#.to_string());
    let startup = input
        .into_startup_config()
        .expect("valid JSON object should succeed");
    let kwargs = startup
        .chat_template_kwargs
        .expect("non-empty kwargs should materialize");
    assert_eq!(kwargs.preserve_thinking(), true);
}

#[test]
fn into_startup_config_rejects_malformed_chat_template_kwargs_json() {
    let mut input = sample_input();
    input.chat_template_kwargs = Some("{not-json".to_string());
    let err = input
        .into_startup_config()
        .expect_err("malformed JSON must error at startup");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("chat-template-kwargs"),
        "error should reference the flag, got: {msg}"
    );
}

#[test]
fn into_startup_config_rejects_non_object_chat_template_kwargs_json() {
    let mut input = sample_input();
    input.chat_template_kwargs = Some("[true]".to_string());
    let err = input
        .into_startup_config()
        .expect_err("arrays must be rejected at startup");
    let msg = format!("{err:#}");
    assert!(msg.contains("chat-template-kwargs"));
    assert!(
        msg.contains("object"),
        "error should mention object, got: {msg}"
    );
}

#[test]
fn into_startup_config_empty_chat_template_kwargs_collapses_to_none() {
    let mut input = sample_input();
    input.chat_template_kwargs = Some("".to_string());
    let startup = input
        .into_startup_config()
        .expect("empty string is a valid no-op");
    assert!(startup.chat_template_kwargs.is_none());

    let mut input = sample_input();
    input.chat_template_kwargs = Some("{}".to_string());
    let startup = input
        .into_startup_config()
        .expect("empty JSON object is a valid no-op");
    assert!(startup.chat_template_kwargs.is_none());
}

// -------------------------------------------------------------------------
// B7 — LLAMA_ARG_LANG_BIAS env-var fallback tests (plan §6.4)
//
// Each test manages the env var explicitly (set + cleanup) and delegates to a
// helper that calls `env_fallback_lang_bias` directly, keeping the env
// mutation inside each test's stack frame for clarity.
//
// NOTE: Rust test threads share a process, so env-var mutations must be
// cleaned up regardless of test outcome. These tests run in-process and are
// intentionally structured to be self-contained.
// -------------------------------------------------------------------------

/// Helper that cleans up `LLAMA_ARG_LANG_BIAS` at drop time.
struct EnvGuard(&'static str);

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        // SAFETY: single-threaded access guaranteed by cargo test (or
        // serial execution with env isolation if run under a test harness).
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var(key, value);
        }
        EnvGuard(key)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var(self.0);
        }
    }
}

/// B7 acceptance test: only `LLAMA_ARG_LANG_BIAS` is set (no CLI flag) →
/// the env value flows into `LangBiasCliArgs.lang_bias` and resolves to the
/// expected `LangBiasConfig`.
#[test]
fn env_var_feeds_parser() {
    use super::env_fallback_lang_bias;
    use crate::lang_bias::parse_lang_bias_entries;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS", "ja=-inf,zh=-5");

    let mut args = LangBiasCliArgs::default(); // lang_bias = None (no CLI flag)
    env_fallback_lang_bias(&mut args);

    // The env var value should now be in args.lang_bias.
    assert_eq!(
        args.lang_bias.as_deref(),
        Some("ja=-inf,zh=-5"),
        "env var value should be copied into lang_bias when CLI flag is absent"
    );

    // Resolve to a LangBiasConfig and verify the bias_set matches the expected pairs.
    let config = args.resolve().unwrap().unwrap();
    let expected = parse_lang_bias_entries("ja=-inf,zh=-5").unwrap();
    assert_eq!(
        config.bias_set.ordered.len(),
        expected.ordered.len(),
        "resolved bias_set should have the same number of entries as the env var"
    );
    for (i, ((got_lang, got_bias), (exp_lang, exp_bias))) in config
        .bias_set
        .ordered
        .iter()
        .zip(expected.ordered.iter())
        .enumerate()
    {
        assert_eq!(
            got_lang, exp_lang,
            "entry {i}: language code mismatch (got {got_lang:?}, expected {exp_lang:?})"
        );
        // Use exact equality for f32 since both come from the same parse path.
        assert_eq!(
            got_bias, exp_bias,
            "entry {i}: bias mismatch (got {got_bias}, expected {exp_bias})"
        );
    }
}

/// B7 acceptance test: `LLAMA_ARG_LANG_BIAS` is set without any CLI flag →
/// resolves to the env-var config.
#[test]
fn env_without_cli_parses() {
    use super::env_fallback_lang_bias;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS", "ko=+5");

    let mut args = LangBiasCliArgs::default();
    env_fallback_lang_bias(&mut args);

    assert_eq!(
        args.lang_bias.as_deref(),
        Some("ko=+5"),
        "env var value should be present when CLI flag is absent"
    );

    let config = args.resolve().unwrap().unwrap();
    assert_eq!(config.bias_set.ordered.len(), 1);
    use mlxcel_core::lang_analyzer::LanguageCode;
    assert_eq!(config.bias_set.ordered[0].0, LanguageCode::Ko);
    assert_eq!(config.bias_set.ordered[0].1, 5.0_f32);
}

/// B7 acceptance test: both CLI `--lang-bias ja=-inf` and env
/// `LLAMA_ARG_LANG_BIAS=ko=+5` are set → CLI wins (env is ignored) and an
/// INFO-level log message is emitted.
///
/// The log message is emitted via `tracing::info!` which writes to the
/// subscriber registered for the current thread; we verify that the CLI value
/// is kept without attempting to capture the log output (subscriber setup is
/// out of scope for a unit test).
#[test]
fn cli_overrides_env() {
    use super::env_fallback_lang_bias;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS", "ko=+5");

    // Simulate CLI providing `--lang-bias ja=-inf`.
    let mut args = LangBiasCliArgs {
        lang_bias: Some("ja=-inf".to_owned()),
        ..Default::default()
    };

    env_fallback_lang_bias(&mut args);

    // CLI value must be preserved; env var must NOT overwrite it.
    assert_eq!(
        args.lang_bias.as_deref(),
        Some("ja=-inf"),
        "CLI --lang-bias should take precedence over LLAMA_ARG_LANG_BIAS env var"
    );

    let config = args.resolve().unwrap().unwrap();
    assert_eq!(config.bias_set.ordered.len(), 1);
    use mlxcel_core::lang_analyzer::LanguageCode;
    assert_eq!(config.bias_set.ordered[0].0, LanguageCode::Ja);
    assert_eq!(config.bias_set.ordered[0].1, f32::NEG_INFINITY);
}

// -------------------------------------------------------------------------
// Issue #405 — LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS env-var fallback
//
// Mirrors the B7 tests above. The env-var fallback for the byte-fragment
// opt-in is permissive about truthiness (accepts `true`/`false`/`1`/`0`) and
// respects CLI precedence.
// -------------------------------------------------------------------------

/// Env var set without CLI flag → `include_byte_fragments` is flipped to `true`.
#[test]
fn byte_fragments_env_var_feeds_flag_true() {
    use super::env_fallback_lang_bias_include_byte_fragments;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS", "true");

    let mut args = LangBiasCliArgs::default();
    assert!(
        !args.include_byte_fragments,
        "default must be false before fallback runs"
    );
    env_fallback_lang_bias_include_byte_fragments(&mut args);
    assert!(
        args.include_byte_fragments,
        "truthy env var must flip include_byte_fragments to true"
    );
}

/// Env var supports `1` / `0` forms too.
#[test]
fn byte_fragments_env_var_accepts_numeric_forms() {
    use super::env_fallback_lang_bias_include_byte_fragments;

    let _env_guard = env_lock();
    // `1` → true
    {
        let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS", "1");
        let mut args = LangBiasCliArgs::default();
        env_fallback_lang_bias_include_byte_fragments(&mut args);
        assert!(args.include_byte_fragments, "`1` must parse as true");
    }
    // `0` → false → no flip when CLI was already false.
    {
        let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS", "0");
        let mut args = LangBiasCliArgs::default();
        env_fallback_lang_bias_include_byte_fragments(&mut args);
        assert!(
            !args.include_byte_fragments,
            "`0` must keep include_byte_fragments=false"
        );
    }
}

/// CLI `--lang-bias-include-byte-fragments` beats the env var.
#[test]
fn byte_fragments_cli_overrides_env() {
    use super::env_fallback_lang_bias_include_byte_fragments;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS", "false");

    // CLI already set include_byte_fragments=true.
    let mut args = LangBiasCliArgs {
        include_byte_fragments: true,
        ..Default::default()
    };
    env_fallback_lang_bias_include_byte_fragments(&mut args);
    assert!(
        args.include_byte_fragments,
        "CLI --lang-bias-include-byte-fragments must win against env 'false'"
    );
}

/// Unparseable env var is ignored (warn-and-drop), leaving CLI default.
#[test]
fn byte_fragments_env_var_unparseable_is_ignored() {
    use super::env_fallback_lang_bias_include_byte_fragments;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS", "maybe");

    let mut args = LangBiasCliArgs::default();
    env_fallback_lang_bias_include_byte_fragments(&mut args);
    assert!(
        !args.include_byte_fragments,
        "unparseable env var must leave the CLI default (false) in place"
    );
}

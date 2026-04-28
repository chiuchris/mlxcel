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

//! Edge-only server startup inputs from CLI compatibility surfaces.
//!
//! `ServerStartupConfig` is the normalized runtime-facing policy object.
//! This module keeps raw CLI concerns such as `--no-slots` overrides and
//! negative seed sentinels at the edge so server startup and request handling do
//! not need to remember llama-server compatibility rules.

use std::net::SocketAddr;
use std::path::PathBuf;

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::lang_analyzer::LangBiasConfig;

use super::chat_template_kwargs::resolve_server_default_kwargs;
use super::thinking_budget::{
    ThinkingBudget, ThinkingBudgetError, resolve_server_default_reasoning_budget,
};
use super::{DecodeStorageBackend, ServerStartupConfig};
use crate::lang_bias::LangBiasCliArgs;

/// Raw server startup input captured from CLI/front-end binaries.
///
/// These fields intentionally preserve edge-specific conventions such as
/// negative seed sentinels and `--no-*` compatibility flags. Normalize them
/// exactly once with [`ServerStartupInput::into_startup_config`].
#[derive(Debug)]
pub struct ServerStartupInput {
    pub model_path: PathBuf,
    pub adapter_path: Option<PathBuf>,
    pub model_alias: Option<String>,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub api_key_file: Option<PathBuf>,
    pub n_parallel: usize,
    pub ctx_size: usize,
    pub n_predict: i32,
    pub timeout: u64,
    pub draft_model_path: Option<PathBuf>,
    pub draft_max: usize,
    pub max_batch_size: Option<usize>,
    pub max_queue_depth: usize,
    pub prefill_chunk_size: usize,
    /// llama-server alias for `--prefill-chunk-size` (`--batch-size` / `-b`).
    ///
    /// When set, maps to `prefill_chunk_size`. If both this and `prefill_chunk_size`
    /// differ from the default, `prefill_chunk_size` takes precedence with a warning.
    pub batch_size: Option<usize>,
    /// llama-server `--ubatch-size`. Accepted but ignored on Apple Silicon.
    pub ubatch_size: Option<usize>,
    pub enable_preemption: bool,
    pub preemption_policy: String,
    /// Disable continuous batching; force the legacy sequential worker.
    pub no_batch: bool,
    /// Maximum number of pending requests to batch together for prefill (default: 1).
    pub max_batch_prefill: usize,
    /// Decode storage backend requested by the CLI. `None` means the startup
    /// layer should use `MLXCEL_SERVER_DECODE_STORAGE` or automatic selection.
    pub decode_storage_backend: Option<DecodeStorageBackend>,
    pub chat_template: Option<String>,
    pub chat_template_file: Option<PathBuf>,
    pub slots: bool,
    pub no_slots: bool,
    pub props: bool,
    pub metrics: bool,
    pub warmup: bool,
    pub no_warmup: bool,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: i64,
    pub repeat_last_n: usize,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_penalty_last_n: i32,
    pub dry_sequence_breakers: Vec<String>,
    pub verbose: bool,
    pub log_disable: bool,
    pub log_file: Option<PathBuf>,

    // Distributed inference.
    /// Path to a TOML cluster configuration file.
    pub distributed_config: Option<PathBuf>,
    /// Role this node plays in the cluster (CLI shorthand).
    pub node_role: Option<String>,
    /// Unique identifier for this node (CLI shorthand).
    pub node_id: Option<String>,
    /// Comma-separated list of peer addresses (CLI shorthand).
    pub peers: Vec<SocketAddr>,
    /// Manual pipeline-parallel layer partition spec (e.g. "0-15,16-31").
    pub pp_layers: Option<String>,
    /// Micro-batch size for in-process pipeline execution.
    pub pp_micro_batch_size: usize,

    // Zero-config multi-machine pipeline bring-up (issue #342).
    /// When set (>= 2), run the zero-config coordinator bring-up and populate
    /// `distributed_config` with a freshly emitted TOML.
    pub pp_auto: Option<u32>,
    /// When `true`, run as a pipeline-stage peer for zero-config bring-up.
    pub pp_peer: bool,
    /// Discovery mode string (parsed into `ClusterDiscoveryMode` at startup).
    pub cluster_discovery: String,
    /// Optional override for the zero-config cluster name.
    pub cluster_name: Option<String>,
    /// Static seed peers for the zero-config bring-up.
    pub cluster_peers: Vec<SocketAddr>,
    /// Optional UDP port for the discovery beacon.
    pub cluster_discovery_port: Option<u16>,
    /// Optional coordinator control-plane bind address.
    pub cluster_control_addr: Option<SocketAddr>,
    /// Optional output path for the emitted cluster TOML.
    pub cluster_config_out: Option<PathBuf>,
    /// When `true`, plan the cluster and exit without starting workers.
    pub dry_run: bool,

    /// Number of tensor-parallel ranks.
    pub tp_size: usize,
    /// MoE expert sharding mode string (parsed at startup).
    pub tp_moe_mode: String,
    /// Embedding sharding mode string (parsed at startup).
    pub tp_embedding_mode: String,
    /// LM head sharding mode string (parsed at startup).
    pub tp_lm_head_mode: String,

    /// Maximum number of cached post-projection image features per loaded model.
    /// `0` disables the cache entirely.
    pub vision_cache_size: usize,

    // Elastic pipeline-parallel repartitioning (issue #349).
    /// Enable `--enable-elastic-pp`. Off by default.
    pub enable_elastic_pp: bool,
    /// Drain timeout seconds for elastic repartitioning.
    pub elastic_pp_drain_timeout: u64,
    /// Memory-pressure trigger fraction for elastic repartitioning.
    pub elastic_pp_pressure_fraction: f64,
    /// Cool-down seconds between memory-pressure triggers.
    pub elastic_pp_cool_down: u64,

    // Observability (issue #350).
    /// Requested port for the Prometheus `/metrics` endpoint. `Some(p)`
    /// enables the endpoint; `p` is informational only in v1 because the
    /// endpoint is multiplexed onto the main HTTP port.
    pub metrics_port: Option<u16>,
    /// Chrome-tracing JSON output path for the pipeline scheduler.
    pub debug_pp_trace: Option<PathBuf>,

    /// Axis B Epic #362 (B8): already-resolved server-wide language-bias
    /// configuration, if the CLI provided `--lang-bias` / `--lang-bias-config`.
    ///
    /// The binary's `serve` command calls `LangBiasCliArgs::resolve()` before
    /// constructing this input. `None` here means "no language steering",
    /// which preserves the bit-exact baseline generation path. B7
    /// (`LLAMA_ARG_LANG_BIAS`) plugs into this same field by calling
    /// [`env_fallback_lang_bias`] on the raw CLI args before `resolve()`, so
    /// the env-var and CLI paths share a single normalization point.
    pub lang_bias_config: Option<LangBiasConfig>,

    /// Issue #409: server-wide thinking-token budget for Qwen3-family models.
    ///
    /// Raw `i32` value preserving llama.cpp semantics:
    /// - `-1` (default) — unrestricted reasoning (bit-exact baseline).
    /// - `0` — immediate `</think>` on the first reasoning token.
    /// - `N > 0` — cap reasoning at `N` tokens.
    ///
    /// Resolved at [`ServerStartupInput::into_startup_config`] time into a
    /// typed `Option<ThinkingBudget>` on `ServerStartupConfig`. Per-request
    /// overrides on `/v1/chat/completions` and `/completion` take precedence.
    /// The env-var fallback `LLAMA_ARG_REASONING_BUDGET` is applied by
    /// [`env_fallback_reasoning_budget`] before this struct is constructed.
    pub reasoning_budget: i32,

    /// Issue #410: raw JSON string for the default chat-template kwargs.
    ///
    /// Mirrors llama.cpp's `--chat-template-kwargs` flag and the
    /// `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` env var. `None` or an empty string
    /// means "no server defaults." Invalid JSON is rejected at
    /// [`ServerStartupInput::into_startup_config`] time with an
    /// `anyhow::bail!`-equivalent error propagated back to the binary entry
    /// point, so the server refuses to start on a clearly malformed flag.
    ///
    /// The env-var fallback is applied by
    /// [`env_fallback_chat_template_kwargs`] in each binary before this
    /// struct is constructed, so the precedence CLI > env is enforced
    /// consistently.
    pub chat_template_kwargs: Option<String>,

    // Issue #484 (B11): llama-server-compatible K/V cache type split flags.
    //
    // These hold the raw CLI strings from `--cache-type-k` / `--cache-type-v`
    // (and their env var aliases `LLAMA_ARG_CACHE_TYPE_K` /
    // `LLAMA_ARG_CACHE_TYPE_V`) before resolution into a `KVCacheMode`.
    //
    // The legacy `--kv-cache-mode` shorthand is stored separately as
    // `kv_cache_mode_legacy`. Resolution priority at
    // `into_startup_config` time:
    //
    //   1. `--cache-type-k` + `--cache-type-v` (split flags) — highest priority.
    //      A warning is emitted when both the split flags and the legacy flag
    //      are present; split flags win.
    //   2. `--kv-cache-mode` (legacy shorthand) — still honored for backward
    //      compatibility.
    //   3. Default: `KVCacheMode::Fp16`.
    //
    // `None` means "not provided by the CLI or env var for this field".
    /// Raw string from `--cache-type-k` or `LLAMA_ARG_CACHE_TYPE_K`.
    /// `None` means the flag was not provided.
    pub cache_type_k: Option<String>,
    /// Raw string from `--cache-type-v` or `LLAMA_ARG_CACHE_TYPE_V`.
    /// `None` means the flag was not provided.
    pub cache_type_v: Option<String>,
    /// Raw string from `--kv-cache-mode` (legacy shorthand).
    /// `None` means the flag was not provided (= default FP16).
    pub kv_cache_mode_legacy: Option<String>,

    // Issue #424: prompt-prefix KV cache knobs.
    // Each field stores the raw CLI/env value before normalization into
    // `PromptCacheConfig`. `None` on the `Option`-typed fields means "not
    // provided by the CLI flag"; defaults come from `PromptCacheConfig::default()`.
    /// Whether the prompt-prefix KV cache is enabled. Defaults to `true`.
    /// Also accepts `MLXCEL_PROMPT_CACHE_ENABLED` and the llama.cpp-compat
    /// alias `LLAMA_ARG_CACHE_REUSE` (parsed as a boolean on/off).
    pub prompt_cache_enabled: bool,

    /// Maximum byte budget for the prompt cache.
    /// `None` means "use the compiled-in default (2 GiB)".
    /// Also accepts `MLXCEL_PROMPT_CACHE_CAPACITY_BYTES`.
    pub prompt_cache_capacity_bytes: Option<usize>,

    /// Maximum number of live cache entries.
    /// `None` means "use the compiled-in default (1024)".
    /// Also accepts `MLXCEL_PROMPT_CACHE_MAX_ENTRIES`.
    pub prompt_cache_max_entries: Option<usize>,

    /// Time-to-live for a cache entry in seconds.
    /// `None` means "use the compiled-in default (3600 s)".
    /// Also accepts `MLXCEL_PROMPT_CACHE_TTL`.
    pub prompt_cache_ttl_seconds: Option<u64>,

    /// Minimum prompt-prefix length (in tokens) before an entry is eligible
    /// for caching. `None` means "use the compiled-in default (32)".
    /// Also accepts `MLXCEL_PROMPT_CACHE_MIN_PREFIX`.
    pub prompt_cache_min_prefix: Option<usize>,
}

impl ServerStartupInput {
    /// Normalize edge-only CLI conventions into runtime startup policy.
    ///
    /// Returns an error when any edge input has a malformed shape that we
    /// cannot reasonably paper over at runtime. Today the only hard-fail
    /// input is `--chat-template-kwargs` (issue #410) because a malformed
    /// JSON object cannot be silently "ignored" without degrading into a
    /// confusing state where the operator thinks they configured kwargs but
    /// didn't.
    pub fn into_startup_config(self) -> anyhow::Result<ServerStartupConfig> {
        let resolution =
            resolve_prefill_chunk_size(self.prefill_chunk_size, self.batch_size, self.ubatch_size);
        // Issue #409: resolve the server-wide thinking budget once, up-front.
        // Invalid values are logged and treated as unbounded so the server
        // still starts (per-request errors are surfaced as 400s at the route).
        let reasoning_budget = match ThinkingBudget::from_raw_i32(self.reasoning_budget) {
            Ok(budget) => budget,
            Err(ThinkingBudgetError::InvalidNegative(v)) => {
                tracing::warn!(
                    "--reasoning-budget {v} is invalid (must be >= -1); ignoring (treating as unbounded)"
                );
                None
            }
            Err(err) => {
                tracing::warn!("--reasoning-budget validation error: {err}; treating as unbounded");
                None
            }
        };
        // Issue #410: resolve the server-wide chat-template kwargs. Invalid
        // JSON is a hard failure (see doc comment above) — return early with
        // a clear error that the binary will surface as an exit code.
        let chat_template_kwargs =
            resolve_server_default_kwargs(self.chat_template_kwargs.as_deref())
                .map_err(|e| anyhow::anyhow!("--chat-template-kwargs: {e}"))?;
        // Issue #424: build PromptCacheConfig from raw CLI/env-resolved values.
        // Any field left at `None` picks up the compiled-in default.
        let prompt_cache = build_prompt_cache_config(
            self.prompt_cache_enabled,
            self.prompt_cache_capacity_bytes,
            self.prompt_cache_max_entries,
            self.prompt_cache_ttl_seconds,
            self.prompt_cache_min_prefix,
        );

        // Issue #484 (B11): resolve the effective KV cache mode from split flags,
        // legacy shorthand, or the default (FP16).
        let kv_cache_mode = resolve_kv_cache_mode(
            self.cache_type_k.as_deref(),
            self.cache_type_v.as_deref(),
            self.kv_cache_mode_legacy.as_deref(),
        )
        .map_err(|e| anyhow::anyhow!("KV cache mode error: {e}"))?;

        Ok(ServerStartupConfig {
            model_path: self.model_path,
            adapter_path: self.adapter_path,
            model_alias: self.model_alias,
            host: self.host,
            port: self.port,
            api_key: self.api_key,
            api_key_file: self.api_key_file,
            n_parallel: self.n_parallel,
            ctx_size: self.ctx_size,
            n_predict: self.n_predict,
            timeout: self.timeout,
            draft_model_path: self.draft_model_path,
            draft_max: self.draft_max,
            max_batch_size: self.max_batch_size,
            max_queue_depth: self.max_queue_depth,
            prefill_chunk_size: resolution.prefill_chunk_size,
            batch_size_conflict: resolution.batch_size_conflict,
            ubatch_size_provided: resolution.ubatch_size_provided,
            enable_preemption: self.enable_preemption,
            preemption_policy: self.preemption_policy,
            no_batch: self.no_batch,
            max_batch_prefill: self.max_batch_prefill,
            decode_storage_backend: self.decode_storage_backend,
            chat_template: self.chat_template,
            chat_template_file: self.chat_template_file,
            enable_slots: resolve_compat_toggle(self.slots, self.no_slots),
            enable_props: self.props,
            enable_metrics: self.metrics || self.metrics_port.is_some(),
            warmup: resolve_compat_toggle(self.warmup, self.no_warmup),
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            seed: resolve_seed(self.seed),
            repeat_last_n: self.repeat_last_n,
            repeat_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_penalty_last_n: self.dry_penalty_last_n,
            dry_sequence_breakers: self.dry_sequence_breakers,
            verbose: self.verbose,
            log_disable: self.log_disable,
            log_file: self.log_file,
            distributed_config: self.distributed_config,
            node_role: self.node_role,
            node_id: self.node_id,
            peers: self.peers,
            pp_layers: self.pp_layers,
            pp_micro_batch_size: self.pp_micro_batch_size,
            pp_auto: self.pp_auto,
            pp_peer: self.pp_peer,
            cluster_discovery: self.cluster_discovery,
            cluster_name: self.cluster_name,
            cluster_peers: self.cluster_peers,
            cluster_discovery_port: self.cluster_discovery_port,
            cluster_control_addr: self.cluster_control_addr,
            cluster_config_out: self.cluster_config_out,
            dry_run: self.dry_run,
            tp_size: self.tp_size,
            tp_moe_mode: self.tp_moe_mode,
            tp_embedding_mode: self.tp_embedding_mode,
            tp_lm_head_mode: self.tp_lm_head_mode,
            vision_cache_size: self.vision_cache_size,
            enable_elastic_pp: self.enable_elastic_pp,
            elastic_pp_drain_timeout: self.elastic_pp_drain_timeout,
            elastic_pp_pressure_fraction: self.elastic_pp_pressure_fraction,
            elastic_pp_cool_down: self.elastic_pp_cool_down,
            metrics_port: self.metrics_port,
            debug_pp_trace: self.debug_pp_trace,
            lang_bias_config: self.lang_bias_config,
            reasoning_budget,
            chat_template_kwargs,
            prompt_cache,
            kv_cache_mode,
        })
    }
}

// ── Issue #484 (B11) — KV cache type split flags ─────────────────────────────

/// Supported K/V combinations and their corresponding `KVCacheMode`.
///
/// | K      | V                 | Mode              |
/// |--------|-------------------|-------------------|
/// | fp16   | fp16              | `Fp16`            |
/// | int8   | int8              | `Int8`            |
/// | fp16   | turbo4 / turbo4-asym | `Turbo4Asym`   |
/// | turbo4 | turbo4            | `Turbo4`          |
/// | fp16   | turbo4-delegated  | `Turbo4Delegated` |
///
/// Any other combination returns an error with a description of valid pairs.
pub fn resolve_kv_cache_mode(
    cache_type_k: Option<&str>,
    cache_type_v: Option<&str>,
    kv_cache_mode_legacy: Option<&str>,
) -> Result<KVCacheMode, String> {
    let have_split = cache_type_k.is_some() || cache_type_v.is_some();
    let have_legacy = kv_cache_mode_legacy.is_some();

    if have_split && have_legacy {
        tracing::warn!(
            "--cache-type-k/--cache-type-v and --kv-cache-mode are both set; \
             --cache-type-k/--cache-type-v take precedence (use one or the other)"
        );
    }

    if have_split {
        // Resolve each side, defaulting the unspecified side to fp16.
        let k_str = cache_type_k.unwrap_or("fp16");
        let v_str = cache_type_v.unwrap_or("fp16");

        let k_mode = k_str
            .parse::<KVCacheMode>()
            .map_err(|_| format!("unrecognised --cache-type-k value \"{k_str}\""))?;
        let v_mode = v_str
            .parse::<KVCacheMode>()
            .map_err(|_| format!("unrecognised --cache-type-v value \"{v_str}\""))?;

        return map_kv_modes_to_cache_mode(k_mode, v_mode);
    }

    if let Some(legacy) = kv_cache_mode_legacy {
        return legacy
            .parse::<KVCacheMode>()
            .map_err(|_| format!("unrecognised --kv-cache-mode value \"{legacy}\""));
    }

    // Default: FP16 (bit-exact baseline).
    Ok(KVCacheMode::Fp16)
}

/// Map a (K-mode, V-mode) pair to the combined `KVCacheMode`.
///
/// Not all combinations are supported. Returns an error with a human-readable
/// message when the pair is unsupported.
fn map_kv_modes_to_cache_mode(k: KVCacheMode, v: KVCacheMode) -> Result<KVCacheMode, String> {
    use KVCacheMode::{Fp16, Int8, Turbo4, Turbo4Asym, Turbo4Delegated};
    match (k, v) {
        (Fp16, Fp16) => Ok(Fp16),
        (Int8, Int8) => Ok(Int8),
        // Asymmetric: FP16 K + Turbo4 V → Turbo4Asym (covers turbo4-asym input on V side)
        (Fp16, Turbo4Asym) | (Fp16, Turbo4) => Ok(Turbo4Asym),
        // Symmetric: Turbo4 K + Turbo4 V → Turbo4 (allowlist-gated inside mlxcel-core)
        (Turbo4, Turbo4) => Ok(Turbo4),
        // Delegated hot/cold: FP16 K + Turbo4Delegated V → Turbo4Delegated
        (Fp16, Turbo4Delegated) => Ok(Turbo4Delegated),
        // Anything else is unsupported.
        (k, v) => Err(format!(
            "unsupported --cache-type-k={k} / --cache-type-v={v} combination; \
             supported pairs:\n  \
             fp16   / fp16              -> fp16 (default)\n  \
             int8   / int8              -> int8\n  \
             fp16   / turbo4            -> fp16+turbo4 (Turbo4Asym)\n  \
             fp16   / turbo4-asym       -> fp16+turbo4 (Turbo4Asym)\n  \
             turbo4 / turbo4            -> turbo4 (symmetric, allowlist-gated)\n  \
             fp16   / turbo4-delegated  -> turbo4-delegated"
        )),
    }
}

/// Apply `LLAMA_ARG_CACHE_TYPE_K` env var fallback to the raw
/// `--cache-type-k` CLI value (issue #484).
///
/// Precedence: CLI flag beats env var. When the CLI flag was not provided
/// (value is `None`) and the env var is set, the env var value is applied.
/// When both are present, the CLI value is kept and an INFO log is emitted.
pub fn env_fallback_cache_type_k(value: &mut Option<String>) {
    apply_optional_string_env_fallback(value, "LLAMA_ARG_CACHE_TYPE_K", "cache-type-k");
}

/// Apply `LLAMA_ARG_CACHE_TYPE_V` env var fallback to the raw
/// `--cache-type-v` CLI value (issue #484).
///
/// Precedence: CLI flag beats env var.  When the CLI flag was not provided
/// (value is `None`) and the env var is set, the env var value is applied.
/// When both are present, the CLI value is kept and an INFO log is emitted.
pub fn env_fallback_cache_type_v(value: &mut Option<String>) {
    apply_optional_string_env_fallback(value, "LLAMA_ARG_CACHE_TYPE_V", "cache-type-v");
}

/// Shared helper: if `value` is `None` and the named env var is set, fill
/// `value` from the env var. If `value` is `Some` (CLI was set) and the env
/// var is also present, log an INFO and keep the CLI value.
fn apply_optional_string_env_fallback(value: &mut Option<String>, key: &str, flag_name: &str) {
    if value.is_some() {
        if std::env::var_os(key).is_some() {
            tracing::info!(
                "{key} env var is set but --{flag_name} CLI flag takes precedence; ignoring {key}"
            );
        }
        return;
    }
    if let Ok(raw) = std::env::var(key) {
        let trimmed = raw.trim().to_string();
        if !trimmed.is_empty() {
            *value = Some(trimmed);
        }
    }
}

/// Apply the `LLAMA_ARG_REASONING_BUDGET` env-var fallback to the raw CLI
/// `--reasoning-budget` value (issue #409).
///
/// Precedence rule (matches existing `LLAMA_ARG_*` precedence helpers):
/// - CLI flag wins over env var.
/// - When CLI is left at the default (`-1`) and the env var is set to a parseable
///   integer, the env value takes effect.
/// - Unparseable env values are logged and ignored (default `-1` is kept).
///
/// This helper mutates the raw `i32` on `Args` / `ServeArgs` before
/// `ServerStartupInput` is constructed so the CLI flag, env var, and
/// request-body paths all converge on the same [`ThinkingBudget::from_raw_i32`]
/// validation point.
pub fn env_fallback_reasoning_budget(cli_value: &mut i32) {
    *cli_value = resolve_server_default_reasoning_budget(*cli_value);
}

/// Apply `LLAMA_ARG_LANG_BIAS` env var fallback to the `lang_bias` field (plan §6.4).
///
/// Precedence rule: CLI flag beats env var.
///
/// - If `args.lang_bias` is `None` and `LLAMA_ARG_LANG_BIAS` is set → fill from env.
/// - If `args.lang_bias` is `Some` (CLI was used) and `LLAMA_ARG_LANG_BIAS` is also set →
///   keep CLI value and log an INFO-level message acknowledging the override.
///
/// Only `LLAMA_ARG_LANG_BIAS` is handled here. Companion env vars
/// (`LLAMA_ARG_LANG_BIAS_POLICY`, `LLAMA_ARG_LANG_BIAS_CONFIG`, etc.) are NOT
/// added because the repo does not yet use `LLAMA_ARG_*` for any of those
/// sibling flags. They are reserved for a follow-up when the convention is
/// established.
pub fn env_fallback_lang_bias(args: &mut LangBiasCliArgs) {
    if args.lang_bias.is_none() {
        if let Ok(v) = std::env::var("LLAMA_ARG_LANG_BIAS") {
            args.lang_bias = Some(v);
        }
    } else if std::env::var_os("LLAMA_ARG_LANG_BIAS").is_some() {
        tracing::info!(
            "LLAMA_ARG_LANG_BIAS env var is set but --lang-bias CLI flag takes precedence; \
             ignoring LLAMA_ARG_LANG_BIAS"
        );
    }
}

/// Apply `LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS` env var fallback to the
/// `include_byte_fragments` field (issue #405).
///
/// Precedence rule: CLI flag beats env var, mirroring the B7 pattern for
/// [`env_fallback_lang_bias`]. The env value is parsed permissively and
/// accepts `true`/`false`/`1`/`0` (case-insensitive). Unparseable values are
/// ignored with a warning so an operator typo does not crash the server.
///
/// - If `args.include_byte_fragments == false` (CLI unset) and the env var is
///   set to a truthy value → flip the field to `true`.
/// - If `args.include_byte_fragments == true` (CLI set) and the env var is
///   also set → keep the CLI value and log an INFO-level message.
pub fn env_fallback_lang_bias_include_byte_fragments(args: &mut LangBiasCliArgs) {
    const KEY: &str = "LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS";
    if !args.include_byte_fragments {
        if let Ok(v) = std::env::var(KEY) {
            match parse_env_bool(&v) {
                Some(true) => {
                    args.include_byte_fragments = true;
                }
                Some(false) => {
                    // Explicitly false — no-op.
                }
                None => {
                    tracing::warn!(
                        "{KEY} env var has unparseable value {:?}; ignoring (expected true/false/1/0)",
                        v
                    );
                }
            }
        }
    } else if std::env::var_os(KEY).is_some() {
        tracing::info!(
            "{KEY} env var is set but --lang-bias-include-byte-fragments CLI flag takes precedence; \
             ignoring {KEY}"
        );
    }
}

/// Parse an env-var bool: `true`/`false`/`1`/`0`, case-insensitive.
///
/// Returns `None` for any other value so the caller can warn once instead of
/// treating garbage as `false`.
fn parse_env_bool(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Resolve a llama-server style `--flag` / `--no-flag` pair into a policy bool.
pub fn resolve_compat_toggle(enabled: bool, disabled: bool) -> bool {
    enabled && !disabled
}

/// Convert the CLI seed sentinel into the runtime representation.
pub fn resolve_seed(seed: i64) -> Option<u64> {
    if seed < 0 { None } else { Some(seed as u64) }
}

// ── Issue #424 — prompt cache config resolution ──────────────────────────────

/// Build a [`crate::server::prompt_cache::PromptCacheConfig`] from the raw
/// values captured from CLI flags and env vars.
///
/// `None` for any numeric field falls back to the compiled-in default from
/// [`crate::server::prompt_cache::PromptCacheConfig::default`].
pub(super) fn build_prompt_cache_config(
    enabled: bool,
    capacity_bytes: Option<usize>,
    max_entries: Option<usize>,
    ttl_seconds: Option<u64>,
    min_prefix: Option<usize>,
) -> crate::server::prompt_cache::PromptCacheConfig {
    use crate::server::prompt_cache::PromptCacheConfig;
    let defaults = PromptCacheConfig::default();
    PromptCacheConfig::new(
        enabled,
        capacity_bytes.unwrap_or(defaults.capacity_bytes),
        max_entries.unwrap_or(defaults.max_entries),
        std::time::Duration::from_secs(ttl_seconds.unwrap_or(defaults.ttl.as_secs())),
        min_prefix.unwrap_or(defaults.min_prefix_tokens),
    )
}

/// Apply `MLXCEL_PROMPT_CACHE_ENABLED` and the llama.cpp-compat
/// `LLAMA_ARG_CACHE_REUSE` alias to the raw CLI bool.
///
/// Precedence:
/// 1. CLI flag — always wins.
/// 2. `MLXCEL_PROMPT_CACHE_ENABLED` env var.
/// 3. `LLAMA_ARG_CACHE_REUSE` env var (llama.cpp compat alias).
/// 4. Compiled-in default (`true`).
///
/// Both env vars accept: `true`/`false`/`1`/`0`/`yes`/`no`/`on`/`off`
/// (case-insensitive). Unparseable values are warn-logged and ignored.
///
/// The `cli_was_set` parameter must be `true` when the binary's clap arg was
/// explicitly provided (not the default), so that a default `true` from clap
/// doesn't shadow an env var that says `false`.
pub fn env_fallback_prompt_cache_enabled(enabled: &mut bool, cli_was_set: bool) {
    const MLXCEL_KEY: &str = "MLXCEL_PROMPT_CACHE_ENABLED";
    const LLAMA_KEY: &str = "LLAMA_ARG_CACHE_REUSE";

    if cli_was_set {
        // CLI explicitly set — log if either env var is also present.
        if std::env::var_os(MLXCEL_KEY).is_some() || std::env::var_os(LLAMA_KEY).is_some() {
            tracing::info!(
                "Prompt-cache-enabled env var(s) are set but the CLI flag takes precedence"
            );
        }
        return;
    }

    // Try MLXCEL_PROMPT_CACHE_ENABLED first.
    if let Ok(raw) = std::env::var(MLXCEL_KEY) {
        match parse_env_bool(&raw) {
            Some(v) => {
                *enabled = v;
                return;
            }
            None => {
                tracing::warn!(
                    "{MLXCEL_KEY} has unparseable value {:?}; ignoring (expected true/false/1/0)",
                    raw
                );
            }
        }
    }

    // Fall back to llama.cpp compat alias LLAMA_ARG_CACHE_REUSE.
    if let Ok(raw) = std::env::var(LLAMA_KEY) {
        match parse_env_bool(&raw) {
            Some(v) => {
                *enabled = v;
            }
            None => {
                tracing::warn!(
                    "{LLAMA_KEY} has unparseable value {:?}; ignoring (expected on/off/true/false/1/0)",
                    raw
                );
            }
        }
    }
}

/// Apply `MLXCEL_PROMPT_CACHE_CAPACITY_BYTES` env var fallback.
///
/// CLI flag beats env var. Unparseable env values are warn-logged and ignored.
pub fn env_fallback_prompt_cache_capacity_bytes(value: &mut Option<usize>) {
    apply_optional_usize_env_fallback(
        value,
        "MLXCEL_PROMPT_CACHE_CAPACITY_BYTES",
        "prompt-cache-capacity-bytes",
    );
}

/// Apply `MLXCEL_PROMPT_CACHE_MAX_ENTRIES` env var fallback.
///
/// CLI flag beats env var. Unparseable env values are warn-logged and ignored.
pub fn env_fallback_prompt_cache_max_entries(value: &mut Option<usize>) {
    apply_optional_usize_env_fallback(
        value,
        "MLXCEL_PROMPT_CACHE_MAX_ENTRIES",
        "prompt-cache-max-entries",
    );
}

/// Apply `MLXCEL_PROMPT_CACHE_TTL` env var fallback (value in seconds).
///
/// CLI flag beats env var. Unparseable env values are warn-logged and ignored.
pub fn env_fallback_prompt_cache_ttl(value: &mut Option<u64>) {
    const KEY: &str = "MLXCEL_PROMPT_CACHE_TTL";
    if value.is_some() {
        if std::env::var_os(KEY).is_some() {
            tracing::info!("{KEY} env var is set but --prompt-cache-ttl CLI flag takes precedence");
        }
        return;
    }
    if let Ok(raw) = std::env::var(KEY) {
        match raw.trim().parse::<u64>() {
            Ok(v) => *value = Some(v),
            Err(_) => tracing::warn!(
                "{KEY} has unparseable value {:?}; ignoring (expected unsigned integer seconds)",
                raw
            ),
        }
    }
}

/// Apply `MLXCEL_PROMPT_CACHE_MIN_PREFIX` env var fallback.
///
/// CLI flag beats env var. Unparseable env values are warn-logged and ignored.
pub fn env_fallback_prompt_cache_min_prefix(value: &mut Option<usize>) {
    apply_optional_usize_env_fallback(
        value,
        "MLXCEL_PROMPT_CACHE_MIN_PREFIX",
        "prompt-cache-min-prefix",
    );
}

/// Shared helper for `Option<usize>` env-var fallbacks.
///
/// If `value` is `Some` (CLI was explicitly set) and the env var is present, an
/// INFO log is emitted. If `value` is `None`, the env var is parsed and applied.
fn apply_optional_usize_env_fallback(value: &mut Option<usize>, key: &str, flag_name: &str) {
    if value.is_some() {
        if std::env::var_os(key).is_some() {
            tracing::info!("{key} env var is set but --{flag_name} CLI flag takes precedence");
        }
        return;
    }
    if let Ok(raw) = std::env::var(key) {
        match raw.trim().parse::<usize>() {
            Ok(v) => *value = Some(v),
            Err(_) => tracing::warn!(
                "{key} has unparseable value {:?}; ignoring (expected unsigned integer)",
                raw
            ),
        }
    }
}

/// Result of resolving the prefill chunk size from the explicit flag and llama-server aliases.
pub struct PrefillChunkResolution {
    /// The effective prefill chunk size to use.
    pub prefill_chunk_size: usize,
    /// True when `--ubatch-size` was provided (always ignored; caller should log a notice).
    pub ubatch_size_provided: bool,
    /// True when both `--batch-size` and an explicit `--prefill-chunk-size` were supplied
    /// with different values (caller should log a warning that `--prefill-chunk-size` wins).
    pub batch_size_conflict: bool,
}

/// Resolve the effective prefill chunk size from the explicit flag and llama-server aliases.
///
/// Resolution rules:
/// - `--ubatch-size` is always ignored on Apple Silicon unified memory (logged at info level).
/// - `--batch-size` is an alias for `--prefill-chunk-size`. If both are provided with
///   different non-default values, `--prefill-chunk-size` takes precedence with a warning.
pub fn resolve_prefill_chunk_size(
    prefill_chunk_size: usize,
    batch_size: Option<usize>,
    ubatch_size: Option<usize>,
) -> PrefillChunkResolution {
    const DEFAULT_PREFILL_CHUNK_SIZE: usize = 512;

    let ubatch_size_provided = ubatch_size.is_some();

    match batch_size {
        None => PrefillChunkResolution {
            prefill_chunk_size,
            ubatch_size_provided,
            batch_size_conflict: false,
        },
        Some(bs) => {
            let explicit_prefill = prefill_chunk_size != DEFAULT_PREFILL_CHUNK_SIZE;
            let conflict = explicit_prefill && bs != prefill_chunk_size;
            PrefillChunkResolution {
                prefill_chunk_size: if explicit_prefill {
                    prefill_chunk_size
                } else {
                    bs
                },
                ubatch_size_provided,
                batch_size_conflict: conflict,
            }
        }
    }
}

#[cfg(test)]
#[path = "cli_input_tests.rs"]
mod tests;

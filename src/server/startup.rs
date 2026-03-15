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

//! Server startup pipeline shared by `mlxcel serve` and `mlxcel-server`.
//!
//! This module keeps process-level side effects such as tracing initialization,
//! chat-template resolution, model warmup, and socket binding out of
//! `server/mod.rs` so the server root can focus on shared types and state.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use tower::Service;

use crate::SamplingConfig;

use super::batch::BatchObservability;
use super::{
    AppState, BatchMetrics, ChatTemplateProcessor, ModelProvider, ServerConfig,
    ServerGenerateOptions, create_app,
};

/// Startup configuration for the server (shared between `mlxcel serve` and `mlxcel-server`).
#[derive(Debug)]
pub struct ServerStartupConfig {
    // Model
    pub model_path: PathBuf,
    pub adapter_path: Option<PathBuf>,
    pub model_alias: Option<String>,

    // Network
    pub host: String,
    pub port: u16,

    // Auth
    pub api_key: Option<String>,
    pub api_key_file: Option<PathBuf>,

    // Limits
    pub n_parallel: usize,
    pub ctx_size: usize,
    pub n_predict: i32, // -1 = unlimited
    pub timeout: u64,

    // Speculative decoding
    pub draft_model_path: Option<PathBuf>,
    pub draft_max: usize,

    // Chat template
    pub chat_template: Option<String>,
    pub chat_template_file: Option<PathBuf>,

    // Endpoint toggles
    pub enable_slots: bool,
    pub enable_props: bool,
    pub enable_metrics: bool,

    // Batch scheduling
    pub max_batch_size: Option<usize>,
    pub max_queue_depth: usize,
    /// Prefill chunk size in tokens (0 = disabled).
    pub prefill_chunk_size: usize,
    /// Set when `--batch-size` and `--prefill-chunk-size` conflict; triggers a startup warning.
    pub batch_size_conflict: bool,
    /// Set when `--ubatch-size` was provided; triggers a startup info notice.
    pub ubatch_size_provided: bool,
    /// Enable preemptive eviction when batch is full.
    pub enable_preemption: bool,
    /// Preemption policy string from CLI (parsed into enum at build_server_config).
    pub preemption_policy: String,
    /// Force the legacy sequential worker, bypassing the batch scheduler.
    pub no_batch: bool,

    // Warmup
    pub warmup: bool,

    // Default sampling
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: Option<u64>,
    pub repeat_last_n: usize,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,

    // DRY
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_penalty_last_n: i32, // -1 = use full context
    pub dry_sequence_breakers: Vec<String>,

    // Logging
    pub verbose: bool,
    pub log_disable: bool,
    pub log_file: Option<PathBuf>,
}

impl Default for ServerStartupConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            adapter_path: None,
            model_alias: None,
            host: "127.0.0.1".to_string(),
            port: 8080,
            api_key: None,
            api_key_file: None,
            n_parallel: 1,
            ctx_size: 0,
            n_predict: -1,
            timeout: 600,
            draft_model_path: None,
            draft_max: 16,
            max_batch_size: None,
            max_queue_depth: 32,
            prefill_chunk_size: 512,
            batch_size_conflict: false,
            ubatch_size_provided: false,
            enable_preemption: false,
            preemption_policy: "longest-first".to_string(),
            no_batch: false,
            chat_template: None,
            chat_template_file: None,
            enable_slots: true,
            enable_props: false,
            enable_metrics: false,
            warmup: true,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.1,
            seed: None,
            repeat_last_n: 64,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: -1,
            dry_sequence_breakers: Vec::new(),
            verbose: false,
            log_disable: false,
            log_file: None,
        }
    }
}

pub(super) fn resolve_default_max_tokens(n_predict: i32) -> usize {
    if n_predict < 0 {
        4096
    } else {
        n_predict as usize
    }
}

pub(super) fn resolve_dry_penalty_last_n(value: i32) -> usize {
    if value < 0 { 0 } else { value as usize }
}

/// Resolve API key from flag or file.
pub(super) fn resolve_api_key(
    api_key: Option<String>,
    api_key_file: Option<&Path>,
) -> Result<Option<String>> {
    if api_key.is_some() {
        return Ok(api_key);
    }
    if let Some(path) = api_key_file {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read API key file: {:?}", path))?;
        let key = content.trim().to_string();
        if key.is_empty() {
            anyhow::bail!("API key file {:?} is empty", path);
        }
        return Ok(Some(key));
    }
    Ok(None)
}

/// Resolve chat template from override string, file, or model's tokenizer metadata.
pub(super) fn resolve_chat_template(
    template_override: Option<&str>,
    template_file: Option<&Path>,
    model_path: &Path,
) -> Result<ChatTemplateProcessor> {
    if let Some(template) = template_override {
        return Ok(ChatTemplateProcessor::with_template(template.to_string()));
    }
    if let Some(path) = template_file {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read chat template file: {:?}", path))?;
        return Ok(ChatTemplateProcessor::with_template(content));
    }
    Ok(ChatTemplateProcessor::from_model_path(model_path)?.unwrap_or_default())
}

/// Parse a preemption policy string from CLI into the enum.
///
/// Accepts "longest-first" (default) and "lowest-priority" (case-insensitive).
fn parse_preemption_policy(s: &str) -> crate::server::PreemptionPolicy {
    match s.trim().to_ascii_lowercase().as_str() {
        "lowest-priority" | "lowestpriority" => crate::server::PreemptionPolicy::LowestPriority,
        _ => crate::server::PreemptionPolicy::LongestFirst,
    }
}

pub(super) fn build_server_config(
    startup: &ServerStartupConfig,
    api_key: Option<String>,
) -> ServerConfig {
    ServerConfig {
        api_key,
        timeout_seconds: startup.timeout,
        model_alias: startup.model_alias.clone(),
        context_size: startup.ctx_size,
        n_parallel: startup.n_parallel,
        enable_slots_endpoint: startup.enable_slots,
        enable_props_endpoint: startup.enable_props,
        enable_metrics_endpoint: startup.enable_metrics,
        default_temperature: startup.temperature,
        default_top_p: startup.top_p,
        default_top_k: startup.top_k,
        default_min_p: startup.min_p,
        default_repetition_penalty: startup.repeat_penalty,
        default_repetition_context_size: startup.repeat_last_n,
        default_max_tokens: resolve_default_max_tokens(startup.n_predict),
        default_seed: startup.seed,
        default_frequency_penalty: startup.frequency_penalty,
        default_presence_penalty: startup.presence_penalty,
        default_dry_multiplier: startup.dry_multiplier,
        default_dry_base: startup.dry_base,
        default_dry_allowed_length: startup.dry_allowed_length,
        default_dry_penalty_last_n: resolve_dry_penalty_last_n(startup.dry_penalty_last_n),
        draft_model_path: startup.draft_model_path.clone(),
        num_draft_tokens: startup.draft_max,
        max_batch_size: startup.max_batch_size.unwrap_or(startup.n_parallel).max(1),
        max_queue_depth: startup.max_queue_depth,
        prefill_chunk_size: startup.prefill_chunk_size,
        enable_preemption: startup.enable_preemption,
        preemption_policy: parse_preemption_policy(&startup.preemption_policy),
        no_batch: startup.no_batch,
    }
}

fn initialize_server_logging(startup: &ServerStartupConfig) -> Result<()> {
    if startup.log_disable {
        return Ok(());
    }

    let filter = if startup.verbose { "debug" } else { "info" };
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter));

    if let Some(ref log_path) = startup.log_file {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .with_context(|| format!("Failed to open log file: {:?}", log_path))?;
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_writer(file)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    }

    Ok(())
}

fn warmup_model(model_provider: &ModelProvider) -> Result<()> {
    model_provider.generate(
        "Hello".to_string(),
        ServerGenerateOptions {
            max_tokens: 1,
            sampling: SamplingConfig::greedy(),
            stop_sequences: None,
            priority: crate::server::batch::RequestPriority::Normal,
        },
    )?;
    Ok(())
}

fn log_endpoints(startup: &ServerStartupConfig, addr: &str) {
    tracing::info!("Starting mlxcel server on {}", addr);
    tracing::info!("Endpoints:");
    tracing::info!("  POST /v1/chat/completions  - OpenAI chat completions");
    tracing::info!("  POST /v1/completions       - OpenAI text completions");
    tracing::info!("  GET  /v1/models            - List models");
    tracing::info!("  POST /completion           - llama-server native completion");
    tracing::info!("  POST /tokenize             - Tokenize text");
    tracing::info!("  POST /detokenize           - Detokenize tokens");
    if startup.enable_props {
        tracing::info!("  GET  /props                - Server properties");
    }
    if startup.enable_slots {
        tracing::info!("  GET  /slots                - Slot status");
    }
    tracing::info!("  GET  /health               - Health check");
}

async fn serve_unix_socket(startup: &ServerStartupConfig, app: axum::Router) -> Result<()> {
    let socket_path = Path::new(&startup.host);

    if socket_path.exists() {
        std::fs::remove_file(socket_path)
            .with_context(|| format!("Failed to remove stale socket: {:?}", socket_path))?;
    }
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create socket directory: {:?}", parent))?;
    }

    log_endpoints(startup, &startup.host);
    let listener = tokio::net::UnixListener::bind(socket_path)
        .with_context(|| format!("Failed to bind Unix socket: {:?}", socket_path))?;

    loop {
        let (socket, _addr) = listener.accept().await?;
        let app = app.clone();
        tokio::spawn(async move {
            let socket = hyper_util::rt::TokioIo::new(socket);
            let hyper_service =
                hyper::service::service_fn(move |request| app.clone().call(request));
            if let Err(err) =
                hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                    .serve_connection(socket, hyper_service)
                    .await
            {
                tracing::debug!("Unix socket connection error: {}", err);
            }
        });
    }
}

async fn serve_tcp(startup: &ServerStartupConfig, app: axum::Router) -> Result<()> {
    let addr = format!("{}:{}", startup.host, startup.port);
    log_endpoints(startup, &addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Start the server with the given startup configuration.
///
/// Shared entry point used by both `mlxcel serve` and `mlxcel-server`.
pub async fn start_server(startup: ServerStartupConfig) -> Result<()> {
    initialize_server_logging(&startup)?;

    if startup.ubatch_size_provided {
        tracing::info!(
            "--ubatch-size is not applicable on Apple Silicon unified memory; ignored"
        );
    }
    if startup.batch_size_conflict {
        tracing::warn!(
            "--batch-size and --prefill-chunk-size both provided; \
             --prefill-chunk-size takes precedence"
        );
    }

    let runtime = crate::initialize_runtime();
    if let Some(invalid) = runtime.invalid_device_override.as_deref() {
        tracing::warn!(
            value = invalid,
            "Ignoring invalid MLXCEL_DEVICE override; using gpu"
        );
    }
    tracing::info!("Runtime device: {}", runtime.device);
    if let Some(max_memory) = runtime.wired_limit_bytes {
        tracing::info!(
            "Wired memory limit: {:.1} GB",
            max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else if runtime.device == crate::RuntimeDevice::Gpu {
        let max_memory = mlxcel_core::gpu_max_memory_size();
        tracing::info!(
            "GPU memory: {:.1} GB (no wired limit)",
            max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    let api_key = resolve_api_key(startup.api_key.clone(), startup.api_key_file.as_deref())?;
    let config = build_server_config(&startup, api_key);
    let chat_template = resolve_chat_template(
        startup.chat_template.as_deref(),
        startup.chat_template_file.as_deref(),
        &startup.model_path,
    )?;
    let tokenizer = crate::tokenizer::load_tokenizer(&startup.model_path)?;

    // Create shared batch metrics and observability that both ModelProvider
    // and AppState read/write.
    let batch_metrics = Arc::new(BatchMetrics::new());
    let batch_observability = Arc::new(BatchObservability::new());

    let model_provider = Arc::new(ModelProvider::new_with_server_config(
        startup.model_path.clone(),
        startup.adapter_path.clone(),
        &config,
        batch_metrics.clone(),
        batch_observability.clone(),
    )?);

    if startup.warmup {
        tracing::info!("Warming up model...");
        match warmup_model(model_provider.as_ref()) {
            Ok(()) => tracing::info!("Warmup complete"),
            Err(err) => tracing::warn!("Warmup failed (non-fatal): {}", err),
        }
    }

    let state = AppState::with_observability(
        model_provider,
        config,
        chat_template,
        tokenizer,
        startup.model_path.clone(),
        batch_metrics,
        batch_observability,
    );
    let app = create_app(state);

    if startup.port == 0 {
        serve_unix_socket(&startup, app).await
    } else {
        serve_tcp(&startup, app).await
    }
}

#[cfg(test)]
#[path = "startup_tests.rs"]
mod tests;

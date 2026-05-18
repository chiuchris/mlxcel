//! OpenAI/llama-server compatible HTTP server for mlxcel

pub mod app;
pub mod chat_template;
pub mod model_provider;
pub mod routes;
pub mod types;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::tokenizer::MlxcelTokenizer;
use anyhow::{Context, Result};
use tokio::sync::Semaphore;
use tower::Service;

pub use app::create_app;
pub use chat_template::ChatTemplateProcessor;
pub use model_provider::{GenerationResult, ModelProvider};

use crate::SamplingConfig;

/// Bridge between server request params and mlxcel-core SamplingConfig
#[derive(Debug, Clone)]
pub struct ServerGenerateOptions {
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_sequences: Option<Vec<String>>,
}

/// Server configuration from CLI arguments
/// Default values match llama-server for compatibility
#[derive(Debug, Clone)]
pub struct ServerConfig {
    // Server options
    pub api_key: Option<String>,
    pub timeout_seconds: u64,

    // Model alias (overrides model_id in API responses)
    pub model_alias: Option<String>,

    // Context size limit (0 = use model default)
    pub context_size: usize,

    // Parallel slots (max concurrent requests)
    pub n_parallel: usize,

    // Endpoint toggles
    pub enable_slots_endpoint: bool,
    pub enable_props_endpoint: bool,
    pub enable_metrics_endpoint: bool,

    // Default sampling parameters (llama-server compatible defaults)
    pub default_temperature: f32,
    pub default_top_p: f32,
    pub default_top_k: i32,
    pub default_min_p: f32,
    pub default_repetition_penalty: f32,
    pub default_repetition_context_size: usize,
    pub default_max_tokens: usize,
    pub default_seed: Option<u64>,

    // OpenAI-compatible penalties
    pub default_frequency_penalty: f32,
    pub default_presence_penalty: f32,

    // Default DRY parameters
    pub default_dry_multiplier: f32,
    pub default_dry_base: f32,
    pub default_dry_allowed_length: usize,
    pub default_dry_penalty_last_n: usize,

    // Speculative decoding
    pub draft_model_path: Option<PathBuf>,
    pub num_draft_tokens: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        // Defaults match llama-server
        Self {
            api_key: None,
            timeout_seconds: 600,
            model_alias: None,
            context_size: 0, // 0 = use model default
            n_parallel: 1,   // Single slot by default (MLX serializes ops anyway)
            enable_slots_endpoint: true,
            enable_props_endpoint: false,
            enable_metrics_endpoint: false,
            default_temperature: 0.8,
            default_top_p: 0.9,
            default_top_k: 40,
            default_min_p: 0.1,
            default_repetition_penalty: 1.0,
            default_repetition_context_size: 64,
            default_max_tokens: 512,
            default_seed: None,
            default_frequency_penalty: 0.0,
            default_presence_penalty: 0.0,
            default_dry_multiplier: 0.0,
            default_dry_base: 1.75,
            default_dry_allowed_length: 2,
            default_dry_penalty_last_n: 0,
            draft_model_path: None,
            num_draft_tokens: 3,
        }
    }
}

/// Server-wide metrics counters (atomic, lock-free)
pub struct Metrics {
    /// Total number of requests received
    pub requests_total: AtomicU64,
    /// Total number of prompt tokens processed
    pub prompt_tokens_total: AtomicU64,
    /// Total number of completion tokens generated
    pub completion_tokens_total: AtomicU64,
    /// Total generation time in milliseconds
    pub generation_time_ms_total: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            prompt_tokens_total: AtomicU64::new(0),
            completion_tokens_total: AtomicU64::new(0),
            generation_time_ms_total: AtomicU64::new(0),
        }
    }

    pub fn record_request(
        &self,
        prompt_tokens: usize,
        completion_tokens: usize,
        generation_time_ms: u64,
    ) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens_total
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.completion_tokens_total
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        self.generation_time_ms_total
            .fetch_add(generation_time_ms, Ordering::Relaxed);
    }
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub model_provider: Arc<ModelProvider>,
    pub config: Arc<ServerConfig>,
    pub chat_template: Arc<ChatTemplateProcessor>,
    /// Tokenizer for tokenize/detokenize endpoints (thread-safe)
    pub tokenizer: Arc<MlxcelTokenizer>,
    /// Model directory path (for props/info)
    pub model_path: PathBuf,
    /// Semaphore to limit concurrent generation requests (parallel slots)
    pub slot_semaphore: Arc<Semaphore>,
    /// Server metrics (request counts, token throughput)
    pub metrics: Arc<Metrics>,
}

impl AppState {
    /// Create AppState with all required components
    pub fn new(
        model_provider: Arc<ModelProvider>,
        config: ServerConfig,
        chat_template: ChatTemplateProcessor,
        tokenizer: MlxcelTokenizer,
        model_path: PathBuf,
    ) -> Self {
        let n_parallel = config.n_parallel.max(1);
        Self {
            model_provider,
            slot_semaphore: Arc::new(Semaphore::new(n_parallel)),
            config: Arc::new(config),
            chat_template: Arc::new(chat_template),
            tokenizer: Arc::new(tokenizer),
            model_path,
            metrics: Arc::new(Metrics::new()),
        }
    }

    /// Get the display model ID (alias if set, otherwise model_provider's ID)
    pub fn display_model_id(&self) -> &str {
        self.config
            .model_alias
            .as_deref()
            .unwrap_or_else(|| self.model_provider.model_id())
    }
}

/// Startup configuration for the server (shared between mlxcel serve and mlxcel-server)
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

/// Resolve API key from flag or file
fn resolve_api_key(api_key: Option<String>, api_key_file: Option<&Path>) -> Result<Option<String>> {
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

/// Resolve chat template from override string, file, or model's tokenizer_config.json
fn resolve_chat_template(
    template_override: Option<&str>,
    template_file: Option<&Path>,
    model_path: &Path,
) -> Result<ChatTemplateProcessor> {
    // 1. Direct template string override
    if let Some(tmpl) = template_override {
        return Ok(ChatTemplateProcessor::with_template(tmpl.to_string()));
    }
    // 2. Template file
    if let Some(path) = template_file {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read chat template file: {:?}", path))?;
        return Ok(ChatTemplateProcessor::with_template(content));
    }
    // 3. From model's tokenizer_config.json
    Ok(ChatTemplateProcessor::from_model_path(model_path)?.unwrap_or_default())
}

/// Start the server with the given startup configuration.
/// Shared entry point used by both `mlxcel serve` and `mlxcel-server`.
pub async fn start_server(startup: ServerStartupConfig) -> Result<()> {
    // 1. Init tracing
    if !startup.log_disable {
        let filter = if startup.verbose { "debug" } else { "info" };
        if let Some(ref log_path) = startup.log_file {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)
                .with_context(|| format!("Failed to open log file: {:?}", log_path))?;
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
                )
                .with_writer(file)
                .init();
        } else {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
                )
                .init();
        }
    }

    // 2. Set wired memory limit for GPU
    let max_memory = mlxcel_core::gpu_max_memory_size();
    mlxcel_core::set_wired_limit(max_memory);
    tracing::info!(
        "Wired memory limit: {:.1} GB",
        max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // 3. Resolve API key
    let api_key = resolve_api_key(startup.api_key, startup.api_key_file.as_deref())?;

    // 4. Build ServerConfig
    let default_max_tokens = if startup.n_predict < 0 {
        512 // default when unlimited
    } else {
        startup.n_predict as usize
    };

    let dry_penalty_last_n = if startup.dry_penalty_last_n < 0 {
        0 // 0 means use full history in the sampling layer
    } else {
        startup.dry_penalty_last_n as usize
    };

    let config = ServerConfig {
        api_key,
        timeout_seconds: startup.timeout,
        model_alias: startup.model_alias,
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
        default_max_tokens,
        default_seed: startup.seed,
        default_frequency_penalty: startup.frequency_penalty,
        default_presence_penalty: startup.presence_penalty,
        default_dry_multiplier: startup.dry_multiplier,
        default_dry_base: startup.dry_base,
        default_dry_allowed_length: startup.dry_allowed_length,
        default_dry_penalty_last_n: dry_penalty_last_n,
        draft_model_path: startup.draft_model_path,
        num_draft_tokens: startup.draft_max,
    };

    // 5. Load chat template
    let chat_template = resolve_chat_template(
        startup.chat_template.as_deref(),
        startup.chat_template_file.as_deref(),
        &startup.model_path,
    )?;

    // 6. Load tokenizer
    let tokenizer = crate::tokenizer::load_tokenizer(&startup.model_path)?;

    // 7. Create ModelProvider
    let model_provider = Arc::new(ModelProvider::new_with_adapter(
        startup.model_path.clone(),
        startup.adapter_path,
    )?);

    // 8. Warmup: generate 1 token and discard
    if startup.warmup {
        tracing::info!("Warming up model...");
        let warmup_result = model_provider.generate(
            "Hello".to_string(),
            ServerGenerateOptions {
                max_tokens: 1,
                sampling: SamplingConfig::greedy(),
                stop_sequences: None,
            },
        );
        match warmup_result {
            Ok(_) => tracing::info!("Warmup complete"),
            Err(e) => tracing::warn!("Warmup failed (non-fatal): {}", e),
        }
    }

    // 9. Create AppState, build app, serve
    let state = AppState::new(
        model_provider,
        config,
        chat_template,
        tokenizer,
        startup.model_path,
    );

    let app = create_app(state);

    let log_endpoints = |addr: &str| {
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
    };

    // Unix socket mode: --host <socket_path> --port 0
    if startup.port == 0 {
        let socket_path = std::path::Path::new(&startup.host);

        // Remove stale socket file if it exists
        if socket_path.exists() {
            std::fs::remove_file(socket_path)
                .with_context(|| format!("Failed to remove stale socket: {:?}", socket_path))?;
        }

        // Ensure parent directory exists
        if let Some(parent) = socket_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create socket directory: {:?}", parent))?;
        }

        log_endpoints(&startup.host);

        let listener = tokio::net::UnixListener::bind(socket_path)
            .with_context(|| format!("Failed to bind Unix socket: {:?}", socket_path))?;

        // axum 0.7 doesn't support UnixListener directly — use hyper-util accept loop
        loop {
            let (socket, _addr) = listener.accept().await?;
            let app = app.clone();
            tokio::spawn(async move {
                let socket = hyper_util::rt::TokioIo::new(socket);
                let hyper_service = hyper::service::service_fn(move |request| {
                    app.clone().call(request)
                });
                if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                    hyper_util::rt::TokioExecutor::new(),
                )
                .serve_connection(socket, hyper_service)
                .await
                {
                    tracing::debug!("Unix socket connection error: {}", e);
                }
            });
        }
    } else {
        let addr = format!("{}:{}", startup.host, startup.port);
        log_endpoints(&addr);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
    }

    Ok(())
}

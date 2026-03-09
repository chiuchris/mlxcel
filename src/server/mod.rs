//! OpenAI/llama-server compatible HTTP server for mlxcel

pub mod app;
pub mod chat_template;
pub mod model_provider;
pub mod routes;
mod startup;
pub mod types;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::tokenizer::MlxcelTokenizer;
use tokio::sync::Semaphore;

pub use app::create_app;
pub use chat_template::ChatTemplateProcessor;
pub use model_provider::{GenerationResult, ModelProvider};
pub use startup::{ServerStartupConfig, start_server};

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

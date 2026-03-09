//! Shared server state and metrics containers.
//!
//! Route handlers only need these coordination structs; separating them from
//! startup/config policy keeps request handling focused on state access rather
//! than construction details.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::Semaphore;

use crate::tokenizer::MlxcelTokenizer;

use super::{ChatTemplateProcessor, ModelProvider, ServerConfig};

/// Server-wide metrics counters (atomic, lock-free).
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub prompt_tokens_total: AtomicU64,
    pub completion_tokens_total: AtomicU64,
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

/// Shared application state passed into route handlers.
#[derive(Clone)]
pub struct AppState {
    pub model_provider: Arc<ModelProvider>,
    pub config: Arc<ServerConfig>,
    pub chat_template: Arc<ChatTemplateProcessor>,
    /// Tokenizer for tokenize/detokenize endpoints (thread-safe).
    pub tokenizer: Arc<MlxcelTokenizer>,
    /// Model directory path (for props/info).
    pub model_path: PathBuf,
    /// Semaphore limiting concurrent generation requests.
    pub slot_semaphore: Arc<Semaphore>,
    /// Server metrics (request counts, token throughput).
    pub metrics: Arc<Metrics>,
}

impl AppState {
    /// Create `AppState` with all required components.
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

    /// Get the display model ID (alias if set, otherwise model provider ID).
    pub fn display_model_id(&self) -> &str {
        self.config
            .model_alias
            .as_deref()
            .unwrap_or_else(|| self.model_provider.model_id())
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;

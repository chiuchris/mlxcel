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

//! Shared server state and metrics containers.
//!
//! Route handlers only need these coordination structs; separating them from
//! startup/config policy keeps request handling focused on state access rather
//! than construction details.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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

/// Batch-level metrics updated by the scheduler thread.
///
/// All fields are atomic for lock-free reads from HTTP handlers and
/// writes from the single scheduler thread.
pub struct BatchMetrics {
    /// Number of sequences currently in the active decode batch.
    pub active_count: AtomicUsize,
    /// Number of requests waiting in the prefill queue.
    pub queue_depth: AtomicUsize,
    /// Cumulative number of sequences that have completed generation.
    pub total_sequences_processed: AtomicU64,
    /// Cumulative number of tokens generated across all sequences.
    pub total_tokens_generated: AtomicU64,
    /// Cumulative number of preemptive evictions.
    pub preemptions_total: AtomicU64,
}

impl Default for BatchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchMetrics {
    pub fn new() -> Self {
        Self {
            active_count: AtomicUsize::new(0),
            queue_depth: AtomicUsize::new(0),
            total_sequences_processed: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
            preemptions_total: AtomicU64::new(0),
        }
    }

    /// Current number of active decode sequences.
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Current prefill queue depth.
    pub fn queue_depth(&self) -> usize {
        self.queue_depth.load(Ordering::Relaxed)
    }

    /// Update active count (called by scheduler thread).
    pub fn set_active_count(&self, count: usize) {
        self.active_count.store(count, Ordering::Relaxed);
    }

    /// Update queue depth (called by scheduler thread).
    pub fn set_queue_depth(&self, depth: usize) {
        self.queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Record a completed sequence (called by scheduler thread).
    pub fn record_sequence_completed(&self, tokens_generated: usize) {
        self.total_sequences_processed
            .fetch_add(1, Ordering::Relaxed);
        self.total_tokens_generated
            .fetch_add(tokens_generated as u64, Ordering::Relaxed);
    }

    /// Record a preemptive eviction (called by scheduler thread).
    pub fn record_preemption(&self) {
        self.preemptions_total.fetch_add(1, Ordering::Relaxed);
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
    /// Batch-level metrics (active sequences, queue depth) for admission
    /// control and status reporting.
    pub batch_metrics: Arc<BatchMetrics>,
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
        batch_metrics: Arc<BatchMetrics>,
    ) -> Self {
        Self {
            model_provider,
            config: Arc::new(config),
            chat_template: Arc::new(chat_template),
            tokenizer: Arc::new(tokenizer),
            model_path,
            batch_metrics,
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

    /// Check whether the server can accept a new request based on queue depth.
    ///
    /// Returns `true` if the current queue depth is below `max_queue_depth`.
    pub fn can_accept_request(&self) -> bool {
        self.batch_metrics.queue_depth() < self.config.max_queue_depth
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;

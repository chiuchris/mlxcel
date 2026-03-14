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

//! Model provider with dedicated generation thread
//!
//! Since MLX operations are not thread-safe, we run the model on a dedicated
//! thread and communicate via channels.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

use anyhow::Result;

use crate::server::ServerGenerateOptions;
use crate::server::batch::BatchObservability;
use crate::server::state::BatchMetrics;

/// Request to the model thread
pub enum ModelRequest {
    Generate {
        prompt: String,
        options: ServerGenerateOptions,
        /// Raw image bytes for VLM (empty for text-only)
        images: Vec<Vec<u8>>,
        response_tx: mpsc::Sender<GenerateEvent>,
    },
    Shutdown,
}

/// Events from generation
pub enum GenerateEvent {
    Token(String),
    Done(GenerationResult),
    Error(String),
}

/// Result of a generation
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub generation_time_ms: u64,
    pub prompt_eval_ms: u64,
    pub generation_only_ms: u64,
    pub finish_reason: String,
}

#[path = "model_worker.rs"]
pub(crate) mod model_worker;

/// Thread-safe model provider using channels
pub struct ModelProvider {
    request_tx: mpsc::Sender<ModelRequest>,
    model_id: String,
    created_at: i64,
    loaded: Arc<AtomicBool>,
    batch_metrics: Arc<BatchMetrics>,
    batch_observability: Arc<BatchObservability>,
    _worker_handle: thread::JoinHandle<()>,
}

impl ModelProvider {
    /// Create and start a new model provider
    pub fn new(model_path: PathBuf) -> Result<Self> {
        Self::new_with_adapter(model_path, None)
    }

    /// Create and start a new model provider with an optional LoRA adapter.
    ///
    /// Uses default batch settings (max_batch_size=1, max_queue_depth=1024).
    pub fn new_with_adapter(model_path: PathBuf, adapter_path: Option<PathBuf>) -> Result<Self> {
        Self::new_with_batch_config(model_path, adapter_path, 1, 1024)
    }

    /// Create and start a new model provider with batch scheduling config.
    pub fn new_with_batch_config(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        max_batch_size: usize,
        max_queue_depth: usize,
    ) -> Result<Self> {
        let batch_metrics = Arc::new(BatchMetrics::new());
        Self::new_with_metrics(
            model_path,
            adapter_path,
            max_batch_size,
            max_queue_depth,
            batch_metrics,
        )
    }

    /// Create and start a new model provider with full server config.
    ///
    /// When `config.no_batch` is true, the legacy sequential worker is spawned
    /// instead of the batch scheduler, regardless of `max_batch_size`.
    pub fn new_with_server_config(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        config: &crate::server::ServerConfig,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self> {
        if config.no_batch {
            Self::new_with_legacy_worker(
                model_path,
                adapter_path,
                batch_metrics,
                batch_observability,
            )
        } else {
            Self::new_with_full_config(
                model_path,
                adapter_path,
                config.max_batch_size,
                config.max_queue_depth,
                config.prefill_chunk_size,
                config.enable_preemption,
                config.preemption_policy,
                batch_metrics,
                batch_observability,
            )
        }
    }

    /// Create and start a new model provider using the legacy sequential worker.
    ///
    /// This is activated by `--no-batch`. The worker uses the `BatchScheduler`
    /// in size-1 mode (no interleaving, no chunked prefill) which is equivalent
    /// to the pre-scheduler sequential request loop.
    pub(crate) fn new_with_legacy_worker(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self> {
        let model_id = model_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let created_at = chrono::Utc::now().timestamp();
        let (request_tx, request_rx) = mpsc::channel::<ModelRequest>();
        let loaded = Arc::new(AtomicBool::new(false));
        let loaded_clone = loaded.clone();
        let worker_model_id = model_id.clone();
        let metrics_clone = batch_metrics.clone();
        let obs_clone = batch_observability.clone();

        let worker_handle = model_worker::spawn_legacy_model_worker(
            model_path,
            adapter_path,
            request_rx,
            loaded_clone,
            worker_model_id,
            metrics_clone,
            obs_clone,
        );

        Ok(Self {
            request_tx,
            model_id,
            created_at,
            loaded,
            batch_metrics,
            batch_observability,
            _worker_handle: worker_handle,
        })
    }

    /// Create and start a new model provider with full scheduler config
    /// and shared batch metrics.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_full_config(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        max_batch_size: usize,
        max_queue_depth: usize,
        prefill_chunk_size: usize,
        enable_preemption: bool,
        preemption_policy: crate::server::config::PreemptionPolicy,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self> {
        let model_id = model_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let created_at = chrono::Utc::now().timestamp();
        let (request_tx, request_rx) = mpsc::channel::<ModelRequest>();
        let loaded = Arc::new(AtomicBool::new(false));
        let loaded_clone = loaded.clone();
        let worker_model_id = model_id.clone();
        let metrics_clone = batch_metrics.clone();
        let obs_clone = batch_observability.clone();

        let sched_config = model_worker::WorkerSchedulerConfig {
            max_batch_size,
            max_queue_depth,
            prefill_chunk_size,
            enable_preemption,
            preemption_policy,
        };

        let worker_handle = model_worker::spawn_model_worker_with_batch_config(
            model_path,
            adapter_path,
            request_rx,
            loaded_clone,
            worker_model_id,
            sched_config,
            metrics_clone,
            obs_clone,
        );

        Ok(Self {
            request_tx,
            model_id,
            created_at,
            loaded,
            batch_metrics,
            batch_observability,
            _worker_handle: worker_handle,
        })
    }

    /// Create and start a new model provider with shared batch metrics.
    pub fn new_with_metrics(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        max_batch_size: usize,
        max_queue_depth: usize,
        batch_metrics: Arc<BatchMetrics>,
    ) -> Result<Self> {
        let model_id = model_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let created_at = chrono::Utc::now().timestamp();

        // Create channel for requests
        let (request_tx, request_rx) = mpsc::channel::<ModelRequest>();

        // Shared loaded flag
        let loaded = Arc::new(AtomicBool::new(false));
        let loaded_clone = loaded.clone();

        // Clone model_id for the worker thread
        let worker_model_id = model_id.clone();
        let metrics_clone = batch_metrics.clone();
        let batch_observability = Arc::new(BatchObservability::new());
        let obs_clone = batch_observability.clone();

        let sched_config = model_worker::WorkerSchedulerConfig {
            max_batch_size,
            max_queue_depth,
            prefill_chunk_size: 0,
            enable_preemption: false,
            preemption_policy: crate::server::config::PreemptionPolicy::default(),
        };

        let worker_handle = model_worker::spawn_model_worker_with_batch_config(
            model_path,
            adapter_path,
            request_rx,
            loaded_clone,
            worker_model_id,
            sched_config,
            metrics_clone,
            obs_clone,
        );

        Ok(Self {
            request_tx,
            model_id,
            created_at,
            loaded,
            batch_metrics,
            batch_observability,
            _worker_handle: worker_handle,
        })
    }

    /// Get a reference to the shared batch metrics.
    pub fn batch_metrics(&self) -> &Arc<BatchMetrics> {
        &self.batch_metrics
    }

    /// Get a reference to the shared batch observability counters.
    pub fn batch_observability(&self) -> &Arc<BatchObservability> {
        &self.batch_observability
    }

    /// Get model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get creation timestamp
    pub fn created_at(&self) -> i64 {
        self.created_at
    }

    /// Check if model is loaded and ready for inference
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::Acquire)
    }

    /// Generate text and return the full result
    pub fn generate(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
    ) -> Result<GenerationResult> {
        self.generate_with_images(prompt, options, Vec::new())
    }

    /// Generate text with optional images and return the full result
    pub fn generate_with_images(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
    ) -> Result<GenerationResult> {
        let response_rx = self.send_generate_request(prompt, options, images)?;
        drain_generation_events(response_rx, |_| {})
    }

    /// Generate text with streaming callback
    pub fn generate_streaming<F>(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        callback: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(String),
    {
        self.generate_streaming_with_images(prompt, options, Vec::new(), callback)
    }

    /// Generate text with optional images and streaming callback
    pub fn generate_streaming_with_images<F>(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        callback: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(String),
    {
        let response_rx = self.send_generate_request(prompt, options, images)?;
        drain_generation_events(response_rx, callback)
    }

    fn send_generate_request(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
    ) -> Result<mpsc::Receiver<GenerateEvent>> {
        let (response_tx, response_rx) = mpsc::channel();

        self.request_tx
            .send(ModelRequest::Generate {
                prompt,
                options,
                images,
                response_tx,
            })
            .map_err(|e| anyhow::anyhow!("Failed to send request: {}", e))?;

        Ok(response_rx)
    }
}

fn send_shutdown_signal(request_tx: &mpsc::Sender<ModelRequest>) -> bool {
    request_tx.send(ModelRequest::Shutdown).is_ok()
}

fn drain_generation_events<F>(
    response_rx: mpsc::Receiver<GenerateEvent>,
    mut on_token: F,
) -> Result<GenerationResult>
where
    F: FnMut(String),
{
    loop {
        match response_rx.recv() {
            Ok(GenerateEvent::Token(token)) => on_token(token),
            Ok(GenerateEvent::Done(result)) => return Ok(result),
            Ok(GenerateEvent::Error(err)) => return Err(anyhow::anyhow!(err)),
            Err(_) => return Err(anyhow::anyhow!("Response channel closed")),
        }
    }
}

impl Drop for ModelProvider {
    fn drop(&mut self) {
        let _ = send_shutdown_signal(&self.request_tx);
    }
}

#[cfg(test)]
#[path = "model_provider_tests.rs"]
mod tests;

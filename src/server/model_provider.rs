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
mod model_worker;

/// Thread-safe model provider using channels
pub struct ModelProvider {
    request_tx: mpsc::Sender<ModelRequest>,
    model_id: String,
    created_at: i64,
    loaded: Arc<AtomicBool>,
    _worker_handle: thread::JoinHandle<()>,
}

impl ModelProvider {
    /// Create and start a new model provider
    pub fn new(model_path: PathBuf) -> Result<Self> {
        Self::new_with_adapter(model_path, None)
    }

    /// Create and start a new model provider with an optional LoRA adapter
    pub fn new_with_adapter(model_path: PathBuf, adapter_path: Option<PathBuf>) -> Result<Self> {
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

        let worker_handle = model_worker::spawn_model_worker(
            model_path,
            adapter_path,
            request_rx,
            loaded_clone,
            worker_model_id,
        );

        Ok(Self {
            request_tx,
            model_id,
            created_at,
            loaded,
            _worker_handle: worker_handle,
        })
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
        // Send shutdown signal
        let _ = self.request_tx.send(ModelRequest::Shutdown);
    }
}

// ModelProvider is Send + Sync because it only contains channels and atomics
unsafe impl Send for ModelProvider {}
unsafe impl Sync for ModelProvider {}

#[cfg(test)]
#[path = "model_provider_tests.rs"]
mod tests;

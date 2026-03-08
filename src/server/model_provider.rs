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
use crate::vlm_runtime::prepare_and_compute_vlm_embeddings;

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

        // Spawn worker thread that owns the model
        let worker_handle = thread::spawn(move || {
            tracing::info!("Model worker thread starting, loading model...");

            // Load model on this thread (with optional adapter)
            let result = if let Some(adapter) = adapter_path {
                tracing::info!("Loading LoRA adapter from {:?}", adapter);
                crate::load_model_with_adapter(&model_path, &adapter)
            } else {
                crate::load_model(&model_path)
            };
            let (model, tokenizer) = match result {
                Ok((m, t)) => {
                    tracing::info!("Model {} loaded successfully", worker_model_id);
                    loaded_clone.store(true, Ordering::Release);
                    (m, t)
                }
                Err(e) => {
                    tracing::error!("Failed to load model: {}", e);
                    return;
                }
            };

            // Read EOS tokens from generation_config.json
            let config_eos = crate::read_eos_token_ids(&model_path);
            if !config_eos.is_empty() {
                tracing::info!("EOS tokens from config: {:?}", config_eos);
            }

            // Create generator (kept alive across requests)
            let num_layers = crate::LanguageModel::num_layers(&model);
            let mut generator = crate::CxxGenerator::new(num_layers);

            // Process requests
            loop {
                match request_rx.recv() {
                    Ok(ModelRequest::Generate {
                        prompt,
                        options,
                        images,
                        response_tx,
                    }) => {
                        let start = std::time::Instant::now();

                        // Tokenize prompt
                        let token_ids = match tokenizer.encode(prompt.as_str(), true) {
                            Ok(ids) => ids,
                            Err(e) => {
                                let _ = response_tx.send(GenerateEvent::Error(format!(
                                    "Tokenization error: {}",
                                    e
                                )));
                                continue;
                            }
                        };
                        let mut prompt_tokens: Vec<i32> =
                            token_ids.iter().map(|&x| x as i32).collect();
                        let prompt_token_count = prompt_tokens.len();

                        // Reset generator state between requests
                        // Uses reset_with_model to also reset model-internal caches
                        // (e.g. sliding window, SSM, hybrid models)
                        generator.reset_with_model(&model);

                        let max_tokens = options.max_tokens;

                        // Inject config-based EOS tokens into sampling config
                        let mut sampling = options.sampling.clone();
                        for &id in &config_eos {
                            if !sampling.stop_token_ids.contains(&id) {
                                sampling.stop_token_ids.push(id);
                            }
                        }

                        // Check if this is a VLM request with images
                        let vlm_embeddings = if !images.is_empty() && model.is_vlm() {
                            // Decode raw bytes to DynamicImage
                            let decoded_images: Vec<image::DynamicImage> = images
                                .iter()
                                .filter_map(|bytes| {
                                    image::load_from_memory(bytes)
                                        .map_err(|e| {
                                            tracing::warn!("Failed to decode image: {}", e);
                                            e
                                        })
                                        .ok()
                                })
                                .collect();

                            if decoded_images.is_empty() {
                                let _ = response_tx.send(GenerateEvent::Error(
                                    "Failed to decode any images".to_string(),
                                ));
                                continue;
                            }

                            match prepare_and_compute_vlm_embeddings(
                                &model,
                                &mut prompt_tokens,
                                &prompt,
                                &decoded_images,
                                |text, add_special| {
                                    tokenizer
                                        .encode(text, add_special)
                                        .unwrap_or_default()
                                        .iter()
                                        .map(|&t| t as i32)
                                        .collect()
                                },
                            ) {
                                Ok(prepared) => prepared.map(|prepared| prepared.embeddings),
                                Err(err) => {
                                    let _ = response_tx.send(GenerateEvent::Error(err.to_string()));
                                    continue;
                                }
                            }
                        } else {
                            None
                        };

                        // Context-aware streaming decode state
                        let mut all_ids: Vec<u32> =
                            prompt_tokens.iter().map(|&x| x as u32).collect();
                        let mut prev_decoded_len = tokenizer
                            .decode(
                                &prompt_tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                                false,
                            )
                            .unwrap_or_default()
                            .len();
                        let mut generated_text = String::new();
                        let mut completion_tokens = 0usize;
                        let mut first_token_time: Option<std::time::Instant> = None;

                        let tx_clone = response_tx.clone();
                        let tokenizer_ref = &tokenizer;

                        let on_token = |token_id: i32| {
                            if first_token_time.is_none() {
                                first_token_time = Some(std::time::Instant::now());
                            }
                            completion_tokens += 1;
                            all_ids.push(token_id as u32);

                            // Context-aware decode: decode all IDs, diff with previous
                            let full_text =
                                tokenizer_ref.decode(&all_ids, false).unwrap_or_default();
                            let new_text = &full_text[prev_decoded_len..];

                            if !new_text.is_empty() {
                                generated_text.push_str(new_text);
                                let _ = tx_clone.send(GenerateEvent::Token(new_text.to_string()));
                                prev_decoded_len = full_text.len();
                            }

                            // Return true to continue generation
                            true
                        };

                        // Use VLM or standard generation
                        if let Some(ref embeddings) = vlm_embeddings {
                            let mask_ref = embeddings
                                .attention_mask_4d
                                .as_ref()
                                .map(|m| m.as_ref().unwrap());
                            generator.generate_streaming_with_embeddings(
                                &model,
                                &prompt_tokens,
                                Some(embeddings.inputs_embeds.as_ref().unwrap()),
                                mask_ref,
                                max_tokens,
                                &sampling,
                                on_token,
                            );
                        } else {
                            generator.generate_streaming(
                                &model,
                                &prompt_tokens,
                                max_tokens,
                                &sampling,
                                on_token,
                            );
                        }

                        let elapsed = start.elapsed();
                        let prompt_eval_ms = first_token_time
                            .map(|t| (t - start).as_millis() as u64)
                            .unwrap_or(elapsed.as_millis() as u64);
                        let generation_only_ms = elapsed.as_millis() as u64 - prompt_eval_ms;

                        let finish_reason = if completion_tokens >= max_tokens {
                            "length".to_string()
                        } else {
                            "stop".to_string()
                        };

                        let _ = response_tx.send(GenerateEvent::Done(GenerationResult {
                            text: generated_text,
                            prompt_tokens: prompt_token_count,
                            completion_tokens,
                            generation_time_ms: elapsed.as_millis() as u64,
                            prompt_eval_ms,
                            generation_only_ms,
                            finish_reason,
                        }));
                    }
                    Ok(ModelRequest::Shutdown) => {
                        tracing::info!("Model worker thread shutting down");
                        break;
                    }
                    Err(_) => {
                        // Channel closed, exit
                        tracing::info!("Request channel closed, worker exiting");
                        break;
                    }
                }
            }
        });

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
        let (response_tx, response_rx) = mpsc::channel();

        self.request_tx
            .send(ModelRequest::Generate {
                prompt,
                options,
                images,
                response_tx,
            })
            .map_err(|e| anyhow::anyhow!("Failed to send request: {}", e))?;

        // Collect all tokens and wait for done
        loop {
            match response_rx.recv() {
                Ok(GenerateEvent::Token(_)) => {
                    // Ignore tokens for non-streaming
                }
                Ok(GenerateEvent::Done(r)) => {
                    return Ok(r);
                }
                Ok(GenerateEvent::Error(e)) => {
                    return Err(anyhow::anyhow!(e));
                }
                Err(_) => {
                    return Err(anyhow::anyhow!("Response channel closed"));
                }
            }
        }
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
        mut callback: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(String),
    {
        let (response_tx, response_rx) = mpsc::channel();

        self.request_tx
            .send(ModelRequest::Generate {
                prompt,
                options,
                images,
                response_tx,
            })
            .map_err(|e| anyhow::anyhow!("Failed to send request: {}", e))?;

        // Process events
        loop {
            match response_rx.recv() {
                Ok(GenerateEvent::Token(token)) => {
                    callback(token);
                }
                Ok(GenerateEvent::Done(r)) => {
                    return Ok(r);
                }
                Ok(GenerateEvent::Error(e)) => {
                    return Err(anyhow::anyhow!(e));
                }
                Err(_) => {
                    return Err(anyhow::anyhow!("Response channel closed"));
                }
            }
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

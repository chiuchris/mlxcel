//! Model provider with dedicated generation thread
//!
//! Since MLX operations are not thread-safe, we run the model on a dedicated
//! thread and communicate via channels.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

use anyhow::Result;

use crate::qwen_vl::insert_qwen_vl_image_tokens;
use crate::server::ServerGenerateOptions;
use crate::vision::processors::ImageProcessor;
use crate::vlm_prompt::apply_image_token_blocks;

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

                            if let Some(info) = model.qwen_vl_prompt_info() {
                                let (pixel_values, grid_thw) =
                                    info.processor.preprocess_with_grid(&decoded_images);
                                let _ = insert_qwen_vl_image_tokens(
                                    &mut prompt_tokens,
                                    &grid_thw,
                                    info.spatial_merge_size,
                                    info.vision_start_token_id,
                                    info.image_token_id,
                                );

                                let input_ids_arr = mlxcel_core::from_slice_i32(
                                    &prompt_tokens,
                                    &[1, prompt_tokens.len() as i32],
                                );
                                let merged = model
                                    .qwen_vl_input_embeddings(
                                        &input_ids_arr,
                                        &pixel_values,
                                        &grid_thw,
                                    )
                                    .expect("Qwen-VL prompt info without matching model");
                                Some(merged)
                            } else if let Some(gemma3n_vl) = model.gemma3n_vl_model() {
                                let _ = model.image_token_block_info().and_then(|info| {
                                    apply_image_token_blocks(
                                        &mut prompt_tokens,
                                        info,
                                        decoded_images.len(),
                                    )
                                });

                                let pixel_values = gemma3n_vl.processor.preprocess(&decoded_images);
                                let input_ids_arr = mlxcel_core::from_slice_i32(
                                    &prompt_tokens,
                                    &[1, prompt_tokens.len() as i32],
                                );
                                let merged =
                                    gemma3n_vl.get_input_embeddings(&input_ids_arr, &pixel_values);
                                Some(merged)
                            } else if let Some(molmo2) = model.molmo2_vl_model() {
                                // Molmo2 VLM: multi-scale preprocessing + additive merge
                                if let Some(img) = decoded_images.first() {
                                    let proc_out = molmo2.processor.preprocess_image(img);

                                    // Generate image token string and re-tokenize
                                    let image_token_str =
                                        molmo2.processor.get_image_tokens(&proc_out.image_grid);
                                    let mut text = prompt.clone();
                                    if text.contains("<|image|>") {
                                        text = text.replace("<|image|>", &image_token_str);
                                    } else {
                                        text = format!("{}{}", image_token_str, text);
                                    }
                                    prompt_tokens = tokenizer
                                        .encode(&text, true)
                                        .unwrap_or_default()
                                        .iter()
                                        .map(|&t| t as i32)
                                        .collect();

                                    let pixel_values = mlxcel_core::from_slice_f32(
                                        &proc_out.pixel_values,
                                        &proc_out.pixel_values_shape,
                                    );
                                    let image_token_pooling = mlxcel_core::from_slice_i32(
                                        &proc_out.image_token_pooling,
                                        &proc_out.image_token_pooling_shape,
                                    );
                                    let image_grids =
                                        mlxcel_core::from_slice_i32(&proc_out.image_grid, &[4]);
                                    let image_num_crops = mlxcel_core::from_slice_i32(
                                        &[proc_out.image_num_crops],
                                        &[1],
                                    );
                                    let input_ids_arr = mlxcel_core::from_slice_i32(
                                        &prompt_tokens,
                                        &[1, prompt_tokens.len() as i32],
                                    );
                                    let merged = molmo2.get_input_embeddings(
                                        &input_ids_arr,
                                        &pixel_values,
                                        &image_token_pooling,
                                        &image_grids,
                                        &image_num_crops,
                                    );
                                    Some(merged)
                                } else {
                                    None
                                }
                            } else if let Some(phi3v) = model.phi3_vl_model() {
                                // Phi-3V VLM: split text around <|image_N|> tags,
                                // tokenize chunks, interleave with negative IDs
                                let num_images = decoded_images.len();

                                // Ensure <|image_N|> tags are in the prompt text
                                let mut text = prompt.clone();
                                let has_image_tags = (1..=num_images)
                                    .any(|i| text.contains(&format!("<|image_{}|>", i)));

                                if !has_image_tags && num_images > 0 {
                                    let image_tags: String = (1..=num_images)
                                        .map(|i| format!("<|image_{}|>\n", i))
                                        .collect();
                                    if let Some(pos) = text.find("<|user|>\n") {
                                        text.insert_str(pos + "<|user|>\n".len(), &image_tags);
                                    } else {
                                        text = format!("{}{}", image_tags, text);
                                    }
                                }

                                // Collect image tag positions sorted by position
                                let mut tag_positions: Vec<(usize, usize, usize)> = Vec::new();
                                for n in 1..=num_images {
                                    let tag = format!("<|image_{}|>", n);
                                    let mut search_from = 0;
                                    while let Some(pos) = text[search_from..].find(&tag) {
                                        let abs_pos = search_from + pos;
                                        tag_positions.push((abs_pos, abs_pos + tag.len(), n));
                                        search_from = abs_pos + tag.len();
                                    }
                                }
                                tag_positions.sort_by_key(|&(start, _, _)| start);

                                if !tag_positions.is_empty() {
                                    let mut new_tokens: Vec<i32> = Vec::new();
                                    let mut last_end = 0;

                                    for (chunk_idx, &(tag_start, tag_end, image_num)) in
                                        tag_positions.iter().enumerate()
                                    {
                                        let before = &text[last_end..tag_start];
                                        if !before.is_empty() {
                                            let add_special = chunk_idx == 0 && last_end == 0;
                                            let tokens = tokenizer
                                                .encode(before, add_special)
                                                .unwrap_or_default();
                                            new_tokens.extend(tokens.iter().map(|&t| t as i32));
                                        }

                                        if image_num <= num_images {
                                            let (w, h) = (
                                                decoded_images[image_num - 1].width(),
                                                decoded_images[image_num - 1].height(),
                                            );
                                            let num_img_tokens =
                                                phi3v.processor.calc_num_image_tokens(w, h);
                                            let neg_id = -(image_num as i32);
                                            for _ in 0..num_img_tokens {
                                                new_tokens.push(neg_id);
                                            }
                                        }

                                        last_end = tag_end;
                                    }

                                    let after = &text[last_end..];
                                    if !after.is_empty() {
                                        let tokens =
                                            tokenizer.encode(after, false).unwrap_or_default();
                                        new_tokens.extend(tokens.iter().map(|&t| t as i32));
                                    }

                                    prompt_tokens = new_tokens;
                                }

                                let (pixel_values, image_sizes) =
                                    phi3v.processor.preprocess(&decoded_images);
                                let input_ids_arr = mlxcel_core::from_slice_i32(
                                    &prompt_tokens,
                                    &[1, prompt_tokens.len() as i32],
                                );
                                let merged = phi3v.get_input_embeddings(
                                    &input_ids_arr,
                                    &pixel_values,
                                    &image_sizes,
                                );
                                Some(merged)
                            } else if let Some(vision_module) = model.vision_module() {
                                let _ = model.image_token_block_info().and_then(|info| {
                                    apply_image_token_blocks(
                                        &mut prompt_tokens,
                                        info,
                                        decoded_images.len(),
                                    )
                                });

                                let pixel_values =
                                    vision_module.processor.preprocess(&decoded_images);
                                let mask = mlxcel_core::ones(
                                    &[1, prompt_tokens.len() as i32],
                                    mlxcel_core::dtype::INT32,
                                );
                                let input_ids_arr = mlxcel_core::from_slice_i32(
                                    &prompt_tokens,
                                    &[1, prompt_tokens.len() as i32],
                                );
                                let merged = vision_module.get_input_embeddings(
                                    &model,
                                    &input_ids_arr,
                                    Some(&pixel_values),
                                    &mask,
                                );
                                Some(merged)
                            } else {
                                None
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

//! Chat completions endpoint

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use futures::stream::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::sampling::{ResolvedSamplingParams, build_sampling_config};
use crate::server::chat_template::ChatMessage;
use crate::server::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ErrorResponse,
    SamplingParams,
};
use crate::server::{AppState, ServerConfig, ServerGenerateOptions};

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request.stream {
        stream_chat_completion(state, request).await.into_response()
    } else {
        non_stream_chat_completion(state, request)
            .await
            .into_response()
    }
}

async fn non_stream_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
) -> Result<Json<ChatCompletionResponse>, ErrorResponse> {
    // Try to acquire a slot permit (non-blocking check for available slots)
    let _permit = state.slot_semaphore.try_acquire().map_err(|_| {
        ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
    })?;

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();

    // Convert messages to chat template format
    let messages: Vec<ChatMessage> = request
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.as_str().to_string(),
            content: m.content.text(),
        })
        .collect();

    // Extract images from multimodal content
    let image_data = extract_image_data(&request);

    // Apply chat template (fallback to simple format on error)
    let prompt = match state.chat_template.apply(&messages) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("Chat template render failed, using fallback: {:#}", e);
            request.to_prompt()
        }
    };

    let options = build_generate_options(&request.params, &state.config);

    // Generate (blocking call handled by model provider's worker thread)
    let result = state
        .model_provider
        .generate_with_images(prompt, options, image_data)
        .map_err(|e| ErrorResponse::new(format!("Generation error: {}", e), "server_error"))?;

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    Ok(Json(ChatCompletionResponse::new(
        request_id,
        model_id,
        result.text,
        result.prompt_tokens,
        result.completion_tokens,
        Some(result.finish_reason),
    )))
}

async fn stream_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Try to acquire a slot permit for streaming
    let permit = state.slot_semaphore.clone().try_acquire_owned().ok();

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();

    // Convert messages to chat template format
    let messages: Vec<ChatMessage> = request
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.as_str().to_string(),
            content: m.content.text(),
        })
        .collect();

    // Extract images from multimodal content
    let image_data = extract_image_data(&request);

    // Apply chat template (fallback to simple format on error for streaming)
    let prompt = state
        .chat_template
        .apply(&messages)
        .unwrap_or_else(|_| request.to_prompt());

    let options = build_generate_options(&request.params, &state.config);

    // Create async channel for SSE events
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(100);

    // Clone for the spawned task
    let request_id_clone = request_id.clone();
    let model_id_clone = model_id.clone();

    // Spawn a blocking task to handle generation
    tokio::task::spawn_blocking(move || {
        // Check if we got a permit
        let _permit = match permit {
            Some(p) => p,
            None => {
                // Send error and return
                let error_chunk = ChatCompletionChunk::finish(
                    request_id_clone,
                    model_id_clone,
                    "error".to_string(),
                );
                let _ = tx.blocking_send(Ok(
                    Event::default().data(serde_json::to_string(&error_chunk).unwrap())
                ));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                return;
            }
        };

        // Send initial chunk with role
        let initial =
            ChatCompletionChunk::initial(request_id_clone.clone(), model_id_clone.clone());
        let _ = tx.blocking_send(Ok(
            Event::default().data(serde_json::to_string(&initial).unwrap())
        ));

        // Use model provider's streaming API
        let tx_clone = tx.clone();
        let request_id_inner = request_id_clone.clone();
        let model_id_inner = model_id_clone.clone();

        let result = state.model_provider.generate_streaming_with_images(
            prompt,
            options,
            image_data,
            |token| {
                let chunk = ChatCompletionChunk::content(
                    request_id_inner.clone(),
                    model_id_inner.clone(),
                    token,
                );
                let _ = tx_clone.blocking_send(Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                ));
            },
        );

        // Send finish chunk
        let finish_reason = match &result {
            Ok(r) => r.finish_reason.clone(),
            Err(_) => "error".to_string(),
        };
        let finish = ChatCompletionChunk::finish(request_id_clone, model_id_clone, finish_reason);
        let _ = tx.blocking_send(Ok(
            Event::default().data(serde_json::to_string(&finish).unwrap())
        ));

        // Send [DONE] marker
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));

        // _permit is dropped here, releasing the slot
    });

    Sse::new(ReceiverStream::new(rx))
}

/// Build ServerGenerateOptions using request params with server config as defaults
pub(crate) fn build_generate_options(
    params: &SamplingParams,
    config: &Arc<ServerConfig>,
) -> ServerGenerateOptions {
    let temperature = params.temperature.unwrap_or(config.default_temperature);
    let top_k = params
        .top_k
        .map(|k| k as i32)
        .unwrap_or(config.default_top_k);
    let top_p = params.top_p.unwrap_or(config.default_top_p);
    let repetition_penalty = params
        .repetition_penalty
        .unwrap_or(config.default_repetition_penalty);
    let min_p = params.min_p.unwrap_or(config.default_min_p);
    let seed = params.seed.or(config.default_seed);
    let frequency_penalty = params
        .frequency_penalty
        .unwrap_or(config.default_frequency_penalty);
    let presence_penalty = params
        .presence_penalty
        .unwrap_or(config.default_presence_penalty);

    let sampling = build_sampling_config(ResolvedSamplingParams {
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
        repetition_penalty,
        dry_multiplier: params
            .dry_multiplier
            .unwrap_or(config.default_dry_multiplier),
        dry_base: params.dry_base.unwrap_or(config.default_dry_base),
        dry_allowed_length: params
            .dry_allowed_length
            .unwrap_or(config.default_dry_allowed_length),
        dry_penalty_last_n: params
            .dry_penalty_last_n
            .unwrap_or(config.default_dry_penalty_last_n),
        dry_sequence_breakers: params.dry_sequence_breakers.clone().unwrap_or_default(),
        frequency_penalty,
        presence_penalty,
        stop_token_ids: Vec::new(),
    });

    ServerGenerateOptions {
        max_tokens: params.max_tokens.unwrap_or(config.default_max_tokens),
        sampling,
        stop_sequences: params.stop.clone(),
    }
}

/// Extract image data from multimodal chat messages
///
/// Supports:
/// - base64 data URIs: `data:image/png;base64,...`
/// - file:// URLs: `file:///path/to/image.jpg`
fn extract_image_data(request: &ChatCompletionRequest) -> Vec<Vec<u8>> {
    let urls = request.image_urls();
    if urls.is_empty() {
        return Vec::new();
    }

    urls.iter()
        .filter_map(|url| {
            if url.starts_with("data:") {
                // Parse base64 data URI: data:image/...;base64,<data>
                if let Some(comma_pos) = url.find(',') {
                    let b64_data = &url[comma_pos + 1..];
                    use base64::Engine;
                    match base64::engine::general_purpose::STANDARD.decode(b64_data) {
                        Ok(bytes) => Some(bytes),
                        Err(e) => {
                            tracing::warn!("Failed to decode base64 image: {}", e);
                            None
                        }
                    }
                } else {
                    tracing::warn!("Invalid data URI format");
                    None
                }
            } else if let Some(path) = url.strip_prefix("file://") {
                // Read from local file
                match std::fs::read(path) {
                    Ok(bytes) => Some(bytes),
                    Err(e) => {
                        tracing::warn!("Failed to read image file {}: {}", path, e);
                        None
                    }
                }
            } else {
                tracing::warn!("Unsupported image URL scheme: {}", url);
                None
            }
        })
        .collect()
}

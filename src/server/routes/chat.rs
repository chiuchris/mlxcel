//! Chat completions endpoint

use axum::{
    Json,
    extract::State,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use futures::stream::Stream;
use std::convert::Infallible;

use crate::server::chat_template::ChatMessage;
use crate::server::media::extract_chat_image_data;
use crate::server::request_options::{RequestOptionOverrides, build_server_generate_options};
use crate::server::streaming::sse_channel;
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
    let image_data = extract_chat_image_data(&request);

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
    let image_data = extract_chat_image_data(&request);

    // Apply chat template (fallback to simple format on error for streaming)
    let prompt = state
        .chat_template
        .apply(&messages)
        .unwrap_or_else(|_| request.to_prompt());

    let options = build_generate_options(&request.params, &state.config);

    let (events, stream) = sse_channel(100);

    // Clone for the spawned task
    let request_id_clone = request_id.clone();
    let model_id_clone = model_id.clone();
    let finish_events = events.clone();

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
                finish_events.json(&error_chunk);
                finish_events.done();
                return;
            }
        };

        // Send initial chunk with role
        let initial =
            ChatCompletionChunk::initial(request_id_clone.clone(), model_id_clone.clone());
        finish_events.json(&initial);

        // Use model provider's streaming API
        let token_events = finish_events.clone();
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
                token_events.json(&chunk);
            },
        );

        // Send finish chunk
        let finish_reason = match &result {
            Ok(r) => r.finish_reason.clone(),
            Err(_) => "error".to_string(),
        };
        let finish = ChatCompletionChunk::finish(request_id_clone, model_id_clone, finish_reason);
        finish_events.json(&finish);
        finish_events.done();

        // _permit is dropped here, releasing the slot
    });

    Sse::new(stream)
}

/// Build ServerGenerateOptions using request params with server config as defaults
pub(crate) fn build_generate_options(
    params: &SamplingParams,
    config: &ServerConfig,
) -> ServerGenerateOptions {
    build_server_generate_options(
        config,
        RequestOptionOverrides {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_k: params.top_k.map(|k| k as i32),
            top_p: params.top_p,
            min_p: params.min_p,
            repetition_penalty: params.repetition_penalty,
            seed: params.seed,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            dry_multiplier: params.dry_multiplier,
            dry_base: params.dry_base,
            dry_allowed_length: params.dry_allowed_length,
            dry_penalty_last_n: params.dry_penalty_last_n,
            dry_sequence_breakers: params.dry_sequence_breakers.clone(),
            stop_sequences: params.stop.clone(),
        },
    )
}

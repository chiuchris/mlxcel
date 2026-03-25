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

//! OpenAI-compatible chat completions adapter.
//!
//! This file should stay thin: it flattens the HTTP request, delegates prompt
//! preparation and option merging to shared helpers, and streams chunk payloads
//! back through `server/streaming.rs`.

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response, sse::Sse},
};

use crate::server::batch::RequestPriority;
use crate::server::chat_request::prepare_chat_request;
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
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let priority = parse_priority_header(&headers);
    if request.stream {
        stream_chat_completion(state, request, priority).await
    } else {
        non_stream_chat_completion(state, request, priority)
            .await
            .into_response()
    }
}

/// Extract the `X-Priority` header value, defaulting to `Normal`.
pub(crate) fn parse_priority_header(headers: &HeaderMap) -> RequestPriority {
    headers
        .get("x-priority")
        .and_then(|v| v.to_str().ok())
        .and_then(RequestPriority::from_header)
        .unwrap_or_default()
}

async fn non_stream_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
    priority: RequestPriority,
) -> Result<Json<ChatCompletionResponse>, ErrorResponse> {
    // Queue-depth admission control: reject when prefill queue is full
    if !state.can_accept_request() {
        return Err(ErrorResponse::service_unavailable(
            "All slots are busy. Please try again later.",
        ));
    }

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();

    let prepared = prepare_chat_request(&state.chat_template, &request).await;
    let mut options = build_generate_options(&request.params, &state.config);
    options.priority = priority;

    // Generate (blocking call handled by model provider's worker thread)
    let result = state
        .model_provider
        .generate_with_images(prepared.prompt, options, prepared.image_data)
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
    priority: RequestPriority,
) -> Response {
    // Queue-depth admission control: return 503 before opening SSE stream
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();
    let prepared = prepare_chat_request(&state.chat_template, &request).await;
    let mut options = build_generate_options(&request.params, &state.config);
    options.priority = priority;

    // Extract include_usage before request is moved into the closure
    let include_usage = request
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    let (events, stream) = sse_channel(100);

    // Clone for the spawned task
    let request_id_clone = request_id.clone();
    let model_id_clone = model_id.clone();
    let finish_events = events.clone();

    // Spawn a blocking task to handle generation
    tokio::task::spawn_blocking(move || {
        // Send initial chunk with role
        let initial =
            ChatCompletionChunk::initial(request_id_clone.clone(), model_id_clone.clone());
        let _ = finish_events.json(&initial);

        // Use model provider's streaming API
        let token_events = finish_events.clone();
        let request_id_inner = request_id_clone.clone();
        let model_id_inner = model_id_clone.clone();

        let result = state.model_provider.generate_streaming_with_images(
            prepared.prompt,
            options,
            prepared.image_data,
            |token| {
                let chunk = ChatCompletionChunk::content(
                    request_id_inner.clone(),
                    model_id_inner.clone(),
                    token,
                );
                let _ = token_events.json(&chunk);
            },
        );

        // Send finish chunk
        let finish_reason = match &result {
            Ok(r) => r.finish_reason.clone(),
            Err(_) => "error".to_string(),
        };
        let finish = ChatCompletionChunk::finish(
            request_id_clone.clone(),
            model_id_clone.clone(),
            finish_reason,
        );
        let _ = finish_events.json(&finish);

        // Send usage chunk if requested (stream_options.include_usage)
        if include_usage && let Ok(ref r) = result {
            let usage_chunk = ChatCompletionChunk::usage(
                request_id_clone.clone(),
                model_id_clone.clone(),
                r.prompt_tokens,
                r.completion_tokens,
            );
            let _ = finish_events.json(&usage_chunk);
        }

        finish_events.done();
    });

    Sse::new(stream).into_response()
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
            priority: RequestPriority::default(),
        },
    )
}

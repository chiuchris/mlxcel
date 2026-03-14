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

//! OpenAI-compatible text completions adapter.
//!
//! Policy for defaults, sampling, and streaming stays in shared server helpers;
//! this module only translates the HTTP request/response shape.

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response, sse::Sse},
};

use crate::server::AppState;
use crate::server::streaming::sse_channel;
use crate::server::types::{CompletionChunk, CompletionRequest, CompletionResponse, ErrorResponse};

use super::chat::build_generate_options;

/// POST /v1/completions
pub async fn completions(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    if request.stream {
        stream_completion(state, request).await
    } else {
        non_stream_completion(state, request).await.into_response()
    }
}

async fn non_stream_completion(
    state: AppState,
    request: CompletionRequest,
) -> Result<Json<CompletionResponse>, ErrorResponse> {
    // Queue-depth admission control: reject when prefill queue is full
    if !state.can_accept_request() {
        return Err(ErrorResponse::service_unavailable(
            "All slots are busy. Please try again later.",
        ));
    }

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();

    let prompt = request.prompt.clone();
    let options = build_generate_options(&request.params, &state.config);

    // Generate (blocking call handled by model provider's worker thread)
    let result = state
        .model_provider
        .generate(prompt, options)
        .map_err(|e| ErrorResponse::new(format!("Generation error: {}", e), "server_error"))?;

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    Ok(Json(CompletionResponse::new(
        request_id,
        model_id,
        result.text,
        result.prompt_tokens,
        result.completion_tokens,
        Some(result.finish_reason),
    )))
}

async fn stream_completion(state: AppState, request: CompletionRequest) -> Response {
    // Queue-depth admission control: return 503 before opening SSE stream
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();
    let prompt = request.prompt.clone();
    let options = build_generate_options(&request.params, &state.config);

    let (events, stream) = sse_channel(100);

    // Clone for the spawned task
    let request_id_clone = request_id.clone();
    let model_id_clone = model_id.clone();
    let finish_events = events.clone();

    // Spawn a blocking task to handle generation
    tokio::task::spawn_blocking(move || {
        // Use model provider's streaming API
        let token_events = finish_events.clone();
        let request_id_inner = request_id_clone.clone();
        let model_id_inner = model_id_clone.clone();

        let result = state
            .model_provider
            .generate_streaming(prompt, options, |token| {
                let chunk = CompletionChunk::content(
                    request_id_inner.clone(),
                    model_id_inner.clone(),
                    token,
                );
                let _ = token_events.json(&chunk);
            });

        // Send finish chunk
        let finish_reason = match &result {
            Ok(r) => r.finish_reason.clone(),
            Err(_) => "error".to_string(),
        };
        let finish = CompletionChunk::finish(request_id_clone, model_id_clone, finish_reason);
        let _ = finish_events.json(&finish);
        finish_events.done();
    });

    Sse::new(stream).into_response()
}

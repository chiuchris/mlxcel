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

use std::collections::HashMap;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response, sse::Sse},
};

use mlxcel_core::sampling::{LogprobsConfig, TokenLogprobData};

use crate::server::AppState;
use crate::server::batch::RequestPriority;
use crate::server::streaming::sse_channel;
use crate::server::types::response::CompletionLogprobs;
use crate::server::types::{CompletionChunk, CompletionRequest, CompletionResponse, ErrorResponse};
use crate::tokenizer::MlxcelTokenizer;

use super::chat::{build_generate_options, decode_token, parse_priority_header};

/// Build a `CompletionLogprobs` from a list of `TokenLogprobData` (legacy format).
fn build_completion_logprobs(
    tokenizer: &MlxcelTokenizer,
    lp_data: &[TokenLogprobData],
    top_k: usize,
) -> CompletionLogprobs {
    let mut tokens = Vec::with_capacity(lp_data.len());
    let mut token_logprobs = Vec::with_capacity(lp_data.len());
    let mut text_offset = Vec::with_capacity(lp_data.len());
    let mut top_logprobs_list = Vec::with_capacity(lp_data.len());

    let mut char_offset: usize = 0;

    for lp in lp_data {
        let token_text = decode_token(tokenizer, lp.token_id);
        text_offset.push(char_offset);
        char_offset += token_text.len();
        tokens.push(token_text);
        token_logprobs.push(lp.logprob);

        let top_map = if top_k > 0 && !lp.top_alternatives.is_empty() {
            let mut map = HashMap::new();
            for &(alt_id, alt_lp) in lp.top_alternatives.iter().take(top_k) {
                let alt_text = decode_token(tokenizer, alt_id);
                map.insert(alt_text, alt_lp);
            }
            Some(map)
        } else {
            None
        };
        top_logprobs_list.push(top_map);
    }

    CompletionLogprobs {
        tokens,
        token_logprobs,
        text_offset,
        top_logprobs: top_logprobs_list,
    }
}

/// Build a streaming `CompletionLogprobs` for a single token chunk.
fn build_single_token_completion_logprobs(
    tokenizer: &MlxcelTokenizer,
    lp: &TokenLogprobData,
    top_k: usize,
    char_offset: usize,
) -> CompletionLogprobs {
    let token_text = decode_token(tokenizer, lp.token_id);
    let top_map = if top_k > 0 && !lp.top_alternatives.is_empty() {
        let mut map = HashMap::new();
        for &(alt_id, alt_lp) in lp.top_alternatives.iter().take(top_k) {
            let alt_text = decode_token(tokenizer, alt_id);
            map.insert(alt_text, alt_lp);
        }
        Some(map)
    } else {
        None
    };
    CompletionLogprobs {
        tokens: vec![token_text],
        token_logprobs: vec![lp.logprob],
        text_offset: vec![char_offset],
        top_logprobs: vec![top_map],
    }
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CompletionRequest>,
) -> Response {
    // Validate logprobs range per OpenAI spec (0-5 for legacy completions)
    if let Some(top) = request.logprobs
        && top > 5
    {
        return ErrorResponse::new("logprobs must be between 0 and 5", "invalid_request_error")
            .into_response();
    }

    let priority = parse_priority_header(&headers);
    if request.stream {
        stream_completion(state, request, priority).await
    } else {
        non_stream_completion(state, request, priority)
            .await
            .into_response()
    }
}

async fn non_stream_completion(
    state: AppState,
    request: CompletionRequest,
    priority: RequestPriority,
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
    let mut options = build_generate_options(&request.params, &state.config);
    options.priority = priority;

    // In the legacy format, `logprobs` is a number (top-k); 0 means return only
    // the selected token's log-prob, None means don't return logprobs at all.
    let top_k = request.logprobs.unwrap_or(0) as usize;
    let logprobs_enabled = request.logprobs.is_some();
    if logprobs_enabled {
        options.logprobs = LogprobsConfig {
            enabled: true,
            top_k,
        };
    }

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

    // Build legacy-format logprobs if requested
    let logprobs = result.logprobs.as_deref().and_then(|lp_data| {
        if lp_data.is_empty() || !logprobs_enabled {
            None
        } else {
            Some(build_completion_logprobs(&state.tokenizer, lp_data, top_k))
        }
    });

    Ok(Json(CompletionResponse::new_with_logprobs(
        request_id,
        model_id,
        result.text,
        result.prompt_tokens,
        result.completion_tokens,
        Some(result.finish_reason),
        logprobs,
    )))
}

async fn stream_completion(
    state: AppState,
    request: CompletionRequest,
    priority: RequestPriority,
) -> Response {
    // Queue-depth admission control: return 503 before opening SSE stream
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();
    let prompt = request.prompt.clone();
    let mut options = build_generate_options(&request.params, &state.config);
    options.priority = priority;

    // Extract include_usage before request is moved into the closure
    let include_usage = request
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    // In the legacy format, `logprobs` is a number (top-k); 0 means return only
    // the selected token's log-prob, None means don't return logprobs at all.
    let top_k = request.logprobs.unwrap_or(0) as usize;
    let logprobs_enabled = request.logprobs.is_some();
    if logprobs_enabled {
        options.logprobs = LogprobsConfig {
            enabled: true,
            top_k,
        };
    }

    let (events, stream, cancelled) = sse_channel(100);

    // Clone for the spawned task
    let request_id_clone = request_id.clone();
    let model_id_clone = model_id.clone();
    let finish_events = events.clone();
    let tokenizer = state.tokenizer.clone();

    // Spawn a blocking task to handle generation
    tokio::task::spawn_blocking(move || {
        // Use logprobs-aware streaming
        let token_events = finish_events.clone();
        let request_id_inner = request_id_clone.clone();
        let model_id_inner = model_id_clone.clone();
        let mut char_offset: usize = 0;

        let result = state
            .model_provider
            .generate_streaming_with_logprobs_cancellable(
                prompt,
                options,
                Vec::new(),
                Vec::new(),
                cancelled,
                |token, lp_data| {
                    let logprobs = if logprobs_enabled {
                        lp_data.as_ref().map(|lp| {
                            let chunk_lp = build_single_token_completion_logprobs(
                                &tokenizer,
                                lp,
                                top_k,
                                char_offset,
                            );
                            char_offset += token.len();
                            chunk_lp
                        })
                    } else {
                        None
                    };
                    let chunk = CompletionChunk::content_with_logprobs(
                        request_id_inner.clone(),
                        model_id_inner.clone(),
                        token,
                        logprobs,
                    );
                    let _ = token_events.json(&chunk);
                },
            );

        // Send finish chunk
        let finish_reason = match &result {
            Ok(r) => r.finish_reason.clone(),
            Err(_) => "error".to_string(),
        };
        let finish = CompletionChunk::finish(
            request_id_clone.clone(),
            model_id_clone.clone(),
            finish_reason,
        );
        let _ = finish_events.json(&finish);

        // Send usage chunk if requested (stream_options.include_usage)
        if include_usage && let Ok(ref r) = result {
            let usage_chunk = CompletionChunk::usage(
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

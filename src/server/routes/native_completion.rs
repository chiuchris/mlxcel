//! Native completion endpoint (llama-server /completion format)
//!
//! Different from /v1/completions — uses `n_predict` instead of `max_tokens`,
//! returns `{"content": "...", "stop": true, "timings": {...}}`.

use std::convert::Infallible;

use axum::{
    Json,
    extract::State,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use futures::stream::Stream;

use crate::server::request_options::{RequestOptionOverrides, build_server_generate_options};
use crate::server::streaming::sse_channel;
use crate::server::types::{
    ErrorResponse, NativeCompletionRequest, NativeCompletionResponse, TimingInfo,
};
use crate::server::{AppState, ServerConfig, ServerGenerateOptions};

/// POST /completion
pub async fn native_completion(
    State(state): State<AppState>,
    Json(request): Json<NativeCompletionRequest>,
) -> Response {
    if request.stream.unwrap_or(false) {
        stream_native_completion(state, request)
            .await
            .into_response()
    } else {
        non_stream_native_completion(state, request)
            .await
            .into_response()
    }
}

async fn non_stream_native_completion(
    state: AppState,
    request: NativeCompletionRequest,
) -> Result<Json<NativeCompletionResponse>, ErrorResponse> {
    let _permit = state.slot_semaphore.try_acquire().map_err(|_| {
        ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
    })?;

    let options = build_native_options(&request, &state);

    let result = state
        .model_provider
        .generate(request.prompt.clone(), options)
        .map_err(|e| ErrorResponse::new(format!("Generation error: {}", e), "server_error"))?;

    let prompt_ms = result.prompt_eval_ms as f64;
    let gen_ms = result.generation_only_ms as f64;

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    Ok(Json(NativeCompletionResponse {
        content: result.text,
        stop: result.finish_reason == "stop",
        generation_settings: serde_json::json!({}),
        model: state.display_model_id().to_string(),
        tokens_predicted: result.completion_tokens,
        tokens_evaluated: result.prompt_tokens,
        timings: TimingInfo {
            prompt_n: result.prompt_tokens,
            prompt_ms,
            prompt_per_token_ms: if result.prompt_tokens > 0 {
                prompt_ms / result.prompt_tokens as f64
            } else {
                0.0
            },
            prompt_per_second: if prompt_ms > 0.0 {
                result.prompt_tokens as f64 / (prompt_ms / 1000.0)
            } else {
                0.0
            },
            predicted_n: result.completion_tokens,
            predicted_ms: gen_ms,
            predicted_per_token_ms: if result.completion_tokens > 0 {
                gen_ms / result.completion_tokens as f64
            } else {
                0.0
            },
            predicted_per_second: if gen_ms > 0.0 {
                result.completion_tokens as f64 / (gen_ms / 1000.0)
            } else {
                0.0
            },
        },
    }))
}

async fn stream_native_completion(
    state: AppState,
    request: NativeCompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let permit = state.slot_semaphore.clone().try_acquire_owned().ok();

    let options = build_native_options(&request, &state);
    let prompt = request.prompt.clone();

    let (events, stream) = sse_channel(100);
    let finish_events = events.clone();

    tokio::task::spawn_blocking(move || {
        let _permit = match permit {
            Some(p) => p,
            None => {
                let err = serde_json::json!({"content": "", "stop": true});
                finish_events.json(&err);
                return;
            }
        };

        let token_events = finish_events.clone();

        let result = state
            .model_provider
            .generate_streaming(prompt, options, |token| {
                let chunk = serde_json::json!({
                    "content": token,
                    "stop": false,
                });
                token_events.json(&chunk);
            });

        // Send final chunk
        let stop = match &result {
            Ok(r) => r.finish_reason == "stop",
            Err(_) => true,
        };
        let final_chunk = serde_json::json!({
            "content": "",
            "stop": true,
            "stop_type": if stop { "stop" } else { "limit" },
        });
        finish_events.json(&final_chunk);
    });

    Sse::new(stream)
}

fn build_native_options(
    request: &NativeCompletionRequest,
    state: &AppState,
) -> ServerGenerateOptions {
    build_native_generate_options(&state.config, request)
}

fn build_native_generate_options(
    config: &ServerConfig,
    request: &NativeCompletionRequest,
) -> ServerGenerateOptions {
    build_server_generate_options(
        config,
        RequestOptionOverrides {
            max_tokens: request.n_predict,
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            min_p: request.min_p,
            repetition_penalty: request.repeat_penalty,
            seed: request.seed,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            dry_multiplier: request.dry_multiplier,
            dry_base: request.dry_base,
            dry_allowed_length: request.dry_allowed_length,
            dry_penalty_last_n: request.dry_penalty_last_n,
            dry_sequence_breakers: request.dry_sequence_breakers.clone(),
            stop_sequences: request.stop.clone(),
        },
    )
}

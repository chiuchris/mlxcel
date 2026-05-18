//! Text completions endpoint

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
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::server::AppState;
use crate::server::types::{CompletionChunk, CompletionRequest, CompletionResponse, ErrorResponse};

use super::chat::build_generate_options;

/// POST /v1/completions
pub async fn completions(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    if request.stream {
        stream_completion(state, request).await.into_response()
    } else {
        non_stream_completion(state, request).await.into_response()
    }
}

async fn non_stream_completion(
    state: AppState,
    request: CompletionRequest,
) -> Result<Json<CompletionResponse>, ErrorResponse> {
    // Try to acquire a slot permit (non-blocking check for available slots)
    let _permit = state.slot_semaphore.try_acquire().map_err(|_| {
        ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
    })?;

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

async fn stream_completion(
    state: AppState,
    request: CompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Try to acquire a slot permit for streaming
    let permit = state.slot_semaphore.clone().try_acquire_owned().ok();

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();
    let prompt = request.prompt.clone();
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
                let error_chunk =
                    CompletionChunk::finish(request_id_clone, model_id_clone, "error".to_string());
                let _ = tx.blocking_send(Ok(
                    Event::default().data(serde_json::to_string(&error_chunk).unwrap())
                ));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                return;
            }
        };

        // Use model provider's streaming API
        let tx_clone = tx.clone();
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
                let _ = tx_clone.blocking_send(Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                ));
            });

        // Send finish chunk
        let finish_reason = match &result {
            Ok(r) => r.finish_reason.clone(),
            Err(_) => "error".to_string(),
        };
        let finish = CompletionChunk::finish(request_id_clone, model_id_clone, finish_reason);
        let _ = tx.blocking_send(Ok(
            Event::default().data(serde_json::to_string(&finish).unwrap())
        ));

        // Send [DONE] marker
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));

        // _permit is dropped here, releasing the slot
    });

    Sse::new(ReceiverStream::new(rx))
}

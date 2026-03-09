//! Prometheus-compatible metrics endpoint.
//!
//! This route is read-only and should remain separate from generation policy.

use std::sync::atomic::Ordering;

use axum::{
    extract::State,
    http::header,
    response::{IntoResponse, Response},
};

use crate::server::AppState;

/// GET /metrics — Prometheus text format
pub async fn metrics(State(state): State<AppState>) -> Response {
    let m = &state.metrics;

    let requests = m.requests_total.load(Ordering::Relaxed);
    let prompt_tokens = m.prompt_tokens_total.load(Ordering::Relaxed);
    let completion_tokens = m.completion_tokens_total.load(Ordering::Relaxed);
    let gen_time_ms = m.generation_time_ms_total.load(Ordering::Relaxed);

    let slots_total = state.config.n_parallel;
    let slots_available = state.slot_semaphore.available_permits();

    let body = format!(
        "# HELP mlxcel_requests_total Total number of generation requests\n\
         # TYPE mlxcel_requests_total counter\n\
         mlxcel_requests_total {requests}\n\
         # HELP mlxcel_prompt_tokens_total Total prompt tokens processed\n\
         # TYPE mlxcel_prompt_tokens_total counter\n\
         mlxcel_prompt_tokens_total {prompt_tokens}\n\
         # HELP mlxcel_completion_tokens_total Total completion tokens generated\n\
         # TYPE mlxcel_completion_tokens_total counter\n\
         mlxcel_completion_tokens_total {completion_tokens}\n\
         # HELP mlxcel_generation_time_seconds_total Total generation time in seconds\n\
         # TYPE mlxcel_generation_time_seconds_total counter\n\
         mlxcel_generation_time_seconds_total {gen_time_sec:.3}\n\
         # HELP mlxcel_slots_total Total number of parallel slots\n\
         # TYPE mlxcel_slots_total gauge\n\
         mlxcel_slots_total {slots_total}\n\
         # HELP mlxcel_slots_available Available parallel slots\n\
         # TYPE mlxcel_slots_available gauge\n\
         mlxcel_slots_available {slots_available}\n",
        gen_time_sec = gen_time_ms as f64 / 1000.0,
    );

    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

//! Server properties endpoint (llama-server compatible).
//!
//! This route surfaces configuration/state snapshots and should stay as a thin
//! read-only adapter.

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::types::PropsResponse;

/// GET /props
pub async fn props(State(state): State<AppState>) -> Json<PropsResponse> {
    let config = &state.config;

    Json(PropsResponse {
        default_generation_settings: serde_json::json!({
            "n_predict": config.default_max_tokens,
            "temperature": config.default_temperature,
            "top_k": config.default_top_k,
            "top_p": config.default_top_p,
            "min_p": config.default_min_p,
            "repeat_penalty": config.default_repetition_penalty,
            "repeat_last_n": config.default_repetition_context_size,
            "seed": config.default_seed.unwrap_or(u64::MAX),
            "frequency_penalty": config.default_frequency_penalty,
            "presence_penalty": config.default_presence_penalty,
            "dry_multiplier": config.default_dry_multiplier,
            "dry_base": config.default_dry_base,
            "dry_allowed_length": config.default_dry_allowed_length,
            "dry_penalty_last_n": config.default_dry_penalty_last_n,
        }),
        total_slots: config.n_parallel,
    })
}

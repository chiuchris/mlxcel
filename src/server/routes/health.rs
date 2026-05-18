//! Health check endpoint (llama-server compatible)

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

use crate::server::AppState;
use crate::server::types::HealthResponse;

/// GET /health
///
/// Returns llama-server compatible status:
/// - `{"status": "ok"}` when model is loaded and slots available
/// - `{"status": "no slot available"}` when all slots are busy
/// - `{"status": "loading model"}` when model is still loading
pub async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    if !state.model_provider.is_loaded() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthResponse {
                status: "loading model".to_string(),
                model: None,
            }),
        );
    }

    let has_slots = state.slot_semaphore.available_permits() > 0;
    if !has_slots {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthResponse {
                status: "no slot available".to_string(),
                model: Some(state.display_model_id().to_string()),
            }),
        );
    }

    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok".to_string(),
            model: Some(state.display_model_id().to_string()),
        }),
    )
}

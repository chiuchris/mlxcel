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

//! Health check endpoint (llama-server compatible).
//!
//! This route only reports server liveness and slot availability. Slot policy
//! itself stays in shared server state.

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

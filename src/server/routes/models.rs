//! Models endpoint

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::types::{ModelInfo, ModelsResponse};

/// GET /v1/models
pub async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let models = vec![ModelInfo {
        id: state.display_model_id().to_string(),
        object: "model".to_string(),
        created: state.model_provider.created_at(),
        owned_by: "user".to_string(),
    }];

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

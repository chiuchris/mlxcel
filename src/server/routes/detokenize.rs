//! Detokenize endpoint (llama-server compatible)

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::types::{DetokenizeRequest, DetokenizeResponse, ErrorResponse};

/// POST /detokenize
pub async fn detokenize(
    State(state): State<AppState>,
    Json(request): Json<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, ErrorResponse> {
    let ids: Vec<u32> = request.tokens.iter().map(|&x| x as u32).collect();

    let content = state.tokenizer.decode(&ids, false).map_err(|e| {
        ErrorResponse::new(
            format!("Detokenization error: {}", e),
            "invalid_request_error",
        )
    })?;

    Ok(Json(DetokenizeResponse { content }))
}

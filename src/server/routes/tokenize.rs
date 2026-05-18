//! Tokenize endpoint (llama-server compatible)

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::types::{ErrorResponse, TokenizeRequest, TokenizeResponse};

/// POST /tokenize
pub async fn tokenize(
    State(state): State<AppState>,
    Json(request): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ErrorResponse> {
    let add_special = request.add_special.unwrap_or(false);

    let token_ids = state
        .tokenizer
        .encode(request.content.as_str(), add_special)
        .map_err(|e| {
            ErrorResponse::new(
                format!("Tokenization error: {}", e),
                "invalid_request_error",
            )
        })?;

    let tokens: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();

    Ok(Json(TokenizeResponse { tokens }))
}

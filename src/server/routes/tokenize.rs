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

//! Tokenize endpoint (llama-server compatible).
//!
//! This is a thin tokenizer adapter and should not grow generation or chat
//! policy that belongs in other server modules.

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

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

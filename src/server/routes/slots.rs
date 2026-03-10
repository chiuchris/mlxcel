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

//! Slots endpoint (llama-server compatible).
//!
//! Slot accounting lives in shared state; this module only adapts that state
//! into the llama-server-compatible response shape.

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::types::SlotInfo;

/// GET /slots
pub async fn slots(State(state): State<AppState>) -> Json<Vec<SlotInfo>> {
    let total = state.config.n_parallel.max(1);
    let available = state.slot_semaphore.available_permits();

    let slots: Vec<SlotInfo> = (0..total)
        .map(|i| {
            let is_idle = i < available;
            SlotInfo {
                id: i,
                state: if is_idle { 0 } else { 1 },
                model: state.display_model_id().to_string(),
                is_processing: !is_idle,
            }
        })
        .collect();

    Json(slots)
}

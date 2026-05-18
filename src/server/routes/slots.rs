//! Slots endpoint (llama-server compatible)

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

use crate::core::ml::hardware_accel::HardwareStatus;
use crate::state::AppState;
use axum::{extract::State, response::Json};

/// GET /api/v1/hardware/status
pub async fn get_hardware_status(State(state): State<AppState>) -> Json<HardwareStatus> {
    Json(state.hardware_status.clone())
}

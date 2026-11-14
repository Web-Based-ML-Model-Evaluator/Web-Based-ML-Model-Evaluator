use crate::models::StandardErrorResponse;
use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};

/// GET /api/v1/models/comparison/{session_id} (FR-12)
pub async fn get_comparison_dashboard(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Json<serde_json::Value> {
    let history = state.comparison_store.get_history(&session_id);
    let runs = if history.is_empty()
        && (session_id == "all" || session_id == "default" || session_id == "mock")
    {
        if session_id == "mock" && state.comparison_store.get_all_runs().is_empty() {
            state.comparison_store.seed_mock_runs();
        }
        state.comparison_store.get_all_runs()
    } else {
        history
    };

    Json(serde_json::json!({
        "dataset_id": session_id,
        "total_runs": runs.len(),
        "runs": runs
    }))
}

/// POST /api/v1/models/compare (FR-12, BR-7)
pub async fn compare_models(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<StandardErrorResponse>)> {
    let ids: Vec<String> = payload["execution_ids"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    if ids.len() < 2 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(StandardErrorResponse::new(
                "INSUFFICIENT_MODELS_FOR_COMPARISON",
                "Model comparison requires selecting at least two trained models side-by-side (FR-12 / BR-7).",
                Some(serde_json::json!({ "selected_count": ids.len() })),
            )),
        ));
    }

    match state.comparison_store.build_comparison_report(&ids) {
        Ok(report) => Ok(Json(serde_json::to_value(report).unwrap())),
        Err(err_msg) => Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(StandardErrorResponse::new(
                "COMPARISON_FAILED",
                &err_msg,
                Some(serde_json::json!({ "requested_ids": ids })),
            )),
        )),
    }
}

/// DELETE /api/v1/models/comparison/{session_id} (SEC-1, SEC-3, SRS 7.3)
pub async fn purge_session_models(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Json<serde_json::Value> {
    let count = state.comparison_store.purge_session(&session_id);
    Json(serde_json::json!({
        "status": "SESSION_MODELS_PURGED",
        "session_id": session_id,
        "purged_runs_count": count,
        "message": "Model execution runs purged securely from host memory per SEC-1 and SEC-3."
    }))
}


pub mod comparison;
pub mod datasets;
pub mod hardware;
pub mod models;
pub mod routing;

use crate::models::StandardErrorResponse;
use crate::state::AppState;
use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use tower_http::cors::{Any, CorsLayer};

pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "web_ml_unified_backend",
        "version": "1.0.0",
        "team": "Team 13",
        "architecture": "High-Performance Unified In-Memory Pipeline (Rust + Axum + Rayon)"
    }))
}

pub async fn fallback_404_handler() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(StandardErrorResponse::new(
            "RESOURCE_NOT_FOUND",
            "The requested API endpoint does not exist on this server.",
            None,
        )),
    )
}

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_check))
        // Datasets (Portion 1 unified)
        .route("/api/v1/datasets/upload", post(datasets::upload_dataset))
        .route(
            "/api/v1/datasets/:dataset_id/preview",
            get(datasets::get_preview),
        )
        .route(
            "/api/v1/datasets/:dataset_id/preprocess",
            post(datasets::preprocess_dataset),
        )
        .route(
            "/api/v1/datasets/:dataset_id/data",
            get(datasets::get_cleaned_data),
        )
        .route(
            "/api/v1/datasets/:dataset_id/schema",
            get(datasets::get_schema),
        )
        .route(
            "/api/v1/datasets/:dataset_id",
            delete(datasets::purge_dataset),
        )
        .route(
            "/api/v1/datasets/:dataset_id/purge",
            post(datasets::purge_dataset),
        )
        // Configuration & Routing (Portion 2 unified)
        .route("/api/v1/config/tasks", get(routing::list_supported_tasks))
        .route(
            "/api/v1/config/validate",
            post(routing::validate_configuration),
        )
        .route(
            "/api/v1/pipeline/dispatch",
            post(routing::dispatch_pipeline),
        )
        // Model Execution & Comparison (Portion 3 unified)
        .route(
            "/api/v1/models/supported",
            get(models::get_supported_algorithms),
        )
        .route("/api/v1/models/execute", post(models::execute_model))
        .route("/api/v1/models/compare", post(comparison::compare_models))
        .route(
            "/api/v1/models/comparison/:session_id",
            get(comparison::get_comparison_dashboard).delete(comparison::purge_session_models),
        )
        .route(
            "/api/v1/models/session/purge",
            post(models::purge_model_session),
        )
        // Global Session Purge
        .route("/api/v1/session/purge", post(datasets::purge_all_datasets))
        // Hardware & Acceleration Status
        .route("/api/v1/hardware/status", get(hardware::get_hardware_status))
        .route("/api/v1/hardware/benchmark", post(hardware::benchmark_hardware))
        .fallback(fallback_404_handler)
        .layer(cors)
        .with_state(state)
}

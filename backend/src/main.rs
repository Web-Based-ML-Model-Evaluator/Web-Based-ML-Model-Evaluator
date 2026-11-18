use axum::{
    http::Method,
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod api;
mod models;
mod state;

use state::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,web_ml_backend=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let state = AppState::default();

    let cors = CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers(tower_http::cors::Any);

    let app = Router::new()
        // Health & Telemetry
        .route("/health", get(health_check))
        .route("/api/v1/health", get(health_check))
        .route("/api/v1/hardware", get(api::hardware::get_hardware_telemetry))
        // Dataset Ingestion & Dynamic Questions
        .route("/api/v1/datasets/upload", post(api::datasets::upload_dataset))
        .route(
            "/api/v1/datasets/:dataset_id/analysis",
            get(api::datasets::get_dataset_analysis),
        )
        .route(
            "/api/v1/datasets/:dataset_id/questions",
            get(api::datasets::get_dataset_questions),
        )
        // Model Training & Model Download
        .route("/api/v1/models/train", post(api::train::train_model))
        .route(
            "/api/v1/models/download/:model_id",
            get(api::train::download_model_file),
        )
        // Model Evaluation (Universal file formats & CSV test sets)
        .route(
            "/api/v1/models/evaluate-file",
            post(api::evaluate::evaluate_model_file),
        )
        .route(
            "/api/v1/models/evaluate",
            post(api::evaluate::evaluate_model_json),
        )
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("============================================================");
    println!(" Web-Based ML Model Evaluator Backend (Rust Axum Engine)");
    println!(" Running on http://127.0.0.1:{}", port);
    println!(" Endpoints:");
    println!("   POST /api/v1/datasets/upload          (Smart Dataset Ingestion)");
    println!("   GET  /api/v1/datasets/:id/questions   (Dataset-Aware Questions)");
    println!("   POST /api/v1/models/train             (Model Training Engine)");
    println!("   GET  /api/v1/models/download/:id      (Trained Model File Download)");
    println!("   POST /api/v1/models/evaluate-file     (Universal Model Evaluator)");
    println!("============================================================");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({
        "status": "healthy",
        "engine": "web_ml_backend",
        "version": "2.0.0",
        "timestamp": chrono::Utc::now()
    }))
}

mod handlers;
mod models;
mod preprocessing;
mod validator;

use axum::{
    routing::{get, post},
    Router,
};
use handlers::{
    get_cleaned_data, get_preview, get_schema, health_check, preprocess_dataset, upload_dataset,
    AppState,
};
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/datasets/upload", post(upload_dataset))
        .route("/api/v1/datasets/:dataset_id/preview", get(get_preview))
        .route("/api/v1/datasets/:dataset_id/preprocess", post(preprocess_dataset))
        .route("/api/v1/datasets/:dataset_id/data", get(get_cleaned_data))
        .route("/api/v1/datasets/:dataset_id/schema", get(get_schema))
        .layer(cors)
        .with_state(state)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = AppState::default();
    let app = create_router(state);

    let port = 8001;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("============================================================");
    println!(" PORTION 1: DATA INGESTION & PREPROCESSING SERVICE");
    println!(" Assigned Owner: Sneha Ajith (PES2UG24CS480)");
    println!(" Listening on: http://localhost:{}", port);
    println!(" Health Check: http://localhost:{}/health", port);
    println!(" Upload API:   POST http://localhost:{}/api/v1/datasets/upload", port);
    println!("============================================================");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

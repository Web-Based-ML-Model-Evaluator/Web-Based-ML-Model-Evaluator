mod client;
mod handlers;
mod models;
mod router_validation;

use axum::{
    routing::{get, post},
    Router,
};
use client::PipelineClient;
use handlers::{dispatch_pipeline, health_check, list_supported_tasks, validate_configuration, AppState};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/config/tasks", get(list_supported_tasks))
        .route("/api/v1/config/validate", post(validate_configuration))
        .route("/api/v1/pipeline/dispatch", post(dispatch_pipeline))
        .layer(cors)
        .with_state(state)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let p1_url = std::env::var("PORTION1_URL").unwrap_or_else(|_| "http://localhost:8001".to_string());
    let p3_url = std::env::var("PORTION3_URL").unwrap_or_else(|_| "http://localhost:8003".to_string());

    let client = Arc::new(PipelineClient::new(&p1_url, &p3_url));
    let state = AppState { client };
    let app = create_router(state);

    let port = 8002;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("============================================================");
    println!(" PORTION 2: TASK ROUTING & CONFIGURATION SERVICE");
    println!(" Assigned Owner: Satvik Phadke (PES2UG24CS450)");
    println!(" Listening on:   http://localhost:{}", port);
    println!(" Health Check:   http://localhost:{}/health", port);
    println!(" Tasks API:      GET  http://localhost:{}/api/v1/config/tasks", port);
    println!(" Validate API:   POST http://localhost:{}/api/v1/config/validate", port);
    println!(" Dispatch API:   POST http://localhost:{}/api/v1/pipeline/dispatch", port);
    println!(" Upstream P1:    {}", p1_url);
    println!(" Downstream P3:  {}", p3_url);
    println!("============================================================");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

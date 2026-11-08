mod comparison;
mod handlers;
mod metrics;
mod ml_algorithms;
mod models;

use axum::{
    routing::{get, post},
    Router,
};
use handlers::{
    compare_models, execute_model, fallback_404_handler, get_comparison_dashboard,
    get_supported_algorithms, health_check, purge_all_sessions, purge_session_models, AppState,
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
        .route("/api/v1/models/supported", get(get_supported_algorithms))
        .route("/api/v1/models/execute", post(execute_model))
        .route(
            "/api/v1/models/comparison/:session_id",
            get(get_comparison_dashboard).delete(purge_session_models),
        )
        .route("/api/v1/models/compare", post(compare_models))
        .route("/api/v1/models/session/purge", post(purge_all_sessions))
        .fallback(fallback_404_handler)
        .layer(cors)
        .with_state(state)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = AppState::default();
    let app = create_router(state);

    let port = 8003;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("============================================================");
    println!(" PORTION 3: MODEL EXECUTION, METRICS & COMPARISON SERVICE");
    println!(" Assigned Owner: Shreya Patil (PES2UG24CS484)");
    println!(" Listening on:   http://localhost:{}", port);
    println!(" Health Check:   http://localhost:{}/health", port);
    println!(
        " Execute Model:  POST http://localhost:{}/api/v1/models/execute",
        port
    );
    println!(
        " Comparison API: GET  http://localhost:{}/api/v1/models/comparison/:id",
        port
    );
    println!(
        " Supported API:  GET  http://localhost:{}/api/v1/models/supported",
        port
    );
    println!("============================================================");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

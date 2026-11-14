pub mod api;
pub mod core;
pub mod models;
pub mod state;

use api::create_router;
use state::AppState;
use std::net::SocketAddr;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = AppState::default();
    let app = create_router(state);

    let preferred_ports: Vec<u16> = if let Ok(p_str) = std::env::var("PORT") {
        vec![p_str.parse().unwrap_or(8080)]
    } else {
        vec![8080, 8000, 8081, 5000]
    };

    let mut listener_opt = None;
    let mut bound_port = 8080;

    for p in preferred_ports {
        let addr = SocketAddr::from(([127, 0, 0, 1], p));
        if let Ok(l) = tokio::net::TcpListener::bind(addr).await {
            listener_opt = Some(l);
            bound_port = p;
            break;
        }
    }

    let listener = listener_opt.expect("Failed to bind TCP listener to an open port");

    println!("============================================================");
    println!("🚀 Unified ML Backend server running at http://localhost:{}", bound_port);
    println!("============================================================");
    tracing::info!("Unified ML Backend server listening on http://localhost:{}", bound_port);

    axum::serve(listener, app).await.unwrap();
}

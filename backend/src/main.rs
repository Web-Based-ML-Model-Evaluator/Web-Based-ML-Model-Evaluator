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
    let app = create_router(state.clone());

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Unified ML Backend server listening on http://{}", addr);

    // Also run backward-compatibility listeners on 8001, 8002, 8003 if on default port 8000
    if port == 8000 {
        for compat_port in [8001, 8002, 8003] {
            let compat_state = state.clone();
            tokio::spawn(async move {
                let compat_app = create_router(compat_state);
                let compat_addr = SocketAddr::from(([0, 0, 0, 0], compat_port));
                if let Ok(listener) = tokio::net::TcpListener::bind(compat_addr).await {
                    tracing::info!("Backward-compatibility proxy listening on http://{}", compat_addr);
                    let _ = axum::serve(listener, compat_app).await;
                }
            });
        }
    }

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

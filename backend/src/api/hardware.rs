use axum::response::Json;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct HardwareInfo {
    pub cpu_cores: usize,
    pub os: String,
    pub arch: String,
    pub engine_version: String,
    pub webgpu_supported: bool,
    pub status: String,
}

pub async fn get_hardware_telemetry() -> Json<HardwareInfo> {
    Json(HardwareInfo {
        cpu_cores: num_cpus(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        engine_version: "2.0.0 (Rust Axum + SmartCore)".to_string(),
        webgpu_supported: true,
        status: "healthy".to_string(),
    })
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

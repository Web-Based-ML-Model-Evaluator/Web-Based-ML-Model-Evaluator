use serde::{Deserialize, Serialize};

/// What the host machine reports about itself. Every field is measured; anything that
/// cannot be detected is `None` rather than an assumed value.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HardwareStatus {
    /// Logical CPU cores reported by the operating system.
    pub logical_cores: Option<usize>,
    /// AVX2 or SSE4.1 detected at runtime (x86_64 only; `None` on other architectures).
    pub simd_available: Option<bool>,
    /// How models are actually trained.
    pub compute_backend: String,
}

impl HardwareStatus {
    pub fn probe() -> Self {
        let logical_cores = std::thread::available_parallelism().map(|n| n.get()).ok();

        #[cfg(target_arch = "x86_64")]
        let simd_available =
            Some(is_x86_feature_detected!("avx2") || is_x86_feature_detected!("sse4.1"));
        #[cfg(not(target_arch = "x86_64"))]
        let simd_available = None;

        Self {
            logical_cores,
            simd_available,
            compute_backend: "Single-threaded CPU (smartcore and built-in neural networks)"
                .to_string(),
        }
    }
}

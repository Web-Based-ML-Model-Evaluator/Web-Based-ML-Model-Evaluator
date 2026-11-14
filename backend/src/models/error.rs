use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Standard structured error response required by FR-13 / FR-2
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
    pub module: String,
    pub details: Option<serde_json::Value>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StandardErrorResponse {
    pub error: ErrorDetail,
}

impl StandardErrorResponse {
    pub fn new(code: &str, message: &str, details: Option<serde_json::Value>) -> Self {
        Self {
            error: ErrorDetail {
                code: code.to_string(),
                message: message.to_string(),
                module: "web_ml_backend".to_string(),
                details,
                timestamp: Utc::now(),
            },
        }
    }

    pub fn with_module(code: &str, message: &str, module: &str, details: Option<serde_json::Value>) -> Self {
        Self {
            error: ErrorDetail {
                code: code.to_string(),
                message: message.to_string(),
                module: module.to_string(),
                details,
                timestamp: Utc::now(),
            },
        }
    }
}

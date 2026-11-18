use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StandardError {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
    pub details: Option<Value>,
}

impl StandardError {
    pub fn new(code: &str, message: &str, details: Option<Value>) -> Self {
        Self {
            error: ErrorDetail {
                code: code.to_string(),
                message: message.to_string(),
                details,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainRequest {
    pub dataset_id: String,
    pub task_type: String, // "classification", "regression", "clustering"
    pub algorithm: String,
    pub target_column: Option<String>,
    pub feature_columns: Vec<String>,
    pub test_split_ratio: Option<f64>,
    pub stratified_split: Option<bool>,
    pub hyperparameters: Option<HashMap<String, Value>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainResponse {
    pub execution_id: String,
    pub model_id: String,
    pub dataset_id: String,
    pub task_type: String,
    pub algorithm: String,
    pub target_column: Option<String>,
    pub feature_columns: Vec<String>,
    pub train_samples: usize,
    pub test_samples: usize,
    pub train_time_ms: f64,
    pub metrics: Value,
    pub confusion_matrix: Option<Value>,
    pub class_labels: Option<Vec<String>>,
    pub residuals_summary: Option<Value>,
    pub sample_predictions: Vec<Value>,
    pub download_url: String,
    pub executed_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EvaluateResponse {
    pub evaluation_id: String,
    pub model_filename: String,
    pub model_format: String,
    pub dataset_filename: String,
    pub task_type: String,
    pub algorithm: String,
    pub target_column: String,
    pub total_evaluated_samples: usize,
    pub evaluation_time_ms: f64,
    pub metrics: Value,
    pub confusion_matrix: Option<Value>,
    pub class_labels: Option<Vec<String>>,
    pub sample_predictions: Vec<Value>,
}

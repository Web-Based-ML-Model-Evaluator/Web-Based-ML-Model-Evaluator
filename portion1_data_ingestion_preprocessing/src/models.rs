use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
                module: "portion1_data_ingestion_preprocessing".to_string(),
                details,
                timestamp: Utc::now(),
            },
        }
    }
}

/// Metadata summary of a single column
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ColumnSummary {
    pub name: String,
    pub inferred_type: String, // "Numeric", "Categorical", "DateTime"
    pub missing_count: usize,
    pub missing_percentage: f64,
    pub sample_values: Vec<String>,
}

/// Response returned immediately after CSV ingestion (FR-1 & FR-2)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DatasetUploadResponse {
    pub dataset_id: String,
    pub filename: String,
    pub total_rows: usize,
    pub total_columns: usize,
    pub column_names: Vec<String>,
    pub column_summaries: Vec<ColumnSummary>,
    pub preview_rows: Vec<HashMap<String, String>>, // 5-row preview
    pub file_size_bytes: usize,
    pub uploaded_at: DateTime<Utc>,
}

/// Preprocessing imputation strategy for handling missing values (FR-3)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ImputationStrategy {
    DropRowsWithNull,
    DropColumn,
    Mean,
    Median,
    Mode,
    Constant(String),
}

/// Request payload to clean and preprocess the dataset (FR-3)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PreprocessingRequest {
    pub imputation_rules: HashMap<String, ImputationStrategy>,
    pub drop_duplicates: bool,
    pub trim_whitespace: bool,
}

/// Response after preprocessing is completed (FR-3)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PreprocessingResponse {
    pub dataset_id: String,
    pub original_rows: usize,
    pub cleaned_rows: usize,
    pub remaining_columns: Vec<String>,
    pub missing_values_remaining: usize,
    pub preview_rows: Vec<HashMap<String, String>>,
    pub status: String,
}

/// Cleaned dataset representation served to downstream Portion 2 & Portion 3
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CleanedDatasetDTO {
    pub dataset_id: String,
    pub total_rows: usize,
    pub columns: Vec<String>,
    pub data_types: HashMap<String, String>,
    pub records: Vec<HashMap<String, serde_json::Value>>,
}

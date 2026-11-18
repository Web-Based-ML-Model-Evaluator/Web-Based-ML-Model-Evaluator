use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnAnalysis {
    pub name: String,
    pub inferred_type: String, // "Numeric" | "Categorical"
    pub missing_count: usize,
    pub unique_count: usize,
    pub sample_values: Vec<String>,
    pub is_valid_classification_target: bool,
    pub is_valid_regression_target: bool,
    pub min_val: Option<f64>,
    pub max_val: Option<f64>,
    pub mean_val: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetAnalysis {
    pub dataset_id: String,
    pub filename: String,
    pub total_rows: usize,
    pub total_columns: usize,
    pub column_names: Vec<String>,
    pub columns: Vec<ColumnAnalysis>,
    pub preview_rows: Vec<HashMap<String, String>>,
    pub recommended_task: String, // "classification" | "regression" | "clustering"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedModelPackage {
    pub model_id: String,
    pub task_type: String,
    pub algorithm: String,
    pub hyperparameters: HashMap<String, Value>,
    pub target_column: Option<String>,
    pub feature_columns: Vec<String>,
    pub metrics: Value,
    pub confusion_matrix: Option<Value>,
    pub class_labels: Option<Vec<String>>,
    pub weights_json: String,
    pub file_size_bytes: usize,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct AppState {
    pub raw_datasets: Arc<RwLock<HashMap<String, ml_engine::core::preprocessing::RawDataset>>>,
    pub analyses: Arc<RwLock<HashMap<String, DatasetAnalysis>>>,
    pub trained_models: Arc<RwLock<HashMap<String, TrainedModelPackage>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            raw_datasets: Arc::new(RwLock::new(HashMap::new())),
            analyses: Arc::new(RwLock::new(HashMap::new())),
            trained_models: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

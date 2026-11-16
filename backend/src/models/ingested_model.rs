use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IngestedModelMetadata {
    pub model_id: String,
    pub filename: String,
    pub file_format: String, // "PyTorch (.pt)", "Joblib (.joblib)", "Pickle (.pkl)", "HDF5/Keras (.h5)", "ONNX (.onnx)", "JSON Schema"
    pub task_type: String,   // "classification" or "regression"
    pub architecture: String,
    /// Unknown until the file is actually parsed.
    pub input_features: Option<usize>,
    pub output_classes: Option<usize>,
    pub class_labels: Vec<String>,
    pub file_size_bytes: usize,
    pub uploaded_at: DateTime<Utc>,
    pub weights_summary: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IngestedModelEvaluationRequest {
    pub model_id: String,
    pub dataset_id: String,
    pub target_column: Option<String>,
    pub feature_columns: Vec<String>,
}

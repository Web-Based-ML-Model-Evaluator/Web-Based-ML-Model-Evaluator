use crate::models::execution::ModelComparisonEntry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents an individual model's ranking in a side-by-side comparison (FR-12)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ModelRankEntry {
    pub rank: usize,
    pub execution_id: String,
    pub algorithm: String,
    pub primary_metric_name: String,
    pub primary_metric_value: f64,
    pub train_time_ms: f64,
}

/// Comprehensive side-by-side comparison report across multiple trained models (FR-12 / BR-7)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ComparisonReport {
    pub dataset_id: Option<String>,
    pub task_type: String,
    pub compared_models: Vec<ModelComparisonEntry>,
    pub best_model_id: String,
    pub best_model_algorithm: String,
    pub recommendation_reason: String,
    pub metrics_table: HashMap<String, HashMap<String, f64>>,
    pub ranking: Vec<ModelRankEntry>,
}

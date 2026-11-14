use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Incoming Execution Request
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExecutionRequest {
    pub dataset_id: String,
    pub task_type: String, // "classification", "regression", "clustering"
    pub algorithm: String,
    pub target_column: Option<String>,
    pub feature_columns: Vec<String>,
    pub test_split_ratio: Option<f64>,
    pub num_clusters: Option<usize>,
    pub data: Option<serde_json::Value>, // Inline records or fetched from unified state
}

/// Classification performance metrics and confusion matrix (FR-9)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub class_labels: Vec<String>,
}

/// Regression performance metrics (FR-10)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct RegressionMetrics {
    pub mean_absolute_error: f64,     // MAE
    pub mean_squared_error: f64,      // MSE
    pub root_mean_squared_error: f64, // RMSE
    pub r2_score: f64,
}

/// 2D PCA Point for scatter plot rendering (FR-11)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct PcaScatterPoint {
    pub sample_id: usize,
    pub x: f64,
    pub y: f64,
    pub cluster: usize,
}

/// Clustering performance metrics and PCA visuals (FR-11)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ClusteringMetrics {
    pub silhouette_score: f64,
    pub davies_bouldin_index: f64,
    pub num_clusters: usize,
    pub cluster_counts: HashMap<usize, usize>,
    pub pca_scatter_data: Vec<PcaScatterPoint>,
}

/// Unified metrics container
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum MetricResults {
    Classification(ClassificationMetrics),
    Regression(RegressionMetrics),
    Clustering(ClusteringMetrics),
}

/// Stored entry for the side-by-side Model Comparison Dashboard (FR-12)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ModelComparisonEntry {
    pub execution_id: String,
    pub dataset_id: String,
    pub task_type: String,
    pub algorithm: String,
    pub executed_at: DateTime<Utc>,
    pub train_time_ms: f64,
    pub primary_metric_name: String,
    pub primary_metric_value: f64,
    pub summary_metrics: HashMap<String, f64>,
}

/// Full Execution Response
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExecutionResponse {
    pub execution_id: String,
    pub dataset_id: String,
    pub task_type: String,
    pub algorithm: String,
    pub train_time_ms: f64,
    pub train_samples: usize,
    pub test_samples: usize,
    pub metrics: MetricResults,
    pub comparison_entry: ModelComparisonEntry,
}


use serde::{Deserialize, Serialize};

/// Permitted ML Problem Types (FR-4)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Classification,
    Regression,
    Clustering,
}

/// Description of algorithms supported for a task
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskSupportInfo {
    pub task_type: TaskType,
    pub display_name: String,
    pub description: String,
    pub supported_algorithms: Vec<String>,
    pub requires_target: bool,
    pub allows_split: bool,
    pub hyperparameter_fields: Vec<String>,
}

/// Complete User Configuration Payload (FR-4, FR-5, FR-6, FR-7)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskConfiguration {
    pub dataset_id: String,
    pub task_type: TaskType,
    pub algorithm: String,
    pub target_column: Option<String>, // Required for Supervised (FR-5)
    pub feature_columns: Vec<String>,  // Required for all (FR-5, FR-6)
    pub test_split_ratio: Option<f64>, // 0.1 to 0.5 (FR-7)
    pub num_clusters: Option<usize>,   // Required for Clustering k >= 2 (FR-6)
    pub random_seed: Option<u64>,
}

/// Output of configuration validator (FR-13)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub summary: Option<serde_json::Value>,
}

/// Pipeline dispatch request to initiate training
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PipelineDispatchRequest {
    pub config: TaskConfiguration,
    pub mock_mode: Option<bool>,
}

/// Pipeline dispatch response
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PipelineDispatchResponse {
    pub dispatch_id: String,
    pub status: String,
    pub dataset_id: String,
    pub task_type: String,
    pub algorithm: String,
    pub execution_result: Option<serde_json::Value>,
}

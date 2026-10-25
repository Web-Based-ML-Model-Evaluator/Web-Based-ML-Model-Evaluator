use crate::models::{TaskConfiguration, TaskType, ValidationResult};
use std::collections::HashMap;

/// Returns the algorithms supported by the system
pub fn get_supported_algorithms(task: &TaskType) -> Vec<&'static str> {
    match task {
        TaskType::Classification => vec![
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "K-Nearest Neighbors (KNN)",
        ],
        TaskType::Regression => vec![
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
        ],
        TaskType::Clustering => vec!["K-Means Clustering"],
    }
}

/// Validates configuration against dataset schema and business rules
///
/// TODO: Assigned to Satvik Phadke (PES2UG24CS450)
/// Requirements (FR-4, FR-5, FR-6, FR-7, FR-13):
/// 1. Validate that the algorithm is supported for `config.task_type` (FR-4)
/// 2. For Supervised tasks (Classification, Regression):
///    - Validate target column exists (FR-5)
///    - Validate target column is NOT in `config.feature_columns` (FR-5, FR-13)
///    - Validate train/test split ratio is between 0.05 and 0.50 (FR-7)
///    - Validate target column data type is compatible (FR-13)
/// 3. For Clustering tasks:
///    - Validate number of clusters k >= 2 (FR-6)
///    - Ensure target column is omitted or ignored (FR-6)
/// 4. Validate that at least one feature column is selected
pub fn validate_task_config(
    config: &TaskConfiguration,
    _available_columns: Option<&Vec<String>>,
    _column_types: Option<&HashMap<String, String>>,
) -> ValidationResult {
    // SKELETON PLACEHOLDER: Satvik to implement validation rules
    let errors = Vec::new();
    let warnings = Vec::new();

    // TODO: Satvik Phadke - Implement validation checks here

    ValidationResult {
        is_valid: errors.is_empty(),
        errors,
        warnings,
        summary: Some(serde_json::json!({
            "task_type": config.task_type,
            "algorithm": config.algorithm,
            "target": config.target_column,
            "features_count": config.feature_columns.len(),
            "status": "SKELETON_READY_FOR_IMPLEMENTATION"
        })),
    }
}

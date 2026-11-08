use crate::models::{TaskConfiguration, TaskType, ValidationResult};
use std::collections::HashMap;

/// Returns the algorithms supported by the system.
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

/// Validates the selected task, algorithm, target, and feature columns.
pub fn validate_task_config(
    config: &TaskConfiguration,
    available_columns: Option<&Vec<String>>,
    _column_types: Option<&HashMap<String, String>>,
) -> ValidationResult {
    let mut errors = Vec::new();
    let warnings = Vec::new();

    // FR-4: Algorithm must be compatible with the selected task.
    let supported_algorithms = get_supported_algorithms(&config.task_type);

    if !supported_algorithms.contains(&config.algorithm.as_str()) {
        errors.push(format!(
            "Algorithm '{}' is not supported for task type '{:?}'.",
            config.algorithm, config.task_type
        ));
    }

    // FR-5: At least one feature column must be selected.
    if config.feature_columns.is_empty() {
        errors.push("Select at least one feature column.".to_string());
    }

    // FR-5 and FR-6: Supervised tasks require a target.
    match config.task_type {
        TaskType::Classification | TaskType::Regression => {
            match config.target_column.as_deref() {
                None => {
                    errors.push("A target column is required for supervised tasks.".to_string());
                }
                Some(target) => {
                    // FR-5: Target must not also be an input feature.
                    if config
                        .feature_columns
                        .iter()
                        .any(|feature| feature == target)
                    {
                        errors.push(
                            "The target column cannot also be selected as a feature.".to_string(),
                        );
                    }

                    // FR-5: If a schema is supplied, target must exist in it.
                    if let Some(columns) = available_columns {
                        if !columns.iter().any(|column| column == target) {
                            errors.push(format!(
                                "Target column '{}' does not exist in the dataset.",
                                target
                            ));
                        }
                    }
                }
            }
        }
        TaskType::Clustering => {
            // Clustering is unsupervised; target is not required.
            if config.target_column.is_some() {
                errors.push("Target column must be omitted for clustering.".to_string());
            }

            // Validate that selected features exist in the supplied schema.
            if let Some(columns) = available_columns {
                for feature in &config.feature_columns {
                    if !columns.iter().any(|column| column == feature) {
                        errors.push(format!(
                            "Feature column '{}' does not exist in the dataset.",
                            feature
                        ));
                    }
                }
            }
        }
    }

    // Validate selected feature columns against the schema for supervised tasks too.
    if matches!(
        config.task_type,
        TaskType::Classification | TaskType::Regression
    ) {
        if let Some(columns) = available_columns {
            for feature in &config.feature_columns {
                if !columns.iter().any(|column| column == feature) {
                    errors.push(format!(
                        "Feature column '{}' does not exist in the dataset.",
                        feature
                    ));
                }
            }
        }

        // FR-7: Train/test split ratio bounds (5% to 50%)
        if let Some(ratio) = config.test_split_ratio {
            if !(0.05..=0.50).contains(&ratio) {
                errors.push(format!(
                    "Train/test split ratio must be between 0.05 (5%) and 0.50 (50%), got {:.2}.",
                    ratio
                ));
            }
        }
    }

    // FR-6: Clustering parameter k >= 2
    if config.task_type == TaskType::Clustering {
        if let Some(k) = config.num_clusters {
            if k < 2 {
                errors.push(format!(
                    "Clustering requires at least 2 clusters (k >= 2), got {}.",
                    k
                ));
            }
        } else {
            errors.push(
                "Clustering requires specifying the number of clusters (k >= 2).".to_string(),
            );
        }
    }

    let is_valid = errors.is_empty();
    let status = if is_valid { "VALID" } else { "INVALID" };

    ValidationResult {
        is_valid,
        errors,
        warnings,
        summary: Some(serde_json::json!({
            "task_type": config.task_type,
            "algorithm": config.algorithm,
            "target": config.target_column,
            "features_count": config.feature_columns.len(),
            "status": status
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_config(
        task_type: TaskType,
        algorithm: &str,
        target_column: Option<&str>,
        feature_columns: Vec<&str>,
    ) -> TaskConfiguration {
        TaskConfiguration {
            dataset_id: "test-dataset".to_string(),
            task_type,
            algorithm: algorithm.to_string(),
            target_column: target_column.map(str::to_string),
            feature_columns: feature_columns.into_iter().map(str::to_string).collect(),
            test_split_ratio: Some(0.2),
            num_clusters: None,
            random_seed: None,
        }
    }

    #[test]
    fn accepts_supported_algorithm() {
        let config = create_config(
            TaskType::Classification,
            "Random Forest Classifier",
            Some("target"),
            vec!["feature1"],
        );

        let result = validate_task_config(&config, None, None);

        assert!(result.is_valid, "{:?}", result.errors);
    }

    #[test]
    fn rejects_unsupported_algorithm() {
        let config = create_config(
            TaskType::Classification,
            "Random Forest Regressor",
            Some("target"),
            vec!["feature1"],
        );

        let result = validate_task_config(&config, None, None);

        assert!(!result.is_valid);
    }

    #[test]
    fn rejects_missing_supervised_target() {
        let config = create_config(
            TaskType::Classification,
            "Random Forest Classifier",
            None,
            vec!["feature1"],
        );

        let result = validate_task_config(&config, None, None);

        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("target column is required")));
    }

    #[test]
    fn rejects_target_in_features() {
        let config = create_config(
            TaskType::Classification,
            "Random Forest Classifier",
            Some("target"),
            vec!["feature1", "target"],
        );

        let result = validate_task_config(&config, None, None);

        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("cannot also be selected")));
    }

    #[test]
    fn rejects_empty_features() {
        let config = create_config(
            TaskType::Regression,
            "Linear Regression",
            Some("target"),
            vec![],
        );

        let result = validate_task_config(&config, None, None);

        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("feature column")));
    }

    #[test]
    fn rejects_target_missing_from_schema() {
        let config = create_config(
            TaskType::Classification,
            "Random Forest Classifier",
            Some("missing_target"),
            vec!["feature1"],
        );
        let columns = vec!["feature1".to_string(), "target".to_string()];

        let result = validate_task_config(&config, Some(&columns), None);

        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("does not exist")));
    }

    #[test]
    fn rejects_out_of_bounds_split_ratio() {
        let mut config = create_config(
            TaskType::Classification,
            "Random Forest Classifier",
            Some("target"),
            vec!["feature1"],
        );
        config.test_split_ratio = Some(0.85); // 85% exceeds 50% limit
        let result = validate_task_config(&config, None, None);
        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("Train/test split ratio")));
    }

    #[test]
    fn rejects_clustering_with_k_less_than_two() {
        let config = TaskConfiguration {
            dataset_id: "test-dataset".to_string(),
            task_type: TaskType::Clustering,
            algorithm: "K-Means Clustering".to_string(),
            target_column: None,
            feature_columns: vec!["feature1".to_string()],
            test_split_ratio: None,
            num_clusters: Some(1), // k < 2
            random_seed: None,
        };
        let result = validate_task_config(&config, None, None);
        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("at least 2 clusters")));
    }
}

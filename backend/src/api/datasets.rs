use crate::models::StandardError;
use crate::state::{AppState, ColumnAnalysis, DatasetAnalysis};
use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::Json,
};
use ml_engine::core::preprocessing::{is_missing, RawDataset};
use ml_engine::core::validator::{sanitize_filename, validate_csv_upload};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

pub async fn upload_dataset(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<DatasetAnalysis>, (StatusCode, Json<StandardError>)> {
    let mut filename = "dataset.csv".to_string();
    let mut file_bytes: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" || name == "dataset" {
            if let Some(f_name) = field.file_name() {
                filename = sanitize_filename(f_name);
            }
            let data = field.bytes().await.map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(StandardError::new(
                        "UPLOAD_FAILED",
                        &format!("Failed to read uploaded file: {}", e),
                        None,
                    )),
                )
            })?;
            file_bytes = Some(data.to_vec());
            break;
        }
    }

    let bytes = file_bytes.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(StandardError::new(
                "MISSING_FILE_FIELD",
                "Multipart payload must include a 'file' or 'dataset' key.",
                None,
            )),
        )
    })?;

    // Validate size and format
    if let Err(err) = validate_csv_upload(&filename, &bytes) {
        return Err((StatusCode::BAD_REQUEST, Json(StandardError::new(&err.error.code, &err.error.message, err.error.details))));
    }

    let content = bytes
        .strip_prefix(&[0xEF, 0xBB, 0xBF])
        .unwrap_or(&bytes);

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(content);

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(StandardError::new(
                    "CSV_PARSE_ERROR",
                    &format!("Failed to parse CSV header row: {}", e),
                    None,
                )),
            )
        })?
        .iter()
        .map(|s| s.trim().to_string())
        .collect();

    if headers.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(StandardError::new(
                "EMPTY_CSV",
                "The CSV file must contain a header row with at least 2 columns.",
                None,
            )),
        ));
    }

    let mut rows: Vec<Vec<String>> = Vec::new();
    for result in reader.records() {
        if let Ok(rec) = result {
            let row: Vec<String> = rec.iter().map(|s| s.trim().to_string()).collect();
            if row.iter().any(|v| !v.is_empty()) {
                rows.push(row);
            }
        }
    }

    if rows.len() < 5 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(StandardError::new(
                "TOO_FEW_ROWS",
                "Dataset contains fewer than 5 rows. Please provide at least 5 rows for training.",
                None,
            )),
        ));
    }

    let dataset_id = Uuid::new_v4().to_string();

    // Column-by-column smart profiling
    let mut columns_analysis = Vec::with_capacity(headers.len());
    let mut rec_task = "classification".to_string();

    for (col_idx, col_name) in headers.iter().enumerate() {
        let mut missing_cnt = 0;
        let mut distinct_set = HashSet::new();
        let mut numeric_vals = Vec::new();
        let mut is_num = true;

        for row in &rows {
            let val = row.get(col_idx).map(|s| s.as_str()).unwrap_or("");
            if is_missing(val) {
                missing_cnt += 1;
            } else {
                distinct_set.insert(val.to_string());
                if let Ok(num) = val.parse::<f64>() {
                    numeric_vals.push(num);
                } else {
                    is_num = false;
                }
            }
        }

        let unique_count = distinct_set.len();
        let mut sample_values: Vec<String> = distinct_set.into_iter().take(5).collect();
        sample_values.sort();

        let inferred_type = if is_num && !numeric_vals.is_empty() {
            "Numeric".to_string()
        } else {
            "Categorical".to_string()
        };

        let is_valid_classification_target = inferred_type == "Categorical" || (inferred_type == "Numeric" && unique_count <= 20);
        let is_valid_regression_target = inferred_type == "Numeric" && unique_count > 3;

        let (min_val, max_val, mean_val) = if inferred_type == "Numeric" && !numeric_vals.is_empty() {
            let sum: f64 = numeric_vals.iter().sum();
            let mean = sum / (numeric_vals.len() as f64);
            let min = numeric_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = numeric_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (Some(min), Some(max), Some(mean))
        } else {
            (None, None, None)
        };

        let lower = col_name.to_lowercase();
        if (lower.contains("price") || lower.contains("salary") || lower.contains("cost")) && is_valid_regression_target {
            rec_task = "regression".to_string();
        }

        columns_analysis.push(ColumnAnalysis {
            name: col_name.clone(),
            inferred_type,
            missing_count: missing_cnt,
            unique_count,
            sample_values,
            is_valid_classification_target,
            is_valid_regression_target,
            min_val,
            max_val,
            mean_val,
        });
    }

    let preview_rows: Vec<HashMap<String, String>> = rows
        .iter()
        .take(5)
        .map(|row| {
            let mut map = HashMap::new();
            for (col_idx, col_name) in headers.iter().enumerate() {
                let val = row.get(col_idx).cloned().unwrap_or_default();
                map.insert(col_name.clone(), val);
            }
            map
        })
        .collect();

    let analysis = DatasetAnalysis {
        dataset_id: dataset_id.clone(),
        filename: filename.clone(),
        total_rows: rows.len(),
        total_columns: headers.len(),
        column_names: headers.clone(),
        columns: columns_analysis,
        preview_rows,
        recommended_task: rec_task,
    };

    let raw_data = RawDataset {
        dataset_id: dataset_id.clone(),
        filename: filename.clone(),
        headers,
        rows,
    };

    {
        let mut raw_lock = state.raw_datasets.write().unwrap();
        raw_lock.insert(dataset_id.clone(), raw_data);

        let mut an_lock = state.analyses.write().unwrap();
        an_lock.insert(dataset_id.clone(), analysis.clone());
    }

    Ok(Json(analysis))
}

pub async fn get_dataset_analysis(
    Path(dataset_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<DatasetAnalysis>, (StatusCode, Json<StandardError>)> {
    let an_lock = state.analyses.read().unwrap();
    if let Some(analysis) = an_lock.get(&dataset_id) {
        Ok(Json(analysis.clone()))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(StandardError::new(
                "DATASET_NOT_FOUND",
                &format!("Dataset with ID '{}' was not found.", dataset_id),
                None,
            )),
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpec {
    pub name: String,
    pub label: String,
    pub field_type: String, // "number", "slider", "select"
    pub default_value: serde_json::Value,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub options: Option<Vec<String>>,
    pub requirement_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmDefinition {
    pub id: String,
    pub name: String,
    pub task_type: String,
    pub description: String,
    pub complexity: String,
    pub hyperparameters: Vec<HyperparameterSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetQuestionsResponse {
    pub dataset_id: String,
    pub filename: String,
    pub total_rows: usize,
    pub total_columns: usize,
    pub recommended_task: String,
    pub classification_targets: Vec<ColumnAnalysis>,
    pub regression_targets: Vec<ColumnAnalysis>,
    pub candidate_features: Vec<ColumnAnalysis>,
    pub algorithms: Vec<AlgorithmDefinition>,
}

pub async fn get_dataset_questions(
    Path(dataset_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<DatasetQuestionsResponse>, (StatusCode, Json<StandardError>)> {
    let an_lock = state.analyses.read().unwrap();
    let analysis = an_lock.get(&dataset_id).cloned().ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(StandardError::new(
                "DATASET_NOT_FOUND",
                &format!("Dataset with ID '{}' was not found.", dataset_id),
                None,
            )),
        )
    })?;

    let classification_targets: Vec<ColumnAnalysis> = analysis
        .columns
        .iter()
        .filter(|c| c.is_valid_classification_target)
        .cloned()
        .collect();

    let regression_targets: Vec<ColumnAnalysis> = analysis
        .columns
        .iter()
        .filter(|c| c.is_valid_regression_target)
        .cloned()
        .collect();

    let candidate_features: Vec<ColumnAnalysis> = analysis
        .columns
        .iter()
        .filter(|c| c.inferred_type == "Numeric")
        .cloned()
        .collect();

    let algorithms = vec![
        // Classification
        AlgorithmDefinition {
            id: "logistic_regression".to_string(),
            name: "Logistic Regression".to_string(),
            task_type: "classification".to_string(),
            description: "Linear classification model using log-odds. Ideal for linearly separable binary or multi-class targets.".to_string(),
            complexity: "O(n · d)".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "max_iter".to_string(),
                    label: "Maximum Iterations".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(100),
                    min: Some(10.0),
                    max: Some(1000.0),
                    step: Some(10.0),
                    options: None,
                    requirement_text: "Requires integer >= 10. Higher values allow convergence on complex non-linear frontiers.".to_string(),
                },
                HyperparameterSpec {
                    name: "solver".to_string(),
                    label: "Optimization Solver".to_string(),
                    field_type: "select".to_string(),
                    default_value: serde_json::json!("lbfgs"),
                    min: None,
                    max: None,
                    step: None,
                    options: Some(vec!["lbfgs".to_string(), "sgd".to_string()]),
                    requirement_text: "lbfgs is recommended for small-to-medium tabular datasets.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "random_forest_classifier".to_string(),
            name: "Random Forest Classifier".to_string(),
            task_type: "classification".to_string(),
            description: "Ensemble of decorrelated decision trees. Excellent accuracy and robustness to outliers.".to_string(),
            complexity: "O(M · n · log(n))".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "n_trees".to_string(),
                    label: "Number of Trees".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(10),
                    min: Some(1.0),
                    max: Some(100.0),
                    step: Some(1.0),
                    options: None,
                    requirement_text: "Range: 1 to 100. More trees reduce variance but increase training time.".to_string(),
                },
                HyperparameterSpec {
                    name: "max_depth".to_string(),
                    label: "Maximum Tree Depth".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(5),
                    min: Some(1.0),
                    max: Some(20.0),
                    step: Some(1.0),
                    options: None,
                    requirement_text: "Range: 1 to 20. Deeper trees capture complex splits but risk overfitting.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "decision_tree_classifier".to_string(),
            name: "Decision Tree Classifier".to_string(),
            task_type: "classification".to_string(),
            description: "Single interpretable recursive tree splitting on Gini impurity or entropy.".to_string(),
            complexity: "O(n · log(n))".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "max_depth".to_string(),
                    label: "Max Depth".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(4),
                    min: Some(1.0),
                    max: Some(15.0),
                    step: Some(1.0),
                    options: None,
                    requirement_text: "Controls tree height. Keeping depth <= 5 prevents overfitting on small datasets.".to_string(),
                },
                HyperparameterSpec {
                    name: "min_samples_split".to_string(),
                    label: "Min Samples to Split".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(2),
                    min: Some(2.0),
                    max: Some(20.0),
                    step: Some(1.0),
                    options: None,
                    requirement_text: "Minimum internal node samples required to split further.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "knn_classifier".to_string(),
            name: "K-Nearest Neighbors (KNN)".to_string(),
            task_type: "classification".to_string(),
            description: "Non-parametric instance-based classifier. Classifies queries by majority vote of k neighbors.".to_string(),
            complexity: "O(n · d)".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "k".to_string(),
                    label: "Number of Neighbors (k)".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(3),
                    min: Some(1.0),
                    max: Some(15.0),
                    step: Some(1.0),
                    options: None,
                    requirement_text: "Odd numbers (3, 5, 7) recommended to avoid ties in binary classification.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "neural_network".to_string(),
            name: "Neural Network (MLP)".to_string(),
            task_type: "classification".to_string(),
            description: "Multi-layer perceptron neural network with hidden layers and backpropagation.".to_string(),
            complexity: "O(epochs · n · layers)".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "epochs".to_string(),
                    label: "Training Epochs".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(50),
                    min: Some(10.0),
                    max: Some(300.0),
                    step: Some(10.0),
                    options: None,
                    requirement_text: "Number of forward/backward passes through the dataset.".to_string(),
                },
                HyperparameterSpec {
                    name: "learning_rate".to_string(),
                    label: "Learning Rate".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(0.01),
                    min: Some(0.001),
                    max: Some(0.1),
                    step: Some(0.005),
                    options: None,
                    requirement_text: "Gradient descent step size. Lower values provide stable convergence.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "svm_classifier".to_string(),
            name: "Support Vector Machine (SVC)".to_string(),
            task_type: "classification".to_string(),
            description: "Max-margin hyperplane classifier. Highly effective in high-dimensional feature spaces.".to_string(),
            complexity: "O(n² · d)".to_string(),
            hyperparameters: vec![],
        },
        AlgorithmDefinition {
            id: "naive_bayes".to_string(),
            name: "Gaussian Naive Bayes".to_string(),
            task_type: "classification".to_string(),
            description: "Probabilistic classifier applying Bayes theorem under strong feature independence assumptions.".to_string(),
            complexity: "O(n · d)".to_string(),
            hyperparameters: vec![],
        },
        AlgorithmDefinition {
            id: "cnn_classifier".to_string(),
            name: "1D Convolutional Neural Network (CNN)".to_string(),
            task_type: "classification".to_string(),
            description: "Deep 1D convolution filters extracting local sequential and feature correlation patterns.".to_string(),
            complexity: "O(epochs · n · filters)".to_string(),
            hyperparameters: vec![],
        },
        // Regression
        AlgorithmDefinition {
            id: "linear_regression".to_string(),
            name: "Linear Regression".to_string(),
            task_type: "regression".to_string(),
            description: "Standard ordinary least squares regression. Models linear relationships with continuous targets.".to_string(),
            complexity: "O(d³ + n · d²)".to_string(),
            hyperparameters: vec![],
        },
        AlgorithmDefinition {
            id: "random_forest_regressor".to_string(),
            name: "Random Forest Regressor".to_string(),
            task_type: "regression".to_string(),
            description: "Ensemble of regression trees that averages continuous predictions to minimize MSE.".to_string(),
            complexity: "O(M · n · log(n))".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "n_trees".to_string(),
                    label: "Number of Trees".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(10),
                    min: Some(1.0),
                    max: Some(100.0),
                    step: Some(1.0),
                    options: None,
                    requirement_text: "Number of trees in forest. 10-30 balances accuracy and low latency.".to_string(),
                },
                HyperparameterSpec {
                    name: "max_depth".to_string(),
                    label: "Maximum Depth".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(5),
                    min: Some(1.0),
                    max: Some(20.0),
                    step: Some(1.0),
                    options: None,
                    requirement_text: "Maximum depth limit to avoid overfitting continuous target noise.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "decision_tree_regressor".to_string(),
            name: "Decision Tree Regressor".to_string(),
            task_type: "regression".to_string(),
            description: "Single recursive regression tree minimizing squared error at each partition.".to_string(),
            complexity: "O(n · log(n))".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "max_depth".to_string(),
                    label: "Max Depth".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(4),
                    min: Some(1.0),
                    max: Some(15.0),
                    step: Some(1.0),
                    options: None,
                    requirement_text: "Depth ceiling for regression partitions.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "ridge_regression".to_string(),
            name: "Ridge Regression (L2)".to_string(),
            task_type: "regression".to_string(),
            description: "Linear regression with L2 Tikhonov regularization. Prevents extreme weights on multicollinear features.".to_string(),
            complexity: "O(d³ + n · d²)".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "alpha".to_string(),
                    label: "Regularization Strength (alpha)".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(1.0),
                    min: Some(0.01),
                    max: Some(10.0),
                    step: Some(0.1),
                    options: None,
                    requirement_text: "Alpha > 0. Larger values enforce stronger shrinkage on feature weights.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "lasso_regression".to_string(),
            name: "Lasso Regression (L1)".to_string(),
            task_type: "regression".to_string(),
            description: "Linear regression with L1 norm penalty. Drives redundant feature coefficients exactly to zero.".to_string(),
            complexity: "O(n · d · iter)".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "alpha".to_string(),
                    label: "L1 Penalty (alpha)".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(1.0),
                    min: Some(0.01),
                    max: Some(10.0),
                    step: Some(0.1),
                    options: None,
                    requirement_text: "Alpha > 0. Higher penalty forces sparsity in feature selection.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "knn_regressor".to_string(),
            name: "K-Nearest Neighbors Regressor".to_string(),
            task_type: "regression".to_string(),
            description: "Local interpolation predicting continuous target by averaging the k closest training points.".to_string(),
            complexity: "O(n · d)".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "k".to_string(),
                    label: "Number of Neighbors (k)".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(3),
                    min: Some(1.0),
                    max: Some(15.0),
                    step: Some(1.0),
                    options: None,
                    requirement_text: "Number of neighbors to average. 3 to 7 works best for smooth continuous predictions.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "svr".to_string(),
            name: "Support Vector Regressor (SVR)".to_string(),
            task_type: "regression".to_string(),
            description: "Support Vector Regression with epsilon-tube insensitive loss. Highly robust against outliers.".to_string(),
            complexity: "O(n² · d)".to_string(),
            hyperparameters: vec![],
        },
        AlgorithmDefinition {
            id: "neural_network_regressor".to_string(),
            name: "Neural Network Regressor (MLP)".to_string(),
            task_type: "regression".to_string(),
            description: "Non-linear multi-layer neural network with MSE loss for continuous numeric prediction.".to_string(),
            complexity: "O(epochs · n · layers)".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "epochs".to_string(),
                    label: "Training Epochs".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(50),
                    min: Some(10.0),
                    max: Some(300.0),
                    step: Some(10.0),
                    options: None,
                    requirement_text: "Training iterations for MSE minimization.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "cnn_regressor".to_string(),
            name: "1D Convolutional Neural Network (CNN)".to_string(),
            task_type: "regression".to_string(),
            description: "1D Convolution filters for continuous numeric target regression.".to_string(),
            complexity: "O(epochs · n · filters)".to_string(),
            hyperparameters: vec![],
        },
        // Clustering
        AlgorithmDefinition {
            id: "kmeans".to_string(),
            name: "K-Means Clustering".to_string(),
            task_type: "clustering".to_string(),
            description: "Unsupervised centroid partitioning with PCA 2D dimensionality reduction.".to_string(),
            complexity: "O(k · n · d · iter)".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "num_clusters".to_string(),
                    label: "Number of Clusters (k)".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(3),
                    min: Some(2.0),
                    max: Some(10.0),
                    step: Some(1.0),
                    options: None,
                    requirement_text: "Must be between 2 and 10. Does not require any target column.".to_string(),
                },
            ],
        },
        AlgorithmDefinition {
            id: "dbscan".to_string(),
            name: "DBSCAN Density-Based Clustering".to_string(),
            task_type: "clustering".to_string(),
            description: "Density-based spatial clustering of applications with noise. Automatically determines cluster count and detects outliers.".to_string(),
            complexity: "O(n² · d)".to_string(),
            hyperparameters: vec![
                HyperparameterSpec {
                    name: "eps".to_string(),
                    label: "Neighborhood Radius (epsilon)".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(0.5),
                    min: Some(0.1),
                    max: Some(5.0),
                    step: Some(0.1),
                    options: None,
                    requirement_text: "Maximum distance between two points for them to be considered in the same neighborhood.".to_string(),
                },
                HyperparameterSpec {
                    name: "min_samples".to_string(),
                    label: "Min Neighborhood Samples".to_string(),
                    field_type: "slider".to_string(),
                    default_value: serde_json::json!(3),
                    min: Some(2.0),
                    max: Some(10.0),
                    step: Some(1.0),
                    options: None,
                    requirement_text: "Minimum number of points required to form a dense core cluster.".to_string(),
                },
            ],
        },
    ];

    Ok(Json(DatasetQuestionsResponse {
        dataset_id: analysis.dataset_id,
        filename: analysis.filename,
        total_rows: analysis.total_rows,
        total_columns: analysis.total_columns,
        recommended_task: analysis.recommended_task,
        classification_targets,
        regression_targets,
        candidate_features,
        algorithms,
    }))
}

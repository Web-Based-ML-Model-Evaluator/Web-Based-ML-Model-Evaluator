use crate::comparison::ComparisonStore;
use crate::metrics::{
    compute_classification_metrics, compute_clustering_metrics, compute_regression_metrics,
};
use crate::ml_algorithms::{
    project_pca_2d, run_classification_model, run_kmeans, run_regression_model,
};
use crate::models::{
    ExecutionRequest, ExecutionResponse, MetricResults, ModelComparisonEntry,
    StandardErrorResponse,
};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use chrono::Utc;
use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;

#[derive(Clone, Default)]
pub struct AppState {
    pub comparison_store: ComparisonStore,
}

/// GET /health
pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "portion3_model_execution_evaluation",
        "owner": "Shreya Patil (PES2UG24CS484)",
        "port": 8003
    }))
}

/// GET /api/v1/models/supported
pub async fn get_supported_algorithms() -> impl IntoResponse {
    Json(serde_json::json!({
        "classification": [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "K-Nearest Neighbors (KNN)"
        ],
        "regression": [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor"
        ],
        "clustering": ["K-Means Clustering"]
    }))
}

/// POST /api/v1/models/execute (FR-8, FR-9, FR-10, FR-11, FR-12, NFR-1)
pub async fn execute_model(
    State(state): State<AppState>,
    Json(req): Json<ExecutionRequest>,
) -> Result<Json<ExecutionResponse>, (StatusCode, Json<StandardErrorResponse>)> {
    let start_time = Instant::now();

    // 1. Extract records from payload or mock
    let records = match &req.data {
        Some(serde_json::Value::Array(arr)) => arr.clone(),
        Some(serde_json::Value::Object(map)) => {
            if let Some(serde_json::Value::Array(recs)) = map.get("records") {
                recs.clone()
            } else {
                vec![]
            }
        }
        _ => {
            // Default mock data so Shreya Patil can test without external APIs!
            vec![
                serde_json::json!({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "species": "setosa"}),
                serde_json::json!({"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "species": "setosa"}),
                serde_json::json!({"sepal_length": 4.7, "sepal_width": 3.2, "petal_length": 1.3, "species": "setosa"}),
                serde_json::json!({"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "species": "versicolor"}),
                serde_json::json!({"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "species": "versicolor"}),
                serde_json::json!({"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 4.9, "species": "versicolor"}),
                serde_json::json!({"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "species": "virginica"}),
                serde_json::json!({"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 5.1, "species": "virginica"}),
                serde_json::json!({"sepal_length": 7.1, "sepal_width": 3.0, "petal_length": 5.9, "species": "virginica"}),
                serde_json::json!({"sepal_length": 7.6, "sepal_width": 3.0, "petal_length": 6.6, "species": "virginica"})
            ]
        }
    };

    if records.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(StandardErrorResponse::new(
                "NO_DATA_RECORDS",
                "Dataset contains no records to train on.",
                None,
            )),
        ));
    }

    // 2. Extract feature matrix X
    let mut feature_matrix: Vec<Vec<f64>> = Vec::with_capacity(records.len());
    for rec in &records {
        let mut row = Vec::with_capacity(req.feature_columns.len());
        for f in &req.feature_columns {
            let val = match rec.get(f) {
                Some(serde_json::Value::Number(n)) => n.as_f64().unwrap_or(0.0),
                Some(serde_json::Value::String(s)) => s.parse::<f64>().unwrap_or(0.0),
                _ => 0.0,
            };
            row.push(val);
        }
        feature_matrix.push(row);
    }

    let execution_id = Uuid::new_v4().to_string();
    let total_samples = feature_matrix.len();
    let split_ratio = req.test_split_ratio.unwrap_or(0.20);
    let test_samples = ((total_samples as f64 * split_ratio).round() as usize).max(1);
    let train_samples = total_samples.saturating_sub(test_samples);

    let (metrics_result, comparison_entry) = match req.task_type.to_lowercase().as_str() {
        "classification" => {
            let target_col = req.target_column.as_deref().unwrap_or("target");
            let labels: Vec<String> = records
                .iter()
                .map(|r| {
                    r.get(target_col)
                        .map(|v| match v {
                            serde_json::Value::String(s) => s.clone(),
                            _ => v.to_string(),
                        })
                        .unwrap_or_else(|| "unknown".to_string())
                })
                .collect();

            // Train / test split
            let (x_train, x_test) = feature_matrix.split_at(train_samples);
            let (y_train, y_test) = labels.split_at(train_samples);

            let predictions = run_classification_model(&req.algorithm, x_train, y_train, x_test);
            let class_metrics = compute_classification_metrics(y_test, &predictions);

            let mut summary_map = HashMap::new();
            summary_map.insert("accuracy".to_string(), class_metrics.accuracy);
            summary_map.insert("precision".to_string(), class_metrics.precision);
            summary_map.insert("recall".to_string(), class_metrics.recall);
            summary_map.insert("f1_score".to_string(), class_metrics.f1_score);

            let comp_entry = ModelComparisonEntry {
                execution_id: execution_id.clone(),
                dataset_id: req.dataset_id.clone(),
                task_type: "Classification".to_string(),
                algorithm: req.algorithm.clone(),
                executed_at: Utc::now(),
                train_time_ms: 0.0,
                primary_metric_name: "Accuracy".to_string(),
                primary_metric_value: class_metrics.accuracy,
                summary_metrics: summary_map,
            };

            (MetricResults::Classification(class_metrics), comp_entry)
        }
        "regression" => {
            let target_col = req.target_column.as_deref().unwrap_or("target");
            let targets: Vec<f64> = records
                .iter()
                .map(|r| {
                    r.get(target_col)
                        .and_then(|v| match v {
                            serde_json::Value::Number(n) => n.as_f64(),
                            serde_json::Value::String(s) => s.parse::<f64>().ok(),
                            _ => None,
                        })
                        .unwrap_or(0.0)
                })
                .collect();

            let (x_train, x_test) = feature_matrix.split_at(train_samples);
            let (y_train, y_test) = targets.split_at(train_samples);

            let predictions = run_regression_model(&req.algorithm, x_train, y_train, x_test);
            let reg_metrics = compute_regression_metrics(y_test, &predictions);

            let mut summary_map = HashMap::new();
            summary_map.insert("mae".to_string(), reg_metrics.mean_absolute_error);
            summary_map.insert("mse".to_string(), reg_metrics.mean_squared_error);
            summary_map.insert("rmse".to_string(), reg_metrics.root_mean_squared_error);
            summary_map.insert("r2".to_string(), reg_metrics.r2_score);

            let comp_entry = ModelComparisonEntry {
                execution_id: execution_id.clone(),
                dataset_id: req.dataset_id.clone(),
                task_type: "Regression".to_string(),
                algorithm: req.algorithm.clone(),
                executed_at: Utc::now(),
                train_time_ms: 0.0,
                primary_metric_name: "R2 Score".to_string(),
                primary_metric_value: reg_metrics.r2_score,
                summary_metrics: summary_map,
            };

            (MetricResults::Regression(reg_metrics), comp_entry)
        }
        "clustering" => {
            let k = req.num_clusters.unwrap_or(3);
            let assignments = run_kmeans(&feature_matrix, k, 30);
            let pca_points = project_pca_2d(&feature_matrix, &assignments);
            let cluster_metrics =
                compute_clustering_metrics(&feature_matrix, &assignments, k, pca_points);

            let mut summary_map = HashMap::new();
            summary_map.insert("silhouette".to_string(), cluster_metrics.silhouette_score);
            summary_map.insert("davies_bouldin".to_string(), cluster_metrics.davies_bouldin_index);

            let comp_entry = ModelComparisonEntry {
                execution_id: execution_id.clone(),
                dataset_id: req.dataset_id.clone(),
                task_type: "Clustering".to_string(),
                algorithm: req.algorithm.clone(),
                executed_at: Utc::now(),
                train_time_ms: 0.0,
                primary_metric_name: "Silhouette Score".to_string(),
                primary_metric_value: cluster_metrics.silhouette_score,
                summary_metrics: summary_map,
            };

            (MetricResults::Clustering(cluster_metrics), comp_entry)
        }
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(StandardErrorResponse::new(
                    "INVALID_TASK_TYPE",
                    &format!(
                        "Unsupported task '{}'. Must be classification, regression, or clustering.",
                        req.task_type
                    ),
                    None,
                )),
            ));
        }
    };

    let elapsed_ms = (start_time.elapsed().as_secs_f64() * 1000.0 * 10.0).round() / 10.0;

    // Enforce 30-second pipeline latency constraint (PR-1, PR-2, NFR-1)
    if start_time.elapsed().as_secs() >= 30 {
        return Err((
            StatusCode::GATEWAY_TIMEOUT,
            Json(StandardErrorResponse::new(
                "PIPELINE_LATENCY_EXCEEDED",
                &format!(
                    "Model execution exceeded the 30-second latency budget (took {:.2}s, limit 30s) per PR-1, PR-2, and NFR-1.",
                    start_time.elapsed().as_secs_f64()
                ),
                Some(serde_json::json!({
                    "elapsed_ms": elapsed_ms,
                    "max_allowed_seconds": 30
                })),
            )),
        ));
    }

    let mut final_comp_entry = comparison_entry;
    final_comp_entry.train_time_ms = elapsed_ms;

    // Record to Model Comparison Dashboard store (FR-12)
    state.comparison_store.record_run(final_comp_entry.clone());

    Ok(Json(ExecutionResponse {
        execution_id,
        dataset_id: req.dataset_id,
        task_type: req.task_type,
        algorithm: req.algorithm,
        train_time_ms: elapsed_ms,
        train_samples,
        test_samples,
        metrics: metrics_result,
        comparison_entry: final_comp_entry,
    }))
}

/// GET /api/v1/models/comparison/{session_id} (FR-12)
pub async fn get_comparison_dashboard(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Json<serde_json::Value> {
    let history = state.comparison_store.get_history(&session_id);
    Json(serde_json::json!({
        "dataset_id": session_id,
        "total_runs": history.len(),
        "runs": history
    }))
}

/// POST /api/v1/models/compare (FR-12)
pub async fn compare_models(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let ids: Vec<String> = payload["execution_ids"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let comparison = state.comparison_store.compare_runs(&ids);
    Json(serde_json::json!({
        "compared_models": comparison
    }))
}

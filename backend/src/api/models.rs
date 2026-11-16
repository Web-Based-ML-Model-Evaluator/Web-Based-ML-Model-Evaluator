use crate::core::ml::algorithms::{
    project_pca_2d, split_shuffled, split_stratified, try_run_classification_model, try_run_kmeans,
    try_run_regression_model,
};
use crate::core::ml::metrics::{
    compute_classification_metrics, compute_clustering_metrics, compute_regression_metrics,
};
use crate::core::preprocessing::clean_and_impute;
use crate::models::{
    ExecutionRequest, ExecutionResponse, IngestedModelEvaluationRequest, IngestedModelMetadata,
    MetricResults, ModelComparisonEntry, PreprocessingRequest, StandardErrorResponse,
};
use crate::state::AppState;
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use chrono::Utc;
use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;

/// GET /api/v1/models/supported
pub async fn get_supported_algorithms() -> impl IntoResponse {
    Json(serde_json::json!({
        "classification": [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "K-Nearest Neighbors (KNN)",
            "Neural Network (MLP)",
            "Convolutional Neural Network (CNN)"
        ],
        "regression": [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "Neural Network (MLP)",
            "Convolutional Neural Network (CNN)"
        ],
        "clustering": ["K-Means Clustering"]
    }))
}

type ApiError = (StatusCode, StandardErrorResponse);

fn api_error(status: StatusCode, code: &str, message: String) -> ApiError {
    (status, StandardErrorResponse::new(code, &message, None))
}

/// Returns the records of a dataset in the session, as JSON objects.
///
/// Uses the cleaned data when the dataset was preprocessed. A dataset that was uploaded
/// but not preprocessed is only usable when it has no missing values (GitHub #23 lets such
/// datasets skip Step 1's preprocessing). Never substitutes sample data: an unknown id is
/// a `DATASET_NOT_FOUND` error.
pub fn load_dataset_records(
    state: &AppState,
    dataset_id: &str,
) -> Result<Vec<serde_json::Value>, ApiError> {
    let to_values = |records: &[HashMap<String, serde_json::Value>]| -> Vec<serde_json::Value> {
        records
            .iter()
            .map(|rec| serde_json::to_value(rec).unwrap_or_default())
            .collect()
    };

    if let Ok(cleaned) = state.cleaned_datasets.read() {
        if let Some(dto) = cleaned.get(dataset_id) {
            return Ok(to_values(&dto.records));
        }
    }

    let raw = state
        .raw_datasets
        .read()
        .ok()
        .and_then(|raw| raw.get(dataset_id).cloned());
    let raw = match raw {
        Some(r) => r,
        None => {
            return Err(api_error(
                StatusCode::NOT_FOUND,
                "DATASET_NOT_FOUND",
                format!(
                    "Dataset '{}' was not found in this session. Upload it first.",
                    dataset_id
                ),
            ));
        }
    };

    // No rules: just type the values (numbers as numbers) without changing any data
    let no_rules = PreprocessingRequest {
        imputation_rules: HashMap::new(),
        drop_duplicates: false,
        trim_whitespace: true,
    };
    let (dto, summary) =
        clean_and_impute(&raw, &no_rules).map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, e))?;
    if summary.missing_values_remaining > 0 {
        return Err(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "DATASET_NEEDS_PREPROCESSING",
            format!(
                "Dataset '{}' has {} missing value(s). Choose how to handle them in Step 1 (preprocessing) before training.",
                dataset_id, summary.missing_values_remaining
            ),
        ));
    }
    Ok(to_values(&dto.records))
}

/// Reads one cell as a number. Missing columns and non-numeric values are errors,
/// never silently replaced by 0.
fn numeric_cell(record: &serde_json::Value, column: &str, row: usize) -> Result<f64, ApiError> {
    let parsed = match record.get(column) {
        Some(serde_json::Value::Number(n)) => n.as_f64(),
        Some(serde_json::Value::String(s)) => s.trim().parse::<f64>().ok(),
        None => {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "UNKNOWN_COLUMN",
                format!("Column '{}' does not exist in this dataset.", column),
            ));
        }
        _ => None,
    };
    parsed.filter(|v| v.is_finite()).ok_or_else(|| {
        api_error(
            StatusCode::BAD_REQUEST,
            "NON_NUMERIC_VALUE",
            format!(
                "Column '{}' has a non-numeric or missing value in row {}. Only numeric columns can be used here.",
                column,
                row + 1
            ),
        )
    })
}

/// Builds the feature matrix from the selected columns.
pub fn build_feature_matrix(
    records: &[serde_json::Value],
    feature_columns: &[String],
) -> Result<Vec<Vec<f64>>, ApiError> {
    if feature_columns.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "NO_FEATURES_SELECTED",
            "Select at least one feature column.".to_string(),
        ));
    }
    records
        .iter()
        .enumerate()
        .map(|(i, rec)| {
            feature_columns
                .iter()
                .map(|col| numeric_cell(rec, col, i))
                .collect()
        })
        .collect()
}

/// Reads the target column as class labels (any non-missing value).
fn label_column(records: &[serde_json::Value], target: &str) -> Result<Vec<String>, ApiError> {
    records
        .iter()
        .enumerate()
        .map(|(i, rec)| match rec.get(target) {
            Some(serde_json::Value::String(s)) if !s.trim().is_empty() => Ok(s.clone()),
            Some(serde_json::Value::Number(n)) => Ok(n.to_string()),
            Some(serde_json::Value::Bool(b)) => Ok(b.to_string()),
            None => Err(api_error(
                StatusCode::BAD_REQUEST,
                "UNKNOWN_COLUMN",
                format!("Target column '{}' does not exist in this dataset.", target),
            )),
            _ => Err(api_error(
                StatusCode::BAD_REQUEST,
                "MISSING_TARGET_VALUE",
                format!("Target column '{}' is empty in row {}.", target, i + 1),
            )),
        })
        .collect()
}

fn required_target(req_target: &Option<String>) -> Result<&str, ApiError> {
    req_target
        .as_deref()
        .filter(|t| !t.trim().is_empty())
        .ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                "MISSING_TARGET_COLUMN",
                "Select a target column for supervised learning.".to_string(),
            )
        })
}

/// Internal execution helper shared between `/api/v1/models/execute` and `/api/v1/pipeline/dispatch`
pub fn execute_model_internal(
    state: &AppState,
    req: ExecutionRequest,
) -> Result<ExecutionResponse, (StatusCode, StandardErrorResponse)> {
    let start_time = Instant::now();

    // 1. Records come from the request body or from this session's dataset store
    let records = match &req.data {
        Some(serde_json::Value::Array(arr)) => arr.clone(),
        Some(serde_json::Value::Object(map)) => match map.get("records") {
            Some(serde_json::Value::Array(recs)) => recs.clone(),
            _ => vec![],
        },
        _ => load_dataset_records(state, &req.dataset_id)?,
    };

    if records.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            StandardErrorResponse::new(
                "NO_DATA_RECORDS",
                "Dataset contains no records to train on.",
                None,
            ),
        ));
    }

    // 2. Extract feature matrix X
    let feature_matrix = build_feature_matrix(&records, &req.feature_columns)?;

    let execution_id = Uuid::new_v4().to_string();
    let total_samples = feature_matrix.len();
    let split_ratio = req.test_split_ratio.unwrap_or(0.20);
    let test_samples = ((total_samples as f64 * split_ratio).round() as usize).max(1);
    let train_samples = total_samples.saturating_sub(test_samples);

    let (metrics_result, mut comparison_entry) = match req.task_type.to_lowercase().as_str() {
        "classification" => {
            let target_col = required_target(&req.target_column)?;
            let labels = label_column(&records, target_col)?;

            // Train / test split using deterministic shuffle
            let (x_train, x_test) = match split_stratified(&feature_matrix, &labels, train_samples)
            {
                Ok(split) => split,
                Err(err) => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        StandardErrorResponse::new("SPLIT_FAILED", &err.message(), None),
                    ));
                }
            };

            let (y_train, y_test) = match split_stratified(&labels, &labels, train_samples) {
                Ok(split) => split,
                Err(err) => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        StandardErrorResponse::new("SPLIT_FAILED", &err.message(), None),
                    ));
                }
            };

            let predictions =
                match try_run_classification_model(&req.algorithm, &x_train, &y_train, &x_test) {
                    Ok(p) => p,
                    Err(err) => {
                        return Err((
                            StatusCode::BAD_REQUEST,
                            StandardErrorResponse::new("TRAINING_FAILED", &err.message(), None),
                        ));
                    }
                };
            let class_metrics = compute_classification_metrics(&y_test, &predictions);

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
            let target_col = required_target(&req.target_column)?;
            let targets: Vec<f64> = records
                .iter()
                .enumerate()
                .map(|(i, r)| numeric_cell(r, target_col, i))
                .collect::<Result<_, _>>()?;

            let (x_train, x_test) = split_shuffled(&feature_matrix, train_samples);
            let (y_train, y_test) = split_shuffled(&targets, train_samples);

            let predictions =
                match try_run_regression_model(&req.algorithm, &x_train, &y_train, &x_test) {
                    Ok(p) => p,
                    Err(err) => {
                        return Err((
                            StatusCode::BAD_REQUEST,
                            StandardErrorResponse::new("TRAINING_FAILED", &err.message(), None),
                        ));
                    }
                };
            let reg_metrics = compute_regression_metrics(&y_test, &predictions);

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
            let k = req.num_clusters.ok_or_else(|| {
                api_error(
                    StatusCode::BAD_REQUEST,
                    "MISSING_CLUSTER_COUNT",
                    "Choose the number of clusters (k) for K-Means.".to_string(),
                )
            })?;
            let assignments = try_run_kmeans(&feature_matrix, k, 30).map_err(|err| {
                api_error(StatusCode::BAD_REQUEST, "TRAINING_FAILED", err.message())
            })?;
            let pca_points = project_pca_2d(&feature_matrix, &assignments);
            let cluster_metrics =
                compute_clustering_metrics(&feature_matrix, &assignments, k, pca_points);

            let mut summary_map = HashMap::new();
            summary_map.insert("silhouette".to_string(), cluster_metrics.silhouette_score);
            summary_map.insert(
                "davies_bouldin".to_string(),
                cluster_metrics.davies_bouldin_index,
            );

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
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                StandardErrorResponse::new(
                    "UNSUPPORTED_TASK_TYPE",
                    &format!("Task type '{}' is not supported.", other),
                    None,
                ),
            ));
        }
    };

    let elapsed = start_time.elapsed().as_secs_f64() * 1000.0;
    comparison_entry.train_time_ms = (elapsed * 100.0).round() / 100.0;

    // Record to comparison store for side-by-side leaderboard (FR-12)
    state.comparison_store.record_run(comparison_entry.clone());

    Ok(ExecutionResponse {
        execution_id,
        dataset_id: req.dataset_id,
        task_type: req.task_type,
        algorithm: req.algorithm,
        train_time_ms: comparison_entry.train_time_ms,
        train_samples,
        test_samples,
        metrics: metrics_result,
        comparison_entry,
    })
}

/// POST /api/v1/models/execute (FR-8, FR-9, FR-10, FR-11, FR-12, NFR-1)
pub async fn execute_model(
    State(state): State<AppState>,
    Json(req): Json<ExecutionRequest>,
) -> Result<Json<ExecutionResponse>, (StatusCode, Json<StandardErrorResponse>)> {
    execute_model_internal(&state, req)
        .map(Json)
        .map_err(|(code, err)| (code, Json(err)))
}

/// POST /api/v1/models/session/purge
pub async fn purge_model_session(State(state): State<AppState>) -> Json<serde_json::Value> {
    state.comparison_store.purge_all();
    Json(serde_json::json!({
        "status": "PURGED_SUCCESSFULLY",
        "message": "All execution results and comparison histories purged from volatile memory."
    }))
}

/// POST /api/v1/models/upload (Issue #19, #31)
/// Supports uploading pre-trained model files (.pt, .pth, .joblib, .pkl, .h5, .onnx, .json)
pub async fn upload_model(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<IngestedModelMetadata>, (StatusCode, Json<StandardErrorResponse>)> {
    let mut filename = "model.pt".to_string();
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut task_type = "classification".to_string();
    let mut architecture = "Pretrained Neural Network".to_string();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" || name == "model" {
            if let Some(f_name) = field.file_name() {
                filename = crate::core::validator::sanitize_filename(f_name);
            }
            let data = field.bytes().await.map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(StandardErrorResponse::new(
                        "MODEL_UPLOAD_READ_FAILED",
                        &format!("Failed to read uploaded model file: {}", e),
                        None,
                    )),
                )
            })?;
            file_bytes = Some(data.to_vec());
        } else if name == "task_type" {
            if let Ok(val) = field.text().await {
                if !val.trim().is_empty() {
                    task_type = val.trim().to_lowercase();
                }
            }
        } else if name == "architecture" {
            if let Ok(val) = field.text().await {
                if !val.trim().is_empty() {
                    architecture = val.trim().to_string();
                }
            }
        }
    }

    let bytes = match file_bytes {
        Some(b) => b,
        None => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(StandardErrorResponse::new(
                    "MISSING_MODEL_FILE",
                    "No model file found in multipart upload. Field name must be 'file' or 'model'.",
                    None,
                )),
            ));
        }
    };

    let lower_filename = filename.to_lowercase();
    let (file_format, default_arch) = if lower_filename.ends_with(".pt")
        || lower_filename.ends_with(".pth")
    {
        (
            "PyTorch (.pt)",
            "PyTorch Neural Network (TorchScript / JIT)",
        )
    } else if lower_filename.ends_with(".joblib") {
        (
            "Joblib (.joblib)",
            "Scikit-Learn Pipeline (Joblib Serialization)",
        )
    } else if lower_filename.ends_with(".pkl") || lower_filename.ends_with(".pickle") {
        ("Pickle (.pkl)", "Python Pickle Serialized Estimator")
    } else if lower_filename.ends_with(".h5") || lower_filename.ends_with(".hdf5") {
        (
            "HDF5/Keras (.h5)",
            "Keras / TensorFlow HDF5 Deep Learning Model",
        )
    } else if lower_filename.ends_with(".onnx") {
        (
            "ONNX (.onnx)",
            "Open Neural Network Exchange (ONNX) Runtime Graph",
        )
    } else if lower_filename.ends_with(".json") {
        (
            "JSON Model (.json)",
            "Configured Weight Specification (JSON)",
        )
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(StandardErrorResponse::new(
                "UNSUPPORTED_MODEL_FORMAT",
                "Unsupported model file format. Allowed extensions: .pt, .pth, .joblib, .pkl, .h5, .onnx, .json",
                None,
            )),
        ));
    };

    if architecture == "Pretrained Neural Network" {
        architecture = default_arch.to_string();
    }

    let model_id = format!("model-{}", Uuid::new_v4());
    let file_size_bytes = bytes.len();
    let metadata = IngestedModelMetadata {
        model_id: model_id.clone(),
        filename: filename.clone(),
        file_format: file_format.to_string(),
        task_type: task_type.clone(),
        architecture,
        // The file is stored as-is; it is not parsed, so its shape is unknown.
        input_features: None,
        output_classes: None,
        class_labels: vec![],
        file_size_bytes,
        uploaded_at: Utc::now(),
        weights_summary: format!(
            "{file_format} file stored in session memory ({file_size_bytes} bytes). It has not been loaded or inspected."
        ),
    };

    if let Ok(mut lock) = state.ingested_models.write() {
        lock.insert(model_id.clone(), metadata.clone());
    }

    Ok(Json(metadata))
}

/// GET /api/v1/models/ingested
pub async fn list_ingested_models(
    State(state): State<AppState>,
) -> Json<Vec<IngestedModelMetadata>> {
    let models = if let Ok(lock) = state.ingested_models.read() {
        lock.values().cloned().collect()
    } else {
        vec![]
    };
    Json(models)
}

/// POST /api/v1/models/evaluate-ingested (Issue #20, #32)
///
/// Uploaded model files are stored but cannot be executed by this service, so this
/// endpoint returns an explicit error instead of producing metrics. Real on-device
/// evaluation of ONNX models is tracked in #46.
pub async fn evaluate_ingested_model(
    State(state): State<AppState>,
    Json(req): Json<IngestedModelEvaluationRequest>,
) -> Result<Json<ExecutionResponse>, (StatusCode, Json<StandardErrorResponse>)> {
    let known = state
        .ingested_models
        .read()
        .map(|models| models.contains_key(&req.model_id))
        .unwrap_or(false);
    if !known {
        return Err((
            StatusCode::NOT_FOUND,
            Json(StandardErrorResponse::new(
                "MODEL_NOT_FOUND",
                &format!(
                    "Ingested model '{}' was not found in active session.",
                    req.model_id
                ),
                None,
            )),
        ));
    }

    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(StandardErrorResponse::new(
            "MODEL_EXECUTION_UNAVAILABLE",
            "Uploaded model files cannot be executed yet, so no metrics can be produced. Train a built-in model instead.",
            None,
        )),
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::preprocessing::parse_csv_summary;

    /// Loads a CSV straight into the session, as the upload handler would.
    fn state_with_csv(id: &str, csv: &str) -> AppState {
        let state = AppState::default();
        let (raw, summary) = parse_csv_summary(id, "data.csv", csv.as_bytes()).unwrap();
        state
            .raw_datasets
            .write()
            .unwrap()
            .insert(id.to_string(), raw);
        state
            .upload_summaries
            .write()
            .unwrap()
            .insert(id.to_string(), summary);
        state
    }

    fn two_class_csv() -> String {
        let mut csv = String::from("x,y,label\n");
        for i in 0..20 {
            let label = if i % 2 == 0 { "a" } else { "b" };
            csv.push_str(&format!("{},{},{}\n", i, i * 2, label));
        }
        csv
    }

    fn request(dataset_id: &str, task: &str, target: Option<&str>) -> ExecutionRequest {
        ExecutionRequest {
            dataset_id: dataset_id.to_string(),
            task_type: task.to_string(),
            algorithm: "Decision Tree Classifier".to_string(),
            target_column: target.map(String::from),
            feature_columns: vec!["x".to_string(), "y".to_string()],
            test_split_ratio: Some(0.2),
            num_clusters: None,
            data: None,
        }
    }

    #[test]
    fn unknown_dataset_is_an_error_not_sample_data() {
        let state = AppState::default();
        let (status, err) =
            execute_model_internal(&state, request("missing", "classification", Some("label")))
                .unwrap_err();
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(err.error.code, "DATASET_NOT_FOUND");
    }

    #[test]
    fn complete_dataset_trains_without_preprocessing() {
        let state = state_with_csv("d1", &two_class_csv());
        let res =
            execute_model_internal(&state, request("d1", "classification", Some("label"))).unwrap();
        assert_eq!(res.train_samples + res.test_samples, 20);
    }

    #[test]
    fn dataset_with_gaps_must_be_preprocessed_first() {
        let csv = two_class_csv().replacen("0,0,a", "0,,a", 1);
        let state = state_with_csv("d1", &csv);
        let (status, err) =
            execute_model_internal(&state, request("d1", "classification", Some("label")))
                .unwrap_err();
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(err.error.code, "DATASET_NEEDS_PREPROCESSING");
    }

    #[test]
    fn text_feature_is_rejected_instead_of_becoming_zero() {
        let state = state_with_csv("d1", &two_class_csv());
        let mut req = request("d1", "classification", Some("x"));
        req.feature_columns = vec!["label".to_string()];
        let (_, err) = execute_model_internal(&state, req).unwrap_err();
        assert_eq!(err.error.code, "NON_NUMERIC_VALUE");
    }

    #[test]
    fn missing_target_column_is_rejected() {
        let state = state_with_csv("d1", &two_class_csv());
        let (_, err) =
            execute_model_internal(&state, request("d1", "classification", None)).unwrap_err();
        assert_eq!(err.error.code, "MISSING_TARGET_COLUMN");

        let (_, err) =
            execute_model_internal(&state, request("d1", "classification", Some("nope")))
                .unwrap_err();
        assert_eq!(err.error.code, "UNKNOWN_COLUMN");
    }

    #[test]
    fn clustering_requires_explicit_k() {
        let state = state_with_csv("d1", &two_class_csv());
        let mut req = request("d1", "clustering", None);
        req.algorithm = "K-Means Clustering".to_string();
        let (_, err) = execute_model_internal(&state, req.clone()).unwrap_err();
        assert_eq!(err.error.code, "MISSING_CLUSTER_COUNT");

        req.num_clusters = Some(2);
        assert!(execute_model_internal(&state, req).is_ok());
    }

    #[tokio::test]
    async fn uploaded_model_evaluation_never_invents_metrics() {
        let state = AppState::default();
        let meta = IngestedModelMetadata {
            model_id: "m1".to_string(),
            filename: "model.onnx".to_string(),
            file_format: "ONNX (.onnx)".to_string(),
            task_type: "classification".to_string(),
            architecture: "test".to_string(),
            input_features: None,
            output_classes: None,
            class_labels: vec![],
            file_size_bytes: 3,
            uploaded_at: Utc::now(),
            weights_summary: String::new(),
        };
        state
            .ingested_models
            .write()
            .unwrap()
            .insert("m1".to_string(), meta);

        let req = IngestedModelEvaluationRequest {
            model_id: "m1".to_string(),
            dataset_id: "d1".to_string(),
            target_column: None,
            feature_columns: vec![],
        };
        let (status, Json(err)) = evaluate_ingested_model(State(state), Json(req))
            .await
            .unwrap_err();
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(err.error.code, "MODEL_EXECUTION_UNAVAILABLE");
    }
}

use crate::models::{EvaluateResponse, StandardError};
use crate::state::AppState;
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::Json,
};
use ml_engine::core::ml::algorithms::{split_shuffled, split_stratified};
use ml_engine::core::ml::metrics::{compute_classification_metrics, compute_regression_metrics};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluateJsonRequest {
    pub dataset_id: Option<String>,
    pub dataset_csv: Option<String>,
    pub model_filename: String,
    pub task_type: String,
    pub algorithm: String,
    pub target_column: Option<String>,
    pub feature_columns: Option<Vec<String>>,
}

fn detect_model_format(filename: &str) -> Result<String, String> {
    let lower = filename.to_lowercase();
    let ext = lower.split('.').last().unwrap_or("");
    match ext {
        "pt" => Ok("PyTorch Model Tensor Checkpoint (.pt)".to_string()),
        "pth" => Ok("PyTorch Weights State Dict (.pth)".to_string()),
        "pt2" => Ok("PyTorch 2.0 Exported Program (.pt2)".to_string()),
        "pte" => Ok("ExecuTorch Edge Executable (.pte)".to_string()),
        "ptl" => Ok("TorchScript Lite Mobile Package (.ptl)".to_string()),
        "pkt" => Ok("SmartCore Model Package (.pkt)".to_string()),
        "pkl" => Ok("Python Pickle Model Archive (.pkl)".to_string()),
        "bin" => Ok("Compiled Binary Tensor Weights (.bin)".to_string()),
        "json" => Ok("Structured JSON Manifest & Weights (.json)".to_string()),
        "onnx" => Ok("Open Neural Network Exchange (.onnx)".to_string()),
        "h5" | "hdf5" => Ok("Hierarchical Data Format (.h5)".to_string()),
        "joblib" => Ok("Joblib Serialized Estimator (.joblib)".to_string()),
        _ => Err(format!(
            "Unsupported model format '.{}'. Supported formats: .pt, .pth, .pt2, .pte, .ptl, .pkt, .pkl, .bin, .json, .onnx, .h5, .joblib",
            ext
        )),
    }
}

pub async fn evaluate_model_file(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<EvaluateResponse>, (StatusCode, Json<StandardError>)> {
    let start_time = Instant::now();

    let mut model_filename = String::new();
    let mut model_bytes: Vec<u8> = Vec::new();
    let mut dataset_filename = String::new();
    let mut dataset_csv = String::new();
    let mut dataset_id = String::new();
    let mut task_type = String::new();
    let mut algorithm = String::new();
    let mut target_column = String::new();
    let mut feature_columns: Vec<String> = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "model_file" => {
                if let Some(fname) = field.file_name() {
                    model_filename = fname.to_string();
                }
                if let Ok(bytes) = field.bytes().await {
                    model_bytes = bytes.to_vec();
                }
            }
            "dataset_file" => {
                if let Some(fname) = field.file_name() {
                    dataset_filename = fname.to_string();
                }
                if let Ok(text) = field.text().await {
                    dataset_csv = text;
                }
            }
            "dataset_id" => {
                if let Ok(val) = field.text().await {
                    dataset_id = val;
                }
            }
            "task_type" => {
                if let Ok(val) = field.text().await {
                    task_type = val;
                }
            }
            "algorithm" => {
                if let Ok(val) = field.text().await {
                    algorithm = val;
                }
            }
            "target_column" => {
                if let Ok(val) = field.text().await {
                    target_column = val;
                }
            }
            "feature_columns" => {
                if let Ok(val) = field.text().await {
                    if let Ok(parsed) = serde_json::from_str::<Vec<String>>(&val) {
                        feature_columns = parsed;
                    }
                }
            }
            _ => {}
        }
    }

    if model_filename.is_empty() && !model_bytes.is_empty() {
        model_filename = "model.pt".to_string();
    }

    if model_filename.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(StandardError::new(
                "MISSING_MODEL_FILE",
                "Please upload a model file (.pt, .pth, .pt2, .pte, .ptl, .pkt, etc.).",
                None,
            )),
        ));
    }

    let model_format = detect_model_format(&model_filename).map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            Json(StandardError::new("UNSUPPORTED_FORMAT", &err, None)),
        )
    })?;

    // Load dataset CSV
    let (csv_content, d_name) = if !dataset_csv.is_empty() {
        (
            dataset_csv,
            if dataset_filename.is_empty() {
                "uploaded_dataset.csv".to_string()
            } else {
                dataset_filename
            },
        )
    } else if !dataset_id.is_empty() {
        let lock = state.raw_datasets.read().unwrap();
        let raw = lock.get(&dataset_id).cloned().ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(StandardError::new(
                    "DATASET_NOT_FOUND",
                    &format!("Dataset '{}' not found in session.", dataset_id),
                    None,
                )),
            )
        })?;
        let mut wtr = csv::Writer::from_writer(vec![]);
        let _ = wtr.write_record(&raw.headers);
        for row in &raw.rows {
            let _ = wtr.write_record(row);
        }
        let csv_bytes = wtr.into_inner().unwrap_or_default();
        let text = String::from_utf8(csv_bytes).unwrap_or_default();
        (text, raw.filename.clone())
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(StandardError::new(
                "MISSING_DATASET",
                "Please provide a test dataset CSV file or an existing dataset_id.",
                None,
            )),
        ));
    };

    // Run evaluation
    execute_evaluation(
        &model_filename,
        &model_format,
        &d_name,
        &csv_content,
        &task_type,
        &algorithm,
        if target_column.is_empty() {
            None
        } else {
            Some(target_column)
        },
        feature_columns,
        start_time,
    )
}

pub async fn evaluate_model_json(
    State(state): State<AppState>,
    Json(req): Json<EvaluateJsonRequest>,
) -> Result<Json<EvaluateResponse>, (StatusCode, Json<StandardError>)> {
    let start_time = Instant::now();
    let model_format = detect_model_format(&req.model_filename).map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            Json(StandardError::new("UNSUPPORTED_FORMAT", &err, None)),
        )
    })?;

    let (csv_content, d_name) = if let Some(csv) = req.dataset_csv {
        (csv, "dataset.csv".to_string())
    } else if let Some(id) = req.dataset_id {
        let lock = state.raw_datasets.read().unwrap();
        let raw = lock.get(&id).cloned().ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(StandardError::new(
                    "DATASET_NOT_FOUND",
                    &format!("Dataset '{}' not found.", id),
                    None,
                )),
            )
        })?;
        let mut wtr = csv::Writer::from_writer(vec![]);
        let _ = wtr.write_record(&raw.headers);
        for row in &raw.rows {
            let _ = wtr.write_record(row);
        }
        let csv_bytes = wtr.into_inner().unwrap_or_default();
        let text = String::from_utf8(csv_bytes).unwrap_or_default();
        (text, raw.filename.clone())
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(StandardError::new(
                "MISSING_DATASET",
                "Please provide dataset_csv or dataset_id.",
                None,
            )),
        ));
    };

    execute_evaluation(
        &req.model_filename,
        &model_format,
        &d_name,
        &csv_content,
        &req.task_type,
        &req.algorithm,
        req.target_column,
        req.feature_columns.unwrap_or_default(),
        start_time,
    )
}

fn execute_evaluation(
    model_filename: &str,
    model_format: &str,
    dataset_filename: &str,
    csv_content: &str,
    task_type_raw: &str,
    algorithm_raw: &str,
    target_column_opt: Option<String>,
    feature_columns: Vec<String>,
    start_time: Instant,
) -> Result<Json<EvaluateResponse>, (StatusCode, Json<StandardError>)> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(csv_content.as_bytes());

    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(StandardError::new(
                    "CSV_PARSE_ERROR",
                    &format!("Failed to read CSV headers: {}", e),
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
            Json(StandardError::new("EMPTY_CSV", "CSV has no columns.", None)),
        ));
    }

    let mut rows: Vec<Vec<String>> = Vec::new();
    for rec in rdr.records() {
        if let Ok(record) = rec {
            rows.push(record.iter().map(|s| s.trim().to_string()).collect());
        }
    }

    if rows.len() < 2 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(StandardError::new(
                "INSUFFICIENT_ROWS",
                "Test dataset must have at least 2 rows for evaluation.",
                None,
            )),
        ));
    }

    let col_indices: HashMap<String, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();

    // Determine target column
    let target_col = match target_column_opt {
        Some(col) if col_indices.contains_key(&col) => col,
        _ => headers.last().cloned().unwrap_or_else(|| "target".to_string()),
    };

    let target_idx = *col_indices.get(&target_col).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(StandardError::new(
                "TARGET_NOT_FOUND",
                &format!("Target column '{}' not found in test dataset.", target_col),
                None,
            )),
        )
    })?;

    // Determine feature columns
    let feats = if !feature_columns.is_empty() {
        feature_columns
            .into_iter()
            .filter(|f| col_indices.contains_key(f) && f != &target_col)
            .collect::<Vec<_>>()
    } else {
        headers
            .iter()
            .filter(|&h| h != &target_col)
            .cloned()
            .collect::<Vec<_>>()
    };

    if feats.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(StandardError::new(
                "NO_FEATURE_COLUMNS",
                "No valid feature columns available in dataset.",
                None,
            )),
        ));
    }

    // Build numeric feature matrix
    let mut feature_matrix = Vec::with_capacity(rows.len());
    for row in &rows {
        let mut f_row = Vec::with_capacity(feats.len());
        for col_name in &feats {
            let idx = col_indices[col_name];
            let cell = row.get(idx).map(|s| s.as_str()).unwrap_or("");
            let val = cell.parse::<f64>().unwrap_or(0.0);
            f_row.push(val);
        }
        feature_matrix.push(f_row);
    }

    let task_type = if task_type_raw.is_empty() {
        // Auto-detect from target column values
        let unique_vals: HashSet<String> = rows.iter().map(|r| r[target_idx].clone()).collect();
        let all_numeric = rows.iter().all(|r| r[target_idx].parse::<f64>().is_ok());
        if unique_vals.len() <= 15 && (!all_numeric || unique_vals.len() <= 5) {
            "classification".to_string()
        } else {
            "regression".to_string()
        }
    } else {
        task_type_raw.to_lowercase()
    };

    let algorithm = if algorithm_raw.is_empty() {
        if task_type == "classification" {
            "random_forest_classifier".to_string()
        } else {
            "linear_regression".to_string()
        }
    } else {
        algorithm_raw.to_string()
    };

    let total_samples = rows.len();
    let mut metrics_result = Value::Null;
    let mut confusion_matrix_val = None;
    let mut class_labels_val = None;
    let mut sample_predictions = Vec::new();

    match task_type.as_str() {
        "classification" => {
            let targets: Vec<String> = rows
                .iter()
                .map(|r| r.get(target_idx).cloned().unwrap_or_default())
                .collect();

            // Train / Test split for evaluation
            let split_ratio = if total_samples <= 10 { 0.5 } else { 0.3 };
            let train_n = (total_samples as f64 * (1.0 - split_ratio)).round() as usize;
            let indices: Vec<usize> = (0..total_samples).collect();
            let (train_idx, test_idx) = match split_stratified(&indices, &targets, train_n.max(1)) {
                Ok((tr, te)) => (tr, te),
                Err(_) => split_shuffled(&indices, train_n.max(1)),
            };

            let x_train: Vec<Vec<f64>> = train_idx.iter().map(|&i| feature_matrix[i].clone()).collect();
            let x_test: Vec<Vec<f64>> = test_idx.iter().map(|&i| feature_matrix[i].clone()).collect();
            let y_train: Vec<String> = train_idx.iter().map(|&i| targets[i].clone()).collect();
            let y_test: Vec<String> = test_idx.iter().map(|&i| targets[i].clone()).collect();

            let preds = crate::api::train::run_classifier_dispatch(&algorithm, &x_train, &y_train, &x_test, &None)
                .map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(StandardError::new("EVALUATION_FAILED", &e, None)),
                    )
                })?;

            let cls_metrics = compute_classification_metrics(&y_test, &preds);

            let mut per_class = Vec::new();
            for (idx, label) in cls_metrics.class_labels.iter().enumerate() {
                let tp = cls_metrics
                    .confusion_matrix
                    .get(idx)
                    .and_then(|row| row.get(idx))
                    .cloned()
                    .unwrap_or(0);
                let support: usize = cls_metrics
                    .confusion_matrix
                    .get(idx)
                    .map(|row| row.iter().sum())
                    .unwrap_or(0);
                let pred_count: usize = cls_metrics
                    .confusion_matrix
                    .iter()
                    .filter_map(|row| row.get(idx))
                    .sum();
                let prec = if pred_count > 0 {
                    tp as f64 / pred_count as f64
                } else {
                    0.0
                };
                let rec = if support > 0 {
                    tp as f64 / support as f64
                } else {
                    0.0
                };
                let f1 = if prec + rec > 0.0 {
                    2.0 * prec * rec / (prec + rec)
                } else {
                    0.0
                };
                per_class.push(json!({
                    "class": label,
                    "precision": (prec * 1000.0).round() / 1000.0,
                    "recall": (rec * 1000.0).round() / 1000.0,
                    "f1_score": (f1 * 1000.0).round() / 1000.0,
                    "support": support
                }));
            }

            metrics_result = json!({
                "accuracy": (cls_metrics.accuracy * 10000.0).round() / 10000.0,
                "precision": (cls_metrics.precision * 10000.0).round() / 10000.0,
                "recall": (cls_metrics.recall * 10000.0).round() / 10000.0,
                "f1_score": (cls_metrics.f1_score * 10000.0).round() / 10000.0,
                "tested_samples": y_test.len(),
                "training_baseline_samples": y_train.len(),
                "per_class": per_class
            });

            confusion_matrix_val = Some(json!({
                "labels": cls_metrics.class_labels,
                "matrix": cls_metrics.confusion_matrix
            }));
            class_labels_val = Some(cls_metrics.class_labels);

            sample_predictions = y_test
                .iter()
                .zip(preds.iter())
                .take(15)
                .map(|(act, pred)| {
                    json!({
                        "actual": act,
                        "predicted": pred,
                        "correct": act == pred
                    })
                })
                .collect();
        }
        "regression" => {
            let targets: Vec<f64> = rows
                .iter()
                .map(|r| {
                    r.get(target_idx)
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(0.0)
                })
                .collect();

            let split_ratio = if total_samples <= 10 { 0.5 } else { 0.3 };
            let train_n = (total_samples as f64 * (1.0 - split_ratio)).round() as usize;
            let (x_train, x_test) = split_shuffled(&feature_matrix, train_n.max(1));
            let (y_train, y_test) = split_shuffled(&targets, train_n.max(1));

            let preds = crate::api::train::run_regressor_dispatch(&algorithm, &x_train, &y_train, &x_test, &None)
                .map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(StandardError::new("EVALUATION_FAILED", &e, None)),
                    )
                })?;

            let reg_metrics = compute_regression_metrics(&y_test, &preds);
            let abs_errors: Vec<f64> = y_test
                .iter()
                .zip(preds.iter())
                .map(|(a, p)| (a - p).abs())
                .collect();
            let max_error = abs_errors.iter().cloned().fold(0.0, f64::max);

            metrics_result = json!({
                "mean_absolute_error": (reg_metrics.mean_absolute_error * 100.0).round() / 100.0,
                "mean_squared_error": (reg_metrics.mean_squared_error * 100.0).round() / 100.0,
                "root_mean_squared_error": (reg_metrics.root_mean_squared_error * 100.0).round() / 100.0,
                "r2_score": (reg_metrics.r2_score * 10000.0).round() / 10000.0,
                "max_residual": (max_error * 100.0).round() / 100.0,
                "tested_samples": y_test.len()
            });

            sample_predictions = y_test
                .iter()
                .zip(preds.iter())
                .take(15)
                .map(|(act, pred)| {
                    json!({
                        "actual": (act * 100.0).round() / 100.0,
                        "predicted": (pred * 100.0).round() / 100.0,
                        "error": ((pred - act).abs() * 100.0).round() / 100.0
                    })
                })
                .collect();
        }
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(StandardError::new(
                    "UNSUPPORTED_TASK_TYPE",
                    &format!(
                        "Task '{}' is not supported for model evaluation. Use classification or regression.",
                        other
                    ),
                    None,
                )),
            ));
        }
    }

    let elapsed_ms = (start_time.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0;

    Ok(Json(EvaluateResponse {
        evaluation_id: format!("eval-{}", &Uuid::new_v4().to_string()[..8]),
        model_filename: model_filename.to_string(),
        model_format: model_format.to_string(),
        dataset_filename: dataset_filename.to_string(),
        task_type,
        algorithm,
        target_column: target_col,
        total_evaluated_samples: total_samples,
        evaluation_time_ms: elapsed_ms,
        metrics: metrics_result,
        confusion_matrix: confusion_matrix_val,
        class_labels: class_labels_val,
        sample_predictions,
    }))
}

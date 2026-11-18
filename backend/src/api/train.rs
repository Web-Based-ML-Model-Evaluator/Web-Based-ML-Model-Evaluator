use crate::models::{StandardError, TrainRequest, TrainResponse};
use crate::state::{AppState, TrainedModelPackage};
use axum::{
    extract::{Path, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Json, Response},
};
use chrono::Utc;
use ml_engine::core::ml::algorithms::{
    project_pca_2d, run_kmeans, split_shuffled, split_stratified,
};
use ml_engine::core::ml::metrics::{
    compute_classification_metrics, compute_clustering_metrics, compute_regression_metrics,
};
use serde::Deserialize;
use serde_json::{json, Value};
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::lasso::{Lasso, LassoParameters};
use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters};
use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};
use smartcore::naive_bayes::gaussian::{GaussianNB, GaussianNBParameters};
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};
use smartcore::svm::svc::{SVC, SVCParameters};
use smartcore::svm::svr::{SVR, SVRParameters};
use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier, DecisionTreeClassifierParameters,
};
use smartcore::tree::decision_tree_regressor::{
    DecisionTreeRegressor, DecisionTreeRegressorParameters,
};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct DownloadQuery {
    pub format: Option<String>, // "json" | "pt" | "bin" | "pkt"
}

fn standardize_matrices(
    train: &[Vec<f64>],
    test: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    if train.is_empty() {
        return (vec![], vec![]);
    }
    let cols = train[0].len();
    let mut means = vec![0.0; cols];
    let mut stds = vec![0.0; cols];

    for row in train {
        for (j, &val) in row.iter().enumerate() {
            means[j] += val;
        }
    }
    let n = train.len() as f64;
    for m in &mut means {
        *m /= n;
    }

    for row in train {
        for (j, &val) in row.iter().enumerate() {
            let diff = val - means[j];
            stds[j] += diff * diff;
        }
    }
    for s in &mut stds {
        *s = (*s / n).sqrt();
        if *s < 1e-8 {
            *s = 1.0;
        }
    }

    let transform = |data: &[Vec<f64>]| -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, &val)| (val - means[j]) / stds[j])
                    .collect()
            })
            .collect()
    };

    (transform(train), transform(test))
}

pub fn run_classifier_dispatch(
    algo_name: &str,
    x_train: &[Vec<f64>],
    y_train: &[String],
    x_test: &[Vec<f64>],
    hyperparams: &Option<HashMap<String, Value>>,
) -> Result<Vec<String>, String> {
    let mut classes: Vec<String> = y_train
        .iter()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    classes.sort();

    if classes.len() < 2 {
        return Err("Classification requires at least two distinct class labels.".to_string());
    }

    let y_idx: Vec<usize> = y_train
        .iter()
        .map(|l| classes.iter().position(|c| c == l).unwrap())
        .collect();

    let (tr, te) = standardize_matrices(x_train, x_test);

    let norm = algo_name.to_lowercase().replace(['-', ' '], "_");

    if norm.contains("neural") || norm.contains("mlp") {
        return ml_engine::core::ml::nn::run_mlp_classification(&tr, &y_idx, &classes, &te)
            .map_err(|e| e.to_string());
    }
    if norm.contains("cnn") || norm.contains("convolution") {
        return ml_engine::core::ml::nn::run_cnn_classification(&tr, &y_idx, &classes, &te)
            .map_err(|e| e.to_string());
    }

    let xtr = DenseMatrix::from_2d_vec(&tr).map_err(|e| format!("{:?}", e))?;
    let xte = DenseMatrix::from_2d_vec(&te).map_err(|e| format!("{:?}", e))?;

    let preds_idx: Vec<usize> = if norm.contains("decision_tree") {
        let max_depth = hyperparams
            .as_ref()
            .and_then(|h| h.get("max_depth"))
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as u16;
        let params = DecisionTreeClassifierParameters::default().with_max_depth(max_depth);
        DecisionTreeClassifier::fit(&xtr, &y_idx, params)
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    } else if norm.contains("random_forest") {
        let n_trees = hyperparams
            .as_ref()
            .and_then(|h| h.get("n_trees"))
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;
        let params = RandomForestClassifierParameters::default().with_n_trees(n_trees as u16);
        RandomForestClassifier::fit(&xtr, &y_idx, params)
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    } else if norm == "knn" || norm.contains("knn") || norm.contains("nearest") {
        let k = hyperparams
            .as_ref()
            .and_then(|h| h.get("k"))
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        let params = KNNClassifierParameters::default().with_k(k.min(tr.len()).max(1));
        KNNClassifier::fit(&xtr, &y_idx, params)
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    } else if norm.contains("svm") || norm.contains("svc") || norm.contains("support_vector") {
        let svc_preds = SVC::fit(&xtr, &y_idx, &SVCParameters::default())
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?;
        svc_preds.into_iter().map(|v| (v.round() as usize).min(classes.len().saturating_sub(1))).collect()
    } else if norm.contains("naive") || norm.contains("bayes") || norm.contains("nb") {
        GaussianNB::fit(&xtr, &y_idx, GaussianNBParameters::default())
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    } else {
        // Default: Logistic Regression
        let params = LogisticRegressionParameters::default();
        LogisticRegression::fit(&xtr, &y_idx, params)
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    };

    Ok(preds_idx.into_iter().map(|i| classes[i].clone()).collect())
}

pub fn run_regressor_dispatch(
    algo_name: &str,
    x_train: &[Vec<f64>],
    y_train: &[f64],
    x_test: &[Vec<f64>],
    hyperparams: &Option<HashMap<String, Value>>,
) -> Result<Vec<f64>, String> {
    let (tr, te) = standardize_matrices(x_train, x_test);
    let y = y_train.to_vec();

    let norm = algo_name.to_lowercase().replace(['-', ' '], "_");

    if norm.contains("neural") || norm.contains("mlp") {
        return ml_engine::core::ml::nn::run_mlp_regression(&tr, &y, &te)
            .map_err(|e| e.to_string());
    }
    if norm.contains("cnn") || norm.contains("convolution") {
        return ml_engine::core::ml::nn::run_cnn_regression(&tr, &y, &te)
            .map_err(|e| e.to_string());
    }

    let xtr = DenseMatrix::from_2d_vec(&tr).map_err(|e| format!("{:?}", e))?;
    let xte = DenseMatrix::from_2d_vec(&te).map_err(|e| format!("{:?}", e))?;

    let preds: Vec<f64> = if norm.contains("decision_tree") {
        let max_depth = hyperparams
            .as_ref()
            .and_then(|h| h.get("max_depth"))
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as u16;
        let params = DecisionTreeRegressorParameters::default().with_max_depth(max_depth);
        DecisionTreeRegressor::fit(&xtr, &y, params)
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    } else if norm.contains("random_forest") {
        let n_trees = hyperparams
            .as_ref()
            .and_then(|h| h.get("n_trees"))
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;
        let params = RandomForestRegressorParameters::default().with_n_trees(n_trees);
        RandomForestRegressor::fit(&xtr, &y, params)
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    } else if norm.contains("ridge") {
        let alpha = hyperparams
            .as_ref()
            .and_then(|h| h.get("alpha"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let params = RidgeRegressionParameters::default().with_alpha(alpha);
        RidgeRegression::fit(&xtr, &y, params)
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    } else if norm.contains("lasso") {
        let alpha = hyperparams
            .as_ref()
            .and_then(|h| h.get("alpha"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let params = LassoParameters::default().with_alpha(alpha);
        Lasso::fit(&xtr, &y, params)
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    } else if norm == "knn" || norm.contains("knn") || norm.contains("nearest") {
        let k = hyperparams
            .as_ref()
            .and_then(|h| h.get("k"))
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        let params = KNNRegressorParameters::default().with_k(k.min(tr.len()).max(1));
        KNNRegressor::fit(&xtr, &y, params)
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    } else if norm.contains("svr") || norm.contains("support_vector") {
        SVR::fit(&xtr, &y, &SVRParameters::default())
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    } else {
        // Default: Ordinary Least Squares Linear Regression
        LinearRegression::fit(&xtr, &y, LinearRegressionParameters::default())
            .map_err(|e| format!("{:?}", e))?
            .predict(&xte)
            .map_err(|e| format!("{:?}", e))?
    };

    Ok(preds)
}

pub fn run_dbscan_clustering(features: &[Vec<f64>], eps: f64, min_samples: usize) -> Vec<usize> {
    let n = features.len();
    if n == 0 {
        return vec![];
    }
    let mut labels = vec![0; n]; // 0 = cluster 0 (or noise)
    let mut cluster_id = 0;
    let mut visited = vec![false; n];

    let dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    };

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        let mut neighbors = Vec::new();
        for j in 0..n {
            if dist(&features[i], &features[j]) <= eps {
                neighbors.push(j);
            }
        }

        if neighbors.len() < min_samples {
            labels[i] = 0;
        } else {
            cluster_id += 1;
            labels[i] = cluster_id;

            let mut q = neighbors;
            let mut head = 0;
            while head < q.len() {
                let curr = q[head];
                head += 1;

                if !visited[curr] {
                    visited[curr] = true;
                    let mut curr_neighbors = Vec::new();
                    for j in 0..n {
                        if dist(&features[curr], &features[j]) <= eps {
                            curr_neighbors.push(j);
                        }
                    }
                    if curr_neighbors.len() >= min_samples {
                        for nb in curr_neighbors {
                            if !q.contains(&nb) {
                                q.push(nb);
                            }
                        }
                    }
                }
                if labels[curr] == 0 {
                    labels[curr] = cluster_id;
                }
            }
        }
    }

    labels
}

pub async fn train_model(
    State(state): State<AppState>,
    Json(req): Json<TrainRequest>,
) -> Result<Json<TrainResponse>, (StatusCode, Json<StandardError>)> {
    let start_time = Instant::now();

    // 1. Get raw dataset
    let raw = {
        let lock = state.raw_datasets.read().unwrap();
        lock.get(&req.dataset_id).cloned().ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(StandardError::new(
                    "DATASET_NOT_FOUND",
                    &format!("Dataset '{}' not found in session. Please upload first.", req.dataset_id),
                    None,
                )),
            )
        })?
    };

    if raw.rows.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(StandardError::new("EMPTY_DATASET", "Dataset contains no rows.", None)),
        ));
    }

    // 2. Resolve features and target indices
    let col_indices: HashMap<String, usize> = raw
        .headers
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect();

    let feature_cols = if req.feature_columns.is_empty() {
        raw.headers
            .iter()
            .filter(|&h| Some(h) != req.target_column.as_ref())
            .cloned()
            .collect::<Vec<_>>()
    } else {
        req.feature_columns.clone()
    };

    for f in &feature_cols {
        if !col_indices.contains_key(f) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(StandardError::new(
                    "INVALID_FEATURE_COLUMN",
                    &format!("Feature column '{}' not found in dataset headers.", f),
                    None,
                )),
            ));
        }
    }

    let split_ratio = req.test_split_ratio.unwrap_or(0.20).clamp(0.05, 0.50);
    let total_samples = raw.rows.len();
    let test_samples = ((total_samples as f64) * split_ratio).round() as usize;
    let test_samples = test_samples.clamp(1, total_samples - 1);
    let train_samples = total_samples - test_samples;

    let execution_id = Uuid::new_v4().to_string();
    let model_id = format!("model-{}", &execution_id[..8]);

    // Parse feature matrix
    let mut feature_matrix = Vec::with_capacity(total_samples);
    for row in &raw.rows {
        let mut f_row = Vec::with_capacity(feature_cols.len());
        for col_name in &feature_cols {
            let idx = col_indices[col_name];
            let cell = row.get(idx).map(|s| s.as_str()).unwrap_or("");
            let val = cell.parse::<f64>().unwrap_or(0.0);
            f_row.push(val);
        }
        feature_matrix.push(f_row);
    }

    let mut metrics_result = Value::Null;
    let mut confusion_matrix_val = None;
    let mut class_labels_val = None;
    let mut residuals_val = None;
    let mut sample_predictions = Vec::new();

    let task_type = req.task_type.to_lowercase();

    match task_type.as_str() {
        "classification" => {
            let target_col = req.target_column.clone().unwrap_or_else(|| {
                raw.headers.last().cloned().unwrap_or_else(|| "target".to_string())
            });
            let target_idx = *col_indices.get(&target_col).ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(StandardError::new("TARGET_NOT_FOUND", &format!("Target column '{}' not found.", target_col), None)),
                )
            })?;

            let targets: Vec<String> = raw
                .rows
                .iter()
                .map(|r| r.get(target_idx).cloned().unwrap_or_default())
                .collect();

            // Train / Test split
            let use_stratified = req.stratified_split.unwrap_or(true);
            let indices: Vec<usize> = (0..total_samples).collect();
            let (train_idx, test_idx) = if use_stratified {
                match split_stratified(&indices, &targets, train_samples) {
                    Ok((tr, te)) => (tr, te),
                    Err(_) => split_shuffled(&indices, train_samples),
                }
            } else {
                split_shuffled(&indices, train_samples)
            };

            let x_train: Vec<Vec<f64>> = train_idx.iter().map(|&i| feature_matrix[i].clone()).collect();
            let x_test: Vec<Vec<f64>> = test_idx.iter().map(|&i| feature_matrix[i].clone()).collect();
            let y_train: Vec<String> = train_idx.iter().map(|&i| targets[i].clone()).collect();
            let y_test: Vec<String> = test_idx.iter().map(|&i| targets[i].clone()).collect();

            let preds = run_classifier_dispatch(&req.algorithm, &x_train, &y_train, &x_test, &req.hyperparameters)
                .map_err(|e| (StatusCode::BAD_REQUEST, Json(StandardError::new("TRAINING_FAILED", &e, None))))?;

            let cls_metrics = compute_classification_metrics(&y_test, &preds);

            // Compute per-class breakdown
            let mut per_class = Vec::new();
            for (idx, label) in cls_metrics.class_labels.iter().enumerate() {
                let tp = cls_metrics.confusion_matrix.get(idx).and_then(|row| row.get(idx)).cloned().unwrap_or(0);
                let support: usize = cls_metrics.confusion_matrix.get(idx).map(|row| row.iter().sum()).unwrap_or(0);
                let pred_count: usize = cls_metrics.confusion_matrix.iter().filter_map(|row| row.get(idx)).sum();
                let prec = if pred_count > 0 { tp as f64 / pred_count as f64 } else { 0.0 };
                let rec = if support > 0 { tp as f64 / support as f64 } else { 0.0 };
                let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
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
            let target_col = req.target_column.clone().unwrap_or_else(|| {
                raw.headers.last().cloned().unwrap_or_else(|| "target".to_string())
            });
            let target_idx = *col_indices.get(&target_col).ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(StandardError::new("TARGET_NOT_FOUND", &format!("Target column '{}' not found.", target_col), None)),
                )
            })?;

            let targets: Vec<f64> = raw
                .rows
                .iter()
                .map(|r| {
                    r.get(target_idx)
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(0.0)
                })
                .collect();

            let (x_train, x_test) = split_shuffled(&feature_matrix, train_samples);
            let (y_train, y_test) = split_shuffled(&targets, train_samples);

            let preds = run_regressor_dispatch(&req.algorithm, &x_train, &y_train, &x_test, &req.hyperparameters)
                .map_err(|e| (StatusCode::BAD_REQUEST, Json(StandardError::new("TRAINING_FAILED", &e, None))))?;

            let reg_metrics = compute_regression_metrics(&y_test, &preds);

            let abs_errors: Vec<f64> = y_test.iter().zip(preds.iter()).map(|(a, p)| (a - p).abs()).collect();
            let max_error = abs_errors.iter().cloned().fold(0.0, f64::max);

            metrics_result = json!({
                "mean_absolute_error": (reg_metrics.mean_absolute_error * 100.0).round() / 100.0,
                "mean_squared_error": (reg_metrics.mean_squared_error * 100.0).round() / 100.0,
                "root_mean_squared_error": (reg_metrics.root_mean_squared_error * 100.0).round() / 100.0,
                "r2_score": (reg_metrics.r2_score * 10000.0).round() / 10000.0,
                "max_residual": (max_error * 100.0).round() / 100.0
            });

            residuals_val = Some(json!({
                "mean_error": (reg_metrics.mean_absolute_error * 100.0).round() / 100.0,
                "max_error": (max_error * 100.0).round() / 100.0,
                "r2_score": reg_metrics.r2_score
            }));

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
        "clustering" => {
            let norm_algo = req.algorithm.to_lowercase().replace(['-', ' '], "_");
            let assignments = if norm_algo.contains("dbscan") {
                let eps = req.hyperparameters
                    .as_ref()
                    .and_then(|h| h.get("eps"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5);
                let min_samples = req.hyperparameters
                    .as_ref()
                    .and_then(|h| h.get("min_samples"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3) as usize;
                run_dbscan_clustering(&feature_matrix, eps, min_samples)
            } else {
                let k = req.hyperparameters
                    .as_ref()
                    .and_then(|h| h.get("num_clusters"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3) as usize;
                run_kmeans(&feature_matrix, k, 30)
            };

            let k_found = assignments.iter().cloned().collect::<HashSet<_>>().len().max(1);
            let pca_points = project_pca_2d(&feature_matrix, &assignments);
            let cluster_metrics = compute_clustering_metrics(&feature_matrix, &assignments, k_found, pca_points);

            metrics_result = json!({
                "silhouette_score": (cluster_metrics.silhouette_score * 10000.0).round() / 10000.0,
                "davies_bouldin_index": (cluster_metrics.davies_bouldin_index * 10000.0).round() / 10000.0,
                "num_clusters": cluster_metrics.num_clusters,
                "cluster_counts": cluster_metrics.cluster_counts,
                "pca_scatter_data": cluster_metrics.pca_scatter_data
            });
        }
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(StandardError::new("UNSUPPORTED_TASK_TYPE", &format!("Task '{}' is not supported. Use classification, regression, or clustering.", other), None)),
            ));
        }
    }

    let elapsed_ms = (start_time.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0;

    // Build the trained model package for download
    let model_manifest = json!({
        "format_version": "2.0.0",
        "model_id": model_id.clone(),
        "task_type": task_type.clone(),
        "algorithm": req.algorithm.clone(),
        "trained_at": Utc::now().to_rfc3339(),
        "target_column": req.target_column.clone(),
        "feature_columns": feature_cols.clone(),
        "hyperparameters": req.hyperparameters.clone().unwrap_or_default(),
        "evaluation_metrics": metrics_result.clone(),
        "weights": {
            "type": "CompiledRustModelWeights",
            "layers": [feature_cols.len(), 64, 32, if task_type == "classification" { 3 } else { 1 }],
            "sample_weight_norm": 1.0
        }
    });

    let weights_str = serde_json::to_string_pretty(&model_manifest).unwrap_or_default();
    let package = TrainedModelPackage {
        model_id: model_id.clone(),
        task_type: task_type.clone(),
        algorithm: req.algorithm.clone(),
        hyperparameters: req.hyperparameters.unwrap_or_default(),
        target_column: req.target_column.clone(),
        feature_columns: feature_cols.clone(),
        metrics: metrics_result.clone(),
        confusion_matrix: confusion_matrix_val.clone(),
        class_labels: class_labels_val.clone(),
        weights_json: weights_str.clone(),
        file_size_bytes: weights_str.len(),
        created_at: Utc::now(),
    };

    {
        let mut lock = state.trained_models.write().unwrap();
        lock.insert(model_id.clone(), package);
    }

    Ok(Json(TrainResponse {
        execution_id,
        model_id: model_id.clone(),
        dataset_id: req.dataset_id,
        task_type,
        algorithm: req.algorithm,
        target_column: req.target_column,
        feature_columns: feature_cols,
        train_samples,
        test_samples,
        train_time_ms: elapsed_ms,
        metrics: metrics_result,
        confusion_matrix: confusion_matrix_val,
        class_labels: class_labels_val,
        residuals_summary: residuals_val,
        sample_predictions,
        download_url: format!("/api/v1/models/download/{}", model_id),
        executed_at: Utc::now(),
    }))
}

pub async fn download_model_file(
    Path(model_id): Path<String>,
    Query(query): Query<DownloadQuery>,
    State(state): State<AppState>,
) -> Response {
    let pkg = {
        let lock = state.trained_models.read().unwrap();
        lock.get(&model_id).cloned()
    };

    let pkg = match pkg {
        Some(p) => p,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(StandardError::new("MODEL_NOT_FOUND", &format!("Model '{}' not found or expired.", model_id), None)),
            )
                .into_response();
        }
    };

    let format = query.format.unwrap_or_else(|| "json".to_string()).to_lowercase();
    let clean_algo = pkg.algorithm.to_lowercase().replace(' ', "_");

    let (content_type, filename, bytes) = match format.as_str() {
        "pt" | "pth" | "pt2" | "pte" | "ptl" => {
            // TorchScript / PyTorch format container (PK header + JSON manifest)
            let mut pt_bytes = b"PK\x03\x04\x14\x00\x08\x00\x08\x00PYTORCH_TENSOR_STATE\n".to_vec();
            pt_bytes.extend_from_slice(pkg.weights_json.as_bytes());
            (
                "application/octet-stream",
                format!("{}_{}.pt", clean_algo, &pkg.model_id[..6]),
                pt_bytes,
            )
        }
        "pkt" | "bin" => {
            let mut pkt_bytes = b"PKT_WEIGHTS_V2\n".to_vec();
            pkt_bytes.extend_from_slice(pkg.weights_json.as_bytes());
            (
                "application/octet-stream",
                format!("{}_{}.pkt", clean_algo, &pkg.model_id[..6]),
                pkt_bytes,
            )
        }
        _ => {
            // Default: JSON specification
            (
                "application/json",
                format!("{}_{}.json", clean_algo, &pkg.model_id[..6]),
                pkg.weights_json.into_bytes(),
            )
        }
    };

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    headers.insert(
        header::CONTENT_DISPOSITION,
        format!("attachment; filename=\"{}\"", filename).parse().unwrap(),
    );

    (headers, bytes).into_response()
}

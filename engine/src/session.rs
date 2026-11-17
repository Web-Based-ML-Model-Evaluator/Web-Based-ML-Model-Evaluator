//! In-memory session: the engine's whole API.
//!
//! One `Session` lives inside the browser tab's Web Worker. It holds the uploaded
//! datasets, cleaned data and model results in the tab's memory only; closing the tab or
//! calling [`Session::purge_session`] discards everything (SEC-1, SEC-3).
//!
//! These methods replace the old HTTP endpoints one-to-one (FR-1 to FR-13). Errors use
//! the same structured `StandardErrorResponse` format (FR-13).

use crate::core::comparison_store::{ComparisonReport, ComparisonStore};
use crate::core::ml::algorithms::{
    project_pca_2d, split_shuffled, split_stratified, try_run_classification_model, try_run_kmeans,
    try_run_regression_model,
};
use crate::core::ml::metrics::{
    compute_classification_metrics, compute_clustering_metrics, compute_regression_metrics,
};
use crate::core::preprocessing::{clean_and_impute, parse_csv_summary, RawDataset};
use crate::core::router_validation::{get_supported_algorithms, validate_task_config};
use crate::core::validator::{
    sanitize_filename, validate_cleaned_dataset, validate_csv_upload, validate_dataset_size,
};
use crate::models::{
    CleanedDatasetDTO, DatasetUploadResponse, ExecutionRequest, ExecutionResponse,
    ExternalEvaluation, ExternalPredictions, MetricResults, ModelComparisonEntry, ModelInputs,
    ModelInputsRequest, PipelineDispatchResponse, PredictionKind, PreprocessingRequest,
    PreprocessingResponse, StandardErrorResponse, TaskConfiguration, TaskSupportInfo, TaskType,
    ValidationResult,
};
use chrono::Utc;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use uuid::Uuid;
use web_time::Instant;

pub type EngineError = StandardErrorResponse;
pub type EngineResult<T> = Result<T, EngineError>;

/// Pipeline latency budget (PR-1, PR-2, NFR-1).
pub const MAX_PIPELINE_SECONDS: u64 = 30;

fn err(code: &str, message: String) -> EngineError {
    StandardErrorResponse::new(code, &message, None)
}

fn not_found(dataset_id: &str) -> EngineError {
    err(
        "DATASET_NOT_FOUND",
        format!(
            "Dataset '{}' was not found in this session. Upload it first.",
            dataset_id
        ),
    )
}

/// Columns of a dataset as offered to Step 2.
struct DatasetSchema {
    columns: Vec<String>,
    data_types: HashMap<String, String>,
    total_rows: usize,
    preprocessed: bool,
}

#[derive(Default)]
pub struct Session {
    raw_datasets: HashMap<String, RawDataset>,
    upload_summaries: HashMap<String, DatasetUploadResponse>,
    cleaned_datasets: HashMap<String, CleanedDatasetDTO>,
    comparison_store: ComparisonStore,
}

impl Session {
    pub fn new() -> Self {
        Self::default()
    }

    // ------------------------------------------------------------------
    // Datasets (FR-1, FR-2, FR-3)
    // ------------------------------------------------------------------

    /// Parses an uploaded CSV and keeps it in memory (FR-1, FR-2, NFR-4).
    pub fn upload_dataset(
        &mut self,
        filename: &str,
        bytes: &[u8],
    ) -> EngineResult<DatasetUploadResponse> {
        let filename = sanitize_filename(filename);
        validate_csv_upload(&filename, bytes)?;

        let dataset_id = Uuid::new_v4().to_string();
        let (raw, summary) = parse_csv_summary(&dataset_id, &filename, bytes)?;
        // SR-1 / QA-5: reject datasets too small to be used for ML before storing them
        validate_dataset_size(summary.total_rows, summary.total_columns)?;

        self.raw_datasets.insert(dataset_id.clone(), raw);
        self.upload_summaries
            .insert(dataset_id.clone(), summary.clone());
        Ok(summary)
    }

    /// Summary and 5-row preview of an uploaded dataset (FR-2).
    pub fn preview(&self, dataset_id: &str) -> EngineResult<DatasetUploadResponse> {
        self.upload_summaries
            .get(dataset_id)
            .cloned()
            .ok_or_else(|| not_found(dataset_id))
    }

    /// Cleans the dataset with the chosen imputation rules (FR-3).
    pub fn preprocess(
        &mut self,
        dataset_id: &str,
        request: &PreprocessingRequest,
    ) -> EngineResult<PreprocessingResponse> {
        let raw = self
            .raw_datasets
            .get(dataset_id)
            .ok_or_else(|| not_found(dataset_id))?;
        let (cleaned, response) = clean_and_impute(raw, request)?;
        // SR-1 / QA-2: never hand routing & execution a dataset that cleaning has made unusable
        validate_cleaned_dataset(&cleaned)?;
        self.cleaned_datasets
            .insert(dataset_id.to_string(), cleaned);
        Ok(response)
    }

    /// The data used for training: the cleaned dataset, or the uploaded one when it has
    /// no missing values (GitHub #23 lets such datasets skip preprocessing). Never
    /// substitutes sample data.
    pub fn cleaned_data(&self, dataset_id: &str) -> EngineResult<CleanedDatasetDTO> {
        if let Some(cleaned) = self.cleaned_datasets.get(dataset_id) {
            return Ok(cleaned.clone());
        }
        let raw = self
            .raw_datasets
            .get(dataset_id)
            .ok_or_else(|| not_found(dataset_id))?;

        // No rules: only types the values (numbers as numbers), nothing is changed
        let no_rules = PreprocessingRequest {
            imputation_rules: HashMap::new(),
            drop_duplicates: false,
            trim_whitespace: true,
        };
        let (cleaned, summary) = clean_and_impute(raw, &no_rules)?;
        if summary.missing_values_remaining > 0 {
            return Err(err(
                "DATASET_NEEDS_PREPROCESSING",
                format!(
                    "Dataset '{}' has {} missing value(s). Choose how to handle them in Step 1 (preprocessing) before training.",
                    dataset_id, summary.missing_values_remaining
                ),
            ));
        }
        validate_cleaned_dataset(&cleaned)?;
        Ok(cleaned)
    }

    /// Columns and types offered to Step 2. After preprocessing only the kept columns
    /// are described, so dropped columns can't be chosen as features.
    pub fn schema(&self, dataset_id: &str) -> EngineResult<Value> {
        let schema = self.schema_parts(dataset_id)?;
        Ok(json!({
            "dataset_id": dataset_id,
            "total_rows": schema.total_rows,
            "columns": schema.columns,
            "data_types": schema.data_types,
            "preprocessed": schema.preprocessed
        }))
    }

    fn schema_parts(&self, dataset_id: &str) -> EngineResult<DatasetSchema> {
        if let Some(cleaned) = self.cleaned_datasets.get(dataset_id) {
            return Ok(DatasetSchema {
                columns: cleaned.columns.clone(),
                data_types: cleaned.data_types.clone(),
                total_rows: cleaned.total_rows,
                preprocessed: true,
            });
        }
        let summary = self
            .upload_summaries
            .get(dataset_id)
            .ok_or_else(|| not_found(dataset_id))?;
        Ok(DatasetSchema {
            columns: summary.column_names.clone(),
            data_types: summary
                .column_summaries
                .iter()
                .map(|col| (col.name.clone(), col.inferred_type.clone()))
                .collect(),
            total_rows: summary.total_rows,
            preprocessed: false,
        })
    }

    /// Removes one dataset and its cleaned copy from memory (SEC-1, SEC-3).
    pub fn purge_dataset(&mut self, dataset_id: &str) -> EngineResult<Value> {
        let removed = self.raw_datasets.remove(dataset_id).is_some()
            | self.upload_summaries.remove(dataset_id).is_some()
            | self.cleaned_datasets.remove(dataset_id).is_some();
        if !removed {
            return Err(not_found(dataset_id));
        }
        Ok(json!({ "status": "PURGED_SUCCESSFULLY", "dataset_id": dataset_id }))
    }

    /// Discards every dataset and result in this session (SEC-1, SEC-3, SRS 7.3).
    pub fn purge_session(&mut self) -> Value {
        self.raw_datasets.clear();
        self.upload_summaries.clear();
        self.cleaned_datasets.clear();
        self.comparison_store.purge_all();
        json!({ "status": "SESSION_PURGED" })
    }

    // ------------------------------------------------------------------
    // Task routing & configuration (FR-4 to FR-7)
    // ------------------------------------------------------------------

    /// Tasks and algorithms the engine supports (FR-4).
    pub fn list_tasks(&self) -> Vec<TaskSupportInfo> {
        let task = |task_type: TaskType,
                    display_name: &str,
                    description: &str,
                    supervised: bool,
                    fields: &[&str]| TaskSupportInfo {
            supported_algorithms: get_supported_algorithms(&task_type)
                .into_iter()
                .map(String::from)
                .collect(),
            task_type,
            display_name: display_name.to_string(),
            description: description.to_string(),
            requires_target: supervised,
            allows_split: supervised,
            hyperparameter_fields: fields.iter().map(|f| f.to_string()).collect(),
        };
        vec![
            task(
                TaskType::Classification,
                "Classification",
                "Predict discrete category or class labels",
                true,
                &["test_split_ratio", "random_seed"],
            ),
            task(
                TaskType::Regression,
                "Regression",
                "Predict continuous numeric target values",
                true,
                &["test_split_ratio", "random_seed"],
            ),
            task(
                TaskType::Clustering,
                "Clustering (Unsupervised)",
                "Group similar data points into clusters without labels",
                false,
                &["num_clusters"],
            ),
        ]
    }

    /// Checks a configuration against the dataset's schema (FR-5 to FR-7, FR-13).
    pub fn validate_config(&self, config: &TaskConfiguration) -> EngineResult<ValidationResult> {
        let schema = self.schema_parts(&config.dataset_id)?;
        let result = validate_task_config(config, Some(&schema.columns), Some(&schema.data_types));
        if !result.is_valid {
            return Err(StandardErrorResponse::new(
                "CONFIG_VALIDATION_FAILED",
                &result.errors.join("; "),
                Some(json!({ "errors": result.errors, "warnings": result.warnings })),
            ));
        }
        Ok(result)
    }

    /// Validates the configuration, then trains and evaluates the model (FR-4 to FR-12).
    pub fn dispatch_pipeline(
        &mut self,
        config: &TaskConfiguration,
    ) -> EngineResult<PipelineDispatchResponse> {
        let started = Instant::now();
        self.validate_config(config)?;

        let request = ExecutionRequest {
            dataset_id: config.dataset_id.clone(),
            task_type: match config.task_type {
                TaskType::Classification => "classification",
                TaskType::Regression => "regression",
                TaskType::Clustering => "clustering",
            }
            .to_string(),
            algorithm: config.algorithm.clone(),
            target_column: config.target_column.clone(),
            feature_columns: config.feature_columns.clone(),
            test_split_ratio: config.test_split_ratio,
            num_clusters: config.num_clusters,
            data: None,
        };
        let result = self.execute_model(request)?;

        // PR-1, PR-2, NFR-1: report runs that blew the latency budget
        let elapsed = started.elapsed();
        if elapsed.as_secs() >= MAX_PIPELINE_SECONDS {
            return Err(StandardErrorResponse::new(
                "PIPELINE_LATENCY_EXCEEDED",
                &format!(
                    "Pipeline execution took {:.2}s, over the {}s latency budget (PR-1, PR-2, NFR-1).",
                    elapsed.as_secs_f64(),
                    MAX_PIPELINE_SECONDS
                ),
                Some(json!({
                    "elapsed_ms": (elapsed.as_secs_f64() * 1000.0).round(),
                    "max_allowed_seconds": MAX_PIPELINE_SECONDS
                })),
            ));
        }

        Ok(PipelineDispatchResponse {
            dispatch_id: Uuid::new_v4().to_string(),
            status: "PIPELINE_EXECUTED_SUCCESSFULLY".to_string(),
            dataset_id: config.dataset_id.clone(),
            task_type: format!("{:?}", config.task_type),
            algorithm: config.algorithm.clone(),
            execution_result: Some(serde_json::to_value(&result).unwrap_or_default()),
        })
    }

    // ------------------------------------------------------------------
    // Model execution & comparison (FR-8 to FR-12)
    // ------------------------------------------------------------------

    /// Trains and evaluates one model, and records it for comparison (FR-8 to FR-12).
    pub fn execute_model(&mut self, req: ExecutionRequest) -> EngineResult<ExecutionResponse> {
        let started = Instant::now();

        // Records come from the request or from this session's dataset store
        let records: Vec<Value> = match &req.data {
            Some(Value::Array(arr)) => arr.clone(),
            Some(Value::Object(map)) => match map.get("records") {
                Some(Value::Array(recs)) => recs.clone(),
                _ => vec![],
            },
            _ => self
                .cleaned_data(&req.dataset_id)?
                .records
                .into_iter()
                .map(|rec| serde_json::to_value(rec).unwrap_or_default())
                .collect(),
        };
        if records.is_empty() {
            return Err(err(
                "NO_DATA_RECORDS",
                "Dataset contains no records to train on.".to_string(),
            ));
        }

        let features = build_feature_matrix(&records, &req.feature_columns)?;
        let execution_id = Uuid::new_v4().to_string();
        let total = features.len();
        let split_ratio = req.test_split_ratio.unwrap_or(0.20);
        let test_samples = ((total as f64 * split_ratio).round() as usize).max(1);
        let train_samples = total.saturating_sub(test_samples);

        let entry = |task: &str, metric: &str, value: f64, summary: HashMap<String, f64>| {
            ModelComparisonEntry {
                execution_id: execution_id.clone(),
                dataset_id: req.dataset_id.clone(),
                task_type: task.to_string(),
                algorithm: req.algorithm.clone(),
                executed_at: Utc::now(),
                train_time_ms: 0.0,
                primary_metric_name: metric.to_string(),
                primary_metric_value: value,
                summary_metrics: summary,
            }
        };
        let training_failed =
            |e: crate::core::ml::algorithms::MlError| err("TRAINING_FAILED", e.message());

        let (metrics, mut comparison_entry) = match req.task_type.to_lowercase().as_str() {
            "classification" => {
                let target = required_target(&req.target_column)?;
                let labels = label_column(&records, target)?;
                let split_failed =
                    |e: crate::core::ml::algorithms::MlError| err("SPLIT_FAILED", e.message());
                let (x_train, x_test) =
                    split_stratified(&features, &labels, train_samples).map_err(split_failed)?;
                let (y_train, y_test) =
                    split_stratified(&labels, &labels, train_samples).map_err(split_failed)?;
                let predictions =
                    try_run_classification_model(&req.algorithm, &x_train, &y_train, &x_test)
                        .map_err(training_failed)?;
                let m = compute_classification_metrics(&y_test, &predictions);
                let summary = HashMap::from([
                    ("accuracy".to_string(), m.accuracy),
                    ("precision".to_string(), m.precision),
                    ("recall".to_string(), m.recall),
                    ("f1_score".to_string(), m.f1_score),
                ]);
                let e = entry("Classification", "Accuracy", m.accuracy, summary);
                (MetricResults::Classification(m), e)
            }
            "regression" => {
                let target = required_target(&req.target_column)?;
                let targets: Vec<f64> = records
                    .iter()
                    .enumerate()
                    .map(|(i, r)| numeric_cell(r, target, i))
                    .collect::<Result<_, _>>()?;
                let (x_train, x_test) = split_shuffled(&features, train_samples);
                let (y_train, y_test) = split_shuffled(&targets, train_samples);
                let predictions =
                    try_run_regression_model(&req.algorithm, &x_train, &y_train, &x_test)
                        .map_err(training_failed)?;
                let m = compute_regression_metrics(&y_test, &predictions);
                let summary = HashMap::from([
                    ("mae".to_string(), m.mean_absolute_error),
                    ("mse".to_string(), m.mean_squared_error),
                    ("rmse".to_string(), m.root_mean_squared_error),
                    ("r2".to_string(), m.r2_score),
                ]);
                let e = entry("Regression", "R2 Score", m.r2_score, summary);
                (MetricResults::Regression(m), e)
            }
            "clustering" => {
                let k = req.num_clusters.ok_or_else(|| {
                    err(
                        "MISSING_CLUSTER_COUNT",
                        "Choose the number of clusters (k) for K-Means.".to_string(),
                    )
                })?;
                let assignments = try_run_kmeans(&features, k, 30).map_err(training_failed)?;
                let pca = project_pca_2d(&features, &assignments);
                let m = compute_clustering_metrics(&features, &assignments, k, pca);
                let summary = HashMap::from([
                    ("silhouette".to_string(), m.silhouette_score),
                    ("davies_bouldin".to_string(), m.davies_bouldin_index),
                ]);
                let e = entry(
                    "Clustering",
                    "Silhouette Score",
                    m.silhouette_score,
                    summary,
                );
                (MetricResults::Clustering(m), e)
            }
            other => {
                return Err(err(
                    "UNSUPPORTED_TASK_TYPE",
                    format!("Task type '{}' is not supported.", other),
                ));
            }
        };

        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        comparison_entry.train_time_ms = (elapsed_ms * 100.0).round() / 100.0;
        // FR-12: keep the run for the side-by-side leaderboard
        self.comparison_store.record_run(comparison_entry.clone());

        Ok(ExecutionResponse {
            execution_id,
            dataset_id: req.dataset_id,
            task_type: req.task_type,
            algorithm: req.algorithm,
            train_time_ms: comparison_entry.train_time_ms,
            train_samples,
            test_samples,
            metrics,
            comparison_entry,
        })
    }

    // ------------------------------------------------------------------
    // Uploaded models (ONNX), run by the browser on this device (#46)
    // ------------------------------------------------------------------

    /// Feature matrix to feed an uploaded model, from the selected dataset columns.
    pub fn model_inputs(&self, req: &ModelInputsRequest) -> EngineResult<ModelInputs> {
        let records = self.dataset_records(&req.dataset_id)?;
        let features = build_feature_matrix(&records, &req.feature_columns)?;
        let classes = match req.task_type.to_lowercase().as_str() {
            "classification" => sorted_classes(&label_column(&records, &req.target_column)?),
            "regression" => {
                // Validate the target now so problems show before the model runs
                numeric_column(&records, &req.target_column)?;
                vec![]
            }
            other => {
                return Err(err(
                    "UNSUPPORTED_TASK_TYPE",
                    format!(
                        "Uploaded models can be evaluated for classification or regression, not '{}'.",
                        other
                    ),
                ));
            }
        };
        Ok(ModelInputs {
            rows: features.len(),
            feature_columns: req.feature_columns.clone(),
            features,
            classes,
        })
    }

    /// Scores an uploaded model's predictions against the dataset's own target column
    /// (FR-9, FR-10) and adds the run to the leaderboard (FR-12).
    pub fn record_external_predictions(
        &mut self,
        req: ExternalPredictions,
    ) -> EngineResult<ExternalEvaluation> {
        let records = self.dataset_records(&req.dataset_id)?;
        if req.predictions.len() != records.len() {
            return Err(err(
                "PREDICTION_COUNT_MISMATCH",
                format!(
                    "The model returned {} predictions for {} rows.",
                    req.predictions.len(),
                    records.len()
                ),
            ));
        }
        let execution_id = format!("onnx-{}", Uuid::new_v4());
        let inference_ms = (req.inference_ms.max(0.0) * 100.0).round() / 100.0;
        let entry = |task: &str, metric: &str, value: f64, summary: HashMap<String, f64>| {
            ModelComparisonEntry {
                execution_id: execution_id.clone(),
                dataset_id: req.dataset_id.clone(),
                task_type: task.to_string(),
                algorithm: req.model_name.clone(),
                executed_at: Utc::now(),
                train_time_ms: inference_ms,
                primary_metric_name: metric.to_string(),
                primary_metric_value: value,
                summary_metrics: summary,
            }
        };

        let (metrics, comparison_entry, class_mapping) = match req.task_type.to_lowercase().as_str()
        {
            "classification" => {
                let truth = label_column(&records, &req.target_column)?;
                let (predicted, mapping) =
                    predictions_as_labels(&req.predictions, req.kind, &sorted_classes(&truth))?;
                let m = compute_classification_metrics(&truth, &predicted);
                let summary = HashMap::from([
                    ("accuracy".to_string(), m.accuracy),
                    ("precision".to_string(), m.precision),
                    ("recall".to_string(), m.recall),
                    ("f1_score".to_string(), m.f1_score),
                ]);
                let e = entry("Classification", "Accuracy", m.accuracy, summary);
                (MetricResults::Classification(m), e, mapping)
            }
            "regression" => {
                let truth = numeric_column(&records, &req.target_column)?;
                let predicted = req
                    .predictions
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        p.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                            err(
                                "INVALID_PREDICTION",
                                format!("Prediction {} is not a finite number.", i + 1),
                            )
                        })
                    })
                    .collect::<EngineResult<Vec<f64>>>()?;
                let m = compute_regression_metrics(&truth, &predicted);
                let summary = HashMap::from([
                    ("mae".to_string(), m.mean_absolute_error),
                    ("mse".to_string(), m.mean_squared_error),
                    ("rmse".to_string(), m.root_mean_squared_error),
                    ("r2".to_string(), m.r2_score),
                ]);
                let e = entry("Regression", "R2 Score", m.r2_score, summary);
                (MetricResults::Regression(m), e, None)
            }
            other => {
                return Err(err(
                    "UNSUPPORTED_TASK_TYPE",
                    format!(
                        "Task type '{}' is not supported for uploaded models.",
                        other
                    ),
                ));
            }
        };

        self.comparison_store.record_run(comparison_entry.clone());
        Ok(ExternalEvaluation {
            execution: ExecutionResponse {
                execution_id,
                dataset_id: req.dataset_id,
                task_type: req.task_type,
                algorithm: req.model_name,
                train_time_ms: inference_ms,
                train_samples: 0,
                test_samples: records.len(),
                metrics,
                comparison_entry,
            },
            class_mapping,
        })
    }

    fn dataset_records(&self, dataset_id: &str) -> EngineResult<Vec<Value>> {
        Ok(self
            .cleaned_data(dataset_id)?
            .records
            .into_iter()
            .map(|rec| serde_json::to_value(rec).unwrap_or_default())
            .collect())
    }

    /// Runs executed in this session for one dataset; "all" or "" lists every run (FR-12).
    pub fn comparison_history(&self, dataset_id: &str) -> Value {
        let runs = self.comparison_store.get_history(dataset_id);
        json!({ "dataset_id": dataset_id, "total_runs": runs.len(), "runs": runs })
    }

    /// Side-by-side ranking of two or more runs of the same task (FR-12, BR-7).
    pub fn compare_models(&self, execution_ids: &[String]) -> EngineResult<ComparisonReport> {
        if execution_ids.len() < 2 {
            return Err(StandardErrorResponse::new(
                "INSUFFICIENT_MODELS_FOR_COMPARISON",
                "Model comparison requires selecting at least two trained models side-by-side (FR-12 / BR-7).",
                Some(json!({ "selected_count": execution_ids.len() })),
            ));
        }
        self.comparison_store
            .build_comparison_report(execution_ids)
            .map_err(|message| err("COMPARISON_FAILED", message))
    }

    // ------------------------------------------------------------------
    // Message entry point used by the Web Worker
    // ------------------------------------------------------------------

    /// Runs one engine method by name with JSON arguments; `bytes` carries file contents.
    /// This is the single entry point the browser calls (see `wasm.rs`).
    pub fn call(&mut self, method: &str, args: Value, bytes: Option<&[u8]>) -> EngineResult<Value> {
        let dataset_id = || -> EngineResult<String> { arg(&args, "dataset_id") };
        match method {
            "upload_dataset" => {
                let filename: String = arg(&args, "filename")?;
                let bytes = bytes.ok_or_else(|| {
                    err(
                        "MISSING_FILE",
                        "No file contents were provided.".to_string(),
                    )
                })?;
                to_json(self.upload_dataset(&filename, bytes)?)
            }
            "preview" => to_json(self.preview(&dataset_id()?)?),
            "preprocess" => {
                let request: PreprocessingRequest = arg(&args, "request")?;
                to_json(self.preprocess(&dataset_id()?, &request)?)
            }
            "cleaned_data" => to_json(self.cleaned_data(&dataset_id()?)?),
            "schema" => self.schema(&dataset_id()?),
            "purge_dataset" => self.purge_dataset(&dataset_id()?),
            "purge_session" => Ok(self.purge_session()),
            "list_tasks" => to_json(self.list_tasks()),
            "validate_config" => to_json(self.validate_config(&parse(args)?)?),
            "dispatch_pipeline" => {
                let config: TaskConfiguration = arg(&args, "config")?;
                to_json(self.dispatch_pipeline(&config)?)
            }
            "execute_model" => to_json(self.execute_model(parse(args)?)?),
            "comparison_history" => {
                let id: Option<String> = arg(&args, "dataset_id").ok();
                Ok(self.comparison_history(id.as_deref().unwrap_or("all")))
            }
            "compare_models" => {
                let ids: Vec<String> = arg(&args, "execution_ids")?;
                to_json(self.compare_models(&ids)?)
            }
            "model_inputs" => to_json(self.model_inputs(&parse(args)?)?),
            "record_external_predictions" => {
                to_json(self.record_external_predictions(parse(args)?)?)
            }
            other => Err(err(
                "UNKNOWN_METHOD",
                format!("The engine has no method named '{}'.", other),
            )),
        }
    }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn parse<T: DeserializeOwned>(value: Value) -> EngineResult<T> {
    serde_json::from_value(value).map_err(|e| err("INVALID_REQUEST", e.to_string()))
}

fn arg<T: DeserializeOwned>(args: &Value, key: &str) -> EngineResult<T> {
    match args.get(key) {
        Some(v) => parse(v.clone()),
        None => Err(err(
            "INVALID_REQUEST",
            format!("Missing argument '{}'.", key),
        )),
    }
}

fn to_json<T: Serialize>(value: T) -> EngineResult<Value> {
    serde_json::to_value(value).map_err(|e| err("SERIALIZATION_FAILED", e.to_string()))
}

/// Reads one cell as a number. Missing columns and non-numeric values are errors,
/// never silently replaced by 0.
fn numeric_cell(record: &Value, column: &str, row: usize) -> EngineResult<f64> {
    let parsed = match record.get(column) {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
        None => {
            return Err(err(
                "UNKNOWN_COLUMN",
                format!("Column '{}' does not exist in this dataset.", column),
            ));
        }
        _ => None,
    };
    parsed.filter(|v| v.is_finite()).ok_or_else(|| {
        err(
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
    records: &[Value],
    feature_columns: &[String],
) -> EngineResult<Vec<Vec<f64>>> {
    if feature_columns.is_empty() {
        return Err(err(
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
fn label_column(records: &[Value], target: &str) -> EngineResult<Vec<String>> {
    records
        .iter()
        .enumerate()
        .map(|(i, rec)| match rec.get(target) {
            Some(Value::String(s)) if !s.trim().is_empty() => Ok(s.clone()),
            Some(Value::Number(n)) => Ok(n.to_string()),
            Some(Value::Bool(b)) => Ok(b.to_string()),
            None => Err(err(
                "UNKNOWN_COLUMN",
                format!("Target column '{}' does not exist in this dataset.", target),
            )),
            _ => Err(err(
                "MISSING_TARGET_VALUE",
                format!("Target column '{}' is empty in row {}.", target, i + 1),
            )),
        })
        .collect()
}

/// Reads the target column as numbers (regression).
fn numeric_column(records: &[Value], target: &str) -> EngineResult<Vec<f64>> {
    records
        .iter()
        .enumerate()
        .map(|(i, r)| numeric_cell(r, target, i))
        .collect()
}

/// Distinct labels in sorted order (the order label encoders such as scikit-learn's use).
fn sorted_classes(labels: &[String]) -> Vec<String> {
    let mut classes = labels.to_vec();
    classes.sort();
    classes.dedup();
    classes
}

/// Turns raw model outputs into labels comparable with the dataset's target column.
///
/// Text outputs are used as-is. Numbers are matched to numeric labels by value, or, when
/// the labels are text, read as positions in the sorted class list; that second case is
/// reported back so the user can see how outputs were interpreted.
fn predictions_as_labels(
    predictions: &[Value],
    kind: PredictionKind,
    classes: &[String],
) -> EngineResult<(Vec<String>, Option<String>)> {
    let numeric_classes: Option<Vec<(f64, &String)>> = classes
        .iter()
        .map(|c| c.trim().parse::<f64>().ok().map(|v| (v, c)))
        .collect();
    let by_index = kind == PredictionKind::ClassIndex || numeric_classes.is_none();
    let mut used_index = false;

    let labels = predictions
        .iter()
        .enumerate()
        .map(|(i, p)| match p {
            Value::String(s) => Ok(s.clone()),
            Value::Bool(b) => Ok(b.to_string()),
            Value::Number(n) => {
                let v = n.as_f64().unwrap_or(f64::NAN);
                if by_index {
                    used_index = true;
                    let idx = v as usize;
                    if v.fract() != 0.0 || v < 0.0 || idx >= classes.len() {
                        return Err(err(
                            "INVALID_PREDICTION",
                            format!(
                                "Prediction {} is class number {}, but the target column has {} classes.",
                                i + 1,
                                v,
                                classes.len()
                            ),
                        ));
                    }
                    Ok(classes[idx].clone())
                } else {
                    // Numeric labels: match by value; an unknown value simply counts as wrong
                    let matched = numeric_classes
                        .as_ref()
                        .and_then(|nc| nc.iter().find(|(c, _)| (c - v).abs() < 1e-9))
                        .map(|(_, label)| (*label).clone());
                    Ok(matched.unwrap_or_else(|| v.to_string()))
                }
            }
            _ => Err(err(
                "INVALID_PREDICTION",
                format!("Prediction {} is not a label or a number.", i + 1),
            )),
        })
        .collect::<EngineResult<Vec<String>>>()?;

    let mapping = used_index.then(|| {
        let pairs: Vec<String> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{} = {}", i, c))
            .collect();
        format!(
            "The model outputs class numbers; they were matched to the target's labels in sorted order: {}.",
            pairs.join(", ")
        )
    });
    Ok((labels, mapping))
}

fn required_target(target: &Option<String>) -> EngineResult<&str> {
    target
        .as_deref()
        .filter(|t| !t.trim().is_empty())
        .ok_or_else(|| {
            err(
                "MISSING_TARGET_COLUMN",
                "Select a target column for supervised learning.".to_string(),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ImputationStrategy;

    fn two_class_csv() -> String {
        let mut csv = String::from("x,y,label\n");
        for i in 0..20 {
            let label = if i % 2 == 0 { "a" } else { "b" };
            csv.push_str(&format!("{},{},{}\n", i, i * 2, label));
        }
        csv
    }

    fn session_with(csv: &str) -> (Session, String) {
        let mut s = Session::new();
        let id = s
            .upload_dataset("data.csv", csv.as_bytes())
            .unwrap()
            .dataset_id;
        (s, id)
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

    fn config(dataset_id: &str) -> TaskConfiguration {
        TaskConfiguration {
            dataset_id: dataset_id.to_string(),
            task_type: TaskType::Classification,
            algorithm: "Random Forest Classifier".to_string(),
            target_column: Some("label".to_string()),
            feature_columns: vec!["x".to_string(), "y".to_string()],
            test_split_ratio: Some(0.25),
            num_clusters: None,
            random_seed: None,
        }
    }

    // ---- datasets ----

    #[test]
    fn upload_rejects_non_csv_and_tiny_files() {
        let mut s = Session::new();
        assert!(s.upload_dataset("data.xlsx", b"a,b\n1,2\n").is_err());
        let e = s.upload_dataset("data.csv", b"a,b\n1,2\n").unwrap_err();
        assert!(!e.error.code.is_empty());
    }

    #[test]
    fn schema_for_unknown_dataset_is_not_found() {
        let s = Session::new();
        assert_eq!(
            s.schema("missing").unwrap_err().error.code,
            "DATASET_NOT_FOUND"
        );
    }

    #[test]
    fn schema_describes_uploaded_data_before_preprocessing() {
        let (s, id) = session_with(&two_class_csv());
        let res = s.schema(&id).unwrap();
        assert_eq!(res["preprocessed"], false);
        assert_eq!(res["total_rows"], 20);
        assert_eq!(res["columns"], json!(["x", "y", "label"]));
        assert_eq!(res["data_types"]["x"], "Numeric");
        assert_eq!(res["data_types"]["label"], "Categorical");
    }

    #[test]
    fn schema_omits_columns_dropped_during_preprocessing() {
        let (mut s, id) = session_with(&two_class_csv());
        let req = PreprocessingRequest {
            imputation_rules: HashMap::from([("y".to_string(), ImputationStrategy::DropColumn)]),
            drop_duplicates: false,
            trim_whitespace: true,
        };
        s.preprocess(&id, &req).unwrap();
        let res = s.schema(&id).unwrap();
        assert_eq!(res["preprocessed"], true);
        assert_eq!(res["columns"], json!(["x", "label"]));
    }

    #[test]
    fn purge_session_forgets_everything() {
        let (mut s, id) = session_with(&two_class_csv());
        s.execute_model(request(&id, "classification", Some("label")))
            .unwrap();
        s.purge_session();
        assert_eq!(s.preview(&id).unwrap_err().error.code, "DATASET_NOT_FOUND");
        assert_eq!(s.comparison_history("all")["total_runs"], 0);
    }

    #[test]
    fn purge_dataset_removes_only_that_dataset() {
        let (mut s, id) = session_with(&two_class_csv());
        let other = s
            .upload_dataset("b.csv", two_class_csv().as_bytes())
            .unwrap()
            .dataset_id;
        s.purge_dataset(&id).unwrap();
        assert!(s.preview(&id).is_err());
        assert!(s.preview(&other).is_ok());
        assert!(s.purge_dataset(&id).is_err());
    }

    // ---- execution: no sample data, no silent zeros (#43) ----

    #[test]
    fn unknown_dataset_is_an_error_not_sample_data() {
        let mut s = Session::new();
        let e = s
            .execute_model(request("missing", "classification", Some("label")))
            .unwrap_err();
        assert_eq!(e.error.code, "DATASET_NOT_FOUND");
    }

    #[test]
    fn complete_dataset_trains_without_preprocessing() {
        let (mut s, id) = session_with(&two_class_csv());
        let res = s
            .execute_model(request(&id, "classification", Some("label")))
            .unwrap();
        assert_eq!(res.train_samples + res.test_samples, 20);
        assert_eq!(s.comparison_history(&id)["total_runs"], 1);
    }

    #[test]
    fn dataset_with_gaps_must_be_preprocessed_first() {
        let csv = two_class_csv().replacen("0,0,a", "0,,a", 1);
        let (mut s, id) = session_with(&csv);
        let e = s
            .execute_model(request(&id, "classification", Some("label")))
            .unwrap_err();
        assert_eq!(e.error.code, "DATASET_NEEDS_PREPROCESSING");
    }

    #[test]
    fn text_feature_is_rejected_instead_of_becoming_zero() {
        let (mut s, id) = session_with(&two_class_csv());
        let mut req = request(&id, "classification", Some("x"));
        req.feature_columns = vec!["label".to_string()];
        assert_eq!(
            s.execute_model(req).unwrap_err().error.code,
            "NON_NUMERIC_VALUE"
        );
    }

    #[test]
    fn missing_target_column_is_rejected() {
        let (mut s, id) = session_with(&two_class_csv());
        let e = s
            .execute_model(request(&id, "classification", None))
            .unwrap_err();
        assert_eq!(e.error.code, "MISSING_TARGET_COLUMN");
        let e = s
            .execute_model(request(&id, "classification", Some("nope")))
            .unwrap_err();
        assert_eq!(e.error.code, "UNKNOWN_COLUMN");
    }

    #[test]
    fn clustering_requires_explicit_k() {
        let (mut s, id) = session_with(&two_class_csv());
        let mut req = request(&id, "clustering", None);
        req.algorithm = "K-Means Clustering".to_string();
        assert_eq!(
            s.execute_model(req.clone()).unwrap_err().error.code,
            "MISSING_CLUSTER_COUNT"
        );
        req.num_clusters = Some(2);
        assert!(s.execute_model(req).is_ok());
    }

    // ---- routing & comparison ----

    #[test]
    fn invalid_config_is_rejected_before_training() {
        let (mut s, id) = session_with(&two_class_csv());
        let mut c = config(&id);
        c.feature_columns = vec!["label".to_string(), "x".to_string()];
        let e = s.dispatch_pipeline(&c).unwrap_err();
        assert_eq!(e.error.code, "CONFIG_VALIDATION_FAILED");
        assert_eq!(s.comparison_history("all")["total_runs"], 0);
    }

    #[test]
    fn dispatch_trains_and_compare_ranks_real_runs() {
        let (mut s, id) = session_with(&two_class_csv());
        let first = s.dispatch_pipeline(&config(&id)).unwrap();
        let mut c = config(&id);
        c.algorithm = "Logistic Regression".to_string();
        let second = s.dispatch_pipeline(&c).unwrap();

        let ids: Vec<String> = [first, second]
            .iter()
            .map(|d| {
                d.execution_result.as_ref().unwrap()["execution_id"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        let report = s.compare_models(&ids).unwrap();
        assert_eq!(report.ranking.len(), 2);
        assert!(s.compare_models(&ids[..1]).is_err());
    }

    #[test]
    fn list_tasks_covers_all_three_tasks() {
        let tasks = Session::new().list_tasks();
        assert_eq!(tasks.len(), 3);
        assert!(tasks[0]
            .supported_algorithms
            .iter()
            .any(|a| a.contains("Random Forest")));
        assert!(!tasks[2].requires_target);
    }

    // ---- the browser entry point ----

    #[test]
    fn call_routes_methods_and_reports_bad_input() {
        let mut s = Session::new();
        let up = s
            .call(
                "upload_dataset",
                json!({ "filename": "d.csv" }),
                Some(two_class_csv().as_bytes()),
            )
            .unwrap();
        let id = up["dataset_id"].as_str().unwrap().to_string();
        assert_eq!(
            s.call("schema", json!({ "dataset_id": id }), None).unwrap()["total_rows"],
            20
        );

        assert_eq!(
            s.call("nope", json!({}), None).unwrap_err().error.code,
            "UNKNOWN_METHOD"
        );
        assert_eq!(
            s.call("schema", json!({}), None).unwrap_err().error.code,
            "INVALID_REQUEST"
        );
        assert_eq!(
            s.call("upload_dataset", json!({ "filename": "d.csv" }), None)
                .unwrap_err()
                .error
                .code,
            "MISSING_FILE"
        );
    }

    // ---- uploaded (ONNX) models: inputs and scoring (#46) ----

    fn predictions(
        dataset_id: &str,
        task: &str,
        target: &str,
        preds: Vec<Value>,
    ) -> ExternalPredictions {
        ExternalPredictions {
            dataset_id: dataset_id.to_string(),
            model_name: "Uploaded ONNX: model.onnx".to_string(),
            task_type: task.to_string(),
            target_column: target.to_string(),
            predictions: preds,
            kind: PredictionKind::Label,
            inference_ms: 3.5,
        }
    }

    #[test]
    fn model_inputs_give_features_and_sorted_classes() {
        let (s, id) = session_with(&two_class_csv());
        let inputs = s
            .model_inputs(&ModelInputsRequest {
                dataset_id: id,
                task_type: "classification".to_string(),
                target_column: "label".to_string(),
                feature_columns: vec!["y".to_string(), "x".to_string()],
            })
            .unwrap();
        assert_eq!(inputs.rows, 20);
        assert_eq!(inputs.features[3], vec![6.0, 3.0]);
        assert_eq!(inputs.classes, vec!["a", "b"]);
    }

    #[test]
    fn model_predictions_are_scored_against_the_dataset() {
        let (mut s, id) = session_with(&two_class_csv());
        // Correct except the first row
        let preds: Vec<Value> = (0..20)
            .map(|i| json!(if (i % 2 == 0) == (i != 0) { "a" } else { "b" }))
            .collect();
        let res = s
            .record_external_predictions(predictions(&id, "classification", "label", preds))
            .unwrap();
        match &res.execution.metrics {
            MetricResults::Classification(m) => assert!((m.accuracy - 0.95).abs() < 1e-9),
            other => panic!("unexpected {other:?}"),
        }
        assert!(res.class_mapping.is_none());
        assert_eq!(
            s.comparison_history(&res.execution.dataset_id)["total_runs"],
            1
        );
    }

    #[test]
    fn class_numbers_map_to_sorted_text_labels_and_are_reported() {
        let (mut s, id) = session_with(&two_class_csv());
        let preds: Vec<Value> = (0..20).map(|i| json!(i % 2)).collect();
        let mut req = predictions(&id, "classification", "label", preds);
        req.kind = PredictionKind::ClassIndex;
        let res = s.record_external_predictions(req).unwrap();
        match &res.execution.metrics {
            MetricResults::Classification(m) => assert_eq!(m.accuracy, 1.0),
            other => panic!("unexpected {other:?}"),
        }
        assert!(res.class_mapping.unwrap().contains("0 = a, 1 = b"));
    }

    #[test]
    fn out_of_range_class_numbers_and_wrong_counts_are_errors() {
        let (mut s, id) = session_with(&two_class_csv());
        let preds: Vec<Value> = (0..20).map(|_| json!(7)).collect();
        let mut req = predictions(&id, "classification", "label", preds);
        req.kind = PredictionKind::ClassIndex;
        assert_eq!(
            s.record_external_predictions(req).unwrap_err().error.code,
            "INVALID_PREDICTION"
        );
        let req = predictions(&id, "classification", "label", vec![json!("a")]);
        assert_eq!(
            s.record_external_predictions(req).unwrap_err().error.code,
            "PREDICTION_COUNT_MISMATCH"
        );
    }

    #[test]
    fn regression_predictions_are_scored() {
        let (mut s, id) = session_with(&two_class_csv());
        // y = 2x exactly
        let preds: Vec<Value> = (0..20).map(|i| json!(i as f64 * 2.0)).collect();
        let res = s
            .record_external_predictions(predictions(&id, "regression", "y", preds))
            .unwrap();
        match &res.execution.metrics {
            MetricResults::Regression(m) => {
                assert!(m.mean_absolute_error < 1e-9);
                assert!((m.r2_score - 1.0).abs() < 1e-9);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(res.execution.train_time_ms, 3.5);
    }
}

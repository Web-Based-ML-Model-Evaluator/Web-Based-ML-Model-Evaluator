use crate::core::router_validation::{get_supported_algorithms, validate_task_config};
use crate::models::{
    ExecutionRequest, PipelineDispatchRequest, PipelineDispatchResponse, StandardErrorResponse,
    TaskConfiguration, TaskSupportInfo, TaskType, ValidationResult,
};
use crate::state::AppState;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
};
use uuid::Uuid;

/// GET /api/v1/config/tasks (FR-4)
pub async fn list_supported_tasks() -> Json<Vec<TaskSupportInfo>> {
    let tasks = vec![
        TaskSupportInfo {
            task_type: TaskType::Classification,
            display_name: "Classification".to_string(),
            description: "Predict discrete category or class labels".to_string(),
            supported_algorithms: get_supported_algorithms(&TaskType::Classification)
                .into_iter()
                .map(String::from)
                .collect(),
            requires_target: true,
            allows_split: true,
            hyperparameter_fields: vec!["test_split_ratio".to_string(), "random_seed".to_string()],
        },
        TaskSupportInfo {
            task_type: TaskType::Regression,
            display_name: "Regression".to_string(),
            description: "Predict continuous numeric target values".to_string(),
            supported_algorithms: get_supported_algorithms(&TaskType::Regression)
                .into_iter()
                .map(String::from)
                .collect(),
            requires_target: true,
            allows_split: true,
            hyperparameter_fields: vec!["test_split_ratio".to_string(), "random_seed".to_string()],
        },
        TaskSupportInfo {
            task_type: TaskType::Clustering,
            display_name: "Clustering (Unsupervised)".to_string(),
            description: "Group similar data points into clusters without labels".to_string(),
            supported_algorithms: get_supported_algorithms(&TaskType::Clustering)
                .into_iter()
                .map(String::from)
                .collect(),
            requires_target: false,
            allows_split: false,
            hyperparameter_fields: vec!["num_clusters".to_string()],
        },
    ];

    Json(tasks)
}

/// Helper to get schema from unified in-memory state
pub fn get_schema_from_state(
    state: &AppState,
    dataset_id: &str,
) -> Result<(Vec<String>, std::collections::HashMap<String, String>), StandardErrorResponse> {
    // Check cleaned dataset first
    if let Ok(cleaned_lock) = state.cleaned_datasets.read() {
        if let Some(cleaned) = cleaned_lock.get(dataset_id) {
            return Ok((cleaned.columns.clone(), cleaned.data_types.clone()));
        }
    }

    // Check upload summaries
    if let Ok(summary_lock) = state.upload_summaries.read() {
        if let Some(summary) = summary_lock.get(dataset_id) {
            let types = summary
                .column_summaries
                .iter()
                .map(|col| (col.name.clone(), col.inferred_type.clone()))
                .collect();
            return Ok((summary.column_names.clone(), types));
        }
    }

    // Fallback default for mock or demo runs if dataset_id begins with "demo" or "dataset-001"
    if dataset_id.starts_with("dataset-001") || dataset_id.starts_with("demo") {
        return Ok((
            vec![
                "sepal_length".to_string(),
                "sepal_width".to_string(),
                "petal_length".to_string(),
                "petal_width".to_string(),
                "species".to_string(),
            ],
            std::collections::HashMap::from([
                ("sepal_length".to_string(), "Numeric".to_string()),
                ("sepal_width".to_string(), "Numeric".to_string()),
                ("petal_length".to_string(), "Numeric".to_string()),
                ("petal_width".to_string(), "Numeric".to_string()),
                ("species".to_string(), "Categorical".to_string()),
            ]),
        ));
    }

    Err(StandardErrorResponse::new(
        "DATASET_NOT_FOUND",
        &format!("Dataset with ID '{}' was not found in active session.", dataset_id),
        None,
    ))
}

/// POST /api/v1/config/validate (FR-5, FR-6, FR-7, FR-13)
pub async fn validate_configuration(
    State(state): State<AppState>,
    Json(config): Json<TaskConfiguration>,
) -> Result<Json<ValidationResult>, (StatusCode, Json<StandardErrorResponse>)> {
    let (columns, types) = get_schema_from_state(&state, &config.dataset_id)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(e)))?;

    let result = validate_task_config(&config, Some(&columns), Some(&types));

    if !result.is_valid {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(StandardErrorResponse::new(
                "CONFIG_VALIDATION_FAILED",
                &result.errors.join("; "),
                Some(serde_json::json!({
                    "errors": result.errors,
                    "warnings": result.warnings
                })),
            )),
        ));
    }

    Ok(Json(result))
}

/// POST /api/v1/pipeline/dispatch
pub async fn dispatch_pipeline(
    State(state): State<AppState>,
    Json(payload): Json<PipelineDispatchRequest>,
) -> Result<Json<PipelineDispatchResponse>, (StatusCode, Json<StandardErrorResponse>)> {
    let pipeline_start = std::time::Instant::now();
    let config = &payload.config;

    // 1. Validate configuration against unified state schema
    let (columns, types) = get_schema_from_state(&state, &config.dataset_id)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(e)))?;

    let validation = validate_task_config(config, Some(&columns), Some(&types));
    if !validation.is_valid {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(StandardErrorResponse::new(
                "DISPATCH_VALIDATION_FAILED",
                &validation.errors.join("; "),
                Some(serde_json::json!({ "errors": validation.errors })),
            )),
        ));
    }

    // 2. Prepare execution request from configuration
    let exec_req = ExecutionRequest {
        dataset_id: config.dataset_id.clone(),
        task_type: match config.task_type {
            TaskType::Classification => "classification".to_string(),
            TaskType::Regression => "regression".to_string(),
            TaskType::Clustering => "clustering".to_string(),
        },
        algorithm: config.algorithm.clone(),
        target_column: config.target_column.clone(),
        feature_columns: config.feature_columns.clone(),
        test_split_ratio: config.test_split_ratio,
        num_clusters: config.num_clusters,
        data: None, // Will be fetched directly from unified in-memory state
    };

    // 3. Execute directly in-memory via unified execution engine!
    let exec_response = crate::api::models::execute_model_internal(&state, exec_req)
        .map_err(|(code, err)| (code, Json(err)))?;

    // Enforce 30-second pipeline latency constraint (PR-1, PR-2, NFR-1)
    if pipeline_start.elapsed().as_secs() >= 30 {
        return Err((
            StatusCode::GATEWAY_TIMEOUT,
            Json(StandardErrorResponse::new(
                "PIPELINE_LATENCY_EXCEEDED",
                &format!(
                    "Pipeline execution exceeded the 30-second latency budget (took {:.2}s, limit 30s) per PR-1, PR-2, and NFR-1.",
                    pipeline_start.elapsed().as_secs_f64()
                ),
                Some(serde_json::json!({
                    "elapsed_ms": (pipeline_start.elapsed().as_secs_f64() * 1000.0).round(),
                    "max_allowed_seconds": 30
                })),
            )),
        ));
    }

    Ok(Json(PipelineDispatchResponse {
        dispatch_id: Uuid::new_v4().to_string(),
        status: "PIPELINE_EXECUTED_SUCCESSFULLY".to_string(),
        dataset_id: config.dataset_id.clone(),
        task_type: format!("{:?}", config.task_type),
        algorithm: config.algorithm.clone(),
        execution_result: Some(serde_json::to_value(&exec_response).unwrap_or_default()),
    }))
}

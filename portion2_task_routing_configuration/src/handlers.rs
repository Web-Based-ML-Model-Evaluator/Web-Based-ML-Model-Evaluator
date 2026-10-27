use crate::client::PipelineClient;
use crate::models::{
    PipelineDispatchRequest, PipelineDispatchResponse, StandardErrorResponse, TaskConfiguration,
    TaskSupportInfo, TaskType, ValidationResult,
};
use crate::router_validation::{get_supported_algorithms, validate_task_config};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub client: Arc<PipelineClient>,
}

/// GET /health
pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "portion2_task_routing_configuration",
        "owner": "Satvik Phadke (PES2UG24CS450)",
        "port": 8002
    }))
}

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

/// POST /api/v1/config/validate (FR-5, FR-6, FR-7, FR-13)
pub async fn validate_configuration(
    State(state): State<AppState>,
    Json(config): Json<TaskConfiguration>,
) -> Result<Json<ValidationResult>, (StatusCode, Json<StandardErrorResponse>)> {
    // Attempt fetching schema from Portion 1 (or fallback)
    let (columns, types) = state
        .client
        .fetch_dataset_schema(&config.dataset_id)
        .await
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

    // 1. Validate configuration first
    let (columns, types) = state
        .client
        .fetch_dataset_schema(&config.dataset_id)
        .await
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

    // 2. Fetch cleaned data from Portion 1
    let cleaned_data = state
        .client
        .fetch_cleaned_data(&config.dataset_id)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, Json(e)))?;

    // 3. Dispatch to Portion 3 for execution
    let execution_result = state
        .client
        .dispatch_to_portion3(config, &cleaned_data)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, Json(e)))?;

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
        execution_result: Some(execution_result),
    }))
}

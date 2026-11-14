use crate::core::preprocessing::{clean_and_impute, parse_csv_summary};
use crate::core::validator::{
    sanitize_filename, validate_cleaned_dataset, validate_csv_upload, validate_dataset_size,
};
use crate::models::{
    CleanedDatasetDTO, DatasetUploadResponse, PreprocessingRequest, PreprocessingResponse,
    StandardErrorResponse,
};
use crate::state::AppState;
use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::Json,
};
use std::collections::HashMap;
use uuid::Uuid;

/// POST /api/v1/datasets/upload (FR-1, FR-2, NFR-4)
pub async fn upload_dataset(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<DatasetUploadResponse>, (StatusCode, Json<StandardErrorResponse>)> {
    let mut filename = "uploaded_dataset.csv".to_string();
    let mut file_bytes: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" || name == "dataset" {
            if let Some(f_name) = field.file_name() {
                filename = sanitize_filename(f_name);
            }
            let data = field.bytes().await.map_err(|e| {
                let status = e.status();
                let (code, message) = if status == StatusCode::PAYLOAD_TOO_LARGE {
                    (
                        "FILE_TOO_LARGE",
                        "Uploaded file exceeds the maximum allowed limit of 10 MB.".to_string(),
                    )
                } else {
                    (
                        "UPLOAD_READ_FAILED",
                        format!("Failed to read multipart file stream: {}", e),
                    )
                };
                (
                    status,
                    Json(StandardErrorResponse::new(code, &message, None)),
                )
            })?;
            file_bytes = Some(data.to_vec());
            break;
        }
    }

    let bytes = match file_bytes {
        Some(b) => b,
        None => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(StandardErrorResponse::new(
                    "MISSING_FILE_FIELD",
                    "No file field found in multipart payload. Key must be 'file'.",
                    None,
                )),
            ));
        }
    };

    // Strict validation (10MB limit, extension, UTF-8)
    if let Err(err) = validate_csv_upload(&filename, &bytes) {
        let status = if err.error.code == "FILE_TOO_LARGE" {
            StatusCode::PAYLOAD_TOO_LARGE
        } else {
            StatusCode::BAD_REQUEST
        };
        return Err((status, Json(err)));
    }

    let dataset_id = Uuid::new_v4().to_string();

    let (raw_data, summary) = parse_csv_summary(&dataset_id, &filename, &bytes)
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, Json(e)))?;

    // SR-1 / QA-5: reject datasets too small to be used for ML before storing them
    validate_dataset_size(summary.total_rows, summary.total_columns)
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, Json(e)))?;

    // In-memory session buffer (SASD 3.1)
    {
        let mut raw_lock = state.raw_datasets.write().unwrap();
        raw_lock.insert(dataset_id.clone(), raw_data);

        let mut summary_lock = state.upload_summaries.write().unwrap();
        summary_lock.insert(dataset_id.clone(), summary.clone());
    }

    Ok(Json(summary))
}

/// GET /api/v1/datasets/{dataset_id}/preview (FR-2)
pub async fn get_preview(
    State(state): State<AppState>,
    Path(dataset_id): Path<String>,
) -> Result<Json<DatasetUploadResponse>, (StatusCode, Json<StandardErrorResponse>)> {
    let summary_lock = state.upload_summaries.read().unwrap();
    if let Some(summary) = summary_lock.get(&dataset_id) {
        Ok(Json(summary.clone()))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(StandardErrorResponse::new(
                "DATASET_NOT_FOUND",
                &format!("Dataset with ID '{}' was not found or expired.", dataset_id),
                None,
            )),
        ))
    }
}

/// POST /api/v1/datasets/{dataset_id}/preprocess (FR-3)
pub async fn preprocess_dataset(
    State(state): State<AppState>,
    Path(dataset_id): Path<String>,
    Json(payload): Json<PreprocessingRequest>,
) -> Result<Json<PreprocessingResponse>, (StatusCode, Json<StandardErrorResponse>)> {
    let raw_data = {
        let raw_lock = state.raw_datasets.read().unwrap();
        raw_lock.get(&dataset_id).cloned().ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(StandardErrorResponse::new(
                    "DATASET_NOT_FOUND",
                    &format!("Dataset with ID '{}' was not found.", dataset_id),
                    None,
                )),
            )
        })?
    };

    let (cleaned_dto, response) =
        clean_and_impute(&raw_data, &payload).map_err(|e| (StatusCode::BAD_REQUEST, Json(e)))?;

    // SR-1 / QA-2: never hand routing & execution a dataset that cleaning has made unusable
    validate_cleaned_dataset(&cleaned_dto)
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, Json(e)))?;

    {
        let mut cleaned_lock = state.cleaned_datasets.write().unwrap();
        cleaned_lock.insert(dataset_id, cleaned_dto);
    }

    Ok(Json(response))
}

/// GET /api/v1/datasets/{dataset_id}/data
pub async fn get_cleaned_data(
    State(state): State<AppState>,
    Path(dataset_id): Path<String>,
) -> Result<Json<CleanedDatasetDTO>, (StatusCode, Json<StandardErrorResponse>)> {
    let cleaned_lock = state.cleaned_datasets.read().unwrap();
    if let Some(cleaned) = cleaned_lock.get(&dataset_id) {
        Ok(Json(cleaned.clone()))
    } else {
        // Check if raw dataset exists and can be served as-is
        let raw_lock = state.raw_datasets.read().unwrap();
        if let Some(raw) = raw_lock.get(&dataset_id) {
            let empty_req = PreprocessingRequest {
                imputation_rules: HashMap::new(),
                drop_duplicates: false,
                trim_whitespace: true,
            };
            if let Ok((cleaned_dto, _)) = clean_and_impute(raw, &empty_req) {
                // SR-1: the same checks apply when preprocessing was skipped
                validate_cleaned_dataset(&cleaned_dto)
                    .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, Json(e)))?;
                return Ok(Json(cleaned_dto));
            }
        }

        Err((
            StatusCode::NOT_FOUND,
            Json(StandardErrorResponse::new(
                "CLEANED_DATA_NOT_FOUND",
                &format!(
                    "Cleaned data for dataset '{}' not found. Run preprocessing first.",
                    dataset_id
                ),
                None,
            )),
        ))
    }
}

/// GET /api/v1/datasets/{dataset_id}/schema
pub async fn get_schema(
    State(state): State<AppState>,
    Path(dataset_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<StandardErrorResponse>)> {
    // After preprocessing, describe the cleaned data so dropped columns are not offered downstream
    {
        let cleaned_lock = state.cleaned_datasets.read().unwrap();
        if let Some(cleaned) = cleaned_lock.get(&dataset_id) {
            return Ok(Json(serde_json::json!({
                "dataset_id": dataset_id,
                "total_rows": cleaned.total_rows,
                "columns": cleaned.columns,
                "data_types": cleaned.data_types,
                "preprocessed": true
            })));
        }
    }

    let summary_lock = state.upload_summaries.read().unwrap();
    if let Some(summary) = summary_lock.get(&dataset_id) {
        let types: HashMap<String, String> = summary
            .column_summaries
            .iter()
            .map(|col| (col.name.clone(), col.inferred_type.clone()))
            .collect();

        Ok(Json(serde_json::json!({
            "dataset_id": dataset_id,
            "total_rows": summary.total_rows,
            "columns": summary.column_names,
            "data_types": types,
            "preprocessed": false
        })))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(StandardErrorResponse::new(
                "DATASET_NOT_FOUND",
                &format!("Dataset with ID '{}' was not found.", dataset_id),
                None,
            )),
        ))
    }
}

/// DELETE /api/v1/datasets/{dataset_id} or POST /api/v1/datasets/{dataset_id}/purge
pub async fn purge_dataset(
    State(state): State<AppState>,
    Path(dataset_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<StandardErrorResponse>)> {
    let mut raw_lock = state.raw_datasets.write().unwrap();
    let mut summary_lock = state.upload_summaries.write().unwrap();
    let mut cleaned_lock = state.cleaned_datasets.write().unwrap();

    let removed_raw = raw_lock.remove(&dataset_id).is_some();
    let removed_summary = summary_lock.remove(&dataset_id).is_some();
    let removed_cleaned = cleaned_lock.remove(&dataset_id).is_some();

    if removed_raw || removed_summary || removed_cleaned {
        Ok(Json(serde_json::json!({
            "status": "PURGED_SUCCESSFULLY",
            "dataset_id": dataset_id,
            "message": "Dataset securely purged from in-memory session buffer per SEC-1 and SEC-3."
        })))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(StandardErrorResponse::new(
                "DATASET_NOT_FOUND",
                &format!(
                    "Dataset with ID '{}' does not exist or has already been purged.",
                    dataset_id
                ),
                None,
            )),
        ))
    }
}

/// POST /api/v1/session/purge (SEC-1, SEC-3, SRS 7.3)
pub async fn purge_all_datasets(State(state): State<AppState>) -> Json<serde_json::Value> {
    state.purge_all();

    Json(serde_json::json!({
        "status": "SESSION_PURGED",
        "message": "All session datasets, models, and comparisons purged securely from host memory per SEC-1 and SEC-3."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ImputationStrategy;

    fn state_with_dataset(id: &str) -> AppState {
        let mut csv = String::from("x,y,label\n");
        for i in 0..12 {
            csv.push_str(&format!(
                "{},{},{}\n",
                i,
                i * 2,
                if i % 2 == 0 { "a" } else { "b" }
            ));
        }
        let (raw, summary) = parse_csv_summary(id, "t.csv", csv.as_bytes()).unwrap();
        let state = AppState::default();
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

    #[tokio::test]
    async fn schema_for_unknown_dataset_is_404() {
        let state = AppState::default();
        let err = get_schema(State(state), Path("missing".to_string()))
            .await
            .unwrap_err();
        assert_eq!(err.0, StatusCode::NOT_FOUND);
        assert_eq!(err.1 .0.error.code, "DATASET_NOT_FOUND");
    }

    #[tokio::test]
    async fn schema_describes_uploaded_data_before_preprocessing() {
        let state = state_with_dataset("d1");
        let res = get_schema(State(state), Path("d1".to_string()))
            .await
            .unwrap()
            .0;

        assert_eq!(res["preprocessed"], false);
        assert_eq!(res["total_rows"], 12);
        let cols: Vec<String> = serde_json::from_value(res["columns"].clone()).unwrap();
        assert_eq!(cols, vec!["x", "y", "label"]);
        assert_eq!(res["data_types"]["x"], "Numeric");
        assert_eq!(res["data_types"]["label"], "Categorical");
    }

    #[tokio::test]
    async fn schema_omits_columns_dropped_during_preprocessing() {
        let state = state_with_dataset("d1");
        let mut rules = HashMap::new();
        rules.insert("y".to_string(), ImputationStrategy::DropColumn);
        let _ = preprocess_dataset(
            State(state.clone()),
            Path("d1".to_string()),
            Json(PreprocessingRequest {
                imputation_rules: rules,
                drop_duplicates: false,
                trim_whitespace: true,
            }),
        )
        .await
        .unwrap();

        let res = get_schema(State(state), Path("d1".to_string()))
            .await
            .unwrap()
            .0;
        assert_eq!(res["preprocessed"], true);
        let cols: Vec<String> = serde_json::from_value(res["columns"].clone()).unwrap();
        assert_eq!(cols, vec!["x", "label"]);
    }
}

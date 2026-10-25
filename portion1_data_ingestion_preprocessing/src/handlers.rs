use crate::models::{
    CleanedDatasetDTO, DatasetUploadResponse, PreprocessingRequest, PreprocessingResponse,
    StandardErrorResponse,
};
use crate::preprocessing::{clean_and_impute, parse_csv_summary, RawDataset};
use crate::validator::validate_csv_upload;
use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

#[derive(Default, Clone)]
pub struct AppState {
    pub raw_datasets: Arc<RwLock<HashMap<String, RawDataset>>>,
    pub upload_summaries: Arc<RwLock<HashMap<String, DatasetUploadResponse>>>,
    pub cleaned_datasets: Arc<RwLock<HashMap<String, CleanedDatasetDTO>>>,
}

/// GET /health
pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "portion1_data_ingestion_preprocessing",
        "owner": "Sneha Ajith (PES2UG24CS480)",
        "port": 8001
    }))
}

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
                filename = f_name.to_string();
            }
            let data = field.bytes().await.map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(StandardErrorResponse::new(
                        "UPLOAD_READ_FAILED",
                        &format!("Failed to read multipart file stream: {}", e),
                        None,
                    )),
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

    let (cleaned_dto, response) = clean_and_impute(&raw_data, &payload)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(e)))?;

    {
        let mut cleaned_lock = state.cleaned_datasets.write().unwrap();
        cleaned_lock.insert(dataset_id, cleaned_dto);
    }

    Ok(Json(response))
}

/// GET /api/v1/datasets/{dataset_id}/data (Internal API for Portion 2 & Portion 3)
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
                return Ok(Json(cleaned_dto));
            }
        }

        Err((
            StatusCode::NOT_FOUND,
            Json(StandardErrorResponse::new(
                "CLEANED_DATA_NOT_FOUND",
                &format!("Cleaned data for dataset '{}' not found. Run preprocessing first.", dataset_id),
                None,
            )),
        ))
    }
}

/// GET /api/v1/datasets/{dataset_id}/schema (Internal API for Portion 2)
pub async fn get_schema(
    State(state): State<AppState>,
    Path(dataset_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<StandardErrorResponse>)> {
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
            "data_types": types
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

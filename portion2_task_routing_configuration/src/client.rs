use crate::models::{StandardErrorResponse, TaskConfiguration};
use reqwest::Client;
use std::collections::HashMap;
use std::time::Duration;

/// Pipeline HTTP client connecting Portion 2 with Portions 1 and 3
///
/// TODO: Assigned to Satvik Phadke (PES2UG24CS450)
#[derive(Clone)]
pub struct PipelineClient {
    #[allow(dead_code)]
    client: Client,
    pub portion1_base_url: String,
    pub portion3_base_url: String,
}

impl PipelineClient {
    pub fn new(p1_url: &str, p3_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        Self {
            client,
            portion1_base_url: p1_url.to_string(),
            portion3_base_url: p3_url.to_string(),
        }
    }

    /// Fetches dataset schema from Portion 1 (Sneha Ajith's service)
    /// TODO: Satvik Phadke - Implement HTTP GET request to `/api/v1/datasets/{id}/schema`
    pub async fn fetch_dataset_schema(
        &self,
        _dataset_id: &str,
    ) -> Result<(Vec<String>, HashMap<String, String>), StandardErrorResponse> {
        // SKELETON MOCK: Provides offline mock schema so Satvik is never blocked
        Ok((
            vec!["col1".to_string(), "col2".to_string(), "target".to_string()],
            HashMap::from([
                ("col1".to_string(), "Numeric".to_string()),
                ("col2".to_string(), "Numeric".to_string()),
                ("target".to_string(), "Categorical".to_string()),
            ]),
        ))
    }

    /// Fetches cleaned records from Portion 1 (Sneha Ajith's service)
    /// TODO: Satvik Phadke - Implement HTTP GET request to `/api/v1/datasets/{id}/data`
    pub async fn fetch_cleaned_data(
        &self,
        dataset_id: &str,
    ) -> Result<serde_json::Value, StandardErrorResponse> {
        // SKELETON MOCK: Provides offline mock data
        Ok(serde_json::json!({
            "dataset_id": dataset_id,
            "records": []
        }))
    }

    /// Dispatches execution request to Portion 3 (Shreya Patil's service)
    /// TODO: Satvik Phadke - Implement HTTP POST request to `/api/v1/models/execute`
    pub async fn dispatch_to_portion3(
        &self,
        config: &TaskConfiguration,
        _data_records: &serde_json::Value,
    ) -> Result<serde_json::Value, StandardErrorResponse> {
        // SKELETON MOCK: Returns dispatch receipt
        Ok(serde_json::json!({
            "execution_id": "skeleton-exec-id",
            "status": "DISPATCHED_TO_PORTION3_SKELETON",
            "algorithm": config.algorithm,
            "task_type": config.task_type
        }))
    }
}

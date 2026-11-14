use crate::core::comparison_store::ComparisonStore;
use crate::core::ml::hardware_accel::HardwareStatus;
use crate::core::preprocessing::RawDataset;
use crate::models::{CleanedDatasetDTO, DatasetUploadResponse};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct AppState {
    pub raw_datasets: Arc<RwLock<HashMap<String, RawDataset>>>,
    pub upload_summaries: Arc<RwLock<HashMap<String, DatasetUploadResponse>>>,
    pub cleaned_datasets: Arc<RwLock<HashMap<String, CleanedDatasetDTO>>>,
    pub comparison_store: ComparisonStore,
    pub hardware_status: HardwareStatus,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            raw_datasets: Arc::new(RwLock::new(HashMap::new())),
            upload_summaries: Arc::new(RwLock::new(HashMap::new())),
            cleaned_datasets: Arc::new(RwLock::new(HashMap::new())),
            comparison_store: ComparisonStore::default(),
            hardware_status: HardwareStatus::probe(),
        }
    }
}

impl AppState {
    pub fn purge_all(&self) {
        if let Ok(mut r) = self.raw_datasets.write() {
            r.clear();
        }
        if let Ok(mut u) = self.upload_summaries.write() {
            u.clear();
        }
        if let Ok(mut c) = self.cleaned_datasets.write() {
            c.clear();
        }
        self.comparison_store.purge_all();
    }
}

use crate::models::ModelComparisonEntry;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Default, Clone)]
pub struct ComparisonStore {
    // Map of session_id / dataset_id -> list of trained model runs
    history: Arc<RwLock<HashMap<String, Vec<ModelComparisonEntry>>>>,
}

impl ComparisonStore {
    pub fn record_run(&self, entry: ModelComparisonEntry) {
        let mut map = self.history.write().unwrap();
        map.entry(entry.dataset_id.clone())
            .or_insert_with(Vec::new)
            .push(entry);
    }

    pub fn get_history(&self, dataset_id: &str) -> Vec<ModelComparisonEntry> {
        let map = self.history.read().unwrap();
        map.get(dataset_id).cloned().unwrap_or_default()
    }

    pub fn compare_runs(&self, execution_ids: &[String]) -> Vec<ModelComparisonEntry> {
        let map = self.history.read().unwrap();
        let mut results = Vec::new();
        for runs in map.values() {
            for r in runs {
                if execution_ids.contains(&r.execution_id) {
                    results.push(r.clone());
                }
            }
        }
        results
    }
}

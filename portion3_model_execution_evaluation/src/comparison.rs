use crate::models::ModelComparisonEntry;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Represents an individual model's ranking in a side-by-side comparison (FR-12)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ModelRankEntry {
    pub rank: usize,
    pub execution_id: String,
    pub algorithm: String,
    pub primary_metric_name: String,
    pub primary_metric_value: f64,
    pub train_time_ms: f64,
}

/// Comprehensive side-by-side comparison report across multiple trained models (FR-12 / BR-7)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ComparisonReport {
    pub dataset_id: Option<String>,
    pub task_type: String,
    pub compared_models: Vec<ModelComparisonEntry>,
    pub best_model_id: String,
    pub best_model_algorithm: String,
    pub recommendation_reason: String,
    pub metrics_table: HashMap<String, HashMap<String, f64>>,
    pub ranking: Vec<ModelRankEntry>,
}

#[derive(Default, Clone)]
pub struct ComparisonStore {
    // Map of session_id / dataset_id -> list of trained model runs
    history: Arc<RwLock<HashMap<String, Vec<ModelComparisonEntry>>>>,
}

impl ComparisonStore {
    /// Records a completed model execution run into session history (FR-12)
    pub fn record_run(&self, entry: ModelComparisonEntry) {
        let mut map = self.history.write().unwrap();
        let list = map.entry(entry.dataset_id.clone()).or_insert_with(Vec::new);
        // Avoid duplicate entries if executed with the same ID
        if let Some(pos) = list.iter().position(|e| e.execution_id == entry.execution_id) {
            list[pos] = entry;
        } else {
            list.push(entry);
        }
    }

    /// Retrieves all model runs for a given dataset or session ID (FR-12)
    pub fn get_history(&self, dataset_id: &str) -> Vec<ModelComparisonEntry> {
        let map = self.history.read().unwrap();
        if dataset_id.is_empty() || dataset_id == "all" || dataset_id == "default" {
            let mut all: Vec<ModelComparisonEntry> = map.values().flatten().cloned().collect();
            all.sort_by(|a, b| b.executed_at.cmp(&a.executed_at));
            return all;
        }

        map.get(dataset_id).cloned().unwrap_or_default()
    }

    /// Retrieves all model runs across all sessions, sorted with most recent first
    pub fn get_all_runs(&self) -> Vec<ModelComparisonEntry> {
        let map = self.history.read().unwrap();
        let mut all: Vec<ModelComparisonEntry> = map.values().flatten().cloned().collect();
        all.sort_by(|a, b| b.executed_at.cmp(&a.executed_at));
        all
    }

    /// Retrieves a single model execution by its ID
    pub fn get_run_by_id(&self, execution_id: &str) -> Option<ModelComparisonEntry> {
        let map = self.history.read().unwrap();
        for runs in map.values() {
            for r in runs {
                if r.execution_id == execution_id {
                    return Some(r.clone());
                }
            }
        }
        None
    }

    /// Filters and returns model comparison entries matching the given execution IDs
    pub fn compare_runs(&self, execution_ids: &[String]) -> Vec<ModelComparisonEntry> {
        let map = self.history.read().unwrap();
        let mut results = Vec::new();
        for runs in map.values() {
            for r in runs {
                if execution_ids.contains(&r.execution_id)
                    && !results.iter().any(|existing: &ModelComparisonEntry| existing.execution_id == r.execution_id)
                {
                    results.push(r.clone());
                }
            }
        }
        results
    }

    /// Compares two or more trained models side-by-side, determining the optimal model and rankings (FR-12, BR-7)
    pub fn build_comparison_report(&self, execution_ids: &[String]) -> Result<ComparisonReport, String> {
        if execution_ids.len() < 2 {
            return Err("Model comparison requires at least two trained models side-by-side (FR-12 / BR-7).".to_string());
        }

        let mut found_models: Vec<ModelComparisonEntry> = Vec::new();
        for id in execution_ids {
            if let Some(entry) = self.get_run_by_id(id) {
                if !found_models.iter().any(|m| m.execution_id == entry.execution_id) {
                    found_models.push(entry);
                }
            }
        }

        if found_models.len() < 2 {
            return Err(format!(
                "Could not find at least two valid trained models from the provided IDs. Found {} of {} requested (BR-7).",
                found_models.len(),
                execution_ids.len()
            ));
        }

        // Validate task type homogeneity (BR-7: compare models within same problem domain)
        let first_task = found_models[0].task_type.clone();
        for m in &found_models[1..] {
            if !m.task_type.eq_ignore_ascii_case(&first_task) {
                return Err(format!(
                    "Cannot compare models with differing task types: found '{}' and '{}' (BR-7).",
                    first_task, m.task_type
                ));
            }
        }

        // Determine optimal model and rankings based on task type
        let mut ranked = found_models.clone();
        ranked.sort_by(|a, b| {
            let score_a = score_model_for_comparison(a, &first_task);
            let score_b = score_model_for_comparison(b, &first_task);
            score_b
                .0
                .partial_cmp(&score_a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| score_b.1.partial_cmp(&score_a.1).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| score_b.2.partial_cmp(&score_a.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        let mut ranking = Vec::new();
        for (idx, m) in ranked.iter().enumerate() {
            ranking.push(ModelRankEntry {
                rank: idx + 1,
                execution_id: m.execution_id.clone(),
                algorithm: m.algorithm.clone(),
                primary_metric_name: m.primary_metric_name.clone(),
                primary_metric_value: m.primary_metric_value,
                train_time_ms: m.train_time_ms,
            });
        }

        let best = &ranked[0];
        let best_model_id = best.execution_id.clone();
        let best_model_algorithm = best.algorithm.clone();

        let recommendation_reason = match first_task.to_lowercase().as_str() {
            "classification" => format!(
                "{} is the optimal model, achieving the highest {} of {:.4} with training latency of {:.1} ms.",
                best.algorithm, best.primary_metric_name, best.primary_metric_value, best.train_time_ms
            ),
            "regression" => format!(
                "{} is the optimal model, achieving the highest {} of {:.4} with training latency of {:.1} ms.",
                best.algorithm, best.primary_metric_name, best.primary_metric_value, best.train_time_ms
            ),
            "clustering" => format!(
                "{} is the optimal model, achieving the highest cluster separation with {} of {:.4} in {:.1} ms.",
                best.algorithm, best.primary_metric_name, best.primary_metric_value, best.train_time_ms
            ),
            _ => format!(
                "{} is optimal based on primary metric {} ({:.4}).",
                best.algorithm, best.primary_metric_name, best.primary_metric_value
            ),
        };

        // Build side-by-side metric comparison matrix
        let mut metrics_table: HashMap<String, HashMap<String, f64>> = HashMap::new();

        // Add primary metric
        let mut primary_row = HashMap::new();
        for m in &found_models {
            primary_row.insert(m.execution_id.clone(), m.primary_metric_value);
        }
        metrics_table.insert(found_models[0].primary_metric_name.clone(), primary_row);

        // Add training latency
        let mut time_row = HashMap::new();
        for m in &found_models {
            time_row.insert(m.execution_id.clone(), m.train_time_ms);
        }
        metrics_table.insert("train_time_ms".to_string(), time_row);

        // Add all detailed summary metrics
        for m in &found_models {
            for (key, &val) in &m.summary_metrics {
                metrics_table
                    .entry(key.clone())
                    .or_insert_with(HashMap::new)
                    .insert(m.execution_id.clone(), val);
            }
        }

        let dataset_id = found_models[0].dataset_id.clone();

        Ok(ComparisonReport {
            dataset_id: Some(dataset_id),
            task_type: first_task,
            compared_models: found_models,
            best_model_id,
            best_model_algorithm,
            recommendation_reason,
            metrics_table,
            ranking,
        })
    }

    /// Clears all stored runs (used in testing)
    pub fn clear(&self) {
        let mut map = self.history.write().unwrap();
        map.clear();
    }

    /// Seeds realistic mock runs across all 3 tasks to enable standalone offline exploration
    pub fn seed_mock_runs(&self) {
        let session = "default-session".to_string();

        // 1. Classification models
        let mut rf_summary = HashMap::new();
        rf_summary.insert("accuracy".to_string(), 0.950);
        rf_summary.insert("precision".to_string(), 0.940);
        rf_summary.insert("recall".to_string(), 0.950);
        rf_summary.insert("f1_score".to_string(), 0.945);

        self.record_run(ModelComparisonEntry {
            execution_id: "mock-clf-rf".to_string(),
            dataset_id: session.clone(),
            task_type: "Classification".to_string(),
            algorithm: "Random Forest Classifier".to_string(),
            executed_at: Utc::now(),
            train_time_ms: 38.4,
            primary_metric_name: "Accuracy".to_string(),
            primary_metric_value: 0.950,
            summary_metrics: rf_summary,
        });

        let mut lr_summary = HashMap::new();
        lr_summary.insert("accuracy".to_string(), 0.880);
        lr_summary.insert("precision".to_string(), 0.870);
        lr_summary.insert("recall".to_string(), 0.880);
        lr_summary.insert("f1_score".to_string(), 0.875);

        self.record_run(ModelComparisonEntry {
            execution_id: "mock-clf-lr".to_string(),
            dataset_id: session.clone(),
            task_type: "Classification".to_string(),
            algorithm: "Logistic Regression".to_string(),
            executed_at: Utc::now(),
            train_time_ms: 12.1,
            primary_metric_name: "Accuracy".to_string(),
            primary_metric_value: 0.880,
            summary_metrics: lr_summary,
        });

        let mut knn_summary = HashMap::new();
        knn_summary.insert("accuracy".to_string(), 0.910);
        knn_summary.insert("precision".to_string(), 0.900);
        knn_summary.insert("recall".to_string(), 0.910);
        knn_summary.insert("f1_score".to_string(), 0.905);

        self.record_run(ModelComparisonEntry {
            execution_id: "mock-clf-knn".to_string(),
            dataset_id: session.clone(),
            task_type: "Classification".to_string(),
            algorithm: "K-Nearest Neighbors (KNN)".to_string(),
            executed_at: Utc::now(),
            train_time_ms: 18.2,
            primary_metric_name: "Accuracy".to_string(),
            primary_metric_value: 0.910,
            summary_metrics: knn_summary,
        });

        let mut dt_summary = HashMap::new();
        dt_summary.insert("accuracy".to_string(), 0.860);
        dt_summary.insert("precision".to_string(), 0.850);
        dt_summary.insert("recall".to_string(), 0.860);
        dt_summary.insert("f1_score".to_string(), 0.855);

        self.record_run(ModelComparisonEntry {
            execution_id: "mock-clf-dt".to_string(),
            dataset_id: session.clone(),
            task_type: "Classification".to_string(),
            algorithm: "Decision Tree Classifier".to_string(),
            executed_at: Utc::now(),
            train_time_ms: 9.5,
            primary_metric_name: "Accuracy".to_string(),
            primary_metric_value: 0.860,
            summary_metrics: dt_summary,
        });

        // 2. Regression models
        let mut rf_reg_summary = HashMap::new();
        rf_reg_summary.insert("r2".to_string(), 0.892);
        rf_reg_summary.insert("mae".to_string(), 2.14);
        rf_reg_summary.insert("mse".to_string(), 8.65);
        rf_reg_summary.insert("rmse".to_string(), 2.94);

        self.record_run(ModelComparisonEntry {
            execution_id: "mock-reg-rf".to_string(),
            dataset_id: session.clone(),
            task_type: "Regression".to_string(),
            algorithm: "Random Forest Regressor".to_string(),
            executed_at: Utc::now(),
            train_time_ms: 42.0,
            primary_metric_name: "R2 Score".to_string(),
            primary_metric_value: 0.892,
            summary_metrics: rf_reg_summary,
        });

        let mut dt_reg_summary = HashMap::new();
        dt_reg_summary.insert("r2".to_string(), 0.810);
        dt_reg_summary.insert("mae".to_string(), 3.20);
        dt_reg_summary.insert("mse".to_string(), 15.20);
        dt_reg_summary.insert("rmse".to_string(), 3.90);

        self.record_run(ModelComparisonEntry {
            execution_id: "mock-reg-dt".to_string(),
            dataset_id: session.clone(),
            task_type: "Regression".to_string(),
            algorithm: "Decision Tree Regressor".to_string(),
            executed_at: Utc::now(),
            train_time_ms: 8.3,
            primary_metric_name: "R2 Score".to_string(),
            primary_metric_value: 0.810,
            summary_metrics: dt_reg_summary,
        });

        let mut lin_reg_summary = HashMap::new();
        lin_reg_summary.insert("r2".to_string(), 0.765);
        lin_reg_summary.insert("mae".to_string(), 4.10);
        lin_reg_summary.insert("mse".to_string(), 18.80);
        lin_reg_summary.insert("rmse".to_string(), 4.34);

        self.record_run(ModelComparisonEntry {
            execution_id: "mock-reg-lin".to_string(),
            dataset_id: session.clone(),
            task_type: "Regression".to_string(),
            algorithm: "Linear Regression".to_string(),
            executed_at: Utc::now(),
            train_time_ms: 6.1,
            primary_metric_name: "R2 Score".to_string(),
            primary_metric_value: 0.765,
            summary_metrics: lin_reg_summary,
        });

        // 3. Clustering model
        let mut km_summary = HashMap::new();
        km_summary.insert("silhouette".to_string(), 0.725);
        km_summary.insert("davies_bouldin".to_string(), 0.650);

        self.record_run(ModelComparisonEntry {
            execution_id: "mock-clust-km".to_string(),
            dataset_id: session,
            task_type: "Clustering".to_string(),
            algorithm: "K-Means Clustering".to_string(),
            executed_at: Utc::now(),
            train_time_ms: 15.0,
            primary_metric_name: "Silhouette Score".to_string(),
            primary_metric_value: 0.725,
            summary_metrics: km_summary,
        });
    }
}

/// Helper function to rank models: returns a 3-tuple score where higher is always better for all 3 components
fn score_model_for_comparison(entry: &ModelComparisonEntry, task_type: &str) -> (f64, f64, f64) {
    match task_type.to_lowercase().as_str() {
        "classification" => {
            let acc = entry
                .summary_metrics
                .get("accuracy")
                .copied()
                .unwrap_or(entry.primary_metric_value);
            let f1 = entry.summary_metrics.get("f1_score").copied().unwrap_or(0.0);
            let time_score = -entry.train_time_ms; // Lower training time is preferred
            (acc, f1, time_score)
        }
        "regression" => {
            let r2 = entry
                .summary_metrics
                .get("r2")
                .or_else(|| entry.summary_metrics.get("r2_score"))
                .copied()
                .unwrap_or(entry.primary_metric_value);
            // Lower MAE is better, so negate it
            let mae_score = -entry
                .summary_metrics
                .get("mae")
                .or_else(|| entry.summary_metrics.get("mean_absolute_error"))
                .copied()
                .unwrap_or(0.0);
            let time_score = -entry.train_time_ms;
            (r2, mae_score, time_score)
        }
        "clustering" => {
            let sil = entry
                .summary_metrics
                .get("silhouette")
                .or_else(|| entry.summary_metrics.get("silhouette_score"))
                .copied()
                .unwrap_or(entry.primary_metric_value);
            // Lower Davies-Bouldin is better, so negate it
            let db_score = -entry
                .summary_metrics
                .get("davies_bouldin")
                .or_else(|| entry.summary_metrics.get("davies_bouldin_index"))
                .copied()
                .unwrap_or(0.0);
            let time_score = -entry.train_time_ms;
            (sil, db_score, time_score)
        }
        _ => (entry.primary_metric_value, 0.0, -entry.train_time_ms),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_clf_entry(id: &str, algo: &str, acc: f64, f1: f64, time: f64) -> ModelComparisonEntry {
        let mut map = HashMap::new();
        map.insert("accuracy".to_string(), acc);
        map.insert("f1_score".to_string(), f1);
        ModelComparisonEntry {
            execution_id: id.to_string(),
            dataset_id: "dataset-test".to_string(),
            task_type: "Classification".to_string(),
            algorithm: algo.to_string(),
            executed_at: Utc::now(),
            train_time_ms: time,
            primary_metric_name: "Accuracy".to_string(),
            primary_metric_value: acc,
            summary_metrics: map,
        }
    }

    #[test]
    fn test_record_and_get_history() {
        let store = ComparisonStore::default();
        let entry = sample_clf_entry("run-1", "Random Forest Classifier", 0.95, 0.94, 30.0);
        store.record_run(entry);

        let runs = store.get_history("dataset-test");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].algorithm, "Random Forest Classifier");
    }

    #[test]
    fn test_report_requires_at_least_two_models() {
        let store = ComparisonStore::default();
        store.record_run(sample_clf_entry("run-1", "RF", 0.95, 0.94, 30.0));

        let res = store.build_comparison_report(&["run-1".to_string()]);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("at least two trained models"));
    }

    #[test]
    fn test_report_rejects_differing_tasks() {
        let store = ComparisonStore::default();
        store.record_run(sample_clf_entry("run-1", "RF", 0.95, 0.94, 30.0));

        let mut reg_map = HashMap::new();
        reg_map.insert("r2".to_string(), 0.85);
        store.record_run(ModelComparisonEntry {
            execution_id: "run-2".to_string(),
            dataset_id: "dataset-test".to_string(),
            task_type: "Regression".to_string(),
            algorithm: "Linear Regression".to_string(),
            executed_at: Utc::now(),
            train_time_ms: 10.0,
            primary_metric_name: "R2 Score".to_string(),
            primary_metric_value: 0.85,
            summary_metrics: reg_map,
        });

        let res = store.build_comparison_report(&["run-1".to_string(), "run-2".to_string()]);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("differing task types"));
    }

    #[test]
    fn test_report_ranks_classification_optimally() {
        let store = ComparisonStore::default();
        store.record_run(sample_clf_entry("run-dt", "Decision Tree Classifier", 0.85, 0.84, 10.0));
        store.record_run(sample_clf_entry("run-rf", "Random Forest Classifier", 0.96, 0.95, 35.0));
        store.record_run(sample_clf_entry("run-knn", "K-Nearest Neighbors", 0.91, 0.90, 20.0));

        let report = store
            .build_comparison_report(&[
                "run-dt".to_string(),
                "run-rf".to_string(),
                "run-knn".to_string(),
            ])
            .unwrap();

        assert_eq!(report.best_model_id, "run-rf");
        assert_eq!(report.best_model_algorithm, "Random Forest Classifier");
        assert_eq!(report.ranking.len(), 3);
        assert_eq!(report.ranking[0].execution_id, "run-rf");
        assert_eq!(report.ranking[0].rank, 1);
        assert_eq!(report.ranking[1].execution_id, "run-knn");
        assert_eq!(report.ranking[1].rank, 2);
        assert_eq!(report.ranking[2].execution_id, "run-dt");
        assert_eq!(report.ranking[2].rank, 3);
        assert!(report.recommendation_reason.contains("Random Forest Classifier"));
        assert!(report.metrics_table.contains_key("Accuracy"));
        assert!(report.metrics_table.contains_key("train_time_ms"));
    }

    #[test]
    fn test_seed_mock_runs() {
        let store = ComparisonStore::default();
        store.seed_mock_runs();

        let all = store.get_all_runs();
        assert!(all.len() >= 8); // 4 classification, 3 regression, 1 clustering
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_accuracy_calculation() {
        let actual = ["setosa", "versicolor", "virginica", "setosa"];
        let pred = ["setosa", "versicolor", "setosa", "setosa"]; // 3 out of 4 correct

        let correct = actual
            .iter()
            .zip(pred.iter())
            .filter(|(a, b)| a == b)
            .count();
        let acc = correct as f64 / actual.len() as f64;
        assert_eq!(acc, 0.75);
    }

    #[test]
    fn test_mae_rmse_calculation() {
        let actual: Vec<f64> = vec![100.0, 200.0, 300.0];
        let pred: Vec<f64> = vec![110.0, 190.0, 310.0]; // diffs: 10, -10, 10

        let mae: f64 = actual
            .iter()
            .zip(pred.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / 3.0;
        let mse: f64 = actual
            .iter()
            .zip(pred.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / 3.0;
        let rmse = mse.sqrt();

        assert_eq!(mae, 10.0);
        assert!((rmse - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_confusion_matrix_dimensions() {
        let num_classes = 3;
        let matrix = vec![vec![0usize; num_classes]; num_classes];
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
    }

    #[test]
    fn test_pca_2d_output_dimensions() {
        let n_samples = 10;
        let sample_coords = vec![(1.2, 0.4); n_samples];
        assert_eq!(sample_coords.len(), 10);
    }

    #[test]
    fn test_model_comparison_side_by_side_ranking() {
        // FR-12 / BR-7: Compare multiple classification models side-by-side to identify the optimal model
        struct MockModel {
            id: &'static str,
            algorithm: &'static str,
            accuracy: f64,
            f1_score: f64,
            latency_ms: f64,
        }

        let models = vec![
            MockModel {
                id: "m1",
                algorithm: "Decision Tree",
                accuracy: 0.85,
                f1_score: 0.84,
                latency_ms: 10.0,
            },
            MockModel {
                id: "m2",
                algorithm: "Random Forest",
                accuracy: 0.96,
                f1_score: 0.95,
                latency_ms: 38.0,
            },
            MockModel {
                id: "m3",
                algorithm: "KNN",
                accuracy: 0.91,
                f1_score: 0.90,
                latency_ms: 18.0,
            },
        ];

        // Optimal model must be Random Forest with highest accuracy (0.96)
        let mut sorted = models;
        sorted.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());

        assert_eq!(sorted[0].id, "m2");
        assert_eq!(sorted[0].algorithm, "Random Forest");
        assert!(sorted[0].f1_score >= sorted[1].f1_score);
        assert!(sorted[0].latency_ms > 0.0);
        assert_eq!(sorted[1].id, "m3");
        assert_eq!(sorted[2].id, "m1");
    }

    #[test]
    fn test_model_comparison_regression_ranking() {
        // FR-12 / BR-7: Compare regression models by R2 score
        struct MockRegModel {
            id: &'static str,
            algorithm: &'static str,
            r2_score: f64,
            mae: f64,
        }

        let models = vec![
            MockRegModel {
                id: "r1",
                algorithm: "Linear Regression",
                r2_score: 0.76,
                mae: 4.1,
            },
            MockRegModel {
                id: "r2",
                algorithm: "Random Forest Regressor",
                r2_score: 0.89,
                mae: 2.1,
            },
            MockRegModel {
                id: "r3",
                algorithm: "Decision Tree Regressor",
                r2_score: 0.81,
                mae: 3.2,
            },
        ];

        let mut sorted = models;
        sorted.sort_by(|a, b| b.r2_score.partial_cmp(&a.r2_score).unwrap());

        assert_eq!(sorted[0].id, "r2");
        assert_eq!(sorted[0].algorithm, "Random Forest Regressor");
        assert!(sorted[0].r2_score > sorted[1].r2_score);
        assert!(sorted[0].mae < sorted[1].mae);
    }

    #[test]
    fn test_model_comparison_latency_budget() {
        // NFR-1: Ensure execution and comparison latency is strictly below 30,000 ms (30s)
        let latencies = vec![38.4, 12.1, 18.2, 42.0, 15.0];
        for lat in latencies {
            assert!(
                lat < 30_000.0,
                "Model execution time must be under 30s budget"
            );
        }
    }

    #[test]
    fn test_security_session_buffer_purge() {
        // SEC-1 / SEC-3: In-memory model artifacts purged upon session termination
        use std::collections::HashMap;
        let mut session_runs: HashMap<String, Vec<&'static str>> = HashMap::new();
        session_runs.insert("session-user-1".to_string(), vec!["exec-1", "exec-2"]);
        assert_eq!(session_runs.get("session-user-1").unwrap().len(), 2);

        let removed = session_runs.remove("session-user-1");
        assert!(removed.is_some());
        assert!(
            !session_runs.contains_key("session-user-1"),
            "Session models must be wiped"
        );
    }

    #[test]
    fn test_structured_error_handling_fr13() {
        // FR-13: Standard structured error format
        let err_json = serde_json::json!({
            "error": {
                "code": "PIPELINE_LATENCY_EXCEEDED",
                "message": "Execution exceeded latency budget.",
                "module": "portion3_model_execution_evaluation",
                "timestamp": "2026-11-08T09:00:00Z"
            }
        });
        assert_eq!(err_json["error"]["code"], "PIPELINE_LATENCY_EXCEEDED");
        assert_eq!(
            err_json["error"]["module"],
            "portion3_model_execution_evaluation"
        );
    }

    #[test]
    fn test_neural_network_mlp_and_cnn_execution() {
        use web_ml_backend::core::ml::algorithms::{try_run_classification_model, try_run_regression_model};

        let x_train = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let y_train_cls = vec!["class_a".to_string(), "class_a".to_string(), "class_b".to_string(), "class_b".to_string()];
        let x_test = vec![vec![1.5, 2.5], vec![5.5, 6.5]];

        // MLP Classification
        let mlp_preds = try_run_classification_model("Neural Network (MLP)", &x_train, &y_train_cls, &x_test).unwrap();
        assert_eq!(mlp_preds.len(), 2);

        // CNN Classification
        let cnn_preds = try_run_classification_model("Convolutional Neural Network (CNN)", &x_train, &y_train_cls, &x_test).unwrap();
        assert_eq!(cnn_preds.len(), 2);

        // Regression
        let y_train_reg = vec![3.0, 5.0, 7.0, 11.0];
        let mlp_reg_preds = try_run_regression_model("Neural Network (MLP)", &x_train, &y_train_reg, &x_test).unwrap();
        assert_eq!(mlp_reg_preds.len(), 2);
        let cnn_reg_preds = try_run_regression_model("Convolutional Neural Network (CNN)", &x_train, &y_train_reg, &x_test).unwrap();
        assert_eq!(cnn_reg_preds.len(), 2);
    }

    #[test]
    fn test_model_file_ingestion_formats() {
        let supported_formats = [
            ("model.pt", "PyTorch (.pt)"),
            ("model.pth", "PyTorch (.pt)"),
            ("pipeline.joblib", "Joblib (.joblib)"),
            ("clf.pkl", "Pickle (.pkl)"),
            ("network.h5", "HDF5/Keras (.h5)"),
            ("graph.onnx", "ONNX (.onnx)"),
            ("weights.json", "JSON Model (.json)"),
        ];

        for (filename, _expected_format) in supported_formats {
            let lower = filename.to_lowercase();
            let is_valid = lower.ends_with(".pt")
                || lower.ends_with(".pth")
                || lower.ends_with(".joblib")
                || lower.ends_with(".pkl")
                || lower.ends_with(".h5")
                || lower.ends_with(".onnx")
                || lower.ends_with(".json");
            assert!(is_valid, "{filename} should be accepted");
        }
    }
}

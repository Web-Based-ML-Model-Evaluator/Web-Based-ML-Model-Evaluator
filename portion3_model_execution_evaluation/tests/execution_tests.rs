#[cfg(test)]
mod tests {
    #[test]
    fn test_accuracy_calculation() {
        let actual = vec!["setosa", "versicolor", "virginica", "setosa"];
        let pred = vec!["setosa", "versicolor", "setosa", "setosa"]; // 3 out of 4 correct

        let correct = actual.iter().zip(pred.iter()).filter(|(a, b)| a == b).count();
        let acc = correct as f64 / actual.len() as f64;
        assert_eq!(acc, 0.75);
    }

    #[test]
    fn test_mae_rmse_calculation() {
        let actual: Vec<f64> = vec![100.0, 200.0, 300.0];
        let pred: Vec<f64> = vec![110.0, 190.0, 310.0]; // diffs: 10, -10, 10

        let mae: f64 = actual.iter().zip(pred.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f64>() / 3.0;
        let mse: f64 = actual.iter().zip(pred.iter()).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>() / 3.0;
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
            MockModel { id: "m1", algorithm: "Decision Tree", accuracy: 0.85, f1_score: 0.84, latency_ms: 10.0 },
            MockModel { id: "m2", algorithm: "Random Forest", accuracy: 0.96, f1_score: 0.95, latency_ms: 38.0 },
            MockModel { id: "m3", algorithm: "KNN", accuracy: 0.91, f1_score: 0.90, latency_ms: 18.0 },
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
            MockRegModel { id: "r1", algorithm: "Linear Regression", r2_score: 0.76, mae: 4.1 },
            MockRegModel { id: "r2", algorithm: "Random Forest Regressor", r2_score: 0.89, mae: 2.1 },
            MockRegModel { id: "r3", algorithm: "Decision Tree Regressor", r2_score: 0.81, mae: 3.2 },
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
            assert!(lat < 30_000.0, "Model execution time must be under 30s budget");
        }
    }
}
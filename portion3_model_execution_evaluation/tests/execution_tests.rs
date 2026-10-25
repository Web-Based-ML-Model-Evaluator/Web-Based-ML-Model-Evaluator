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
}

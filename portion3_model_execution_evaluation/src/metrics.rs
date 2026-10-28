use crate::models::{ClassificationMetrics, ClusteringMetrics, PcaScatterPoint, RegressionMetrics};
use crate::ml_algorithms::MlError;
use std::collections::HashMap;

/// Computes Classification Metrics: Accuracy, Precision, Recall, F1, and Confusion Matrix
/// Requirements (FR-9):
/// - Accuracy = Correct / Total
/// - Macro-averaged Precision, Recall, and F1-Score
/// - Dynamic K x K Confusion Matrix with class labels
pub fn compute_classification_metrics(
    actual: &[String],
    predicted: &[String],
) -> ClassificationMetrics {
    let pairs: Vec<(&String, &String)> = actual.iter().zip(predicted.iter()).collect();

    let class_labels = sorted_labels(&pairs);
    let k = class_labels.len();
    let index_of = |l: &String| class_labels.iter().position(|c| c == l).expect("label collected");

    let mut cm = vec![vec![0usize; k]; k];
    for (a, p) in &pairs {
        cm[index_of(a)][index_of(p)] += 1;
    }

    let total = actual.len();
    let correct = actual.iter().zip(predicted.iter()).filter(|(a, b)| a == b).count();
    let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };

    let (mut p_sum, mut r_sum, mut f_sum) = (0.0, 0.0, 0.0);
    for c in 0..k {
        let tp = cm[c][c] as f64;
        let fp: f64 = (0..k).filter(|&r| r != c).map(|r| cm[r][c] as f64).sum();
        let fn_: f64 = (0..k).filter(|&j| j != c).map(|j| cm[c][j] as f64).sum();
        let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let r = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        p_sum += p;
        r_sum += r;
        f_sum += f;
    }
    let n = k.max(1) as f64;

    ClassificationMetrics {
        accuracy,
        precision: p_sum / n,
        recall: r_sum / n,
        f1_score: f_sum / n,
        confusion_matrix: cm,
        class_labels,
    }
}

fn sorted_labels(pairs: &[(&String, &String)]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut labels: Vec<String> = Vec::new();
    for (a, p) in pairs {
        for l in [*a, *p] {
            if seen.insert(l.clone()) {
                labels.push(l.clone());
            }
        }
    }
    if labels.iter().all(|s| s.trim().parse::<f64>().is_ok()) {
        labels.sort_by(|x, y| {
            x.trim().parse::<f64>().unwrap().total_cmp(&y.trim().parse::<f64>().unwrap())
        });
    } else {
        labels.sort();
    }
    labels
}

/// Computes Regression Metrics: MAE, MSE, RMSE, and R2 Score
///
/// Implemented for Issue 8 (FR-10):
/// - Mean Absolute Error (MAE): (1/n) * sum(|actual - predicted|)
/// - Mean Squared Error (MSE): (1/n) * sum((actual - predicted)^2)
/// - Root Mean Squared Error (RMSE): sqrt(MSE)
/// - R2 Score (Coefficient of Determination): 1 - (SS_res / SS_tot)
pub fn compute_regression_metrics(
    actual: &[f64],
    predicted: &[f64],
) -> RegressionMetrics {
    let n = actual.len().min(predicted.len());
    if n == 0 {
        return RegressionMetrics {
            mean_absolute_error: 0.0,
            mean_squared_error: 0.0,
            root_mean_squared_error: 0.0,
            r2_score: 0.0,
        };
    }

    let mut sum_abs_err = 0.0;
    let mut sum_sq_err = 0.0;
    let mut actual_sum = 0.0;

    for i in 0..n {
        let diff = actual[i] - predicted[i];
        sum_abs_err += diff.abs();
        sum_sq_err += diff * diff;
        actual_sum += actual[i];
    }

    let mae = sum_abs_err / n as f64;
    let mse = sum_sq_err / n as f64;
    let rmse = mse.sqrt();

    let mean_actual = actual_sum / n as f64;
    let mut ss_tot = 0.0;
    for &y in &actual[..n] {
        let diff = y - mean_actual;
        ss_tot += diff * diff;
    }

    let r2 = if ss_tot > 1e-12 {
        1.0 - (sum_sq_err / ss_tot)
    } else if sum_sq_err < 1e-12 {
        // Perfect prediction on constant data
        1.0
    } else {
        0.0
    };

    RegressionMetrics {
        mean_absolute_error: (mae * 10000.0).round() / 10000.0,
        mean_squared_error: (mse * 10000.0).round() / 10000.0,
        root_mean_squared_error: (rmse * 10000.0).round() / 10000.0,
        r2_score: (r2 * 10000.0).round() / 10000.0,
    }
}

fn euclidean_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .fold(0.0_f64, |distance, (a, b)| distance.hypot(a - b))
}

fn silhouette_score(features: &[Vec<f64>], assignments: &[usize]) -> f64 {
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (sample, &cluster) in assignments.iter().enumerate() {
        clusters.entry(cluster).or_default().push(sample);
    }
    if clusters.len() < 2 {
        return 0.0;
    }

    assignments
        .iter()
        .enumerate()
        .map(|(sample, own_cluster)| {
            let own_members = &clusters[own_cluster];
            if own_members.len() < 2 {
                return 0.0;
            }
            let within = own_members
                .iter()
                .filter(|&&other| other != sample)
                .map(|&other| euclidean_distance(&features[sample], &features[other]))
                .sum::<f64>()
                / (own_members.len() - 1) as f64;
            let nearest_other = clusters
                .iter()
                .filter(|(cluster, _)| *cluster != own_cluster)
                .map(|(_, members)| {
                    members
                        .iter()
                        .map(|&other| euclidean_distance(&features[sample], &features[other]))
                        .sum::<f64>()
                        / members.len() as f64
                })
                .fold(f64::INFINITY, f64::min);
            let denominator = within.max(nearest_other);
            if denominator.is_finite() && denominator > 0.0 {
                (nearest_other - within) / denominator
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / assignments.len() as f64
}

fn davies_bouldin_index(features: &[Vec<f64>], assignments: &[usize]) -> f64 {
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (sample, &cluster) in assignments.iter().enumerate() {
        clusters.entry(cluster).or_default().push(sample);
    }
    if clusters.len() < 2 {
        return 0.0;
    }

    let centroids: HashMap<usize, Vec<f64>> = clusters
        .iter()
        .map(|(&cluster, members)| {
            let mut centroid = vec![0.0; features[0].len()];
            for &sample in members {
                for (coordinate, value) in centroid.iter_mut().zip(&features[sample]) {
                    *coordinate += value;
                }
            }
            for coordinate in &mut centroid {
                *coordinate /= members.len() as f64;
            }
            (cluster, centroid)
        })
        .collect();
    let scatter: HashMap<usize, f64> = clusters
        .iter()
        .map(|(&cluster, members)| {
            let mean_distance = members
                .iter()
                .map(|&sample| euclidean_distance(&features[sample], &centroids[&cluster]))
                .sum::<f64>()
                / members.len() as f64;
            (cluster, mean_distance)
        })
        .collect();

    let sum_max_ratios = clusters
        .keys()
        .map(|&cluster| {
            clusters
                .keys()
                .filter(|&&other| other != cluster)
                .map(|&other| {
                    let centroid_distance =
                        euclidean_distance(&centroids[&cluster], &centroids[&other]);
                    let combined_scatter = scatter[&cluster] + scatter[&other];
                    if centroid_distance > 0.0 {
                        combined_scatter / centroid_distance
                    } else if combined_scatter > 0.0 {
                        f64::MAX
                    } else {
                        0.0
                    }
                })
                .fold(0.0_f64, f64::max)
        })
        .sum::<f64>();
    sum_max_ratios / clusters.len() as f64
}

/// Computes Euclidean silhouette and Davies-Bouldin scores plus cluster counts.
pub fn compute_clustering_metrics(
    features: &[Vec<f64>],
    assignments: &[usize],
    k: usize,
    pca_data: Vec<PcaScatterPoint>,
) -> ClusteringMetrics {
    compute_clustering_metrics_checked(features, assignments, k, pca_data)
        .unwrap_or_else(|error| panic!("{error}"))
}

fn compute_clustering_metrics_checked(
    features: &[Vec<f64>],
    assignments: &[usize],
    k: usize,
    pca_data: Vec<PcaScatterPoint>,
) -> Result<ClusteringMetrics, MlError> {
    if features.is_empty() {
        return Err(MlError::InvalidInput(
            "Clustering metrics require at least one data point.".into(),
        ));
    }
    if features.len() != assignments.len() {
        return Err(MlError::InvalidInput(
            "Features and cluster assignments have different lengths.".into(),
        ));
    }
    if k == 0 {
        return Err(MlError::InvalidInput(
            "The number of clusters must be at least one.".into(),
        ));
    }
    let dimensions = features[0].len();
    if dimensions == 0 {
        return Err(MlError::InvalidInput(
            "Clustering metrics require at least one feature.".into(),
        ));
    }
    for (row_index, row) in features.iter().enumerate() {
        if row.len() != dimensions {
            return Err(MlError::InvalidInput(format!(
                "Row {} has {} features; expected {dimensions}.",
                row_index + 1,
                row.len()
            )));
        }
        for (column_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(MlError::InvalidInput(format!(
                    "Feature at row {}, column {} is not finite ({value}).",
                    row_index + 1,
                    column_index + 1
                )));
            }
        }
    }
    if let Some((sample, cluster)) = assignments
        .iter()
        .enumerate()
        .find(|(_, cluster)| **cluster >= k)
    {
        return Err(MlError::InvalidInput(format!(
            "Sample {} has cluster {cluster}; expected a cluster index below {k}.",
            sample + 1
        )));
    }

    let mut cluster_counts: HashMap<usize, usize> = (0..k).map(|cluster| (cluster, 0)).collect();
    for &c in assignments {
        *cluster_counts.entry(c).or_insert(0) += 1;
    }

    Ok(ClusteringMetrics {
        silhouette_score: silhouette_score(features, assignments),
        davies_bouldin_index: davies_bouldin_index(features, assignments),
        num_clusters: k,
        cluster_counts,
        pca_scatter_data: pca_data,
    })
}

#[cfg(test)]
mod classification_tests {
   use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn binary_hand_computed() {
        // actual cat: 2 right 1 wrong | actual dog: 1 wrong 4 right
        let t = v(&["cat", "cat", "cat", "dog", "dog", "dog", "dog", "dog"]);
        let p = v(&["cat", "cat", "dog", "dog", "dog", "dog", "cat", "dog"]);
        let m = compute_classification_metrics(&t, &p);
        assert_eq!(m.class_labels, v(&["cat", "dog"]));
        assert_eq!(m.confusion_matrix, vec![vec![2, 1], vec![1, 4]]);
        close(m.accuracy, 0.75);
        close(m.precision, (2.0 / 3.0 + 0.8) / 2.0);
        close(m.recall, (2.0 / 3.0 + 0.8) / 2.0);
        close(m.f1_score, (2.0 / 3.0 + 0.8) / 2.0);
    }

    #[test]
    fn multiclass_with_never_predicted_class_has_no_nan() {
        let t = v(&["a", "b", "c"]);
        let p = v(&["a", "b", "b"]);
        let m = compute_classification_metrics(&t, &p);
        assert_eq!(m.confusion_matrix, vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 1, 0]]);
        close(m.accuracy, 2.0 / 3.0);
        close(m.precision, (1.0 + 0.5 + 0.0) / 3.0); // class c never predicted -> 0, not NaN
        close(m.recall, (1.0 + 1.0 + 0.0) / 3.0);
        close(m.f1_score, (1.0 + 2.0 / 3.0 + 0.0) / 3.0);
        assert!(m.precision.is_finite() && m.recall.is_finite() && m.f1_score.is_finite());
    }

    #[test]
    fn label_only_in_predictions_still_gets_a_row_and_column() {
        let m = compute_classification_metrics(&v(&["a", "a"]), &v(&["a", "z"]));
        assert_eq!(m.class_labels, v(&["a", "z"]));
        assert_eq!(m.confusion_matrix, vec![vec![1, 1], vec![0, 0]]);
        close(m.accuracy, 0.5);
    }

    #[test]
    fn perfect_predictions() {
        let t = v(&["x", "y", "z", "y", "x"]);
        let m = compute_classification_metrics(&t, &t);
        close(m.accuracy, 1.0);
        close(m.precision, 1.0);
        close(m.recall, 1.0);
        close(m.f1_score, 1.0);
    }

    #[test]
    fn numeric_labels_sort_numerically() {
        let m = compute_classification_metrics(&v(&["10", "2", "10"]), &v(&["10", "2", "2"]));
        assert_eq!(m.class_labels, v(&["2", "10"]));
    }

    #[test]
    fn empty_input_is_safe() {
        let m = compute_classification_metrics(&[], &[]);
        close(m.accuracy, 0.0);
        assert!(m.confusion_matrix.is_empty() && m.class_labels.is_empty());
        assert!(m.f1_score.is_finite());
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-3, "{a} != {b}");
    }

    #[test]
    fn test_regression_known_values() {
        let actual = vec![3.0, -0.5, 2.0, 7.0];
        let predicted = vec![2.5, 0.0, 2.0, 8.0];
        let m = compute_regression_metrics(&actual, &predicted);

        // MAE: (|0.5| + |-0.5| + |0.0| + |-1.0|) / 4 = 0.5
        close(m.mean_absolute_error, 0.5);
        // MSE: (0.25 + 0.25 + 0.0 + 1.0) / 4 = 0.375
        close(m.mean_squared_error, 0.375);
        // RMSE: sqrt(0.375) ≈ 0.6124
        close(m.root_mean_squared_error, 0.6124);
        // R2: 1 - (1.5 / 29.1875) ≈ 0.9486
        close(m.r2_score, 0.9486);
    }

    #[test]
    fn test_regression_perfect_predictions() {
        let actual = vec![10.0, 20.0, 30.0, 40.0];
        let predicted = vec![10.0, 20.0, 30.0, 40.0];
        let m = compute_regression_metrics(&actual, &predicted);

        close(m.mean_absolute_error, 0.0);
        close(m.mean_squared_error, 0.0);
        close(m.root_mean_squared_error, 0.0);
        close(m.r2_score, 1.0);
    }

    #[test]
    fn test_regression_empty() {
        let m = compute_regression_metrics(&[], &[]);
        assert_eq!(m.mean_absolute_error, 0.0);
        assert_eq!(m.mean_squared_error, 0.0);
        assert_eq!(m.root_mean_squared_error, 0.0);
        assert_eq!(m.r2_score, 0.0);
    }

    #[test]
    fn test_regression_constant_target() {
        let actual = vec![5.0, 5.0, 5.0];
        let predicted = vec![5.0, 5.0, 5.0];
        let m = compute_regression_metrics(&actual, &predicted);
        close(m.r2_score, 1.0);
    }
}

#[cfg(test)]
mod clustering_metrics_tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn computes_silhouette_and_davies_bouldin_for_separated_clusters() {
        let features = vec![vec![0.0], vec![2.0], vec![10.0], vec![12.0]];
        let assignments = vec![0, 0, 1, 1];
        let metrics = compute_clustering_metrics(&features, &assignments, 2, vec![]);

        close(metrics.silhouette_score, (9.0 / 11.0 + 7.0 / 9.0) / 2.0);
        close(metrics.davies_bouldin_index, 0.2);
    }

    #[test]
    fn includes_empty_cluster_counts_and_preserves_scatter_points() {
        let features = vec![vec![0.0], vec![1.0], vec![10.0]];
        let assignments = vec![0, 0, 2];
        let scatter = vec![PcaScatterPoint {
            sample_id: 1,
            x: 2.5,
            y: -3.0,
            cluster: 0,
        }];

        let metrics = compute_clustering_metrics(&features, &assignments, 3, scatter);

        assert_eq!(metrics.num_clusters, 3);
        assert_eq!(metrics.cluster_counts.get(&0), Some(&2));
        assert_eq!(metrics.cluster_counts.get(&1), Some(&0));
        assert_eq!(metrics.cluster_counts.get(&2), Some(&1));
        assert_eq!(metrics.pca_scatter_data.len(), 1);
        assert_eq!(metrics.pca_scatter_data[0].sample_id, 1);
        close(metrics.pca_scatter_data[0].x, 2.5);
        close(metrics.pca_scatter_data[0].y, -3.0);
    }

    #[test]
    fn undefined_scores_are_zero_for_one_observed_cluster() {
        let metrics =
            compute_clustering_metrics(&[vec![0.0], vec![1.0]], &[0, 0], 2, vec![]);

        assert_eq!(metrics.silhouette_score, 0.0);
        assert_eq!(metrics.davies_bouldin_index, 0.0);
        assert_eq!(metrics.cluster_counts.get(&1), Some(&0));
    }

    #[test]
    fn checked_metrics_reject_inconsistent_inputs() {
        assert!(matches!(
            compute_clustering_metrics_checked(&[], &[], 1, vec![]),
            Err(MlError::InvalidInput(_))
        ));
        assert!(matches!(
            compute_clustering_metrics_checked(&[vec![0.0]], &[], 1, vec![]),
            Err(MlError::InvalidInput(_))
        ));
        assert!(matches!(
            compute_clustering_metrics_checked(&[vec![0.0]], &[1], 1, vec![]),
            Err(MlError::InvalidInput(_))
        ));
        assert!(matches!(
            compute_clustering_metrics_checked(&[vec![f64::NAN]], &[0], 1, vec![]),
            Err(MlError::InvalidInput(_))
        ));
    }
}
use crate::models::{ClassificationMetrics, ClusteringMetrics, PcaScatterPoint, RegressionMetrics};
use std::collections::HashMap;

/// Computes Classification Metrics: Accuracy, Precision, Recall, F1, and Confusion Matrix
///
/// TODO: Assigned to Shreya Patil (PES2UG24CS484)
/// Requirements (FR-9):
/// - Accuracy = Correct / Total
/// - Macro-averaged Precision, Recall, and F1-Score
/// - Dynamic K x K Confusion Matrix with class labels
pub fn compute_classification_metrics(
    actual: &[String],
    predicted: &[String],
) -> ClassificationMetrics {
    // SKELETON PLACEHOLDER: Shreya to implement classification metric calculations
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
        precision: 0.0,
        recall: 0.0,
        f1_score: 0.0,
        confusion_matrix: vec![],
        class_labels: vec![],
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
/// TODO: Assigned to Shreya Patil (PES2UG24CS484)
/// Requirements (FR-10):
/// - Mean Absolute Error (MAE)
/// - Mean Squared Error (MSE)
/// - Root Mean Squared Error (RMSE)
/// - R2 Score (Coefficient of Determination)
pub fn compute_regression_metrics(
    _actual: &[f64],
    _predicted: &[f64],
) -> RegressionMetrics {
    // SKELETON PLACEHOLDER: Shreya to implement regression metric calculations
    RegressionMetrics {
        mean_absolute_error: 0.0,
        mean_squared_error: 0.0,
        root_mean_squared_error: 0.0,
        r2_score: 0.0,
    }
}

/// Computes Clustering Metrics: Silhouette Score, Davies-Bouldin Index, and Cluster Distribution
///
/// TODO: Assigned to Shreya Patil (PES2UG24CS484)
/// Requirements (FR-11):
/// - Silhouette Score (intra-cluster distance a vs nearest-cluster distance b)
/// - Davies-Bouldin Index
/// - Cluster distribution counts
pub fn compute_clustering_metrics(
    _features: &[Vec<f64>],
    assignments: &[usize],
    k: usize,
    pca_data: Vec<PcaScatterPoint>,
) -> ClusteringMetrics {
    // SKELETON PLACEHOLDER: Shreya to implement clustering metric calculations
    let mut cluster_counts: HashMap<usize, usize> = HashMap::new();
    for &c in assignments {
        *cluster_counts.entry(c).or_insert(0) += 1;
    }

    ClusteringMetrics {
        silhouette_score: 0.0,
        davies_bouldin_index: 0.0,
        num_clusters: k,
        cluster_counts,
        pca_scatter_data: pca_data,
    }
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
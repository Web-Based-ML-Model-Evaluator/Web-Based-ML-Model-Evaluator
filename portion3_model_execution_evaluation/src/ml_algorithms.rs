use crate::models::PcaScatterPoint;
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters};
use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier, DecisionTreeClassifierParameters,
};
use smartcore::tree::decision_tree_regressor::{
    DecisionTreeRegressor, DecisionTreeRegressorParameters,
};
use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Principal Component Analysis (PCA) 2D projection for scatter plot rendering
///
/// TODO: Assigned to Shreya Patil (PES2UG24CS484)
/// Requirements (FR-11):
/// - Center feature matrix around column means
/// - Compute 2D principal component projections (PC1, PC2)
/// - Return list of (x, y) coordinates with cluster assignments
pub fn project_pca_2d(_features: &[Vec<f64>], assignments: &[usize]) -> Vec<PcaScatterPoint> {
    // SKELETON PLACEHOLDER: Shreya to implement 2D PCA projection algorithm
    let mut points = Vec::new();
    for (i, &c) in assignments.iter().enumerate() {
        points.push(PcaScatterPoint {
            sample_id: i,
            x: 0.0,
            y: 0.0,
            cluster: c,
        });
    }
    points
}

/// K-Means Clustering Algorithm
///
/// TODO: Assigned to Shreya Patil (PES2UG24CS484)
/// Requirements (FR-8, FR-11):
/// - Initialize k centroids
/// - Assign points to nearest centroid
/// - Recalculate centroids until convergence

/// Runs deterministic K-means and returns one cluster index per input row.
pub fn run_kmeans(features: &[Vec<f64>], k: usize, max_iter: usize) -> Vec<usize> {
    try_run_kmeans(features, k, max_iter).unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible K-means implementation for callers that can handle invalid input.
pub fn try_run_kmeans(
    features: &[Vec<f64>],
    k: usize,
    max_iter: usize,
) -> Result<Vec<usize>, MlError> {
    if features.is_empty() {
        return Err(MlError::InvalidInput(
            "Clustering requires at least one data point.".into(),
        ));
    }
    if k == 0 || k > features.len() {
        return Err(MlError::InvalidInput(format!(
            "The number of clusters must be between 1 and the number of data points ({}).",
            features.len()
        )));
    }
    let dimensions = features[0].len();
    if dimensions == 0 {
        return Err(MlError::InvalidInput(
            "Select at least one feature column for clustering.".into(),
        ));
    }
    if features.iter().any(|row| row.len() != dimensions) {
        return Err(MlError::InvalidInput(
            "All rows must have the same number of features.".into(),
        ));
    }
    if features.iter().flatten().any(|value| !value.is_finite()) {
        return Err(MlError::InvalidInput(
            "Features contain NaN or infinite values.".into(),
        ));
    }

    let squared_distance = |left: &[f64], right: &[f64]| {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
    };

    // K-means++ seeding with a fixed PRNG seed keeps runs repeatable.
    let mut random_state = 42_u64;
    let mut next_random = || {
        random_state ^= random_state >> 12;
        random_state ^= random_state << 25;
        random_state ^= random_state >> 27;
        random_state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let first = (next_random() as usize) % features.len();
    let mut selected = vec![first];
    let mut centroids = vec![features[first].clone()];

    while centroids.len() < k {
        let distances: Vec<f64> = features
            .iter()
            .map(|point| {
                centroids
                    .iter()
                    .map(|center| squared_distance(point, center))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let total_distance = distances.iter().sum::<f64>();
        let next_index = if total_distance.is_finite() && total_distance > 0.0 {
            let threshold =
                (next_random() as f64 / u64::MAX as f64) * total_distance;
            let mut cumulative = 0.0;
            distances
                .iter()
                .position(|distance| {
                    cumulative += distance;
                    cumulative >= threshold
                })
                .filter(|index| !selected.contains(index))
                .or_else(|| (0..features.len()).find(|index| !selected.contains(index)))
        } else {
            (0..features.len()).find(|index| !selected.contains(index))
        }
        .ok_or_else(|| {
            MlError::InvalidInput("Could not initialize distinct cluster centers.".into())
        })?;

        selected.push(next_index);
        centroids.push(features[next_index].clone());
    }

    let mut assignments = vec![0; features.len()];
    let mut previous_assignments: Option<Vec<usize>> = None;
    for _ in 0..max_iter.max(1) {
        for (index, point) in features.iter().enumerate() {
            assignments[index] = centroids
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    squared_distance(point, left).total_cmp(&squared_distance(point, right))
                })
                .map(|(cluster, _)| cluster)
                .expect("K-means initialized at least one centroid");
        }

        if previous_assignments.as_ref() == Some(&assignments) {
            break;
        }
        previous_assignments = Some(assignments.clone());

        let mut sums = vec![vec![0.0; dimensions]; k];
        let mut counts = vec![0usize; k];
        for (point, &cluster) in features.iter().zip(&assignments) {
            counts[cluster] += 1;
            for (sum, value) in sums[cluster].iter_mut().zip(point) {
                *sum += value;
            }
        }
        for cluster in 0..k {
            if counts[cluster] > 0 {
                for value in &mut sums[cluster] {
                    *value /= counts[cluster] as f64;
                }
                centroids[cluster] = std::mem::take(&mut sums[cluster]);
            }
        }
    }

    // Reassign against the final centroids, including when max_iter was reached.
    for (index, point) in features.iter().enumerate() {
        assignments[index] = centroids
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                squared_distance(point, left).total_cmp(&squared_distance(point, right))
            })
            .map(|(cluster, _)| cluster)
            .expect("K-means initialized at least one centroid");
    }
    Ok(assignments)
}

/// Errors from supervised model execution. `message()` is safe to show to the
/// client (no library internals); `Display` carries the raw detail for logs.
#[derive(Debug, Clone, PartialEq)]
pub enum MlError {
    /// Algorithm name is unknown or does not belong to the chosen task type.
    UnsupportedAlgorithm(String),
    /// The data cannot be used (empty split, single class, ragged rows...).
    InvalidInput(String),
    /// The ML library failed or panicked. Detail is internal only.
    TrainingFailed(String),
}

impl MlError {
    pub fn message(&self) -> String {
        match self {
            MlError::UnsupportedAlgorithm(m) | MlError::InvalidInput(m) => m.clone(),
            MlError::TrainingFailed(_) => {
                "The model could not be trained on this data. Check the selected columns and dataset."
                    .to_string()
            }
        }
    }
}

impl std::fmt::Display for MlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MlError::UnsupportedAlgorithm(m) | MlError::InvalidInput(m) => write!(f, "{m}"),
            MlError::TrainingFailed(d) => write!(f, "training failed: {d}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ClassifierAlgo {
    LogisticRegression,
    DecisionTree,
    RandomForest,
    Knn,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RegressorAlgo {
    LinearRegression,
    DecisionTree,
    RandomForest,
}

/// "Random Forest Classifier" / "random-forest" / "K-Nearest Neighbors (KNN)" -> lowercase
/// words joined by '_', so the UI display names and short names both match.
fn normalize(s: &str) -> String {
    let mut out = String::new();
    for c in s.trim().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

fn parse_classifier(name: &str) -> Result<ClassifierAlgo, MlError> {
    let n = normalize(name);
    if n.contains("logistic") {
        Ok(ClassifierAlgo::LogisticRegression)
    } else if n.contains("decision_tree") {
        Ok(ClassifierAlgo::DecisionTree)
    } else if n.contains("random_forest") {
        Ok(ClassifierAlgo::RandomForest)
    } else if n == "knn" || n.contains("nearest") || n.contains("knn") {
        Ok(ClassifierAlgo::Knn)
    } else {
        Err(MlError::UnsupportedAlgorithm(format!(
            "Algorithm '{name}' is not available for classification. Choose one of: \
             Logistic Regression, Decision Tree Classifier, Random Forest Classifier, K-Nearest Neighbors (KNN)."
        )))
    }
}

fn parse_regressor(name: &str) -> Result<RegressorAlgo, MlError> {
    let n = normalize(name);
    if n.contains("linear") {
        Ok(RegressorAlgo::LinearRegression)
    } else if n.contains("decision_tree") {
        Ok(RegressorAlgo::DecisionTree)
    } else if n.contains("random_forest") {
        Ok(RegressorAlgo::RandomForest)
    } else {
        Err(MlError::UnsupportedAlgorithm(format!(
            "Algorithm '{name}' is not available for regression. Choose one of: \
             Linear Regression, Decision Tree Regressor, Random Forest Regressor."
        )))
    }
}

/// Seed for the train/test shuffle. Fixed so the same request gives the same split.
pub const SPLIT_SEED: u64 = 42;

/// Deterministic shuffle, then split into (train, test) with `train_n` train rows.
/// Call it with the same `train_n` on X and y: the permutation depends only on the
/// length, so rows stay aligned. (The skeleton's sequential split put whole classes
/// in the test set whenever the CSV was sorted by class, e.g. iris.)
pub fn split_shuffled<T: Clone>(items: &[T], train_n: usize) -> (Vec<T>, Vec<T>) {
    let n = items.len();
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = SPLIT_SEED.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    let train_n = train_n.min(n);
    // The first `test_n` shuffled rows are test, the rest train.
    let test_n = n - train_n;
    let test = idx[..test_n].iter().map(|&i| items[i].clone()).collect();
    let train = idx[test_n..].iter().map(|&i| items[i].clone()).collect();
    (train, test)
}

/// z-score using TRAIN mean/std (never test statistics). Zero-variance columns become 0.
fn standardize(train: &[Vec<f64>], test: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = train[0].len();
    let n = train.len() as f64;
    let mut tr = train.to_vec();
    let mut te = test.to_vec();
    for j in 0..m {
        let mean = train.iter().map(|r| r[j]).sum::<f64>() / n;
        let var = train.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / n;
        let std = if var.sqrt() > 1e-12 { var.sqrt() } else { 1.0 };
        for r in tr.iter_mut().chain(te.iter_mut()) {
            r[j] = (r[j] - mean) / std;
        }
    }
    (tr, te)
}

/// Shared checks: non-empty train/test, consistent row widths, finite numbers.
fn validate_matrices(x_train: &[Vec<f64>], n_labels: usize, x_test: &[Vec<f64>]) -> Result<(), MlError> {
    if x_train.is_empty() {
        return Err(MlError::InvalidInput(
            "The training set is empty. Use a smaller test split or a larger dataset.".into(),
        ));
    }
    if x_train.len() != n_labels {
        return Err(MlError::InvalidInput("Features and target have different lengths.".into()));
    }
    let width = x_train[0].len();
    if width == 0 {
        return Err(MlError::InvalidInput("Select at least one feature column.".into()));
    }
    if x_train.iter().chain(x_test.iter()).any(|r| r.len() != width) {
        return Err(MlError::InvalidInput("All rows must have the same number of features.".into()));
    }
    if x_train.iter().chain(x_test.iter()).flatten().any(|v| !v.is_finite()) {
        return Err(MlError::InvalidInput("Features contain NaN or infinite values.".into()));
    }
    Ok(())
}

/// SADS 3.7: a panic inside the ML library must become a structured error, not a crashed server.
fn guarded<T>(f: impl FnOnce() -> Result<T, MlError>) -> Result<T, MlError> {
    catch_unwind(AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err(MlError::TrainingFailed("model library panicked".into())))
}

fn internal<E: std::fmt::Debug>(e: E) -> MlError {
    MlError::TrainingFailed(format!("{e:?}"))
}

/// Classification Algorithm Dispatcher (FR-8, Issue #6)
///
/// Fits the chosen classifier on the training rows and returns one predicted label per
/// test row. Supported: Logistic Regression, Decision Tree Classifier, Random Forest
/// Classifier, K-Nearest Neighbors (KNN). Features are standardised with train statistics.
#[allow(dead_code)]
pub fn run_classification_model(
    algorithm: &str,
    x_train: &[Vec<f64>],
    y_train: &[String],
    x_test: &[Vec<f64>],
) -> Vec<String> {
    try_run_classification_model(algorithm, x_train, y_train, x_test)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible classification dispatcher for callers that can handle invalid input and
/// training failures directly.
pub fn try_run_classification_model(
    algorithm: &str,
    x_train: &[Vec<f64>],
    y_train: &[String],
    x_test: &[Vec<f64>],
) -> Result<Vec<String>, MlError> {
    let algo = parse_classifier(algorithm)?;
    validate_matrices(x_train, y_train.len(), x_test)?;

    // class labels <-> indices (sorted so results are stable)
    let mut classes: Vec<String> = y_train.iter().cloned().collect::<HashSet<_>>().into_iter().collect();
    classes.sort();
    if classes.len() < 2 {
        return Err(MlError::InvalidInput(
            "The training data contains only one class. Classification needs at least two; use a smaller test split.".into(),
        ));
    }
    if x_test.is_empty() {
        return Ok(vec![]);
    }
    let y_idx: Vec<usize> = y_train
        .iter()
        .map(|l| classes.iter().position(|c| c == l).expect("label is in classes"))
        .collect();

    let (tr, te) = standardize(x_train, x_test);
    let pred: Vec<usize> = guarded(|| {
        let xtr = DenseMatrix::from_2d_vec(&tr).map_err(internal)?;
        let xte = DenseMatrix::from_2d_vec(&te).map_err(internal)?;
        let p = match algo {
            ClassifierAlgo::LogisticRegression => {
                LogisticRegression::fit(&xtr, &y_idx, LogisticRegressionParameters::default())
                    .map_err(internal)?
                    .predict(&xte)
                    .map_err(internal)?
            }
            ClassifierAlgo::DecisionTree => {
                DecisionTreeClassifier::fit(&xtr, &y_idx, DecisionTreeClassifierParameters::default())
                    .map_err(internal)?
                    .predict(&xte)
                    .map_err(internal)?
            }
            ClassifierAlgo::RandomForest => {
                // default seed is fixed, so results are reproducible
                RandomForestClassifier::fit(&xtr, &y_idx, RandomForestClassifierParameters::default())
                    .map_err(internal)?
                    .predict(&xte)
                    .map_err(internal)?
            }
            ClassifierAlgo::Knn => {
                // k = 3 (or fewer when the training set is tiny), Euclidean distance
                let params = KNNClassifierParameters::default().with_k(3.min(tr.len()));
                KNNClassifier::fit(&xtr, &y_idx, params)
                    .map_err(internal)?
                    .predict(&xte)
                    .map_err(internal)?
            }
        };
        Ok(p)
    })?;

    Ok(pred.into_iter().map(|i| classes[i].clone()).collect())
}

/// Regression Algorithm Dispatcher (FR-8, Issue #6)
///
/// Fits the chosen regressor on the training rows and returns one prediction per test
/// row. Supported: Linear Regression, Decision Tree Regressor, Random Forest Regressor.
#[allow(dead_code)]
pub fn run_regression_model(
    algorithm: &str,
    x_train: &[Vec<f64>],
    y_train: &[f64],
    x_test: &[Vec<f64>],
) -> Vec<f64> {
    try_run_regression_model(algorithm, x_train, y_train, x_test)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible regression dispatcher for callers that can handle invalid input and
/// training failures directly.
pub fn try_run_regression_model(
    algorithm: &str,
    x_train: &[Vec<f64>],
    y_train: &[f64],
    x_test: &[Vec<f64>],
) -> Result<Vec<f64>, MlError> {
    let algo = parse_regressor(algorithm)?;
    validate_matrices(x_train, y_train.len(), x_test)?;
    if y_train.iter().any(|v| !v.is_finite()) {
        return Err(MlError::InvalidInput("The target column contains NaN or infinite values.".into()));
    }
    if x_test.is_empty() {
        return Ok(vec![]);
    }

    let (tr, te) = standardize(x_train, x_test);
    let y = y_train.to_vec();
    guarded(|| {
        let xtr = DenseMatrix::from_2d_vec(&tr).map_err(internal)?;
        let xte = DenseMatrix::from_2d_vec(&te).map_err(internal)?;
        let p = match algo {
            RegressorAlgo::LinearRegression => {
                LinearRegression::fit(&xtr, &y, LinearRegressionParameters::default())
                    .map_err(internal)?
                    .predict(&xte)
                    .map_err(internal)?
            }
            RegressorAlgo::DecisionTree => {
                DecisionTreeRegressor::fit(&xtr, &y, DecisionTreeRegressorParameters::default())
                    .map_err(internal)?
                    .predict(&xte)
                    .map_err(internal)?
            }
            RegressorAlgo::RandomForest => {
                // consider all features at each split (scikit-learn's regression default)
                let params = RandomForestRegressorParameters::default().with_m(tr[0].len());
                RandomForestRegressor::fit(&xtr, &y, params)
                    .map_err(internal)?
                    .predict(&xte)
                    .map_err(internal)?
            }
        };
        Ok(p)
    })
}

#[cfg(test)]
mod supervised_tests {
    use super::*;
    use crate::metrics::compute_classification_metrics;

    /// The shared team fixture: 20 iris rows, SORTED BY CLASS (the hard case for splitting).
    fn iris() -> (Vec<Vec<f64>>, Vec<String>) {
        let raw = include_str!("../../contracts/fixtures/mock_cleaned_dataset.json");
        let v: serde_json::Value = serde_json::from_str(raw).unwrap();
        let cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"];
        let mut x = vec![];
        let mut y = vec![];
        for r in v["records"].as_array().unwrap() {
            x.push(cols.iter().map(|c| r[*c].as_f64().unwrap()).collect());
            y.push(r["species"].as_str().unwrap().to_string());
        }
        (x, y)
    }

    fn run_clf(algo: &str, test_split: f64) -> Result<(Vec<String>, Vec<String>), MlError> {
        let (x, y) = iris();
        let n_test = ((x.len() as f64 * test_split).round() as usize).max(1);
        let n_train = x.len() - n_test;
        let (xtr, xte) = split_shuffled(&x, n_train);
        let (ytr, yte) = split_shuffled(&y, n_train);
        try_run_classification_model(algo, &xtr, &ytr, &xte).map(|p| (yte, p))
    }

    #[test]
    fn all_four_classifiers_run_and_are_accurate_on_sorted_iris() {
        for algo in [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "K-Nearest Neighbors (KNN)",
        ] {
            let (actual, pred) = run_clf(algo, 0.25).unwrap_or_else(|e| panic!("{algo}: {e}"));
            assert_eq!(pred.len(), actual.len(), "{algo}: one prediction per test row");
            let m = compute_classification_metrics(&actual, &pred);
            // Only 5 test rows and 3 virginica rows in training, so this is a sanity floor
            // (chance is 0.33), not a quality bar. Exactness is checked by the oracle test below.
            assert!(m.accuracy >= 0.6, "{algo}: accuracy {} too low", m.accuracy);
            assert!(m.precision.is_finite() && m.recall.is_finite() && m.f1_score.is_finite());
        }
    }

    #[cfg(test)]
    mod kmeans_tests {
        use super::*;

        #[test]
        fn groups_nearby_points_not_adjacent_rows() {
            let features = vec![
                vec![0.0, 0.0],
                vec![100.0, 100.0],
                vec![1.0, 1.0],
                vec![101.0, 99.0],
                vec![-1.0, 1.0],
                vec![99.0, 101.0],
            ];
            let assignments = try_run_kmeans(&features, 2, 100).unwrap();

            assert_eq!(assignments.len(), features.len());
            assert_eq!(assignments[0], assignments[2]);
            assert_eq!(assignments[0], assignments[4]);
            assert_eq!(assignments[1], assignments[3]);
            assert_eq!(assignments[1], assignments[5]);
            assert_ne!(assignments[0], assignments[1]);
        }

        #[test]
        fn results_are_deterministic_and_single_cluster_labels_all_rows_zero() {
            let features = vec![vec![0.0], vec![2.0], vec![10.0], vec![12.0]];
            let first = try_run_kmeans(&features, 2, 50).unwrap();
            let second = try_run_kmeans(&features, 2, 50).unwrap();
            assert_eq!(first, second);
            assert_eq!(run_kmeans(&features, 1, 50), vec![0; features.len()]);
        }

        #[test]
        fn identical_points_are_supported() {
            let features = vec![vec![3.0, 3.0]; 4];
            let assignments = try_run_kmeans(&features, 3, 20).unwrap();
            assert_eq!(assignments.len(), features.len());
            assert!(assignments.iter().all(|&cluster| cluster < 3));
        }

        #[test]
        fn rejects_empty_invalid_or_malformed_data() {
            assert!(matches!(
                try_run_kmeans(&[], 1, 10),
                Err(MlError::InvalidInput(_))
            ));
            let features = vec![vec![0.0], vec![1.0]];
            assert!(matches!(
                try_run_kmeans(&features, 0, 10),
                Err(MlError::InvalidInput(_))
            ));
            assert!(matches!(
                try_run_kmeans(&features, 3, 10),
                Err(MlError::InvalidInput(_))
            ));
            assert!(matches!(
                try_run_kmeans(&[vec![0.0], vec![1.0, 2.0]], 1, 10),
                Err(MlError::InvalidInput(_))
            ));
            assert!(matches!(
                try_run_kmeans(&[vec![f64::NAN], vec![1.0]], 1, 10),
                Err(MlError::InvalidInput(_))
            ));
        }
    }

    /// Oracle check: an independent 3-nearest-neighbour implementation (Euclidean distance on
    /// train-standardised features, majority vote, ties broken alphabetically) must agree with
    /// SmartCore's KNN on every test row.
    #[test]
    fn knn_matches_a_hand_written_3nn() {
        let (x, y) = iris();
        let (xtr, xte) = split_shuffled(&x, 15);
        let (ytr, _) = split_shuffled(&y, 15);
        let ours = try_run_classification_model("KNN", &xtr, &ytr, &xte).unwrap();

        let (tr, te) = standardize(&xtr, &xte);
        let oracle: Vec<String> = te
            .iter()
            .map(|q| {
                let mut d: Vec<(f64, &String)> = tr
                    .iter()
                    .zip(&ytr)
                    .map(|(r, l)| (r.iter().zip(q).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt(), l))
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut votes: std::collections::BTreeMap<&String, usize> = Default::default();
                for (_, l) in d.iter().take(3) {
                    *votes.entry(l).or_default() += 1;
                }
                let best = votes.values().copied().max().unwrap();
                votes.into_iter().find(|(_, c)| *c == best).unwrap().0.clone()
            })
            .collect();
        assert_eq!(ours, oracle);
    }

    #[test]
    fn predictions_only_use_labels_seen_in_training() {
        let (_, pred) = run_clf("Random Forest Classifier", 0.25).unwrap();
        assert!(pred.iter().all(|p| ["setosa", "versicolor", "virginica"].contains(&p.as_str())));
    }

    #[test]
    fn classifier_results_are_deterministic() {
        for algo in ["Random Forest Classifier", "Decision Tree Classifier", "K-Nearest Neighbors (KNN)"] {
            assert_eq!(run_clf(algo, 0.25).unwrap(), run_clf(algo, 0.25).unwrap(), "{algo}");
        }
    }

    #[test]
    fn algorithm_names_are_forgiving() {
        for name in ["knn", "KNN", "k-nearest-neighbors", "K-Nearest Neighbors (KNN)", "  K Nearest Neighbours "] {
            assert_eq!(parse_classifier(name), Ok(ClassifierAlgo::Knn), "{name}");
        }
        assert_eq!(parse_classifier("random_forest"), Ok(ClassifierAlgo::RandomForest));
        assert_eq!(parse_classifier("Decision Tree"), Ok(ClassifierAlgo::DecisionTree));
        assert_eq!(parse_classifier("logistic regression"), Ok(ClassifierAlgo::LogisticRegression));
        assert_eq!(parse_regressor("Linear Regression"), Ok(RegressorAlgo::LinearRegression));
        assert_eq!(parse_regressor("Decision Tree Regressor"), Ok(RegressorAlgo::DecisionTree));
        assert_eq!(parse_regressor("random forest regressor"), Ok(RegressorAlgo::RandomForest));
    }

    #[test]
    fn unknown_or_wrong_task_algorithm_is_rejected() {
        assert!(matches!(run_clf("Neural Network", 0.25), Err(MlError::UnsupportedAlgorithm(_))));
        // a regression-only algorithm for classification, and vice versa
        assert!(matches!(run_clf("Linear Regression", 0.25), Err(MlError::UnsupportedAlgorithm(_))));
        let r = try_run_regression_model("Logistic Regression", &[vec![1.0]], &[1.0], &[vec![1.0]]);
        assert!(matches!(r, Err(MlError::UnsupportedAlgorithm(_))));
        let r = try_run_regression_model("K-Nearest Neighbors (KNN)", &[vec![1.0]], &[1.0], &[vec![1.0]]);
        assert!(matches!(r, Err(MlError::UnsupportedAlgorithm(_))));
    }

    #[test]
    fn bad_inputs_return_errors_instead_of_panicking() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y: Vec<String> = ["a", "b", "a"].iter().map(|s| s.to_string()).collect();
        // empty training set
        assert!(matches!(try_run_classification_model("KNN", &[], &[], &x), Err(MlError::InvalidInput(_))));
        // length mismatch
        assert!(matches!(try_run_classification_model("KNN", &x, &y[..2], &x), Err(MlError::InvalidInput(_))));
        // single class in training data
        let one: Vec<String> = vec!["a".into(); 3];
        assert!(matches!(try_run_classification_model("Logistic Regression", &x, &one, &x), Err(MlError::InvalidInput(_))));
        // ragged rows
        let ragged = vec![vec![1.0], vec![2.0, 3.0], vec![4.0]];
        assert!(matches!(try_run_classification_model("KNN", &ragged, &y, &x), Err(MlError::InvalidInput(_))));
        // NaN feature
        let nan = vec![vec![f64::NAN], vec![2.0], vec![3.0]];
        assert!(matches!(try_run_classification_model("KNN", &nan, &y, &x), Err(MlError::InvalidInput(_))));
        // no feature columns
        let none = vec![vec![], vec![], vec![]];
        assert!(matches!(try_run_classification_model("KNN", &none, &y, &none), Err(MlError::InvalidInput(_))));
    }

    #[test]
    fn tiny_train_set_and_zero_variance_column_do_not_crash() {
        let x = vec![vec![1.0, 5.0], vec![2.0, 5.0], vec![9.0, 5.0], vec![10.0, 5.0]];
        let y: Vec<String> = ["a", "a", "b", "b"].iter().map(|s| s.to_string()).collect();
        for algo in ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"] {
            let p = try_run_classification_model(algo, &x, &y, &[vec![1.5, 5.0], vec![9.5, 5.0]])
                .unwrap_or_else(|e| panic!("{algo}: {e}"));
            assert_eq!(p.len(), 2, "{algo}");
        }
    }

    #[test]
    fn empty_test_set_returns_empty_predictions() {
        let (x, y) = iris();
        assert_eq!(try_run_classification_model("KNN", &x, &y, &[]).unwrap(), Vec::<String>::new());
    }

    // ---- regression -------------------------------------------------------

    /// y = 3*a + 2*b + 10 with a small deterministic wobble. Standardised features make
    /// linear regression near-perfect; trees and forests should still track it well.
    fn linear_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut x = vec![];
        let mut y = vec![];
        for i in 0..60 {
            let a = (i % 12) as f64 + 1.0;
            let b = ((i * 7) % 10) as f64;
            x.push(vec![a, b]);
            y.push(3.0 * a + 2.0 * b + 10.0 + if i % 2 == 0 { 0.1 } else { -0.1 });
        }
        (x, y)
    }

    fn r2(actual: &[f64], pred: &[f64]) -> f64 {
        let mean = actual.iter().sum::<f64>() / actual.len() as f64;
        let sst: f64 = actual.iter().map(|y| (y - mean).powi(2)).sum();
        let sse: f64 = actual.iter().zip(pred).map(|(a, p)| (a - p).powi(2)).sum();
        1.0 - sse / sst
    }

    #[test]
    fn all_three_regressors_fit_linear_data() {
        let (x, y) = linear_data();
        let (xtr, xte) = split_shuffled(&x, 48);
        let (ytr, yte) = split_shuffled(&y, 48);
        for (algo, min_r2) in [
            ("Linear Regression", 0.99),
            ("Decision Tree Regressor", 0.85),
            ("Random Forest Regressor", 0.90),
        ] {
            let pred = try_run_regression_model(algo, &xtr, &ytr, &xte).unwrap_or_else(|e| panic!("{algo}: {e}"));
            assert_eq!(pred.len(), yte.len(), "{algo}");
            let score = r2(&yte, &pred);
            assert!(score >= min_r2, "{algo}: R2 {score} below {min_r2}");
        }
    }

    #[test]
    fn regression_rejects_bad_input() {
        let x = vec![vec![1.0], vec![2.0]];
        assert!(matches!(try_run_regression_model("Linear Regression", &[], &[], &x), Err(MlError::InvalidInput(_))));
        assert!(matches!(try_run_regression_model("Linear Regression", &x, &[1.0], &x), Err(MlError::InvalidInput(_))));
        assert!(matches!(
            try_run_regression_model("Linear Regression", &x, &[1.0, f64::NAN], &x),
            Err(MlError::InvalidInput(_))
        ));
    }

    // ---- split ------------------------------------------------------------

    #[test]
    fn split_is_deterministic_aligned_and_complete() {
        let idx: Vec<usize> = (0..20).collect();
        let doubled: Vec<usize> = idx.iter().map(|i| i * 2).collect();
        let (tr_a, te_a) = split_shuffled(&idx, 15);
        let (tr_b, te_b) = split_shuffled(&doubled, 15);
        assert_eq!((tr_a.len(), te_a.len()), (15, 5));
        // rows stay aligned between X and y
        assert!(tr_a.iter().zip(&tr_b).all(|(a, b)| a * 2 == *b));
        assert!(te_a.iter().zip(&te_b).all(|(a, b)| a * 2 == *b));
        // nothing lost
        let mut all: Vec<usize> = tr_a.iter().chain(&te_a).cloned().collect();
        all.sort();
        assert_eq!(all, idx);
        // same call, same result
        assert_eq!(split_shuffled(&idx, 15), (tr_a, te_a));
        // it really shuffles (a sorted-by-class file must not give a class-only test set)
        assert_ne!(split_shuffled(&idx, 15).1, vec![0, 1, 2, 3, 4]);
    }
}

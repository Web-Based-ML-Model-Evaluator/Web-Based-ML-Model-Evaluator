"""
Tests for Classification algorithms execution and evaluation metrics.
Covers:
  - Logistic Regression, Decision Tree, Random Forest, KNN, MLP, and CNN.
  - Metrics contract: accuracy, precision, recall, f1_score, confusion_matrix, class_labels.
  - Stratified train/test split preservation.
"""

import pytest

CLASSIFIERS = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
    "K-Nearest Neighbors (KNN)",
    "Neural Network (MLP)",
    "Convolutional Neural Network (CNN)"
]

@pytest.mark.parametrize("algorithm", CLASSIFIERS)
def test_all_classifiers_execute_successfully(client, iris_csv_bytes, algorithm):
    """Executes each supported classification algorithm on the Iris dataset and validates metrics."""
    up_res = client.upload_dataset("iris_clf.csv", iris_csv_bytes)
    assert up_res.status_code == 200
    dataset_id = up_res.json()["dataset_id"]

    exec_req = {
        "dataset_id": dataset_id,
        "task_type": "classification",
        "algorithm": algorithm,
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "test_split_ratio": 0.2,
        "stratified_split": True
    }

    res = client.execute_model(exec_req)
    assert res.status_code == 200, f"Algorithm {algorithm} failed with: {res.text}"
    data = res.json()

    assert data["task_type"] == "classification"
    assert data["algorithm"] == algorithm
    assert "metrics" in data

    metrics = data["metrics"]
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert "class_labels" in metrics

    cm = metrics["confusion_matrix"]
    assert isinstance(cm, list)
    assert len(cm) == len(metrics["class_labels"])
    assert len(cm[0]) == len(metrics["class_labels"])

def test_stratified_split_flag(client, iris_csv_bytes):
    """Verifies that stratified splitting executes and yields balanced multi-class metrics."""
    up_res = client.upload_dataset("iris_strat.csv", iris_csv_bytes)
    assert up_res.status_code == 200
    dataset_id = up_res.json()["dataset_id"]

    exec_req = {
        "dataset_id": dataset_id,
        "task_type": "classification",
        "algorithm": "Random Forest Classifier",
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "test_split_ratio": 0.2,
        "stratified_split": True
    }

    res = client.execute_model(exec_req)
    assert res.status_code == 200
    metrics = res.json()["metrics"]
    assert metrics["accuracy"] >= 0.85

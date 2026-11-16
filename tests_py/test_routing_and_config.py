"""
Tests for ML task routing, algorithm catalog, and configuration validation.
"""

def test_list_supported_tasks(client):
    """Verifies that the tasks catalog returns classification, regression, and clustering."""
    res = client.get("/api/v1/config/tasks")
    assert res.status_code == 200
    tasks = res.json()
    task_types = [t["task_type"] for t in tasks]
    assert "classification" in task_types
    assert "regression" in task_types
    assert "clustering" in task_types

def test_validate_valid_classification_config(client, iris_csv_bytes):
    """Verifies that a valid classification config passes validation."""
    up_res = client.upload_dataset("iris_val.csv", iris_csv_bytes)
    dataset_id = up_res.json()["dataset_id"]

    config = {
        "dataset_id": dataset_id,
        "task_type": "classification",
        "algorithm": "Decision Tree Classifier",
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "test_split_ratio": 0.2
    }
    res = client.post("/api/v1/config/validate", json=config)
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True

def test_validate_target_in_features_fails(client, iris_csv_bytes):
    """Verifies that having the target column inside feature columns fails validation."""
    up_res = client.upload_dataset("iris_invalid.csv", iris_csv_bytes)
    dataset_id = up_res.json()["dataset_id"]

    config = {
        "dataset_id": dataset_id,
        "task_type": "classification",
        "algorithm": "Random Forest Classifier",
        "target_column": "species",
        "feature_columns": ["sepal_length", "species"],
        "test_split_ratio": 0.2
    }
    res = client.post("/api/v1/config/validate", json=config)
    assert res.status_code == 422
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "CONFIG_VALIDATION_FAILED"

def test_validate_clustering_requires_k_ge_2(client, clustering_csv_bytes):
    """Verifies that clustering configuration rejects num_clusters < 2."""
    up_res = client.upload_dataset("cluster_val.csv", clustering_csv_bytes)
    dataset_id = up_res.json()["dataset_id"]

    config = {
        "dataset_id": dataset_id,
        "task_type": "clustering",
        "algorithm": "K-Means Clustering",
        "feature_columns": ["x1", "x2", "x3", "x4"],
        "hyperparameters": {
            "num_clusters": 1
        }
    }
    res = client.post("/api/v1/config/validate", json=config)
    assert res.status_code == 422
    data = res.json()
    assert data["error"]["code"] == "CONFIG_VALIDATION_FAILED"

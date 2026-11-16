"""
Tests for Model Comparison Dashboard, side-by-side comparison, and ranking (FR-12, BR-7).
"""

def test_mock_comparison_dashboard(client):
    """Verifies that the comparison dashboard seeds mock runs and returns metrics history."""
    res = client.get("/api/v1/models/comparison/mock")
    assert res.status_code == 200
    data = res.json()
    assert "total_runs" in data
    assert data["total_runs"] >= 2
    assert "runs" in data

def test_compare_models_requires_at_least_two(client):
    """Verifies that comparing fewer than 2 models is rejected (FR-12 / BR-7)."""
    res = client.post("/api/v1/models/compare", json={"execution_ids": ["exec-1"]})
    assert res.status_code == 400
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "INSUFFICIENT_MODELS_FOR_COMPARISON"

def test_compare_trained_models_side_by_side(client, iris_csv_bytes):
    """Trains two models on the same dataset and generates a side-by-side comparison report."""
    up_res = client.upload_dataset("iris_compare.csv", iris_csv_bytes)
    dataset_id = up_res.json()["dataset_id"]

    # Model 1: Decision Tree
    req1 = {
        "dataset_id": dataset_id,
        "task_type": "classification",
        "algorithm": "Decision Tree Classifier",
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "test_split_ratio": 0.2
    }
    m1_res = client.execute_model(req1)
    assert m1_res.status_code == 200
    exec1 = m1_res.json()["execution_id"]

    # Model 2: Random Forest
    req2 = {
        "dataset_id": dataset_id,
        "task_type": "classification",
        "algorithm": "Random Forest Classifier",
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "test_split_ratio": 0.2
    }
    m2_res = client.execute_model(req2)
    assert m2_res.status_code == 200
    exec2 = m2_res.json()["execution_id"]

    # Compare them
    cmp_res = client.compare_models({"execution_ids": [exec1, exec2]})
    assert cmp_res.status_code == 200, f"Comparison failed: {cmp_res.text}"
    report = cmp_res.json()

    assert "ranking" in report
    assert len(report["ranking"]) == 2
    assert "best_model_id" in report
    assert report["best_model_id"] in [exec1, exec2]
    assert "recommendation_reason" in report
    assert "metrics_table" in report

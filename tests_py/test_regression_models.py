"""
Tests for Regression algorithms execution and metrics evaluation.
Covers:
  - Linear Regression, Decision Tree Regressor, Random Forest Regressor, MLP, and CNN.
  - Metrics contract: MAE, MSE, RMSE, R-squared score.
"""

import pytest

REGRESSORS = [
    "Linear Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Neural Network (MLP)",
    "Convolutional Neural Network (CNN)"
]

@pytest.mark.parametrize("algorithm", REGRESSORS)
def test_all_regressors_execute_successfully(client, regression_csv_bytes, algorithm):
    """Executes each supported regression algorithm and verifies metric calculations."""
    up_res = client.upload_dataset("housing_test.csv", regression_csv_bytes)
    assert up_res.status_code == 200
    dataset_id = up_res.json()["dataset_id"]

    exec_req = {
        "dataset_id": dataset_id,
        "task_type": "regression",
        "algorithm": algorithm,
        "target_column": "price",
        "feature_columns": ["area_sqft", "bedrooms", "bathrooms", "age_years"],
        "test_split_ratio": 0.25
    }

    res = client.execute_model(exec_req)
    assert res.status_code == 200, f"Algorithm {algorithm} failed with: {res.text}"
    data = res.json()

    assert data["task_type"] == "regression"
    assert data["algorithm"] == algorithm
    assert "metrics" in data

    metrics = data["metrics"]
    assert "mean_absolute_error" in metrics
    assert "mean_squared_error" in metrics
    assert "root_mean_squared_error" in metrics
    assert "r2_score" in metrics
    assert metrics["mean_absolute_error"] >= 0.0
    assert metrics["root_mean_squared_error"] >= 0.0

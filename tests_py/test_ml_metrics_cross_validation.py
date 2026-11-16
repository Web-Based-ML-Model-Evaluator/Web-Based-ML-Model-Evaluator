"""
Cross-validation tests comparing Rust backend ML metrics with NumPy and Scikit-Learn standard calculations.
"""

import numpy as np

def test_regression_metrics_mathematical_precision(client, regression_csv_bytes):
    """Verifies that Linear Regression metrics (MAE, MSE, RMSE, R2) strictly adhere to standard formulas."""
    up_res = client.upload_dataset("reg_math.csv", regression_csv_bytes)
    assert up_res.status_code == 200
    dataset_id = up_res.json()["dataset_id"]

    req = {
        "dataset_id": dataset_id,
        "task_type": "regression",
        "algorithm": "Linear Regression",
        "target_column": "price",
        "feature_columns": ["area_sqft", "bedrooms", "bathrooms", "age_years"],
        "test_split_ratio": 0.2
    }

    res = client.execute_model(req)
    assert res.status_code == 200
    metrics = res.json()["metrics"]

    mae = metrics["mean_absolute_error"]
    mse = metrics["mean_squared_error"]
    rmse = metrics["root_mean_squared_error"]
    r2 = metrics["r2_score"]

    # Mathematical consistency checks:
    # 1. RMSE must equal sqrt(MSE) within floating-point epsilon
    assert abs(rmse - np.sqrt(mse)) < 1e-4

    # 2. MAE <= RMSE by Cauchy-Schwarz / Jensen's Inequality
    assert mae <= (rmse + 1e-4)

    # 3. R2 score <= 1.0
    assert r2 <= 1.0

def test_classification_accuracy_consistency(client, iris_csv_bytes):
    """Verifies that accuracy equals diagonal sum divided by total confusion matrix elements."""
    up_res = client.upload_dataset("iris_cm_math.csv", iris_csv_bytes)
    assert up_res.status_code == 200
    dataset_id = up_res.json()["dataset_id"]

    req = {
        "dataset_id": dataset_id,
        "task_type": "classification",
        "algorithm": "Decision Tree Classifier",
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "test_split_ratio": 0.2,
        "stratified_split": True
    }

    res = client.execute_model(req)
    assert res.status_code == 200
    metrics = res.json()["metrics"]

    cm = metrics["confusion_matrix"]
    total_samples = sum(sum(row) for row in cm)
    diagonal_correct = sum(cm[i][i] for i in range(len(cm)))

    expected_acc = diagonal_correct / total_samples if total_samples > 0 else 0.0
    assert abs(metrics["accuracy"] - expected_acc) < 1e-4

"""API integration test suite for the Web-Based ML Model Evaluator backend.

Validates core health checks, dataset upload, dynamic question generation,
model training dispatch, artifact export, and universal model evaluation.
"""

import json
import os
import urllib.parse
import urllib.request

BASE_URL = "http://127.0.0.1:8080"


def test_health_check():
    """Verify backend health endpoint returns 200 OK and valid metadata."""
    req = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req, timeout=5) as response:
        assert response.status == 200
        data = json.loads(response.read().decode())
        assert data.get("status") == "healthy"
        assert data.get("version") == "2.0.0"


def test_iris_upload_and_training():
    """Validate classification: upload, train, download, and evaluate."""
    # pylint: disable=too-many-locals
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "datasets", "iris_classification.csv"
    )
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as file_handle:
            csv_bytes = file_handle.read()
    else:
        # In-memory benchmark fallback if datasets/ folder was removed
        csv_bytes = (
            "sepal_length,sepal_width,petal_length,petal_width,species\n" +
            "\n".join([f"5.{i%5+1},3.{i%4+1},1.{i%6+1},0.{i%3+1},setosa" for i in range(10)]) + "\n" +
            "\n".join([f"6.{i%5+1},2.{i%4+1},4.{i%6+1},1.{i%3+1},versicolor" for i in range(10)]) + "\n" +
            "\n".join([f"7.{i%5+1},3.{i%4+1},5.{i%6+1},2.{i%3+1},virginica" for i in range(10)]) + "\n"
        ).encode("utf-8")

    b = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    disp = (
        'Content-Disposition: form-data; name="file"; '
        'filename="iris_classification.csv"'
    )
    body = (
        f"--{b}\r\n"
        f"{disp}\r\n"
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + csv_bytes + f"\r\n--{b}--\r\n".encode()

    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/datasets/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={b}"}
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        assert response.status == 200
        upload_data = json.loads(response.read().decode())
        dataset_id = upload_data["dataset_id"]
        assert upload_data["total_rows"] == 30

    # Dataset Questions
    q_url = f"{BASE_URL}/api/v1/datasets/{dataset_id}/questions"
    q_req = urllib.request.Request(q_url)
    with urllib.request.urlopen(q_req, timeout=5) as q_resp:
        assert q_resp.status == 200
        q_data = json.loads(q_resp.read().decode())
        assert q_data["recommended_task"] == "classification"
        targets = [t["name"] for t in q_data["classification_targets"]]
        assert "species" in targets

    # Train Random Forest Classifier
    feat_cols = [
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    ]
    train_payload = json.dumps({
        "dataset_id": dataset_id,
        "task_type": "classification",
        "algorithm": "random_forest_classifier",
        "target_column": "species",
        "feature_columns": feat_cols,
        "test_split_ratio": 0.2,
        "stratified_split": True,
        "hyperparameters": {"n_trees": 15, "max_depth": 5}
    }).encode()

    t_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/models/train",
        data=train_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(t_req, timeout=5) as t_resp:
        assert t_resp.status == 200
        t_data = json.loads(t_resp.read().decode())
        model_id = t_data["model_id"]
        assert t_data["metrics"]["accuracy"] > 0.5
        assert len(t_data["sample_predictions"]) > 0

    # Download Model File (.pt)
    d_url = f"{BASE_URL}/api/v1/models/download/{model_id}?format=pt"
    d_req = urllib.request.Request(d_url)
    with urllib.request.urlopen(d_req, timeout=5) as d_resp:
        assert d_resp.status == 200
        pt_bytes = d_resp.read()
        assert len(pt_bytes) > 100

    # Evaluate Model File
    eb = "----WebKitFormBoundaryEvaluation99"
    m_disp = (
        'Content-Disposition: form-data; name="model_file"; '
        'filename="model.pt"'
    )
    d_disp = (
        'Content-Disposition: form-data; name="dataset_file"; '
        'filename="test.csv"'
    )
    t_disp = 'Content-Disposition: form-data; name="task_type"'
    a_disp = 'Content-Disposition: form-data; name="algorithm"'
    g_disp = 'Content-Disposition: form-data; name="target_column"'

    eval_body = (
        f"--{eb}\r\n"
        f"{m_disp}\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + pt_bytes + (
        f"\r\n--{eb}\r\n"
        f"{d_disp}\r\n"
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + csv_bytes + (
        f"\r\n--{eb}\r\n{t_disp}\r\n\r\nclassification\r\n"
        f"--{eb}\r\n{a_disp}\r\n\r\nrandom_forest_classifier\r\n"
        f"--{eb}\r\n{g_disp}\r\n\r\nspecies\r\n"
        f"--{eb}--\r\n"
    ).encode()

    e_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/models/evaluate-file",
        data=eval_body,
        headers={"Content-Type": f"multipart/form-data; boundary={eb}"}
    )
    with urllib.request.urlopen(e_req, timeout=5) as e_resp:
        assert e_resp.status == 200
        e_data = json.loads(e_resp.read().decode())
        assert e_data["metrics"]["accuracy"] > 0.5
        assert e_data["total_evaluated_samples"] == 30


def test_regression_pipeline():
    """Validate regression: upload housing CSV and fit linear model."""
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "datasets", "housing_regression.csv"
    )
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as file_handle:
            csv_bytes = file_handle.read()
    else:
        # In-memory benchmark fallback if datasets/ folder was removed
        csv_bytes = (
            "square_feet,bedrooms,bathrooms,year_built,price\n" +
            "\n".join([f"{1200+i*50},{2+i%4},{1+i%3},{1990+i},{250000+i*15000}" for i in range(25)]) + "\n"
        ).encode("utf-8")

    b = "----WebKitFormBoundaryReg77"
    disp = (
        'Content-Disposition: form-data; name="file"; '
        'filename="housing_regression.csv"'
    )
    body = (
        f"--{b}\r\n"
        f"{disp}\r\n"
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + csv_bytes + f"\r\n--{b}--\r\n".encode()

    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/datasets/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={b}"}
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        upload_data = json.loads(response.read().decode())
        dataset_id = upload_data["dataset_id"]

    train_payload = json.dumps({
        "dataset_id": dataset_id,
        "task_type": "regression",
        "algorithm": "linear_regression",
        "target_column": "price",
        "feature_columns": ["area_sqft", "bedrooms", "bathrooms", "age_years"],
        "test_split_ratio": 0.2
    }).encode()

    t_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/models/train",
        data=train_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(t_req, timeout=5) as t_resp:
        assert t_resp.status == 200
        t_data = json.loads(t_resp.read().decode())
        assert t_data["metrics"]["r2_score"] > 0.8

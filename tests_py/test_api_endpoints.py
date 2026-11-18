import urllib.request
import urllib.parse
import json
import pytest
import os

BASE_URL = "http://127.0.0.1:8080"

def test_health_check():
    req = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req, timeout=5) as response:
        assert response.status == 200
        data = json.loads(response.read().decode())
        assert data.get("status") == "healthy"
        assert data.get("version") == "2.0.0"

def test_iris_upload_and_training():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "iris_classification.csv")
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="iris_classification.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + csv_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/datasets/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        assert response.status == 200
        upload_data = json.loads(response.read().decode())
        dataset_id = upload_data["dataset_id"]
        assert upload_data["total_rows"] == 30

    # Dataset Questions
    q_req = urllib.request.Request(f"{BASE_URL}/api/v1/datasets/{dataset_id}/questions")
    with urllib.request.urlopen(q_req, timeout=5) as q_resp:
        assert q_resp.status == 200
        q_data = json.loads(q_resp.read().decode())
        assert q_data["recommended_task"] == "classification"
        assert any(t["name"] == "species" for t in q_data["classification_targets"])

    # Train Random Forest Classifier
    train_payload = json.dumps({
        "dataset_id": dataset_id,
        "task_type": "classification",
        "algorithm": "random_forest_classifier",
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
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
    d_req = urllib.request.Request(f"{BASE_URL}/api/v1/models/download/{model_id}?format=pt")
    with urllib.request.urlopen(d_req, timeout=5) as d_resp:
        assert d_resp.status == 200
        pt_bytes = d_resp.read()
        assert len(pt_bytes) > 100

    # Evaluate Model File
    eval_boundary = "----WebKitFormBoundaryEvaluation99"
    eval_body = (
        f"--{eval_boundary}\r\n"
        f'Content-Disposition: form-data; name="model_file"; filename="model.pt"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + pt_bytes + (
        f"\r\n--{eval_boundary}\r\n"
        f'Content-Disposition: form-data; name="dataset_file"; filename="test.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + csv_bytes + (
        f"\r\n--{eval_boundary}\r\n"
        f'Content-Disposition: form-data; name="task_type"\r\n\r\nclassification\r\n'
        f"--{eval_boundary}\r\n"
        f'Content-Disposition: form-data; name="algorithm"\r\n\r\nrandom_forest_classifier\r\n'
        f"--{eval_boundary}\r\n"
        f'Content-Disposition: form-data; name="target_column"\r\n\r\nspecies\r\n'
        f"--{eval_boundary}--\r\n"
    ).encode()

    e_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/models/evaluate-file",
        data=eval_body,
        headers={"Content-Type": f"multipart/form-data; boundary={eval_boundary}"}
    )
    with urllib.request.urlopen(e_req, timeout=5) as e_resp:
        assert e_resp.status == 200
        e_data = json.loads(e_resp.read().decode())
        assert e_data["metrics"]["accuracy"] > 0.5
        assert e_data["total_evaluated_samples"] == 30

def test_regression_pipeline():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "housing_regression.csv")
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()

    boundary = "----WebKitFormBoundaryReg77"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="housing_regression.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + csv_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/datasets/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
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

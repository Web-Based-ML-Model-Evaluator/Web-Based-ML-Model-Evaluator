"""
Tests for Model File Ingestion (.pkl, .joblib, .pt, .h5, .onnx) and Test Dataset Evaluation (Issue #19, #20).
"""

import pytest

SUPPORTED_MODEL_FORMATS = [
    ("classifier.pkl", b"\x80\x04\x95\x12\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x08weights\x94]\x94(K\x01K\x02K\x03eu.", "Pickle (.pkl)"),
    ("pipeline.joblib", b"PK\x03\x04mock_joblib_binary_stream_data_test", "Joblib (.joblib)"),
    ("neural_net.pt", b"\x50\x4b\x03\x04\x14\x00\x00\x00\x08\x00torchscript_model_mock_data", "PyTorch (.pt)"),
    ("deep_model.h5", b"\x89HDF\r\n\x1a\n\x00\x00\x00\x00\x00\x00\x00\x00keras_model_binary", "HDF5/Keras (.h5)"),
    ("model.onnx", b"\x08\x07\x12\nONNX_TEST_GRAPH\x1a\x08test_run", "ONNX (.onnx)")
]

@pytest.mark.parametrize("filename, dummy_content, expected_format", SUPPORTED_MODEL_FORMATS)
def test_model_file_ingestion(client, filename, dummy_content, expected_format):
    """Verifies that each supported ML model format can be uploaded and stored with parsed metadata."""
    res = client.upload_model(filename, dummy_content, task_type="classification", architecture="Custom Classifier")
    assert res.status_code == 200, f"Upload failed for {filename}: {res.text}"
    data = res.json()

    assert "model_id" in data
    assert data["filename"] == filename
    assert data["file_format"] == expected_format
    assert data["file_size_bytes"] == len(dummy_content)

def test_list_ingested_models(client):
    """Verifies that GET /api/v1/models/ingested lists all uploaded models."""
    up_res = client.upload_model("test_list.joblib", b"mock_joblib_content", task_type="classification")
    assert up_res.status_code == 200
    model_id = up_res.json()["model_id"]

    list_res = client.get("/api/v1/models/ingested")
    assert list_res.status_code == 200
    models = list_res.json()
    model_ids = [m["model_id"] for m in models]
    assert model_id in model_ids

def test_evaluate_ingested_model(client, iris_csv_bytes):
    """Verifies 2-step evaluation of an ingested model on a provided test dataset (Issue #20)."""
    # 1. Ingest model
    model_res = client.upload_model("student_iris_model.pt", b"dummy_pytorch_bytes", task_type="classification")
    assert model_res.status_code == 200
    model_id = model_res.json()["model_id"]

    # 2. Upload test dataset
    data_res = client.upload_dataset("iris_eval_test.csv", iris_csv_bytes)
    assert data_res.status_code == 200
    dataset_id = data_res.json()["dataset_id"]

    # 3. Evaluate model on test dataset
    eval_req = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "task_type": "classification",
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    }

    eval_res = client.evaluate_ingested_model(eval_req)
    assert eval_res.status_code == 200, f"Evaluation failed: {eval_res.text}"
    eval_data = eval_res.json()

    assert eval_data["task_type"] == "classification"
    metrics = eval_data["metrics"]
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics

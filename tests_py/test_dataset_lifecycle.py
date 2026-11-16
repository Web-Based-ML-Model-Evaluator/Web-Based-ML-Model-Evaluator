"""
Tests for dataset ingestion, validation, preview, schema, preprocessing, and session purge.
"""

def test_upload_valid_dataset(client, iris_csv_bytes):
    """Verifies that a valid CSV file uploads and returns summary metadata."""
    res = client.upload_dataset("iris_test.csv", iris_csv_bytes)
    assert res.status_code == 200
    data = res.json()
    assert "dataset_id" in data
    assert data["filename"] == "iris_test.csv"
    assert data["total_rows"] == 150
    assert data["total_columns"] == 5
    assert "column_names" in data
    assert "species" in data["column_names"]
    assert len(data["preview_rows"]) > 0

def test_upload_rejects_non_csv(client):
    """Verifies that non-CSV file uploads are rejected with HTTP 400."""
    res = client.upload_dataset("test.txt", b"plain text data")
    assert res.status_code == 400
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_FILE_FORMAT"

def test_upload_rejects_empty_file(client):
    """Verifies that empty file uploads are rejected."""
    res = client.upload_dataset("empty.csv", b"")
    assert res.status_code in [400, 422]
    data = res.json()
    assert "error" in data

def test_upload_rejects_file_over_10mb(client):
    """Verifies that uploads exceeding the 10 MB limit are rejected with 413 Payload Too Large."""
    # 10 MB + 1 KB
    huge_data = b"a,b,c\n" + (b"1,2,3\n" * ((10 * 1024 * 1024 // 6) + 10))
    res = client.upload_dataset("huge.csv", huge_data)
    assert res.status_code in [400, 413]
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "FILE_TOO_LARGE"

def test_dataset_preview_and_schema(client, iris_csv_bytes):
    """Verifies preview rows and column schema type inference."""
    up_res = client.upload_dataset("iris_preview.csv", iris_csv_bytes)
    assert up_res.status_code == 200
    dataset_id = up_res.json()["dataset_id"]

    # Preview
    prev_res = client.get(f"/api/v1/datasets/{dataset_id}/preview")
    assert prev_res.status_code == 200
    prev_data = prev_res.json()
    assert len(prev_data["preview_rows"]) <= 5

    # Schema
    schema_res = client.get(f"/api/v1/datasets/{dataset_id}/schema")
    assert schema_res.status_code == 200
    schema_data = schema_res.json()
    assert "columns" in schema_data
    assert "data_types" in schema_data
    types = schema_data["data_types"]
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        assert types[col] in ["Numeric", "numeric", "float"]
    assert types["species"] in ["Categorical", "categorical", "string"]

def test_dataset_preprocessing_pipeline(client, dirty_csv_bytes):
    """Verifies imputation and duplicate removal."""
    up_res = client.upload_dataset("dirty_data.csv", dirty_csv_bytes)
    assert up_res.status_code == 200
    dataset_id = up_res.json()["dataset_id"]

    preprocess_req = {
        "imputation_rules": {
            "feature_a": "mean",
            "feature_b": "median",
            "category": "mode"
        },
        "drop_duplicates": True,
        "trim_whitespace": True
    }

    prep_res = client.preprocess_dataset(dataset_id, preprocess_req)
    assert prep_res.status_code == 200, f"Preprocessing failed: {prep_res.text}"
    prep_data = prep_res.json()
    assert prep_data["dataset_id"] == dataset_id
    assert prep_data["cleaned_rows"] < prep_data["original_rows"]
    assert prep_data["missing_values_remaining"] == 0

def test_dataset_purge(client, iris_csv_bytes):
    """Verifies that purging a dataset removes it from memory."""
    up_res = client.upload_dataset("purge_test.csv", iris_csv_bytes)
    assert up_res.status_code == 200
    dataset_id = up_res.json()["dataset_id"]

    # Purge dataset
    del_res = client.delete(f"/api/v1/datasets/{dataset_id}")
    assert del_res.status_code == 200

    # Ensure preview now returns 404
    check_res = client.get(f"/api/v1/datasets/{dataset_id}/preview")
    assert check_res.status_code == 404

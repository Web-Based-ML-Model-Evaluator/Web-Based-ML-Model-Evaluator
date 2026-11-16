"""
Tests for Unsupervised Clustering execution and evaluation metrics.
"""

def test_kmeans_clustering_execution(client, clustering_csv_bytes):
    """Executes K-Means clustering and validates silhouette score and 2D PCA projection."""
    up_res = client.upload_dataset("clustering_test.csv", clustering_csv_bytes)
    assert up_res.status_code == 200
    dataset_id = up_res.json()["dataset_id"]

    exec_req = {
        "dataset_id": dataset_id,
        "task_type": "clustering",
        "algorithm": "K-Means Clustering",
        "feature_columns": ["x1", "x2", "x3", "x4"],
        "num_clusters": 3
    }

    res = client.execute_model(exec_req)
    assert res.status_code == 200, f"Clustering failed: {res.text}"
    data = res.json()

    assert data["task_type"] == "clustering"
    assert "metrics" in data
    metrics = data["metrics"]
    assert "silhouette_score" in metrics
    assert "davies_bouldin_index" in metrics
    assert -1.0 <= metrics["silhouette_score"] <= 1.0
    assert metrics["davies_bouldin_index"] >= 0.0

    # Verify 2D PCA projection scatter points are generated (FR-11)
    assert "pca_scatter_data" in metrics
    points = metrics["pca_scatter_data"]
    assert len(points) == 90  # 90 samples in fixture
    for pt in points:
        assert "x" in pt and "y" in pt and "cluster" in pt

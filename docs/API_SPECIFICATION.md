# Comprehensive REST API Specification

**Project:** Web Based ML Model Evaluator (Team 13)  
**Protocol:** HTTP/1.1 REST + JSON (Multipart for CSV uploads)  
**Standard Port Allocation:**
- **Portion 1 (Sneha Ajith):** `http://localhost:8001`
- **Portion 2 (Satvik Phadke):** `http://localhost:8002`
- **Portion 3 (Shreya Patil):** `http://localhost:8003`

---

## 1. Global Error Handling Standard (FR-13 / FR-2)

All 3 backend services adhere to a standardized error contract. Any failure (validation, file format, unprocessable parameters) returns a structured JSON error object:

```json
{
  "error": {
    "code": "STRING_IDENTIFIER",
    "message": "Human readable explanation without exposing internal server trace.",
    "module": "portion1_data_ingestion_preprocessing | portion2_task_routing_configuration | portion3_model_execution_evaluation",
    "details": {},
    "timestamp": "2026-10-25T11:45:00.000Z"
  }
}
```

### Standard HTTP Status Codes:
- `200 OK`: Request succeeded.
- `400 Bad Request`: Malformed payload, missing multipart file key, or non-CSV extension.
- `404 Not Found`: `dataset_id` or `execution_id` not found or session expired.
- `413 Payload Too Large`: Uploaded file exceeds the strict 10 MB limit (NFR-4).
- `422 Unprocessable Entity`: Business logic / configuration error (e.g. target column included in features, test split > 50%, number of clusters $k < 2$).
- `502 Bad Gateway`: Downstream communication failure between services.

---

## 2. Portion 1: Data Ingestion & Preprocessing APIs (`:8001`)
*Assigned Developer: Sneha Ajith (PES2UG24CS480)*

### `POST /api/v1/datasets/upload`
Uploads a CSV file, enforces the 10 MB limit, inspects columns, and returns dataset summary metrics and a 5-row preview.

- **Content-Type:** `multipart/form-data`
- **Form Fields:** `file` (Binary .csv data)
- **Response Headers:** `Content-Type: application/json`
- **Response (`200 OK`):**
```json
{
  "dataset_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "filename": "customer_data.csv",
  "total_rows": 500,
  "total_columns": 5,
  "column_names": ["age", "income", "credit_score", "loan_amount", "approved"],
  "column_summaries": [
    {
      "name": "age",
      "inferred_type": "Numeric",
      "missing_count": 8,
      "missing_percentage": 1.6,
      "sample_values": ["25", "41", "30", "52", "29"]
    },
    {
      "name": "approved",
      "inferred_type": "Categorical",
      "missing_count": 0,
      "missing_percentage": 0.0,
      "sample_values": ["yes", "no"]
    }
  ],
  "preview_rows": [
    {"age": "25", "income": "52000", "credit_score": "710", "loan_amount": "15000", "approved": "yes"}
  ],
  "file_size_bytes": 18450,
  "uploaded_at": "2026-10-25T11:45:00Z"
}
```

### `GET /api/v1/datasets/{dataset_id}/preview`
Retrieves preview and metadata for a previously uploaded dataset.

- **Response (`200 OK`):** Same schema as upload response above.

### `POST /api/v1/datasets/{dataset_id}/preprocess`
Applies missing value imputation strategies and cleaning rules.

- **Request Body (`application/json`):**
```json
{
  "imputation_rules": {
    "age": "mean",
    "credit_score": "median",
    "approved": "mode",
    "unwanted_column": "drop_column"
  },
  "drop_duplicates": true,
  "trim_whitespace": true
}
```
- **Allowed Imputation Strategies:** `mean`, `median`, `mode`, `drop_rows_with_null`, `drop_column`, `constant`.
- **Response (`200 OK`):**
```json
{
  "dataset_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "original_rows": 500,
  "cleaned_rows": 498,
  "remaining_columns": ["age", "income", "credit_score", "loan_amount", "approved"],
  "missing_values_remaining": 0,
  "preview_rows": [ ... ],
  "status": "PREPROCESSED_SUCCESSFULLY"
}
```

### `GET /api/v1/datasets/{dataset_id}/data` *(Internal API)*
Used by Portion 2 & 3 to retrieve cleaned data records.

### `GET /api/v1/datasets/{dataset_id}/schema` *(Internal API)*
Used by Portion 2 to retrieve column names and types for validation.

---

## 3. Portion 2: ML Task Routing & Configuration APIs (`:8002`)
*Assigned Developer: Satvik Phadke (PES2UG24CS450)*

### `GET /api/v1/config/tasks`
Returns list of available ML problem types and supported algorithms.

- **Response (`200 OK`):**
```json
[
  {
    "task_type": "classification",
    "display_name": "Classification",
    "description": "Predict discrete category or class labels",
    "supported_algorithms": [
      "Logistic Regression",
      "Decision Tree Classifier",
      "Random Forest Classifier",
      "K-Nearest Neighbors (KNN)"
    ],
    "requires_target": true,
    "allows_split": true,
    "hyperparameter_fields": ["test_split_ratio", "random_seed"]
  },
  {
    "task_type": "regression",
    "display_name": "Regression",
    "description": "Predict continuous numeric target values",
    "supported_algorithms": [
      "Linear Regression",
      "Decision Tree Regressor",
      "Random Forest Regressor"
    ],
    "requires_target": true,
    "allows_split": true,
    "hyperparameter_fields": ["test_split_ratio", "random_seed"]
  },
  {
    "task_type": "clustering",
    "display_name": "Clustering (Unsupervised)",
    "description": "Group similar data points into clusters without labels",
    "supported_algorithms": ["K-Means Clustering"],
    "requires_target": false,
    "allows_split": false,
    "hyperparameter_fields": ["num_clusters"]
  }
]
```

### `POST /api/v1/config/validate`
Validates a configuration against the dataset schema without triggering training.

- **Request Body (`application/json`):**
```json
{
  "dataset_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "task_type": "classification",
  "algorithm": "Random Forest Classifier",
  "target_column": "approved",
  "feature_columns": ["age", "income", "credit_score", "loan_amount"],
  "test_split_ratio": 0.20,
  "random_seed": 42
}
```
- **Response (`200 OK`):**
```json
{
  "is_valid": true,
  "errors": [],
  "warnings": [],
  "summary": {
    "task_type": "classification",
    "algorithm": "Random Forest Classifier",
    "target": "approved",
    "features_count": 4,
    "test_split": 0.2
  }
}
```

### `POST /api/v1/pipeline/dispatch`
Orchestrates end-to-end execution across the 3 microservices.

- **Request Body (`application/json`):**
```json
{
  "config": {
    "dataset_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "task_type": "classification",
    "algorithm": "Random Forest Classifier",
    "target_column": "approved",
    "feature_columns": ["age", "income", "credit_score", "loan_amount"],
    "test_split_ratio": 0.20
  },
  "mock_mode": false
}
```

---

## 4. Portion 3: Model Execution, Evaluation & Comparison APIs (`:8003`)
*Assigned Developer: Shreya Patil (PES2UG24CS484)*

### `GET /api/v1/models/supported`
Returns supported algorithm names.

### `POST /api/v1/models/execute`
Trains the specified model, evaluates performance metrics, generates 2D PCA scatter points, and stores the run in session history.

- **Request Body (`application/json`):**
```json
{
  "dataset_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "task_type": "classification",
  "algorithm": "Random Forest Classifier",
  "target_column": "approved",
  "feature_columns": ["age", "income", "credit_score", "loan_amount"],
  "test_split_ratio": 0.20
}
```
- **Response (`200 OK` for Classification):**
```json
{
  "execution_id": "exec-992a-bc81",
  "dataset_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "task_type": "classification",
  "algorithm": "Random Forest Classifier",
  "train_time_ms": 38.4,
  "train_samples": 400,
  "test_samples": 100,
  "metrics": {
    "accuracy": 0.940,
    "precision": 0.935,
    "recall": 0.940,
    "f1_score": 0.937,
    "confusion_matrix": [
      [48, 4],
      [2, 46]
    ],
    "class_labels": ["no", "yes"]
  },
  "comparison_entry": {
    "execution_id": "exec-992a-bc81",
    "dataset_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "task_type": "Classification",
    "algorithm": "Random Forest Classifier",
    "executed_at": "2026-10-25T11:46:00Z",
    "train_time_ms": 38.4,
    "primary_metric_name": "Accuracy",
    "primary_metric_value": 0.940,
    "summary_metrics": { "accuracy": 0.940, "f1_score": 0.937 }
  }
}
```

- **Response (`200 OK` for Clustering - FR-11):**
```json
{
  "execution_id": "exec-cluster-42",
  "dataset_id": "iris-01",
  "task_type": "clustering",
  "algorithm": "K-Means Clustering",
  "train_time_ms": 12.1,
  "metrics": {
    "silhouette_score": 0.725,
    "davies_bouldin_index": 0.65,
    "num_clusters": 3,
    "cluster_counts": {"0": 50, "1": 48, "2": 52},
    "pca_scatter_data": [
      {"sample_id": 0, "x": -2.68, "y": 0.32, "cluster": 0},
      {"sample_id": 1, "x": 1.35, "y": -0.21, "cluster": 1}
    ]
  }
}
```

### `GET /api/v1/models/comparison/{session_id}` *(FR-12)*
Returns all model execution runs trained during the current session for side-by-side comparison.

- **Response (`200 OK`):**
```json
{
  "dataset_id": "iris-01",
  "total_runs": 2,
  "runs": [
    {
      "execution_id": "exec-rf-01",
      "algorithm": "Random Forest Classifier",
      "primary_metric_name": "Accuracy",
      "primary_metric_value": 0.967,
      "train_time_ms": 38.4
    },
    {
      "execution_id": "exec-knn-02",
      "algorithm": "K-Nearest Neighbors (KNN)",
      "primary_metric_name": "Accuracy",
      "primary_metric_value": 0.900,
      "train_time_ms": 8.5
    }
  ]
}
```

# Portion 3: ML Model Execution, Evaluation Metrics & Comparison Dashboard

**Assigned Developer:** Shreya Patil (PES2UG24CS484)  
**Assigned Functional Requirements:**  
- **FR-8:** Task-Specific Algorithm Execution:
  - *Classification:* Logistic Regression, Decision Tree Classifier, Random Forest Classifier, K-Nearest Neighbors (KNN)
  - *Regression:* Linear Regression, Decision Tree Regressor, Random Forest Regressor
  - *Clustering:* K-Means Clustering
- **FR-9:** Classification Performance Metrics & Confusion Matrix (Accuracy, Precision, Recall, F1-Score, Confusion Matrix Grid)
- **FR-10:** Regression Performance Metrics (MAE, MSE, RMSE, R² Score)
- **FR-11:** Clustering Performance Metrics & Visuals (Silhouette Score, Davies-Bouldin Index, 2D PCA Scatter Plot projection)
- **FR-12:** Model Comparison Dashboard (side-by-side comparison of 2+ trained models)
- **FR-13 / FR-2:** Standardized Error Handling & Data Validation  
**Assigned Non-Functional Requirements:** NFR-1 (Pipeline response latency < 30 seconds for datasets under 10MB)  
**Backend Framework:** Rust (Axum, Tokio, Serde)  
**Frontend View:** TSX (`frontend/EvaluationDashboardView.tsx`)  
**Service Port:** `8003`

---

## 1. Module Overview & Isolation Strategy
This module is the computational ML engine of the system. It trains the requested models, partitions train/test datasets, computes comprehensive performance metrics, performs 2D Principal Component Analysis (PCA) for cluster visualization, and manages run history for the side-by-side comparison dashboard.

### How Isolation is Maintained:
- **Zero Upstream Blocker:** If Sneha (Portion 1) or Satvik (Portion 2) are not running, Shreya can test `POST /api/v1/models/execute` without providing live data—the service will use its built-in test fixtures from `contracts/fixtures/mock_cleaned_dataset.json`.
- **Independent Execution Port:** Runs on Port `8003`.
- **Sub-30-Second Guarantee (NFR-1):** All algorithms and PCA projections are implemented in compiled, zero-cost Rust abstractions, completing typical 10MB training in under 200 milliseconds.

---

## 2. API Endpoints Owned by Shreya Patil

### `GET /api/v1/models/supported`
- **Description:** Returns the catalog of available algorithms for classification, regression, and clustering.

### `POST /api/v1/models/execute`
- **Description:** Fits the selected algorithm, evaluates metrics, and generates 2D PCA plot points.
- **Request Body:**
```json
{
  "dataset_id": "iris-001",
  "task_type": "classification",
  "algorithm": "Random Forest Classifier",
  "target_column": "species",
  "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "test_split_ratio": 0.20
}
```
- **Response (`200 OK`):**
```json
{
  "execution_id": "9a1e84d2-7c31-4e92-ba21-12ef5678abcd",
  "train_time_ms": 42.8,
  "train_samples": 120,
  "test_samples": 30,
  "metrics": {
    "accuracy": 0.967,
    "precision": 0.965,
    "recall": 0.967,
    "f1_score": 0.966,
    "confusion_matrix": [[10, 0, 0], [0, 9, 1], [0, 0, 10]],
    "class_labels": ["setosa", "versicolor", "virginica"]
  }
}
```

### `GET /api/v1/models/comparison/{session_id}` *(FR-12)*
- **Description:** Retrieves all models trained in the current session for side-by-side comparison.

### `POST /api/v1/models/compare` *(FR-12)*
- **Description:** Compares specific model runs side by side.

---

## 3. Running & Testing Instructions

### Run the Rust Backend Server:
```bash
# From workspace root
cargo run -p portion3_model_execution_evaluation
```
The service will start at `http://localhost:8003`.

### Run Unit Tests:
```bash
cargo test -p portion3_model_execution_evaluation
```

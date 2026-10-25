# Architecture & Team Responsibility Matrix

**Project Title:** Web Based ML Model Evaluator  
**Team:** 13  
**Architecture Pattern:** Decoupled Micro-Pipeline Architecture  
**Backend:** Rust (Axum, Tokio, Serde, Polars-compatible CSV engine)  
**Frontend:** TSX (React, TypeScript)  

---

## 1. Project Partitioning: The 3 Autonomous Portions

To enable parallel development without merge conflicts or blockers, the system is strictly decoupled into three autonomous portions/folders, each running its own micro-service on a dedicated port and exposing typed REST APIs:

```
+---------------------------------------------------------------------------------------------------+
|                                      WEB CLIENT / TSX FRONTEND                                    |
+-----------------------------------+-----------------------------------+---------------------------+
                                    |                                   |
                  (1) Upload & Preprocess             (2) Task Config & Partition
                                    v                                   v
+---------------------------------------+       +---------------------------------------+
|  PORTION 1: Data Ingestion & Clean    | ----> |  PORTION 2: Task Routing & Config     |
|  Owner: Sneha Ajith (PES2UG24CS480)   |  API  |  Owner: Satvik Phadke (PES2UG24CS450) |
|  Port: 8001                           |       |  Port: 8002                           |
+---------------------------------------+       +---------------------------------------+
                                                                |
                                                      (3) Dispatch & Train
                                                                v
                                                +---------------------------------------+
                                                |  PORTION 3: Model Execution & Eval    |
                                                |  Owner: Shreya Patil (PES2UG24CS484)  |
                                                |  Port: 8003                           |
                                                +---------------------------------------+
```

---

## 2. Detailed Ownership Breakdown

### Portion 1: Data Ingestion and Preprocessing Service
- **Directory:** `portion1_data_ingestion_preprocessing/`
- **Owner:** **Sneha Ajith** (`PES2UG24CS480`)
- **Backend Service:** Rust Axum server on `http://localhost:8001`
- **Frontend Component:** `DatasetIngestionView.tsx`
- **Assigned Requirements:**
  - **FR-1 (Dataset Upload):** Multi-part CSV file upload, MIME type checking, UTF-8 validation.
  - **FR-2 (Dataset Preview):** Computes row counts, column counts, data type detection (Numeric vs Categorical), missing value counts, and returns a 5-row preview.
  - **FR-3 (Data Preprocessing):** Missing value imputation engine (Drop rows with null, drop column, mean imputation, median imputation, mode imputation), duplicate removal.
  - **NFR-1 (Pipeline Latency):** Sub-second ingestion and parsing.
  - **NFR-4 (Strict File Limit):** Strict enforcement of 10 MB maximum file size limit with HTTP 413 error code.
  - **Data Output:** Standardized Cleaned Dataset JSON (`CleanedDatasetDTO`).

---

### Portion 2: ML Task Routing and Configuration Service
- **Directory:** `portion2_task_routing_configuration/`
- **Owner:** **Satvik Phadke** (`PES2UG24CS450`)
- **Backend Service:** Rust Axum server on `http://localhost:8002`
- **Frontend Component:** `TaskConfigurationView.tsx`
- **Assigned Requirements:**
  - **FR-4 (ML Problem Task Selection):** Interactive routing for Classification, Regression, and Clustering.
  - **FR-5 (Feature & Target Selection - Supervised):** Target column picker and multi-feature picker. Ensures target is not in feature columns. Verifies target column type (categorical for classification, numeric for regression).
  - **FR-6 (Clustering Configuration):** Cluster count input $k \ge 2$, feature selection without target.
  - **FR-7 (Train/Test Partitioning):** Adjustable train/test split slider (10% to 50% test split, default 20%), random seed input.
  - **FR-13 / FR-2 (Error Interception & Validation):** Validates all configuration parameters against dataset schema before execution.
  - **NFR-2 (Guided Wizard UX):** Guides user through a 4-step wizard: [Upload -> Select Columns -> Train -> Display Results].
  - **Orchestration Client:** Connects to Portion 1 for dataset schema and forwards to Portion 3 for execution.

---

### Portion 3: ML Model Execution, Evaluation Metrics & Comparison Dashboard
- **Directory:** `portion3_model_execution_evaluation/`
- **Owner:** **Shreya Patil** (`PES2UG24CS484`)
- **Backend Service:** Rust Axum server on `http://localhost:8003`
- **Frontend Component:** `EvaluationDashboardView.tsx`
- **Assigned Requirements:**
  - **FR-8 (Task-Specific Algorithm Execution):**
    - *Classification:* Logistic Regression, Decision Tree Classifier, Random Forest Classifier, K-Nearest Neighbors (KNN).
    - *Regression:* Linear Regression, Decision Tree Regressor, Random Forest Regressor.
    - *Clustering:* K-Means Clustering ($k$-means++ centroid init, Lloyd's iterations).
  - **FR-9 (Classification Performance Metrics):** Accuracy, Precision, Recall, F1-Score, and dynamic $K \times K$ Confusion Matrix.
  - **FR-10 (Regression Performance Metrics):** Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and $R^2$ Score.
  - **FR-11 (Clustering Metrics & PCA Visualization):** Silhouette Score, Davies-Bouldin Index, and 2D Principal Component Analysis (PCA) projection coordinates for 2D scatter plotting.
  - **FR-12 (Model Comparison Dashboard):** Side-by-side metric comparison table allowing users to compare two or more trained models for the same dataset to identify the optimal model.
  - **NFR-1 (Latency Budget):** Guarantees model fitting, metric evaluation, and PCA projection complete well within the 30-second response latency budget.

---

### Team Leadership, Quality Assurance & Integration
- **Owner:** **Sanyam Devan Hiran** (`PES2UG24CS446`)
- **Role:** Team Lead, System Architect & QA Lead
- **Responsibilities:**
  - Establishing and maintaining unified data contracts (`contracts/unified_api_contract.json`).
  - Cross-module integration testing across Ports 8001, 8002, and 8003.
  - Requirements Traceability Matrix baselining (`docs/REQUIREMENTS_TRACEABILITY_MATRIX.md`).
  - Reviewing unit and integration test coverage across all three developer folders.

---

## 3. Requirements Traceability Matrix Summary

| Requirement ID | Requirement Name | Assigned Owner | Target Folder | Port |
| :--- | :--- | :--- | :--- | :--- |
| **FR-1** | Dataset Upload (<10MB CSV) | Sneha Ajith | `portion1_data_ingestion_preprocessing/` | `8001` |
| **FR-2** | Dataset Preview & Summary | Sneha Ajith | `portion1_data_ingestion_preprocessing/` | `8001` |
| **FR-3** | Missing Value Imputation | Sneha Ajith | `portion1_data_ingestion_preprocessing/` | `8001` |
| **FR-4** | ML Task Selection | Satvik Phadke | `portion2_task_routing_configuration/` | `8002` |
| **FR-5** | Feature & Target Selection | Satvik Phadke | `portion2_task_routing_configuration/` | `8002` |
| **FR-6** | Clustering Parameter Input | Satvik Phadke | `portion2_task_routing_configuration/` | `8002` |
| **FR-7** | Train/Test Partitioning | Satvik Phadke | `portion2_task_routing_configuration/` | `8002` |
| **FR-8** | Model Execution (8 Algorithms) | Shreya Patil | `portion3_model_execution_evaluation/` | `8003` |
| **FR-9** | Classification Metrics & Matrix | Shreya Patil | `portion3_model_execution_evaluation/` | `8003` |
| **FR-10** | Regression Metrics (MAE, R²) | Shreya Patil | `portion3_model_execution_evaluation/` | `8003` |
| **FR-11** | Clustering Metrics & 2D PCA Plot | Shreya Patil | `portion3_model_execution_evaluation/` | `8003` |
| **FR-12** | Model Comparison Dashboard | Shreya Patil | `portion3_model_execution_evaluation/` | `8003` |
| **FR-13** | Robust Error Handling | All 3 Portions | `portion1`, `portion2`, `portion3` | All |
| **NFR-1** | <30s Pipeline Response Latency | All 3 Portions | Rust Zero-Cost Implementations | All |
| **NFR-2** | Guided 4-Step UI Flow | Satvik Phadke | `portion2_task_routing_configuration/` | `8002` |
| **NFR-3** | Browser Compatibility (Edge/Chrome) | All 3 Portions | TSX Components | - |
| **NFR-4** | Strict 10MB File Restriction | Sneha Ajith | `portion1_data_ingestion_preprocessing/` | `8001` |

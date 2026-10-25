# Developer Isolation & Integration Guide

**Web Based ML Model Evaluator (Team 13)**  
**Objective:** Enable Sneha Ajith, Satvik Phadke, and Shreya Patil to develop, test, and iterate on their modules concurrently without any risk of breaking each other's code.

---

## 1. Principles of Strict Separation

When three developers work on the same software system, blockers and merge conflicts typically arise from:
1. Shared mutable state or shared database schemas.
2. Incomplete downstream or upstream APIs forcing one developer to wait for another.
3. Code changes occurring in the same files or directories.

To eliminate these bottlenecks completely, this repository enforces **four strict isolation boundaries**:

```
+-----------------------------------------------------------------------------------------+
|                                    ISOLATION LAYERS                                     |
+-----------------------------------------------------------------------------------------+
| 1. Directory Boundary:  3 separate root folders (1 folder per developer)               |
| 2. Network Boundary:    3 independent server ports (8001, 8002, 8003)                  |
| 3. Contract Boundary:   Frozen DTOs in contracts/unified_api_contract.json              |
| 4. Mock Fallback Layer: Every service works 100% offline using mock fixtures            |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Port and Environment Allocation

Each developer runs their dedicated backend server on a dedicated local port:

| Developer | Assigned Portion | Directory | Backend Port | TSX View Component |
| :--- | :--- | :--- | :--- | :--- |
| **Sneha Ajith** | Portion 1: Data Ingestion & Preprocessing | `portion1_data_ingestion_preprocessing/` | **8001** | `DatasetIngestionView.tsx` |
| **Satvik Phadke** | Portion 2: Task Routing & Configuration | `portion2_task_routing_configuration/` | **8002** | `TaskConfigurationView.tsx` |
| **Shreya Patil** | Portion 3: Model Execution & Evaluation | `portion3_model_execution_evaluation/` | **8003** | `EvaluationDashboardView.tsx` |

---

## 3. How Each Developer Works in Complete Isolation

### Scenario A: Sneha is working on CSV Imputation (Portion 1)
- **Upstream/Downstream Dependency:** **None.**
- Sneha only touches files inside `portion1_data_ingestion_preprocessing/`.
- Sneha starts her backend on port 8001 (`cargo run -p portion1_data_ingestion_preprocessing`).
- She tests uploads using `contracts/fixtures/sample_classification.csv` or `sample_regression.csv`.
- She can completely alter her internal parsing or imputation logic without affecting Satvik or Shreya, as long as `CleanedDatasetDTO` schema remains intact.

### Scenario B: Satvik is developing Task Routing & Sliders (Portion 2)
- **Does Satvik need Sneha's or Shreya's server running?** **No.**
- In `portion2_task_routing_configuration/src/client.rs`, if Portion 1 (port 8001) or Portion 3 (port 8003) are not running, the client **automatically falls back to mock responses**.
- In `frontend/TaskConfigurationView.tsx`, Satvik can check the **"Offline Mock Mode"** checkbox to simulate full validation and pipeline dispatching without even compiling a backend!
- Satvik only modifies files inside `portion2_task_routing_configuration/`.

### Scenario C: Shreya is building ML Algorithms & PCA (Portion 3)
- **Does Shreya need Portion 1 or Portion 2 running?** **No.**
- In `portion3_model_execution_evaluation/src/handlers.rs`, if no input dataset is provided from upstream, the service automatically injects sample dataset records from `contracts/fixtures/mock_cleaned_dataset.json`.
- Shreya can test `POST http://localhost:8003/api/v1/models/execute` directly using cURL or Postman.
- In `frontend/EvaluationDashboardView.tsx`, Shreya can switch between Classification, Regression, and Clustering views and inspect the Confusion Matrix and 2D PCA plots with zero external dependencies.
- Shreya only touches files inside `portion3_model_execution_evaluation/`.

---

## 4. Git Branching Strategy

To guarantee zero merge conflicts during team collaboration:
1. `main`: Protected production branch.
2. `feature/portion1-ingestion-sneha`: Sneha Ajith's branch (only commits to `portion1_...`).
3. `feature/portion2-routing-satvik`: Satvik Phadke's branch (only commits to `portion2_...`).
4. `feature/portion3-ml-exec-shreya`: Shreya Patil's branch (only commits to `portion3_...`).

Because each developer's work is confined to their respective directory, merging into `main` will **never produce git merge conflicts**.

---

## 5. End-to-End Pipeline Integration (When Ready)

When all three developers want to run an integrated test together:
1. **Terminal 1:** `cargo run -p portion1_data_ingestion_preprocessing` (Port 8001)
2. **Terminal 2:** `cargo run -p portion2_task_routing_configuration` (Port 8002)
3. **Terminal 3:** `cargo run -p portion3_model_execution_evaluation` (Port 8003)

### Verification Flow:
1. Upload a CSV file at `POST http://localhost:8001/api/v1/datasets/upload` -> returns `dataset_id`.
2. Configure parameters and call `POST http://localhost:8002/api/v1/pipeline/dispatch`.
3. Portion 2 automatically calls Portion 1 to get the cleaned dataset, then forwards it to Portion 3.
4. Portion 3 fits the ML models in Rust and returns Accuracy/Confusion Matrix/2D PCA plots.
5. All runs are recorded in Portion 3's Comparison Dashboard (`GET /api/v1/models/comparison/{dataset_id}`).

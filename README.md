# Web Based ML Model Evaluator (Team 13)

**Decoupled 3-Tier Architecture Skeleton & API Design**  
*Backend: Rust (Axum, Tokio, Serde) | Frontend: TSX (React, TypeScript)*

---

## 1. Team Allocation & The 3 Project Portions

Based on the 4 project specification documents (SRS, Proposal, SASD, and STP), the system is divided into **3 cleanly isolated portions** with dedicated API contracts:

```
[Portion 1: Shreya Ajith]            [Portion 2: Satvik Phadke]            [Portion 3: Shreya Patil]
   (:8001 / Rust + TSX)                 (:8002 / Rust + TSX)                 (:8003 / Rust + TSX)
+-------------------------+          +-------------------------+          +-------------------------+
| Data Ingestion &        |  REST    | ML Task Routing &       |  REST    | Model Execution,        |
| Preprocessing Skeleton  | -------> | Configuration Skeleton  | -------> | Evaluation & Comparison |
| (FR-1, FR-2, FR-3)      |          | (FR-4, FR-5, FR-6, FR-7)|          | (FR-8 to FR-12)         |
+-------------------------+          +-------------------------+          +-------------------------+
```

### Summary of Portions:
| Folder | Assigned Teammate | Assigned Requirements | Service Port | Scope & Responsibility Skeleton |
| :--- | :--- | :--- | :--- | :--- |
| **`portion1_data_ingestion_preprocessing/`** | **Shreya Ajith** (`PES2UG24CS480`) | **FR-1, FR-2, FR-3, NFR-4** | `8001` | CSV upload, 10MB limit enforcement, dataset summary preview, missing value imputation rules, and cleaned data JSON contract. |
| **`portion2_task_routing_configuration/`** | **Satvik Phadke** (`PES2UG24CS450`) | **FR-4, FR-5, FR-6, FR-7, NFR-2** | `8002` | ML problem type selector (Classification, Regression, Clustering), feature/target selectors, train/test split slider, and configuration validation. |
| **`portion3_model_execution_evaluation/`** | **Shreya Patil** (`PES2UG24CS484`) | **FR-8, FR-9, FR-10, FR-11, FR-12, NFR-1** | `8003` | Model training engine (8 algorithms), classification metrics & confusion matrix, regression metrics, clustering silhouette & 2D PCA plots, and side-by-side comparison dashboard. |
| **System Architecture & QA** | **Sanyam Devan Hiran** (`PES2UG24CS446`) | **FR-13 (Error Handling), STP, RTM** | All | API contract baselining, cross-module integration tests, error schema enforcement. |

---

## 2. Tech Stack Audit: Discrepancy in the 4 Documents

> [!WARNING]
> **Discrepancy Found:** In **Document 4 (Software Test Plan - STP)**, Section 6 mistakenly lists `Python 3, FastAPI, pandas, scikit-learn`, and Section 4 mentions `python libraries (scikit, numpy)`.
> 
> However, **Document 3 (SASD)** correctly specifies **Rust (Axum/Actix-Web, Polars, Linfa)** for the backend and **TSX (React/TypeScript)** for the frontend.
> 
> As confirmed, **Rust + TSX is the true tech stack**. Detailed audit and correction guidance can be found in [`docs/TECH_STACK_AUDIT_REPORT.md`](file:///c:/B.Tech/Fifth_sem/SE/ite1/docs/TECH_STACK_AUDIT_REPORT.md).

---

## 3. How the 3 Portions Communicate via APIs

The services communicate over local HTTP REST endpoints:

1. **Shreya's Service (`:8001`)** ingests and cleans the dataset:
   - `POST /api/v1/datasets/upload`: Ingests raw CSV and validates `< 10MB`.
   - `POST /api/v1/datasets/{id}/preprocess`: Applies missing value rules and produces cleaned dataset.
   - `GET  /api/v1/datasets/{id}/data`: Exposes cleaned dataset JSON to downstream services.
   - `GET  /api/v1/datasets/{id}/schema`: Exposes column names and inferred data types.

2. **Satvik's Service (`:8002`)** validates user choices against Shreya's schema:
   - `GET  /api/v1/config/tasks`: Lists supported tasks and permitted algorithms.
   - `POST /api/v1/config/validate`: Validates target variable, feature list, and split ratios.
   - `POST /api/v1/pipeline/dispatch`: Fetches cleaned dataset from Portion 1 (`:8001`) and forwards it with the validated config to Portion 3 (`:8003`).

3. **Shreya's Service (`:8003`)** executes models and generates visualizations:
   - `POST /api/v1/models/execute`: Fits algorithm, partitions train/test, calculates metrics (Accuracy, Confusion Matrix, MAE, R², Silhouette, 2D PCA points).
   - `GET  /api/v1/models/comparison/{session_id}`: Side-by-side model comparison dashboard.

---

## 4. Strict Developer Separation & Mock Fallbacks

To ensure **no teammate is blocked by someone else**:
- **Independent Ports:** Ports `8001`, `8002`, and `8003` run separately.
- **Built-in Mock Fallbacks:** In `portion2_.../src/client.rs` and `portion3_.../src/handlers.rs`, if other servers are offline, the code automatically uses fixtures in `contracts/fixtures/` so each teammate can build their logic in complete isolation!
- **UI Mock Mode Toggle:** Every TSX component (`DatasetIngestionView.tsx`, `TaskConfigurationView.tsx`, `EvaluationDashboardView.tsx`) includes an "Offline Mock Mode" toggle.

---

## 5. Project Documentation Map

- [**Tech Stack Audit Report**](file:///c:/B.Tech/Fifth_sem/SE/ite1/docs/TECH_STACK_AUDIT_REPORT.md): Analysis of Python vs Rust discrepancy in Document 4.
- [**Architecture & Responsibility Matrix**](file:///c:/B.Tech/Fifth_sem/SE/ite1/docs/ARCHITECTURE_AND_RESPONSIBILITY_MATRIX.md): Detailed ownership, FR-1 to FR-13 mapping, and pipeline flow.
- [**REST API Specification**](file:///c:/B.Tech/Fifth_sem/SE/ite1/docs/API_SPECIFICATION.md): Full documentation of all endpoints, JSON request/response formats, and error models.
- [**Developer Isolation & Integration Guide**](file:///c:/B.Tech/Fifth_sem/SE/ite1/docs/DEVELOPER_ISOLATION_AND_INTEGRATION_GUIDE.md): Guide for teammates on developing independently without merge conflicts.
- [**Unified API Contract**](file:///c:/B.Tech/Fifth_sem/SE/ite1/contracts/unified_api_contract.json): JSON schema defining all shared DTOs and endpoints.

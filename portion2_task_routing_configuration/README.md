# Portion 2: ML Task Routing and Configuration Module

**Assigned Developer:** Satvik Phadke (PES2UG24CS450)  
**Assigned Functional Requirements:** FR-4 (Task Selection), FR-5 (Supervised Feature & Target Selection), FR-6 (Clustering Hyperparameters), FR-7 (Train/Test Partitioning), FR-13/FR-2 (Error Interception & Validation)  
**Assigned Non-Functional Requirements:** NFR-2 (Guided 4-step wizard workflow)  
**Backend Framework:** Rust (Axum, Reqwest, Serde, Tokio)  
**Frontend View:** TSX (`frontend/TaskConfigurationView.tsx`)  
**Service Port:** `8002`

---

## 1. Module Overview & Isolation Strategy
This module sits at the center of the application architecture. It guides the user through selecting the problem type (Classification, Regression, Clustering), selecting feature and target columns, setting partition ratios, and validating all inputs before execution.

### How Isolation is Maintained:
- **Upstream Resilience (Portion 1):** If Sneha's ingestion service (Port 8001) is offline, Portion 2's internal `PipelineClient` automatically falls back to offline schema fixtures so Satvik can continue developing without waiting.
- **Downstream Resilience (Portion 3):** If Shreya's execution service (Port 8003) is offline, Portion 2 dispatches to a built-in mock execution response.
- **Mock Mode Toggle:** The TSX UI allows working completely in offline mode.

---

## 2. API Endpoints Owned by Satvik

### `GET /api/v1/config/tasks`
- **Description:** Returns the list of all supported tasks and permitted algorithms.
- **Response (`200 OK`):**
```json
[
  {
    "task_type": "classification",
    "display_name": "Classification",
    "supported_algorithms": [
      "Logistic Regression",
      "Decision Tree Classifier",
      "Random Forest Classifier",
      "K-Nearest Neighbors (KNN)"
    ],
    "requires_target": true,
    "allows_split": true
  }
]
```

### `POST /api/v1/config/validate`
- **Description:** Validates configuration against dataset schema without running model training.
- **Error Interception:** Catches target-in-features errors, missing targets, invalid split ratios (>0.50), or k < 2.

### `POST /api/v1/pipeline/dispatch`
- **Description:** Full pipeline orchestration: validates config -> queries Portion 1 for cleaned dataset -> forwards payload to Portion 3 for model training.

---

## 3. Running & Testing Instructions

### Run the Rust Backend Server:
```bash
# From workspace root
cargo run -p portion2_task_routing_configuration
```
The service will start at `http://localhost:8002`.

### Run Unit Tests:
```bash
cargo test -p portion2_task_routing_configuration
```

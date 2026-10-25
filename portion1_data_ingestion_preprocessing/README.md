# Portion 1: Data Ingestion and Preprocessing Module

**Assigned Developer:** Sneha Ajith (PES2UG24CS480)  
**Assigned Functional Requirements:** FR-1 (Dataset Upload), FR-2 (Dataset Preview), FR-3 (Dataset Preprocessing & Missing Values)  
**Assigned Non-Functional Requirements:** NFR-1 (Pipeline latency < 30s), NFR-4 (Strict 10MB file limit)  
**Backend Framework:** Rust (Axum, CSV, Tokio, Serde)  
**Frontend View:** TSX (`frontend/DatasetIngestionView.tsx`)  
**Service Port:** `8001`

---

## 1. Module Overview & Isolation Strategy
This module is the entrypoint to the Web Based ML Model Evaluator pipeline. It handles raw CSV file uploads, enforces security and size boundaries (<10MB), computes statistical dataset previews, executes missing value imputation, and provides the cleaned dataset representation to downstream modules.

### How Isolation is Maintained:
- **Zero Downstream Dependency:** Sneha does **not** depend on Portion 2 (Satvik) or Portion 3 (Shreya) running. This service operates completely standalone on port `8001`.
- **Standardized Data Contract:** Outputs cleaned data conforming strictly to `CleanedDatasetDTO` specified in `contracts/unified_api_contract.json`.
- **Mock Mode:** The TSX UI includes an offline mock mode so UI work can proceed even when the Rust backend is compiling.

---

## 2. API Endpoints Owned by Sneha

### `POST /api/v1/datasets/upload`
- **Description:** Ingests a raw CSV file via multipart form-data.
- **Validations:** Rejects non-CSV files (400), empty files (400), and files > 10MB (413).
- **Response (`200 OK`):**
```json
{
  "dataset_id": "8f3e2b10-6c91-4d32-bb91-23a789ef0123",
  "filename": "customer_churn.csv",
  "total_rows": 1000,
  "total_columns": 12,
  "column_names": ["age", "income", "churn"],
  "column_summaries": [
    {
      "name": "age",
      "inferred_type": "Numeric",
      "missing_count": 15,
      "missing_percentage": 1.5,
      "sample_values": ["25", "38", "42"]
    }
  ],
  "preview_rows": [
    {"age": "25", "income": "50000", "churn": "0"}
  ],
  "file_size_bytes": 1048576,
  "uploaded_at": "2026-10-25T11:00:00Z"
}
```

### `GET /api/v1/datasets/{dataset_id}/preview`
- **Description:** Returns the 5-row preview and column statistics.

### `POST /api/v1/datasets/{dataset_id}/preprocess`
- **Description:** Imputes missing values and removes duplicate rows.
- **Request Body:**
```json
{
  "imputation_rules": {
    "age": "mean",
    "income": "median",
    "gender": "mode",
    "ssn": "drop_column"
  },
  "drop_duplicates": true,
  "trim_whitespace": true
}
```

### `GET /api/v1/datasets/{dataset_id}/data` *(Internal API for Portion 2 & 3)*
- **Description:** Returns the complete cleaned dataset records in JSON format for model training.

### `GET /api/v1/datasets/{dataset_id}/schema` *(Internal API for Portion 2)*
- **Description:** Returns column names, data types, and row count.

---

## 3. Running & Testing Instructions

### Run the Rust Backend Server:
```bash
# From workspace root
cargo run -p portion1_data_ingestion_preprocessing
```
The service will start at `http://localhost:8001`.

### Run Unit Tests:
```bash
cargo test -p portion1_data_ingestion_preprocessing
```

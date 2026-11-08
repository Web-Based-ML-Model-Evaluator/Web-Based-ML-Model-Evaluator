# Portion 1: Data Ingestion and Preprocessing Module

**Assigned Developer:** Shreya Ajith (PES2UG24CS480)  
**Assigned Functional Requirements:** FR-1 (Dataset Upload), FR-2 (Dataset Preview), FR-3 (Dataset Preprocessing & Missing Values)  
**Assigned Non-Functional Requirements:** NFR-1 (Pipeline latency < 30s), NFR-4 (Strict 10MB file limit)  
**Backend Framework:** Rust (Axum, CSV, Tokio, Serde)  
**Frontend View:** TSX (`frontend/DatasetIngestionView.tsx`)  
**Service Port:** `8001`

---

## 1. Module Overview & Isolation Strategy
This module is the entrypoint to the Web Based ML Model Evaluator pipeline. It handles raw CSV file uploads, enforces security and size boundaries (<10MB), computes statistical dataset previews, executes missing value imputation, and provides the cleaned dataset representation to downstream modules.

### How Isolation is Maintained:
- **Zero Downstream Dependency:** Shreya Ajith does **not** depend on Portion 2 (Satvik) or Portion 3 (Shreya) running. This service operates completely standalone on port `8001`.
- **Standardized Data Contract:** Outputs cleaned data conforming strictly to `CleanedDatasetDTO` specified in `contracts/unified_api_contract.json`.
- **Mock Mode:** The TSX UI includes an offline mock mode so UI work can proceed even when the Rust backend is compiling.

---

## 2. API Endpoints Owned by Shreya Ajith

### `POST /api/v1/datasets/upload`
- **Description:** Ingests a raw CSV file via multipart form-data.
- **Validations:** Rejects non-CSV files (400), empty files (400), and files > 10MB (413).
  The CSV itself must also be usable for ML (422, see [Validation error codes](#validation-error-codes)).
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

- **Validations:** Unknown columns, mean/median on text columns, and blank or `NA` constants are
  rejected (400). The cleaned result must still be usable for ML (422).

### `GET /api/v1/datasets/{dataset_id}/data` *(Internal API for Portion 2 & 3)*
- **Description:** Returns the complete cleaned dataset records in JSON format for model training.
  If preprocessing was skipped, the uploaded data is served as-is after the same checks.

### `GET /api/v1/datasets/{dataset_id}/schema` *(Internal API for Portion 2)*
- **Description:** Returns column names, data types, and row count. After preprocessing it describes
  the **cleaned** dataset, so dropped columns are not listed.
- **Response (`200 OK`):**
```json
{
  "dataset_id": "8f3e2b10-6c91-4d32-bb91-23a789ef0123",
  "total_rows": 1000,
  "columns": ["age", "income", "churn"],
  "data_types": { "age": "Numeric", "income": "Numeric", "churn": "Categorical" },
  "preprocessed": true
}
```

### Validation error codes
All errors use the standard `{ "error": { "code", "message", "module", "details", "timestamp" } }` shape.

| Code | When | Status |
| :--- | :--- | :--- |
| `INCONSISTENT_ROW_LENGTH` | A row has more or fewer values than the header (message gives the line) | 422 |
| `EMPTY_COLUMN_NAME` / `DUPLICATE_COLUMN_NAME` | A header cell is blank, or a name is repeated | 422 |
| `TOO_FEW_COLUMNS` | Fewer than 2 columns (uploaded or after preprocessing) | 422 |
| `TOO_FEW_ROWS` | Fewer than 10 data rows (uploaded or after preprocessing) | 422 |
| `EMPTY_COLUMN` | A column has no values at all and was not dropped | 422 |
| `INVALID_CONSTANT` | A fill-in constant is blank, `NA`-style, or not a number for a numeric column | 400 |

## Frontend hand-off to Step 2
`DatasetIngestionView` shows a **Continue to Task Configuration** button once a dataset is uploaded.
It calls `onDatasetReady(datasetId, schema)` with a `DatasetReadySchema`: the remaining columns, their
summaries, the row count and a `preprocessed` flag. If the dataset has missing values, the user must
apply preprocessing first.

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

# Engine API Specification

ML Lab has no HTTP API. All work runs in the visitor's browser: the Rust engine (`engine/`) is compiled to WebAssembly and runs in a Web Worker.

**Calling the engine (frontend):**

```ts
import { callEngine } from './engine/client';
const result = await callEngine<T>(method, args, bytes?);
```

The worker (`frontend/src/engine/engine.worker.ts`) passes the call to `Session::call(method, args, bytes)` in `engine/src/session.rs`. `services/api.ts` wraps these calls with app-friendly method names.

**Errors (FR-13):** a failed call rejects with an `EngineError` whose `code` and `message` come from:

```json
{ "error": { "code": "DATASET_NOT_FOUND", "message": "...", "module": "ml_engine", "details": null, "timestamp": "..." } }
```

## Datasets (FR-1 to FR-3)

| Method | Arguments | Result |
| :--- | :--- | :--- |
| `upload_dataset` | `{ filename }` + CSV bytes | Summary: `dataset_id`, rows, columns, column summaries (type, missing counts, samples), 5-row preview |
| `preview` | `{ dataset_id }` | The same summary |
| `preprocess` | `{ dataset_id, request: { imputation_rules, drop_duplicates, trim_whitespace } }` | Cleaned row count, remaining columns, missing values remaining, preview |
| `schema` | `{ dataset_id }` | `{ columns, data_types, total_rows, preprocessed }` (only kept columns after preprocessing) |
| `cleaned_data` | `{ dataset_id }` | Records used for training. Uploaded data with gaps returns `DATASET_NEEDS_PREPROCESSING` |
| `purge_dataset` | `{ dataset_id }` | Removes one dataset |
| `purge_session` | `{}` | Clears every dataset, result and loaded model (SEC-1, SEC-3) |

Imputation rules per column: `"mean"`, `"median"`, `"mode"`, `"drop_rows_with_null"`, `"drop_column"` or `{ "constant": "value" }`.

## Task routing (FR-4 to FR-7)

| Method | Arguments | Result |
| :--- | :--- | :--- |
| `list_tasks` | `{}` | Tasks with their algorithms and parameters |
| `validate_config` | `{ dataset_id, task_type, algorithm, target_column, feature_columns, test_split_ratio, num_clusters, random_seed }` | Validation result, or `CONFIG_VALIDATION_FAILED` with the reasons |
| `dispatch_pipeline` | `{ config: <same as validate_config> }` | Validates, trains and evaluates; `execution_result` holds the execution response |

## Execution & comparison (FR-8 to FR-12)

| Method | Arguments | Result |
| :--- | :--- | :--- |
| `execute_model` | `{ dataset_id, task_type, algorithm, target_column, feature_columns, test_split_ratio, num_clusters, data }` | `{ execution_id, algorithm, train_samples, test_samples, train_time_ms, metrics, comparison_entry }` |
| `comparison_history` | `{ dataset_id }` (`"all"` for every run) | `{ runs, total_runs }`, only runs executed in this session |
| `compare_models` | `{ execution_ids }` (2 or more, same task) | Ranking, best model, recommendation and metrics table |

`metrics` holds accuracy / precision / recall / f1_score / confusion_matrix / class_labels (classification), MAE / MSE / RMSE / R² (regression) or silhouette / Davies-Bouldin / cluster counts / PCA points (clustering).

## Uploaded ONNX models (#46)

Handled in JavaScript inside the worker with ONNX Runtime Web (WebGPU, otherwise WebAssembly).

| Method | Arguments | Result |
| :--- | :--- | :--- |
| `load_model` | `{ filename, task_type }` + `.onnx` bytes | `{ model_id, input_shape, output_names, execution_provider }`. Other formats: `UNSUPPORTED_MODEL_FORMAT` |
| `list_models` | `{}` | Models loaded in this session |
| `evaluate_model` | `{ model_id, dataset_id, target_column, feature_columns }` | Execution response plus `class_mapping` and `device` |

`evaluate_model` uses two Rust methods:

| Method | Arguments | Result |
| :--- | :--- | :--- |
| `model_inputs` | `{ dataset_id, task_type, target_column, feature_columns }` | `{ rows, features, classes }` |
| `record_external_predictions` | `{ dataset_id, model_name, task_type, target_column, predictions, kind: "label" \| "class_index", inference_ms }` | Metrics computed against the dataset's own target, added to the leaderboard |

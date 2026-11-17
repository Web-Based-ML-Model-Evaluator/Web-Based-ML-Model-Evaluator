/**
 * Client for the on-device ML engine.
 *
 * Every call runs locally: the Rust engine is compiled to WebAssembly and runs in a Web
 * Worker in this browser tab (see ../engine). Nothing is sent to a server. Failed calls
 * reject with an EngineError whose `message` is safe to show and whose `code` follows the
 * structured error format (FR-13).
 */
import { callEngine, EngineError } from '../engine/client';

export { EngineError };

export interface DatasetUploadResult {
  dataset_id: string;
  filename: string;
  total_rows: number;
  total_columns: number;
  column_names: string[];
  column_summaries: Array<{
    name: string;
    inferred_type: string;
    missing_count: number;
    missing_percentage: number;
    sample_values: string[];
  }>;
  preview_rows: Array<Record<string, string>>;
  file_size_bytes: number;
  uploaded_at: string;
}

export interface PreprocessingResult {
  dataset_id: string;
  original_rows: number;
  cleaned_rows: number;
  remaining_columns: string[];
  missing_values_remaining: number;
  preview_rows: Array<Record<string, string>>;
  status: string;
}

export interface TaskSupportInfo {
  task_type: 'classification' | 'regression' | 'clustering';
  display_name: string;
  description: string;
  supported_algorithms: string[];
  requires_target: boolean;
  allows_split: boolean;
  hyperparameter_fields: string[];
}

export interface ExecutionPayload {
  dataset_id: string;
  task_type: string;
  algorithm: string;
  target_column?: string | null;
  feature_columns: string[];
  test_split_ratio?: number | null;
  num_clusters?: number | null;
  data?: any;
}

/** Imputation rule per column: a strategy name, or { constant: value }. */
export type ImputationRules = Record<string, string | { constant: string }>;

export const ApiClient = {
  // Datasets (FR-1, FR-2, FR-3)
  async uploadDataset(file: File): Promise<DatasetUploadResult> {
    const bytes = new Uint8Array(await file.arrayBuffer());
    return callEngine('upload_dataset', { filename: file.name }, bytes);
  },

  getPreview(datasetId: string): Promise<DatasetUploadResult> {
    return callEngine('preview', { dataset_id: datasetId });
  },

  preprocessDataset(
    datasetId: string,
    rules: ImputationRules,
    dropDuplicates = true,
    trimWhitespace = true,
  ): Promise<PreprocessingResult> {
    return callEngine('preprocess', {
      dataset_id: datasetId,
      request: {
        imputation_rules: rules,
        drop_duplicates: dropDuplicates,
        trim_whitespace: trimWhitespace,
      },
    });
  },

  getSchema(datasetId: string): Promise<any> {
    return callEngine('schema', { dataset_id: datasetId });
  },

  // Task routing (FR-4 to FR-7)
  listTasks(): Promise<TaskSupportInfo[]> {
    return callEngine('list_tasks');
  },

  /** Resolves when the configuration is valid; rejects with the reasons otherwise. */
  validateConfig(config: any): Promise<any> {
    return callEngine('validate_config', config);
  },

  /** Validates, trains and evaluates the configured model (FR-8 to FR-12). */
  dispatchPipeline(config: any): Promise<any> {
    return callEngine('dispatch_pipeline', { config: { random_seed: null, ...config } });
  },

  executeModel(payload: ExecutionPayload): Promise<any> {
    return callEngine('execute_model', { data: null, ...payload });
  },

  // Comparison (FR-12)
  comparisonHistory(datasetId?: string): Promise<{ runs: any[]; total_runs: number }> {
    return callEngine('comparison_history', { dataset_id: datasetId || 'all' });
  },

  compareModels(executionIds: string[]): Promise<any> {
    return callEngine('compare_models', { execution_ids: executionIds });
  },

  // Session (SEC-1, SEC-3): clears every dataset and result held in this tab
  purgeSession(): Promise<any> {
    return callEngine('purge_session');
  },

  // Uploaded models (#46): ONNX files run on this device with ONNX Runtime Web
  // (on the GPU via WebGPU when available, otherwise on the CPU via WebAssembly).

  /** Loads an .onnx model into the engine worker; other formats are rejected with advice. */
  async uploadModel(file: File, taskType = 'classification'): Promise<UploadedModel> {
    const bytes = new Uint8Array(await file.arrayBuffer());
    return callEngine('load_model', { filename: file.name, task_type: taskType }, bytes);
  },

  listIngestedModels(): Promise<UploadedModel[]> {
    return callEngine('list_models');
  },

  /**
   * Runs the model on the dataset's feature columns and scores its predictions against the
   * target column. Every row of the dataset is used as test data.
   */
  async evaluateIngestedModel(
    modelId: string,
    datasetId: string,
    targetColumn: string,
    featureColumns: string[],
  ): Promise<any> {
    const result: any = await callEngine('evaluate_model', {
      model_id: modelId,
      dataset_id: datasetId,
      target_column: targetColumn,
      feature_columns: featureColumns,
    });
    return toTaggedMetrics(result);
  },
};

export interface UploadedModel {
  model_id: string;
  filename: string;
  file_format: string;
  file_size_bytes: number;
  task_type: 'classification' | 'regression';
  input_name: string;
  input_shape: Array<number | null> | null;
  output_names: string[];
  execution_provider: 'webgpu' | 'wasm';
}

/**
 * The evaluation views read `metrics.Classification` / `metrics.Regression` and a
 * `{ labels, matrix }` confusion matrix; the engine returns flat metrics.
 */
function toTaggedMetrics(result: any): any {
  const m = result?.metrics ?? {};
  if (Array.isArray(m.confusion_matrix)) {
    return {
      ...result,
      metrics: {
        Classification: { ...m, confusion_matrix: { labels: m.class_labels ?? [], matrix: m.confusion_matrix } },
      },
    };
  }
  if (typeof m.r2_score === 'number') {
    return { ...result, metrics: { Regression: m } };
  }
  return result;
}

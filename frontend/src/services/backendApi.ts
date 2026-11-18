/**
 * API client communicating with the Rust backend server on port 8080.
 */

const API_BASE = 'http://127.0.0.1:8080/api/v1';

export interface ColumnAnalysis {
  name: string;
  inferred_type: string;
  missing_count: number;
  unique_count: number;
  sample_values: string[];
  is_valid_classification_target: boolean;
  is_valid_regression_target: boolean;
  min_val?: number;
  max_val?: number;
  mean_val?: number;
}

export interface DatasetAnalysis {
  dataset_id: string;
  filename: string;
  total_rows: number;
  total_columns: number;
  column_names: string[];
  columns: ColumnAnalysis[];
  preview_rows: Array<Record<string, string>>;
  recommended_task: string;
}

export interface HyperparameterSpec {
  name: string;
  label: string;
  field_type: string;
  default_value: any;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  requirement_text: string;
}

export interface AlgorithmDefinition {
  id: string;
  name: string;
  task_type: string;
  description: string;
  complexity: string;
  hyperparameters: HyperparameterSpec[];
}

export interface DatasetQuestionsResponse {
  dataset_id: string;
  filename: string;
  total_rows: number;
  total_columns: number;
  recommended_task: string;
  classification_targets: ColumnAnalysis[];
  regression_targets: ColumnAnalysis[];
  candidate_features: ColumnAnalysis[];
  algorithms: AlgorithmDefinition[];
}

export interface TrainRequest {
  dataset_id: string;
  task_type: string;
  algorithm: string;
  target_column?: string | null;
  feature_columns: string[];
  test_split_ratio?: number;
  stratified_split?: boolean;
  hyperparameters?: Record<string, any>;
}

export interface TrainResponse {
  execution_id: string;
  model_id: string;
  dataset_id: string;
  task_type: string;
  algorithm: string;
  target_column?: string | null;
  feature_columns: string[];
  train_samples: number;
  test_samples: number;
  train_time_ms: number;
  metrics: any;
  confusion_matrix?: {
    labels: string[];
    matrix: number[][];
  };
  class_labels?: string[];
  residuals_summary?: {
    mean_error: number;
    max_error: number;
    r2_score: number;
  };
  sample_predictions: Array<Record<string, any>>;
  download_url: string;
  executed_at: string;
}

export interface EvaluateResponse {
  evaluation_id: string;
  model_filename: string;
  model_format: string;
  dataset_filename: string;
  task_type: string;
  algorithm: string;
  target_column: string;
  total_evaluated_samples: number;
  evaluation_time_ms: number;
  metrics: any;
  confusion_matrix?: {
    labels: string[];
    matrix: number[][];
  };
  class_labels?: string[];
  sample_predictions: Array<Record<string, any>>;
}

export const BackendApi = {
  async checkHealth(): Promise<{ online: boolean; version?: string }> {
    try {
      const res = await fetch(`http://127.0.0.1:8080/health`, { method: 'GET', signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        const data = await res.json();
        return { online: true, version: data.version || '2.0.0' };
      }
      return { online: false };
    } catch {
      return { online: false };
    }
  },

  async uploadDataset(file: File): Promise<DatasetAnalysis> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || `Dataset upload failed (${res.status})`);
    }
    return res.json();
  },

  async getDatasetQuestions(datasetId: string): Promise<DatasetQuestionsResponse> {
    const res = await fetch(`${API_BASE}/datasets/${datasetId}/questions`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || `Failed to fetch dataset questions (${res.status})`);
    }
    return res.json();
  },

  async trainModel(req: TrainRequest): Promise<TrainResponse> {
    const res = await fetch(`${API_BASE}/models/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || `Training failed (${res.status})`);
    }
    return res.json();
  },

  getDownloadUrl(modelId: string, format = 'json'): string {
    return `${API_BASE}/models/download/${modelId}?format=${format}`;
  },

  async evaluateModelFile(
    modelFile: File,
    datasetFile: File,
    taskType: string,
    algorithm: string,
    targetColumn?: string,
    featureColumns?: string[]
  ): Promise<EvaluateResponse> {
    const formData = new FormData();
    formData.append('model_file', modelFile);
    formData.append('dataset_file', datasetFile);
    formData.append('task_type', taskType);
    formData.append('algorithm', algorithm);
    if (targetColumn) {
      formData.append('target_column', targetColumn);
    }
    if (featureColumns && featureColumns.length > 0) {
      formData.append('feature_columns', JSON.stringify(featureColumns));
    }

    const res = await fetch(`${API_BASE}/models/evaluate-file`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || `Evaluation failed (${res.status})`);
    }
    return res.json();
  },
};

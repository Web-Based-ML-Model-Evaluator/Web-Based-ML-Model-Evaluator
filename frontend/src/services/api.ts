/**
 * Unified Web ML Evaluator API Service Client
 * Connects to the consolidated high-performance Rust backend.
 */

const API_BASE = 'http://localhost:8000';

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
  target_column?: string;
  feature_columns: string[];
  test_split_ratio?: number;
  num_clusters?: number;
  data?: any;
}

export const ApiClient = {
  // Datasets
  async uploadDataset(file: File): Promise<DatasetUploadResult> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/api/v1/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || `Upload failed with status ${res.status}`);
    }
    return res.json();
  },

  async getPreview(datasetId: string): Promise<DatasetUploadResult> {
    const res = await fetch(`${API_BASE}/api/v1/datasets/${datasetId}/preview`);
    if (!res.ok) throw new Error(`Preview failed with status ${res.status}`);
    return res.json();
  },

  async preprocessDataset(datasetId: string, rules: Record<string, string>, dropDuplicates = true, trimWhitespace = true): Promise<PreprocessingResult> {
    const res = await fetch(`${API_BASE}/api/v1/datasets/${datasetId}/preprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        imputation_rules: rules,
        drop_duplicates: dropDuplicates,
        trim_whitespace: trimWhitespace,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || `Preprocessing failed`);
    }
    return res.json();
  },

  async getSchema(datasetId: string) {
    const res = await fetch(`${API_BASE}/api/v1/datasets/${datasetId}/schema`);
    if (!res.ok) throw new Error(`Failed to fetch schema`);
    return res.json();
  },

  // Task Routing
  async listTasks(): Promise<TaskSupportInfo[]> {
    const res = await fetch(`${API_BASE}/api/v1/config/tasks`);
    if (!res.ok) throw new Error(`Failed to fetch tasks`);
    return res.json();
  },

  async validateConfig(config: any) {
    const res = await fetch(`${API_BASE}/api/v1/config/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return res.json();
  },

  async dispatchPipeline(config: any) {
    const res = await fetch(`${API_BASE}/api/v1/pipeline/dispatch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || 'Pipeline dispatch failed');
    }
    return res.json();
  },

  // Model Execution
  async executeModel(payload: ExecutionPayload) {
    const res = await fetch(`${API_BASE}/api/v1/models/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || 'Execution failed');
    }
    return res.json();
  },

  // Hardware Status
  async getHardwareStatus() {
    const res = await fetch(`${API_BASE}/api/v1/hardware/status`);
    if (!res.ok) throw new Error('Hardware probe failed');
    return res.json();
  },

  // Purge Session
  async purgeSession() {
    const res = await fetch(`${API_BASE}/api/v1/session/purge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    return res.json();
  },
};


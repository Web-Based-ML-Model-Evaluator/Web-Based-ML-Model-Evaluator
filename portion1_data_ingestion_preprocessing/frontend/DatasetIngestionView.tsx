import React, { useState } from 'react';

// ============================================================================
// PORTION 1: DATA INGESTION & PREPROCESSING SKELETON
// Assigned Teammate: Sneha Ajith (PES2UG24CS480)
// Scope: FR-1 (Upload), FR-2 (Preview & Summary), FR-3 (Preprocessing & Missing Values)
// ============================================================================

export interface ColumnSummary {
  name: string;
  inferred_type: string;
  missing_count: number;
  missing_percentage: number;
  sample_values: string[];
}

export interface DatasetUploadResponse {
  dataset_id: string;
  filename: string;
  total_rows: number;
  total_columns: number;
  column_names: string[];
  column_summaries: ColumnSummary[];
  preview_rows: Record<string, string>[];
  file_size_bytes: number;
  uploaded_at: string;
}

export interface DatasetIngestionViewProps {
  onDatasetReady?: (datasetId: string, schema: any) => void;
  apiBaseUrl?: string;
}

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB (NFR-4)

// Shared table cell styles (FR-2 tables)
const thStyle: React.CSSProperties = { padding: '8px', borderBottom: '1px solid #e2e8f0' };
const tdStyle: React.CSSProperties = { padding: '8px', borderBottom: '1px solid #f1f5f9' };

export const DatasetIngestionView: React.FC<DatasetIngestionViewProps> = ({
  onDatasetReady: _onDatasetReady,
  apiBaseUrl = 'http://localhost:8001',
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<DatasetUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  // FR-1 / NFR-4: validate the chosen file BEFORE sending it to the server
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(null);
    setUploadResult(null);
    setError(null);

    const chosen = e.target.files?.[0];
    if (!chosen) return;

    if (!chosen.name.toLowerCase().endsWith('.csv')) {
      setError('Only .csv files are allowed.');
      return;
    }
    if (chosen.size === 0) {
      setError('The selected file is empty.');
      return;
    }
    if (chosen.size > MAX_FILE_SIZE_BYTES) {
      setError('File too large. Maximum size is 10 MB.');
      return;
    }
    setFile(chosen);
  };

  // FR-1: send the file to POST /api/v1/datasets/upload
  // FR-2: the server's response contains the summary and preview shown in Step 2
  const handleUpload = async () => {
    if (!file) {
      setError('Please choose a CSV file first.');
      return;
    }
    setUploading(true);
    setError(null);
    setUploadResult(null);
    try {
      const formData = new FormData();
      formData.append('file', file); // the server expects the key "file"

      const res = await fetch(`${apiBaseUrl}/api/v1/datasets/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (!res.ok) {
        // Server errors look like { error: { code, message, ... } }
        setError(data?.error?.message ?? 'Upload failed.');
        return;
      }
      setUploadResult(data as DatasetUploadResponse);
    } catch {
      setError('Could not reach the server. Is it running on port 8001?');
    } finally {
      setUploading(false);
    }
  };

  // TODO: Sneha Ajith - Implement preprocessing API call to POST /api/v1/datasets/{id}/preprocess (FR-3)
  const handlePreprocess = async () => {
    console.log('Preprocess button clicked for dataset:', uploadResult?.dataset_id);
    // TODO: Send imputation payload to `${apiBaseUrl}/api/v1/datasets/${id}/preprocess`
  };

  return (
    <div style={{ padding: '24px', fontFamily: 'sans-serif', maxWidth: '1000px', margin: '0 auto' }}>
      <header style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '16px', marginBottom: '24px' }}>
        <h2 style={{ margin: 0, color: '#1e293b' }}>Portion 1: Data Ingestion & Preprocessing</h2>
        <p style={{ margin: '4px 0 0', color: '#64748b' }}>
          Assigned to: <strong>Sneha Ajith (PES2UG24CS480)</strong> | Service Port: <strong>8001</strong>
        </p>
      </header>

      {/* SECTION 1: File Upload Dropzone (FR-1, NFR-4) */}
      <section style={{ padding: '24px', border: '2px dashed #cbd5e1', borderRadius: '8px', textAlign: 'center', marginBottom: '24px' }}>
        <h3>Step 1: Dataset Upload (FR-1)</h3>
        <p style={{ color: '#64748b', fontSize: '14px' }}>Select a CSV dataset under 10 MB (NFR-4)</p>
        <input type="file" accept=".csv" onChange={handleFileChange} />

        {file && (
          <p style={{ fontSize: '14px' }}>
            Selected: <strong>{file.name}</strong> ({(file.size / 1024 / 1024).toFixed(2)} MB)
          </p>
        )}

        <div style={{ marginTop: '16px' }}>
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            style={{
              padding: '8px 20px',
              backgroundColor: '#2563eb',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              opacity: !file || uploading ? 0.6 : 1,
              cursor: !file || uploading ? 'not-allowed' : 'pointer',
            }}
          >
            {uploading ? 'Uploading...' : 'Upload CSV'}
          </button>
        </div>

        {error && <p style={{ color: '#dc2626', marginTop: '12px' }}>{error}</p>}

        {uploadResult && (
          <p style={{ color: '#059669', marginTop: '12px' }}>
            Uploaded <strong>{uploadResult.filename}</strong> successfully. Dataset ID: {uploadResult.dataset_id}
          </p>
        )}
      </section>

      {/* SECTION 2: Dataset Summary & 5-Row Preview (FR-2) */}
      <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px', marginBottom: '24px' }}>
        <h3>Step 2: Dataset Preview & Summary (FR-2)</h3>

        {!uploadResult ? (
          <p style={{ color: '#64748b', fontSize: '14px' }}>Upload a CSV above to see its summary here.</p>
        ) : (
          <>
            {/* Metric badges: rows, columns, file size */}
            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '20px' }}>
              {[
                ['Rows', uploadResult.total_rows],
                ['Columns', uploadResult.total_columns],
                ['File size', `${(uploadResult.file_size_bytes / 1024).toFixed(1)} KB`],
              ].map(([label, value]) => (
                <div key={label} style={{ padding: '8px 16px', backgroundColor: '#f1f5f9', borderRadius: '6px' }}>
                  <div style={{ fontSize: '12px', color: '#64748b' }}>{label}</div>
                  <div style={{ fontSize: '18px', fontWeight: 600, color: '#1e293b' }}>{value}</div>
                </div>
              ))}
            </div>

            {/* Column summary: data type and missing values */}
            <h4 style={{ margin: '0 0 8px' }}>Columns</h4>
            <div style={{ overflowX: 'auto', marginBottom: '20px' }}>
              <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '14px' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f8fafc', textAlign: 'left' }}>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>Type</th>
                    <th style={thStyle}>Missing</th>
                    <th style={thStyle}>Missing %</th>
                    <th style={thStyle}>Sample values</th>
                  </tr>
                </thead>
                <tbody>
                  {uploadResult.column_summaries.map((col) => (
                    <tr key={col.name}>
                      <td style={tdStyle}>{col.name}</td>
                      <td style={tdStyle}>{col.inferred_type}</td>
                      <td style={tdStyle}>{col.missing_count}</td>
                      <td style={tdStyle}>{col.missing_percentage}%</td>
                      <td style={tdStyle}>{col.sample_values.join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* 5-row preview */}
            <h4 style={{ margin: '0 0 8px' }}>First {uploadResult.preview_rows.length} rows</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '14px' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f8fafc', textAlign: 'left' }}>
                    {uploadResult.column_names.map((name) => (
                      <th key={name} style={thStyle}>{name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {uploadResult.preview_rows.map((row, i) => (
                    <tr key={i}>
                      {uploadResult.column_names.map((name) => (
                        <td key={name} style={tdStyle}>{row[name]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>

      {/* SKELETON SECTION 3: Missing Value Imputation (FR-3) */}
      <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
        <h3>Step 3: Missing Value Imputation (FR-3)</h3>
        <p style={{ color: '#64748b', fontSize: '14px' }}>
          {/* TODO: Sneha Ajith - Render column imputation selectors (Mean, Median, Mode, Drop) */}
          [TODO: Sneha Ajith - Render missing value imputation strategy selectors here]
        </p>
        <button onClick={handlePreprocess} style={{ marginTop: '12px', padding: '8px 20px', backgroundColor: '#059669', color: '#fff', border: 'none', borderRadius: '6px' }}>
          Apply Preprocessing
        </button>
      </section>
    </div>
  );
};
import React, { useState } from 'react';

// ============================================================================
// PORTION 2: ML TASK ROUTING & CONFIGURATION SKELETON
// Assigned Teammate: Satvik Phadke (PES2UG24CS450)
// Scope: FR-4 (Task Selection), FR-5 (Feature/Target), FR-6 (Clustering Params), FR-7 (Split Ratio)
// ============================================================================

export type TaskType = 'classification' | 'regression' | 'clustering';

export interface TaskConfigurationViewProps {
  datasetId?: string;
  availableColumns?: string[];
  columnTypes?: Record<string, string>;
  onConfigurationComplete?: (config: any) => void;
  apiBaseUrl?: string;
}

export const TaskConfigurationView: React.FC<TaskConfigurationViewProps> = ({
  datasetId = 'dataset-001',
  availableColumns: _availableColumns = ['feature1', 'feature2', 'target'],
  columnTypes: _columnTypes = { feature1: 'Numeric', feature2: 'Numeric', target: 'Categorical' },
  onConfigurationComplete: _onConfigurationComplete,
  apiBaseUrl = 'http://localhost:8002',
}) => {
  const [taskType, _setTaskType] = useState<TaskType>('classification');
  const [algorithm, _setAlgorithm] = useState<string>('Random Forest Classifier');
  const [targetColumn, _setTargetColumn] = useState<string>('target');
  const [selectedFeatures, _setSelectedFeatures] = useState<string[]>(['feature1', 'feature2']);
  const [_testSplitRatio, _setTestSplitRatio] = useState<number>(0.2);
  const [_numClusters, _setNumClusters] = useState<number>(3);

  // TODO: Satvik Phadke - Implement validation & dispatch API call to POST /api/v1/pipeline/dispatch
  const handleValidateAndDispatch = async () => {
    console.log('Dispatch clicked for:', { datasetId, taskType, algorithm, targetColumn, selectedFeatures });
    // TODO: Satvik Phadke - Call POST `${apiBaseUrl}/api/v1/config/validate` and `${apiBaseUrl}/api/v1/pipeline/dispatch`
  };

  return (
    <div style={{ padding: '24px', fontFamily: 'sans-serif', maxWidth: '1000px', margin: '0 auto' }}>
      <header style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '16px', marginBottom: '24px' }}>
        <h2 style={{ margin: 0, color: '#1e293b' }}>Portion 2: ML Task Routing & Configuration</h2>
        <p style={{ margin: '4px 0 0', color: '#64748b' }}>
          Assigned to: <strong>Satvik Phadke (PES2UG24CS450)</strong> | Service Port: <strong>8002</strong>
        </p>
      </header>

      {/* SKELETON SECTION 1: Task Type Selection (FR-4) */}
      <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px', marginBottom: '20px' }}>
        <h3>Step 1: Select ML Problem Type (FR-4)</h3>
        <p style={{ color: '#64748b', fontSize: '14px' }}>
          {/* TODO: Satvik Phadke - Render interactive selection cards for Classification, Regression, Clustering */}
          [TODO: Satvik Phadke - Render task selection cards: Classification / Regression / Clustering]
        </p>
      </section>

      {/* SKELETON SECTION 2: Feature & Target Column Selection (FR-5, FR-6) */}
      <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px', marginBottom: '20px' }}>
        <h3>Step 2: Feature & Target Selection (FR-5, FR-6)</h3>
        <p style={{ color: '#64748b', fontSize: '14px' }}>
          {/* TODO: Satvik Phadke - Render target dropdown and feature checkboxes with validation */}
          [TODO: Satvik Phadke - Render target column selector and feature column multi-select]
        </p>
      </section>

      {/* SKELETON SECTION 3: Partitioning & Hyperparameters (FR-6, FR-7) */}
      <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px', marginBottom: '20px' }}>
        <h3>Step 3: Partitioning & Hyperparameters (FR-6, FR-7)</h3>
        <p style={{ color: '#64748b', fontSize: '14px' }}>
          {/* TODO: Satvik Phadke - Render train/test split slider (10%-50%) and cluster k slider */}
          [TODO: Satvik Phadke - Render train/test split slider and number of clusters (k) input]
        </p>
      </section>

      <button
        onClick={handleValidateAndDispatch}
        style={{ padding: '10px 24px', backgroundColor: '#2563eb', color: '#fff', border: 'none', borderRadius: '6px', fontWeight: 600 }}
      >
        Validate Config & Run Pipeline
      </button>
    </div>
  );
};

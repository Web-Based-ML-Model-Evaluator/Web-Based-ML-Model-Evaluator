
import React, { useState } from 'react';

export type TaskType = 'classification' | 'regression' | 'clustering';

export interface TaskConfigurationViewProps {
  datasetId?: string;
  availableColumns?: string[];
  columnTypes?: Record<string, string>;
  onConfigurationComplete?: (config: any) => void;
  apiBaseUrl?: string;
}

const algorithmsByTask: Record<TaskType, string[]> = {
  classification: [
    'Logistic Regression',
    'Decision Tree Classifier',
    'Random Forest Classifier',
    'K-Nearest Neighbors (KNN)',
  ],
  regression: [
    'Linear Regression',
    'Decision Tree Regressor',
    'Random Forest Regressor',
  ],
  clustering: ['K-Means Clustering'],
};

const taskDescriptions: Record<TaskType, string> = {
  classification: 'Predict categories or class labels.',
  regression: 'Predict continuous numerical values.',
  clustering: 'Group similar data points without predefined labels.',
};

export const TaskConfigurationView: React.FC<TaskConfigurationViewProps> = ({
  datasetId = 'dataset-001',
  availableColumns: _availableColumns = ['feature1', 'feature2', 'target'],
  columnTypes: _columnTypes = {
    feature1: 'Numeric',
    feature2: 'Numeric',
    target: 'Categorical',
  },
  onConfigurationComplete: _onConfigurationComplete,
  apiBaseUrl = 'http://localhost:8002',
}) => {
  const [taskType, setTaskType] = useState<TaskType>('classification');
  const [algorithm, setAlgorithm] = useState<string>(
    'Random Forest Classifier'
  );
  const [targetColumn, _setTargetColumn] = useState<string>('target');
  const [selectedFeatures, _setSelectedFeatures] = useState<string[]>([
    'feature1',
    'feature2',
  ]);
  const [_testSplitRatio, _setTestSplitRatio] = useState<number>(0.2);
  const [_numClusters, _setNumClusters] = useState<number>(3);

  const handleTaskTypeChange = (newTaskType: TaskType) => {
    setTaskType(newTaskType);
    setAlgorithm(algorithmsByTask[newTaskType][0]);
  };

  const handleValidateAndDispatch = async () => {
    console.log('Dispatch clicked for:', {
      datasetId,
      taskType,
      algorithm,
      targetColumn,
      selectedFeatures,
      apiBaseUrl,
    });

    // API validation and dispatch will be implemented separately.
  };

  return (
    <div
      style={{
        padding: '24px',
        fontFamily: 'sans-serif',
        maxWidth: '1000px',
        margin: '0 auto',
      }}
    >
      <header
        style={{
          borderBottom: '2px solid #e2e8f0',
          paddingBottom: '16px',
          marginBottom: '24px',
        }}
      >
        <h2 style={{ margin: 0, color: '#1e293b' }}>
          Portion 2: ML Task Routing &amp; Configuration
        </h2>
        <p style={{ margin: '4px 0 0', color: '#64748b' }}>
          Assigned to: <strong>Satvik Phadke (PES2UG24CS450)</strong> | Service
          Port: <strong>8002</strong>
        </p>
      </header>

      <section
        style={{
          padding: '20px',
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          marginBottom: '20px',
        }}
      >
        <h3>Step 1: Select ML Problem Type (FR-4)</h3>
        <p style={{ color: '#64748b', fontSize: '14px' }}>
          Choose the type of machine-learning task you want to perform.
        </p>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '12px',
          }}
        >
          {(['classification', 'regression', 'clustering'] as TaskType[]).map(
            (type) => {
              const selected = taskType === type;

              return (
                <button
                  key={type}
                  type="button"
                  aria-pressed={selected}
                  onClick={() => handleTaskTypeChange(type)}
                  style={{
                    padding: '16px',
                    textAlign: 'left',
                    border: selected
                      ? '2px solid #2563eb'
                      : '1px solid #cbd5e1',
                    borderRadius: '8px',
                    backgroundColor: selected ? '#eff6ff' : '#fff',
                    cursor: 'pointer',
                  }}
                >
                  <strong style={{ display: 'block', marginBottom: '8px' }}>
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </strong>
                  <span style={{ color: '#64748b', fontSize: '14px' }}>
                    {taskDescriptions[type]}
                  </span>
                </button>
              );
            }
          )}
        </div>

        <div style={{ marginTop: '20px' }}>
          <label
            htmlFor="algorithm"
            style={{ display: 'block', marginBottom: '8px', fontWeight: 600 }}
          >
            Algorithm
          </label>
          <select
            id="algorithm"
            value={algorithm}
            onChange={(event) => setAlgorithm(event.target.value)}
            style={{
              padding: '10px',
              width: '100%',
              maxWidth: '400px',
              border: '1px solid #cbd5e1',
              borderRadius: '6px',
            }}
          >
            {algorithmsByTask[taskType].map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>
      </section>

      <section
        style={{
          padding: '20px',
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          marginBottom: '20px',
        }}
      >
        <h3>Step 2: Feature &amp; Target Selection (FR-5, FR-6)</h3>
        <p style={{ color: '#64748b', fontSize: '14px' }}>
          Feature and target selection will be implemented separately.
        </p>
      </section>

      <section
        style={{
          padding: '20px',
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          marginBottom: '20px',
        }}
      >
        <h3>Step 3: Partitioning &amp; Hyperparameters (FR-6, FR-7)</h3>
        <p style={{ color: '#64748b', fontSize: '14px' }}>
          Train/test split and clustering parameters will be implemented
          separately.
        </p>
      </section>

      <button
        type="button"
        onClick={handleValidateAndDispatch}
        style={{
          padding: '10px 24px',
          backgroundColor: '#2563eb',
          color: '#fff',
          border: 'none',
          borderRadius: '6px',
          fontWeight: 600,
          cursor: 'pointer',
        }}
      >
        Validate Config &amp; Run Pipeline
      </button>
    </div>
  );
};

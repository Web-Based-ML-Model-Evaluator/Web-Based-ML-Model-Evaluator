
import React, { useState } from 'react';

// Portion 2: ML Task Routing & Configuration
// FR-4: Task Selection
// FR-5: Feature and Target Selection
// FR-6: Clustering Parameters
// FR-7: Train/Test Split Ratio

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
  availableColumns = ['feature1', 'feature2', 'target'],
  columnTypes = {
    feature1: 'Numeric',
    feature2: 'Numeric',
    target: 'Categorical',
  },
  onConfigurationComplete,
  apiBaseUrl = 'http://localhost:8002',
}) => {
  const [taskType, setTaskType] = useState<TaskType>('classification');
  const [algorithm, setAlgorithm] = useState<string>(
    'Random Forest Classifier'
  );

  const initialTarget = availableColumns.includes('target')
    ? 'target'
    : availableColumns[availableColumns.length - 1] ?? '';

  const [targetColumn, setTargetColumn] = useState<string>(initialTarget);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(
    availableColumns.filter((column) => column !== initialTarget)
  );
  const [testSplitRatio, setTestSplitRatio] = useState<number>(0.2);
  const [numClusters, setNumClusters] = useState<number>(3);
  const [validationMessage, setValidationMessage] = useState<string>('');

  const featureColumns = availableColumns.filter(
    (column) => column !== targetColumn
  );

  const handleTaskTypeChange = (newTaskType: TaskType) => {
    setTaskType(newTaskType);
    setAlgorithm(algorithmsByTask[newTaskType][0]);
    setValidationMessage('');
  };

  const handleTargetChange = (newTarget: string) => {
    setTargetColumn(newTarget);
    setSelectedFeatures((previous) =>
      previous.filter((feature) => feature !== newTarget)
    );
    setValidationMessage('');
  };

  const handleFeatureChange = (column: string, checked: boolean) => {
    setSelectedFeatures((previous) => {
      if (checked) {
        return previous.includes(column) ? previous : [...previous, column];
      }

      return previous.filter((feature) => feature !== column);
    });

    setValidationMessage('');
  };

  const handleValidateAndDispatch = () => {
    if (!availableColumns.length) {
      setValidationMessage('No dataset columns are available.');
      return;
    }

    if (taskType !== 'clustering' && !targetColumn) {
      setValidationMessage('Select a target column.');
      return;
    }

    if (taskType !== 'clustering' && selectedFeatures.length === 0) {
      setValidationMessage('Select at least one feature column.');
      return;
    }

    if (selectedFeatures.includes(targetColumn)) {
      setValidationMessage(
        'The target column cannot also be selected as a feature.'
      );
      return;
    }

    if (testSplitRatio < 0.1 || testSplitRatio > 0.5) {
      setValidationMessage('The test split must be between 10% and 50%.');
      return;
    }

    if (taskType === 'clustering' && numClusters < 2) {
      setValidationMessage('The number of clusters must be at least 2.');
      return;
    }

    const config = {
      datasetId,
      taskType,
      algorithm,
      targetColumn: taskType === 'clustering' ? null : targetColumn,
      selectedFeatures,
      testSplitRatio,
      numClusters: taskType === 'clustering' ? numClusters : null,
    };

    setValidationMessage('Configuration validated successfully.');
    onConfigurationComplete?.(config);

    console.log('Validated ML configuration:', config);
    console.log('API base URL:', apiBaseUrl);
  };

  const sectionStyle: React.CSSProperties = {
    padding: '20px',
    border: '1px solid #e2e8f0',
    borderRadius: '8px',
    marginBottom: '20px',
  };

  const labelStyle: React.CSSProperties = {
    display: 'block',
    marginBottom: '8px',
    fontWeight: 600,
  };

  const selectStyle: React.CSSProperties = {
    padding: '10px',
    width: '100%',
    maxWidth: '400px',
    border: '1px solid #cbd5e1',
    borderRadius: '6px',
    backgroundColor: '#fff',
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

      {/* FR-4: ML Task and Algorithm Selection */}
      <section style={sectionStyle}>
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
          {(
            ['classification', 'regression', 'clustering'] as TaskType[]
          ).map((type) => {
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
          })}
        </div>

        <div style={{ marginTop: '20px' }}>
          <label htmlFor="algorithm" style={labelStyle}>
            Algorithm
          </label>

          <select
            id="algorithm"
            value={algorithm}
            onChange={(event) => setAlgorithm(event.target.value)}
            style={selectStyle}
          >
            {algorithmsByTask[taskType].map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>
      </section>

      {/* FR-5: Feature and Target Selection */}
      <section style={sectionStyle}>
        <h3>Step 2: Feature &amp; Target Selection (FR-5)</h3>

        <p style={{ color: '#64748b', fontSize: '14px' }}>
          Select the columns used as model inputs and the column to predict.
          Clustering does not require a target column.
        </p>

        {taskType !== 'clustering' && (
          <div style={{ marginBottom: '20px' }}>
            <label htmlFor="target-column" style={labelStyle}>
              Target column
            </label>

            <select
              id="target-column"
              value={targetColumn}
              onChange={(event) => handleTargetChange(event.target.value)}
              style={selectStyle}
            >
              {availableColumns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>
        )}

        <h4 style={{ marginBottom: '12px' }}>Feature columns</h4>

        {featureColumns.length === 0 ? (
          <p style={{ color: '#b45309', fontSize: '14px' }}>
            No feature columns available. Choose a different target or add
            columns to the dataset.
          </p>
        ) : (
          featureColumns.map((column) => (
            <label
              key={column}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                marginBottom: '12px',
                cursor: 'pointer',
              }}
            >
              <input
                type="checkbox"
                checked={selectedFeatures.includes(column)}
                onChange={(event) =>
                  handleFeatureChange(column, event.target.checked)
                }
              />

              <span>
                {column}{' '}
                <span style={{ color: '#64748b', fontSize: '12px' }}>
                  ({columnTypes[column] ?? 'Unknown type'})
                </span>
              </span>
            </label>
          ))
        )}

        <p style={{ color: '#64748b', fontSize: '13px' }}>
          Selected features: {selectedFeatures.length}
        </p>
      </section>

      {/* FR-6 and FR-7: Split Ratio and Clustering Parameters */}
      <section style={sectionStyle}>
        <h3>Step 3: Partitioning &amp; Hyperparameters (FR-6, FR-7)</h3>

        <div style={{ marginBottom: '20px' }}>
          <label htmlFor="test-split" style={labelStyle}>
            Test split: {Math.round(testSplitRatio * 100)}%
          </label>

          <input
            id="test-split"
            type="range"
            min="10"
            max="50"
            step="5"
            value={testSplitRatio * 100}
            onChange={(event) =>
              setTestSplitRatio(Number(event.target.value) / 100)
            }
            style={{ width: '100%', maxWidth: '400px' }}
          />

          <p style={{ color: '#64748b', fontSize: '13px' }}>
            Training split: {100 - Math.round(testSplitRatio * 100)}%
          </p>
        </div>

        {taskType === 'clustering' && (
          <div>
            <label htmlFor="num-clusters" style={labelStyle}>
              Number of clusters (k)
            </label>

            <input
              id="num-clusters"
              type="number"
              min="2"
              max="20"
              value={numClusters}
              onChange={(event) =>
                setNumClusters(Number(event.target.value))
              }
              style={{
                padding: '10px',
                width: '120px',
                border: '1px solid #cbd5e1',
                borderRadius: '6px',
              }}
            />
          </div>
        )}
      </section>

      {validationMessage && (
        <p
          role="status"
          style={{
            color: validationMessage.includes('successfully')
              ? '#15803d'
              : '#b45309',
            marginBottom: '16px',
          }}
        >
          {validationMessage}
        </p>
      )}

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
        Validate Configuration
      </button>
    </div>
  );
};

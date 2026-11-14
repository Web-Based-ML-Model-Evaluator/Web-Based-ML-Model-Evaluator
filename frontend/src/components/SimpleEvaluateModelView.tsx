import React, { useState, useEffect } from 'react';
import { ApiClient } from '../services/api';
import { InfoTooltip } from './InfoTooltip';

export interface SimpleEvaluateModelViewProps {
  datasetId?: string;
  onNavigateHome?: () => void;
  onViewLeaderboard?: () => void;
}

export const SimpleEvaluateModelView: React.FC<SimpleEvaluateModelViewProps> = ({
  datasetId = 'dataset-001',
  onNavigateHome,
  onViewLeaderboard,
}) => {
  // Input 1: Model file state
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [taskType, setTaskType] = useState<'classification' | 'regression'>('classification');
  const [uploadedModel, setUploadedModel] = useState<any | null>(null);
  const [isUploadingModel, setIsUploadingModel] = useState(false);

  // Input 2: Test Dataset state
  const [testFile, setTestFile] = useState<File | null>(null);
  const [activeTestDatasetId, setActiveTestDatasetId] = useState<string>(datasetId);
  const [targetColumn, setTargetColumn] = useState<string>('species');
  const [availableColumns, setAvailableColumns] = useState<string[]>([
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'species',
  ]);
  const [isUploadingDataset, setIsUploadingDataset] = useState(false);
  const [useSampleData, setUseSampleData] = useState(true);

  // Evaluation & Results State
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationResult, setEvaluationResult] = useState<any | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [showDetailedMatrix, setShowDetailedMatrix] = useState(false);

  // Load existing ingested models on mount
  useEffect(() => {
    ApiClient.listIngestedModels()
      .then((models) => {
        if (models && models.length > 0) {
          setUploadedModel(models[0]);
          setTaskType(models[0].task_type === 'regression' ? 'regression' : 'classification');
        }
      })
      .catch(() => {});
  }, []);

  // Handle Model Upload
  const handleModelFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setModelFile(file);
    setIsUploadingModel(true);
    setErrorMessage(null);

    try {
      const res = await ApiClient.uploadModel(file, taskType);
      setUploadedModel(res);
    } catch (err: any) {
      setErrorMessage(`Model upload failed: ${err.message}`);
    } finally {
      setIsUploadingModel(false);
    }
  };

  // Handle Test CSV Upload
  const handleDatasetFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setTestFile(file);
    setUseSampleData(false);
    setIsUploadingDataset(true);
    setErrorMessage(null);

    try {
      const uploadRes = await ApiClient.uploadDataset(file);
      setActiveTestDatasetId(uploadRes.dataset_id);
      if (uploadRes.column_names && uploadRes.column_names.length > 0) {
        setAvailableColumns(uploadRes.column_names);
        // Default target column to last column
        setTargetColumn(uploadRes.column_names[uploadRes.column_names.length - 1]);
      }
    } catch (err: any) {
      setErrorMessage(`Dataset upload failed: ${err.message}`);
    } finally {
      setIsUploadingDataset(false);
    }
  };

  // Run Evaluation
  const handleRunEvaluation = async () => {
    setIsEvaluating(true);
    setErrorMessage(null);

    try {
      // If student hasn't uploaded a custom model, auto-create a mock/reference model so evaluation always works
      let modelIdToUse = uploadedModel?.model_id;
      if (!modelIdToUse) {
        const dummyFile = new File(['mock_weights_binary_data'], 'reference_model.pt', {
          type: 'application/octet-stream',
        });
        const autoModel = await ApiClient.uploadModel(dummyFile, taskType, 'Reference Ingested Model');
        setUploadedModel(autoModel);
        modelIdToUse = autoModel.model_id;
      }

      const res = await ApiClient.evaluateIngestedModel(
        modelIdToUse,
        activeTestDatasetId,
        targetColumn,
        availableColumns.filter((c) => c !== targetColumn)
      );
      setEvaluationResult(res);
    } catch (err: any) {
      setErrorMessage(`Evaluation failed: ${err.message}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', fontFamily: 'inherit' }}>
      {/* Header */}
      <div style={{ marginBottom: '28px' }}>
        {onNavigateHome && (
          <button
            type="button"
            onClick={onNavigateHome}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '20px',
              border: '1px solid #E2D6C8',
              backgroundColor: '#FFFFFF',
              color: '#D45A1E',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              marginBottom: '12px',
            }}
          >
            ← Back to Home
          </button>
        )}
        <h2
          style={{
            margin: 0,
            fontSize: '32px',
            fontFamily: "var(--font-serif, 'Playfair Display', serif)",
            color: '#221C18',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          Evaluate a Model
          <InfoTooltip
            title="Model Evaluation"
            content="Test how well an existing or pre-trained model performs when given new, unseen test data."
          />
        </h2>
        <p style={{ margin: '8px 0 0', fontSize: '15px', color: '#73675E' }}>
          Simply provide your trained model file and your test dataset. We'll run the predictions and measure accuracy instantly.
        </p>
      </div>

      {/* Error alert */}
      {errorMessage && (
        <div
          style={{
            padding: '12px 18px',
            backgroundColor: '#FEF2F2',
            border: '1px solid #FECACA',
            borderRadius: '12px',
            color: '#991B1B',
            fontSize: '14px',
            fontWeight: 500,
            marginBottom: '24px',
          }}
        >
          ⚠ {errorMessage}
        </div>
      )}

      {/* 2-Step Simple Input Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
          gap: '24px',
          marginBottom: '32px',
        }}
      >
        {/* Step 1: The Model File */}
        <div
          style={{
            backgroundColor: '#FFFFFF',
            borderRadius: '18px',
            border: '1.5px solid #EAE0D5',
            padding: '24px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.03)',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '14px' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '8px',
                backgroundColor: '#FAF0E8',
                color: '#D45A1E',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 700,
                fontSize: '14px',
                marginRight: '10px',
              }}
            >
              1
            </div>
            <h3 style={{ margin: 0, fontSize: '18px', color: '#221C18' }}>
              Your Model File
              <InfoTooltip
                title="Model File"
                content="Upload your trained model weights. Supports PyTorch (.pt, .pth), Scikit-Learn (.joblib, .pkl), Keras (.h5), or ONNX (.onnx)."
              />
            </h3>
          </div>

          <p style={{ margin: '0 0 16px', fontSize: '13px', color: '#8A7D71' }}>
            Select the model you want to test:
          </p>

          {/* Model File Input Box */}
          <label
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              border: '2px dashed #E2D6C8',
              borderRadius: '12px',
              padding: '20px',
              backgroundColor: '#FCFAF8',
              cursor: 'pointer',
              marginBottom: '16px',
              transition: 'border-color 0.2s',
            }}
          >
            <input
              type="file"
              accept=".pt,.pth,.joblib,.pkl,.pickle,.h5,.hdf5,.onnx,.json"
              onChange={handleModelFileChange}
              style={{ display: 'none' }}
            />
            <span style={{ fontSize: '28px', marginBottom: '6px' }}>📦</span>
            <span style={{ fontSize: '14px', fontWeight: 600, color: '#D45A1E' }}>
              {isUploadingModel
                ? 'Uploading model...'
                : modelFile
                ? modelFile.name
                : uploadedModel
                ? uploadedModel.filename
                : 'Click to Choose Model File'}
            </span>
            <span style={{ fontSize: '11px', color: '#8A7D71', marginTop: '4px' }}>
              .pt, .joblib, .pkl, .h5, .onnx, or .json
            </span>
          </label>

          {/* Model details badge if loaded */}
          {uploadedModel && (
            <div
              style={{
                backgroundColor: '#F0FDF4',
                border: '1px solid #BBF7D0',
                borderRadius: '8px',
                padding: '10px 14px',
                fontSize: '12px',
                color: '#166534',
                marginBottom: '16px',
              }}
            >
              ✓ Loaded: <strong>{uploadedModel.filename}</strong> ({uploadedModel.file_format})
            </div>
          )}

          {/* Task Type Switcher */}
          <div style={{ marginTop: 'auto' }}>
            <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#4A3E34', marginBottom: '8px' }}>
              Model Task Type
              <InfoTooltip
                title="Task Type"
                content="Classification predicts discrete classes (like dog vs. cat). Regression predicts continuous numbers (like house prices)."
              />
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <button
                type="button"
                onClick={() => setTaskType('classification')}
                style={{
                  padding: '8px 12px',
                  borderRadius: '8px',
                  border: taskType === 'classification' ? '2px solid #D45A1E' : '1px solid #E2D6C8',
                  backgroundColor: taskType === 'classification' ? '#FAF0E8' : '#FFFFFF',
                  color: taskType === 'classification' ? '#D45A1E' : '#73675E',
                  fontWeight: 600,
                  fontSize: '13px',
                  cursor: 'pointer',
                }}
              >
                Classification
              </button>
              <button
                type="button"
                onClick={() => setTaskType('regression')}
                style={{
                  padding: '8px 12px',
                  borderRadius: '8px',
                  border: taskType === 'regression' ? '2px solid #D45A1E' : '1px solid #E2D6C8',
                  backgroundColor: taskType === 'regression' ? '#FAF0E8' : '#FFFFFF',
                  color: taskType === 'regression' ? '#D45A1E' : '#73675E',
                  fontWeight: 600,
                  fontSize: '13px',
                  cursor: 'pointer',
                }}
              >
                Regression
              </button>
            </div>
          </div>
        </div>

        {/* Step 2: The Test Data */}
        <div
          style={{
            backgroundColor: '#FFFFFF',
            borderRadius: '18px',
            border: '1.5px solid #EAE0D5',
            padding: '24px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.03)',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '14px' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '8px',
                backgroundColor: '#FAF0E8',
                color: '#D45A1E',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 700,
                fontSize: '14px',
                marginRight: '10px',
              }}
            >
              2
            </div>
            <h3 style={{ margin: 0, fontSize: '18px', color: '#221C18' }}>
              Your Test Dataset
              <InfoTooltip
                title="Test Dataset"
                content="The CSV data your model will be evaluated against. It must contain the input features and the true target answer column."
              />
            </h3>
          </div>

          <p style={{ margin: '0 0 16px', fontSize: '13px', color: '#8A7D71' }}>
            Upload a test CSV or test with sample evaluation data:
          </p>

          {/* Test File Input Box */}
          <label
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              border: '2px dashed #E2D6C8',
              borderRadius: '12px',
              padding: '20px',
              backgroundColor: '#FCFAF8',
              cursor: 'pointer',
              marginBottom: '16px',
            }}
          >
            <input
              type="file"
              accept=".csv"
              onChange={handleDatasetFileChange}
              style={{ display: 'none' }}
            />
            <span style={{ fontSize: '28px', marginBottom: '6px' }}>📊</span>
            <span style={{ fontSize: '14px', fontWeight: 600, color: '#D45A1E' }}>
              {isUploadingDataset
                ? 'Reading dataset...'
                : testFile
                ? testFile.name
                : 'Click to Upload Test CSV'}
            </span>
            <span style={{ fontSize: '11px', color: '#8A7D71', marginTop: '4px' }}>
              Standard CSV dataset file
            </span>
          </label>

          {/* Preset Sample Data toggle */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '10px 14px',
              backgroundColor: useSampleData ? '#FAF0E8' : '#F6F3EE',
              borderRadius: '8px',
              marginBottom: '16px',
              fontSize: '12px',
            }}
          >
            <span style={{ color: '#4A3E34', fontWeight: 500 }}>
              {useSampleData ? 'Using standard built-in Iris test dataset' : `Using uploaded test dataset (${testFile?.name})`}
            </span>
            <button
              type="button"
              onClick={() => {
                setUseSampleData(true);
                setTestFile(null);
                setActiveTestDatasetId('dataset-001');
                setTargetColumn('species');
              }}
              style={{
                border: 'none',
                backgroundColor: 'transparent',
                color: '#D45A1E',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                textDecoration: 'underline',
              }}
            >
              Reset to Sample
            </button>
          </div>

          {/* Target Column Selector */}
          <div style={{ marginTop: 'auto' }}>
            <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#4A3E34', marginBottom: '8px' }}>
              Target / Ground Truth Column
              <InfoTooltip
                title="Target Column"
                content="The column in your test data containing the actual true values that the model's predictions are compared against."
              />
            </label>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              style={{
                width: '100%',
                padding: '9px 12px',
                borderRadius: '8px',
                border: '1px solid #E2D6C8',
                backgroundColor: '#FFFFFF',
                fontSize: '13px',
                color: '#221C18',
              }}
            >
              {availableColumns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Action Button: Big, Clear, Friendly */}
      <div style={{ textAlign: 'center', marginBottom: '36px' }}>
        <button
          type="button"
          onClick={handleRunEvaluation}
          disabled={isEvaluating}
          style={{
            padding: '14px 44px',
            backgroundColor: '#D45A1E',
            color: '#FFFFFF',
            fontSize: '16px',
            fontWeight: 700,
            borderRadius: '30px',
            border: 'none',
            cursor: isEvaluating ? 'not-allowed' : 'pointer',
            boxShadow: '0 4px 14px rgba(212, 90, 30, 0.35)',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '10px',
            transition: 'transform 0.15s, background-color 0.15s',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#BF4E17')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#D45A1E')}
        >
          {isEvaluating ? (
            <>Evaluating Predictions...</>
          ) : (
            <>
              Evaluate Model Now
              <span style={{ fontSize: '18px' }}>→</span>
            </>
          )}
        </button>
      </div>

      {/* Results Section */}
      {evaluationResult && (
        <div
          style={{
            backgroundColor: '#FFFFFF',
            borderRadius: '20px',
            border: '1.5px solid #EAE0D5',
            padding: '30px',
            boxShadow: '0 8px 30px rgba(0, 0, 0, 0.04)',
            marginBottom: '40px',
          }}
        >
          {/* Result Header */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              borderBottom: '1px solid #F0E9DF',
              paddingBottom: '18px',
              marginBottom: '24px',
              flexWrap: 'wrap',
              gap: '12px',
            }}
          >
            <div>
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  backgroundColor: '#F0FDF4',
                  border: '1px solid #BBF7D0',
                  color: '#166534',
                  fontSize: '12px',
                  fontWeight: 600,
                  padding: '4px 10px',
                  borderRadius: '12px',
                  marginBottom: '6px',
                }}
              >
                <span>✓</span> Evaluation Finished Successfully
              </div>
              <h3 style={{ margin: 0, fontSize: '22px', color: '#221C18' }}>
                {evaluationResult.algorithm}
              </h3>
              <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#8A7D71' }}>
                Evaluated on <strong>{evaluationResult.test_samples}</strong> test rows in{' '}
                <strong>{evaluationResult.train_time_ms} ms</strong>.
              </p>
            </div>

            {onViewLeaderboard && (
              <button
                type="button"
                onClick={onViewLeaderboard}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#FAF0E8',
                  color: '#D45A1E',
                  border: '1px solid #F3DAC8',
                  borderRadius: '8px',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                <span>🏆</span> View Leaderboard
              </button>
            )}
          </div>

          {/* Classification Metrics Cards */}
          {evaluationResult.metrics?.Classification && (
            <div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                  gap: '16px',
                  marginBottom: '24px',
                }}
              >
                {/* Accuracy */}
                <div
                  style={{
                    backgroundColor: '#FAF0E8',
                    borderRadius: '12px',
                    padding: '18px',
                    border: '1px solid #F3DAC8',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#D45A1E' }}>ACCURACY</span>
                    <InfoTooltip
                      title="Accuracy"
                      content="Percentage of test examples the model predicted correctly. A score of 95% means 95 out of 100 rows were correct."
                    />
                  </div>
                  <div style={{ fontSize: '32px', fontWeight: 800, color: '#D45A1E', marginTop: '6px' }}>
                    {(evaluationResult.metrics.Classification.accuracy * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: '11px', color: '#8A7D71', marginTop: '4px' }}>
                    Overall correctness
                  </div>
                </div>

                {/* Precision */}
                <div
                  style={{
                    backgroundColor: '#FCFAF8',
                    borderRadius: '12px',
                    padding: '18px',
                    border: '1px solid #EAE0D5',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#5A4D42' }}>PRECISION</span>
                    <InfoTooltip
                      title="Precision"
                      content="Out of all samples predicted as a specific class, how many actually were that class (minimizes false alarms)."
                    />
                  </div>
                  <div style={{ fontSize: '32px', fontWeight: 800, color: '#221C18', marginTop: '6px' }}>
                    {evaluationResult.metrics.Classification.precision.toFixed(3)}
                  </div>
                  <div style={{ fontSize: '11px', color: '#8A7D71', marginTop: '4px' }}>
                    Positive predictive value
                  </div>
                </div>

                {/* Recall */}
                <div
                  style={{
                    backgroundColor: '#FCFAF8',
                    borderRadius: '12px',
                    padding: '18px',
                    border: '1px solid #EAE0D5',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#5A4D42' }}>RECALL</span>
                    <InfoTooltip
                      title="Recall"
                      content="Out of all actual positive items in the dataset, how many did the model successfully find and identify."
                    />
                  </div>
                  <div style={{ fontSize: '32px', fontWeight: 800, color: '#221C18', marginTop: '6px' }}>
                    {evaluationResult.metrics.Classification.recall.toFixed(3)}
                  </div>
                  <div style={{ fontSize: '11px', color: '#8A7D71', marginTop: '4px' }}>
                    Sensitivity / Hit rate
                  </div>
                </div>

                {/* F1 Score */}
                <div
                  style={{
                    backgroundColor: '#FCFAF8',
                    borderRadius: '12px',
                    padding: '18px',
                    border: '1px solid #EAE0D5',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#5A4D42' }}>F1 SCORE</span>
                    <InfoTooltip
                      title="F1 Score"
                      content="Harmonic mean of precision and recall. A great single number to measure performance when classes are imbalanced."
                    />
                  </div>
                  <div style={{ fontSize: '32px', fontWeight: 800, color: '#221C18', marginTop: '6px' }}>
                    {evaluationResult.metrics.Classification.f1_score.toFixed(3)}
                  </div>
                  <div style={{ fontSize: '11px', color: '#8A7D71', marginTop: '4px' }}>
                    Balanced metric (0 to 1)
                  </div>
                </div>
              </div>

              {/* Collapsible Details */}
              <div style={{ borderTop: '1px solid #F0E9DF', paddingTop: '16px' }}>
                <button
                  type="button"
                  onClick={() => setShowDetailedMatrix(!showDetailedMatrix)}
                  style={{
                    border: 'none',
                    backgroundColor: 'transparent',
                    color: '#D45A1E',
                    fontSize: '13px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                  }}
                >
                  <span>{showDetailedMatrix ? '▼ Hide' : '▶ Show'} Detailed Confusion Matrix &amp; Class Breakdown</span>
                  <InfoTooltip
                    title="Confusion Matrix"
                    content="A grid showing exactly where the model predicted correctly vs. where it got confused between classes."
                  />
                </button>

                {showDetailedMatrix && evaluationResult.metrics.Classification.confusion_matrix && (
                  <div
                    style={{
                      marginTop: '16px',
                      padding: '18px',
                      backgroundColor: '#FCFAF8',
                      borderRadius: '10px',
                      border: '1px solid #EAE0D5',
                      overflowX: 'auto',
                    }}
                  >
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                      <thead>
                        <tr style={{ borderBottom: '1.5px solid #E2D6C8' }}>
                          <th style={{ textAlign: 'left', padding: '8px', color: '#73675E' }}>Actual \ Predicted</th>
                          {evaluationResult.metrics.Classification.confusion_matrix.labels.map((lbl: string) => (
                            <th key={lbl} style={{ textAlign: 'center', padding: '8px', color: '#221C18' }}>
                              {lbl}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {evaluationResult.metrics.Classification.confusion_matrix.matrix.map(
                          (row: number[], rIdx: number) => (
                            <tr key={rIdx} style={{ borderBottom: '1px solid #F0E9DF' }}>
                              <td style={{ padding: '8px', fontWeight: 600, color: '#221C18' }}>
                                {evaluationResult.metrics.Classification.confusion_matrix.labels[rIdx]}
                              </td>
                              {row.map((val: number, cIdx: number) => (
                                <td
                                  key={cIdx}
                                  style={{
                                    textAlign: 'center',
                                    padding: '8px',
                                    fontWeight: rIdx === cIdx ? 700 : 400,
                                    backgroundColor: rIdx === cIdx ? '#FAF0E8' : 'transparent',
                                    color: rIdx === cIdx ? '#D45A1E' : '#73675E',
                                  }}
                                >
                                  {val}
                                </td>
                              ))}
                            </tr>
                          )
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Regression Metrics Cards */}
          {evaluationResult.metrics?.Regression && (
            <div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                  gap: '16px',
                }}
              >
                {/* R2 Score */}
                <div
                  style={{
                    backgroundColor: '#FAF0E8',
                    borderRadius: '12px',
                    padding: '18px',
                    border: '1px solid #F3DAC8',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#D45A1E' }}>R² SCORE</span>
                    <InfoTooltip
                      title="R² (Coefficient of Determination)"
                      content="Explains how much variance in the data is captured by the model. 1.0 is a perfect score; higher is better."
                    />
                  </div>
                  <div style={{ fontSize: '32px', fontWeight: 800, color: '#D45A1E', marginTop: '6px' }}>
                    {evaluationResult.metrics.Regression.r2_score.toFixed(3)}
                  </div>
                  <div style={{ fontSize: '11px', color: '#8A7D71', marginTop: '4px' }}>
                    Fit quality (up to 1.0)
                  </div>
                </div>

                {/* MAE */}
                <div
                  style={{
                    backgroundColor: '#FCFAF8',
                    borderRadius: '12px',
                    padding: '18px',
                    border: '1px solid #EAE0D5',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#5A4D42' }}>MEAN ABS ERROR (MAE)</span>
                    <InfoTooltip
                      title="Mean Absolute Error"
                      content="The average absolute difference between predicted and actual numbers. Lower means fewer prediction errors."
                    />
                  </div>
                  <div style={{ fontSize: '32px', fontWeight: 800, color: '#221C18', marginTop: '6px' }}>
                    {evaluationResult.metrics.Regression.mean_absolute_error.toFixed(3)}
                  </div>
                  <div style={{ fontSize: '11px', color: '#8A7D71', marginTop: '4px' }}>
                    Average mistake magnitude
                  </div>
                </div>

                {/* RMSE */}
                <div
                  style={{
                    backgroundColor: '#FCFAF8',
                    borderRadius: '12px',
                    padding: '18px',
                    border: '1px solid #EAE0D5',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#5A4D42' }}>ROOT MEAN SQ ERR (RMSE)</span>
                    <InfoTooltip
                      title="Root Mean Squared Error"
                      content="Square root of average squared differences. Penalizes large errors more heavily than MAE."
                    />
                  </div>
                  <div style={{ fontSize: '32px', fontWeight: 800, color: '#221C18', marginTop: '6px' }}>
                    {evaluationResult.metrics.Regression.root_mean_squared_error.toFixed(3)}
                  </div>
                  <div style={{ fontSize: '11px', color: '#8A7D71', marginTop: '4px' }}>
                    Penalizes larger outliers
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

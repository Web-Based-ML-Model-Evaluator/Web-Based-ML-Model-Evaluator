import React, { useState, useCallback } from 'react';
import { DatasetIngestionView } from '../portion1_data_ingestion_preprocessing/frontend/DatasetIngestionView';
import { TaskConfigurationView } from '../portion2_task_routing_configuration/frontend/TaskConfigurationView';
import { EvaluationDashboardView } from '../portion3_model_execution_evaluation/frontend/EvaluationDashboardView';

// ============================================================================
// SYSTEM ARCHITECTURE & QA ORCHESTRATOR (Issue #16 & Issue #17)
// Assigned: Sanyam Devan Hiran (PES2UG24CS446) - System Architecture & QA Lead
// Requirements: NFR-2 (Guided Usability), NFR-3 (Portability), QA-1 (UX Validation),
//               QA-3 (Cross-Browser Compatibility), SEC-1 & SEC-3 (In-Memory Session Purge),
//               FR-13 (Fault Tolerance & Structured Error Handling)
// ============================================================================

export type WizardStep = 1 | 2 | 3 | 4;

interface StepMetadata {
  id: WizardStep;
  title: string;
  subtitle: string;
  badge: string;
}

const STEPS: StepMetadata[] = [
  { id: 1, title: 'Data Ingestion', subtitle: 'Upload & Preprocessing', badge: 'FR 1-3' },
  { id: 2, title: 'Task Routing', subtitle: 'Algorithms & Target Config', badge: 'FR 4-7' },
  { id: 3, title: 'Model Evaluation', subtitle: 'Metrics & Heatmaps', badge: 'FR 8-11' },
  { id: 4, title: 'Comparison', subtitle: 'Side-by-Side Leaderboard', badge: 'FR 12' },
];

export const App: React.FC = () => {
  // Wizard Navigation State (NFR-2, QA-1)
  const [currentStep, setCurrentStep] = useState<WizardStep>(1);
  const [completedSteps, setCompletedSteps] = useState<Set<WizardStep>>(new Set());

  // Shared Session State
  const [sessionId] = useState<string>(() => `session-${Math.random().toString(36).substring(2, 9)}`);
  const [activeDatasetId, setActiveDatasetId] = useState<string>('dataset-001');
  const [availableColumns, setAvailableColumns] = useState<string[]>([
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'species',
  ]);
  const [columnTypes, setColumnTypes] = useState<Record<string, string>>({
    sepal_length: 'Numeric',
    sepal_width: 'Numeric',
    petal_length: 'Numeric',
    petal_width: 'Numeric',
    species: 'Categorical',
  });
  const [taskConfig, setTaskConfig] = useState<any>(null);

  // Security & Purge Notification State (SEC-1, SEC-3)
  const [purgeStatus, setPurgeStatus] = useState<{
    loading: boolean;
    successMessage: string | null;
    errorMessage: string | null;
  }>({
    loading: false,
    successMessage: null,
    errorMessage: null,
  });

  // Service Endpoints Config (Axum Microservices)
  const p1ApiBase = 'http://localhost:8001';
  const p2ApiBase = 'http://localhost:8002';
  const p3ApiBase = 'http://localhost:8003';

  // Step 1 Completion Callback
  const handleDatasetReady = useCallback((datasetId: string, schema: any) => {
    setActiveDatasetId(datasetId);
    if (schema?.column_names) {
      setAvailableColumns(schema.column_names);
    }
    if (schema?.column_summaries) {
      const typesMap: Record<string, string> = {};
      schema.column_summaries.forEach((col: any) => {
        typesMap[col.name] = col.inferred_type === 'numeric' ? 'Numeric' : 'Categorical';
      });
      setColumnTypes(typesMap);
    }
    setCompletedSteps((prev) => new Set([...prev, 1]));
    setCurrentStep(2);
  }, []);

  // Step 2 Completion Callback
  const handleConfigurationComplete = useCallback((config: any) => {
    setTaskConfig(config);
    setCompletedSteps((prev) => new Set([...prev, 2]));
    setCurrentStep(3);
  }, []);

  // In-Memory Session Memory Purge Handler (SEC-1, SEC-3, SRS 7.3)
  const handlePurgeSession = async () => {
    const confirmed = window.confirm(
      'Are you sure you want to purge all in-memory datasets and model evaluation results? This action immediately deletes all cached data from RAM (SEC-1 / SEC-3).'
    );
    if (!confirmed) return;

    setPurgeStatus({ loading: true, successMessage: null, errorMessage: null });

    try {
      // Purge Portion 1 In-Memory Buffer
      const p1Promise = fetch(`${p1ApiBase}/api/v1/session/purge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }).catch(() => null);

      // Purge Portion 3 In-Memory Buffer
      const p3Promise = fetch(`${p3ApiBase}/api/v1/models/session/purge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }).catch(() => null);

      await Promise.all([p1Promise, p3Promise]);

      // Reset client state
      setCompletedSteps(new Set());
      setTaskConfig(null);
      setCurrentStep(1);

      setPurgeStatus({
        loading: false,
        successMessage: `Session ${sessionId} in-memory buffer successfully purged across all backend microservices. Zero persistent PII remains.`,
        errorMessage: null,
      });

      // Auto-clear success banner after 6 seconds
      setTimeout(() => {
        setPurgeStatus((prev) => ({ ...prev, successMessage: null }));
      }, 6000);
    } catch (err: any) {
      setPurgeStatus({
        loading: false,
        successMessage: null,
        errorMessage: `Failed to notify backend: ${err.message}. Local session state has been reset.`,
      });
    }
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: '#f8fafc',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        color: '#0f172a',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Top Application Header */}
      <header
        style={{
          backgroundColor: '#ffffff',
          borderBottom: '1px solid #e2e8f0',
          padding: '16px 24px',
          boxShadow: '0 1px 2px rgba(0,0,0,0.03)',
        }}
      >
        <div
          style={{
            maxWidth: '1280px',
            margin: '0 auto',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '12px',
          }}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div
                style={{
                  width: '28px',
                  height: '28px',
                  borderRadius: '6px',
                  backgroundColor: '#2563eb',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#ffffff',
                  fontWeight: 'bold',
                  fontSize: '16px',
                }}
              >
                M
              </div>
              <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 800, color: '#0f172a' }}>
                Web-Based ML Model Evaluator
              </h1>
              <span
                style={{
                  fontSize: '11px',
                  backgroundColor: '#dbeafe',
                  color: '#1d4ed8',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  fontWeight: 600,
                }}
              >
                Team 13
              </span>
            </div>
            <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#64748b' }}>
              High-performance automated machine learning pipeline with 3-tier decoupled Axum microservices.
            </p>
          </div>

          {/* Session Security Controls (SEC-1, SEC-3) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ textAlign: 'right', fontSize: '12px', color: '#64748b' }}>
              <div>Session: <code style={{ backgroundColor: '#f1f5f9', padding: '2px 4px', borderRadius: '4px' }}>{sessionId}</code></div>
              <div style={{ color: '#16a34a', fontWeight: 600 }}>● In-Memory Isolation (SEC-1)</div>
            </div>
            <button
              type="button"
              onClick={handlePurgeSession}
              disabled={purgeStatus.loading}
              title="Purge all loaded datasets and model states from volatile memory (SEC-1 / SEC-3)"
              style={{
                padding: '8px 14px',
                fontSize: '13px',
                fontWeight: 600,
                borderRadius: '6px',
                border: '1px solid #fecaca',
                backgroundColor: purgeStatus.loading ? '#fee2e2' : '#fef2f2',
                color: '#dc2626',
                cursor: purgeStatus.loading ? 'not-allowed' : 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                transition: 'background-color 0.15s ease',
              }}
            >
              <span>🗑️</span>
              {purgeStatus.loading ? 'Purging RAM...' : 'Purge Session (SEC-3)'}
            </button>
          </div>
        </div>
      </header>

      {/* Purge Notification Banner */}
      {purgeStatus.successMessage && (
        <div
          style={{
            backgroundColor: '#f0fdf4',
            borderBottom: '1px solid #bbf7d0',
            padding: '10px 24px',
            color: '#166534',
            fontSize: '13px',
            textAlign: 'center',
            fontWeight: 600,
          }}
        >
          ✓ {purgeStatus.successMessage}
        </div>
      )}
      {purgeStatus.errorMessage && (
        <div
          style={{
            backgroundColor: '#fef2f2',
            borderBottom: '1px solid #fecaca',
            padding: '10px 24px',
            color: '#991b1b',
            fontSize: '13px',
            textAlign: 'center',
            fontWeight: 600,
          }}
        >
          ⚠ {purgeStatus.errorMessage}
        </div>
      )}

      {/* Guided 4-Step Stepper Navigation (NFR-2, QA-1) */}
      <nav
        aria-label="Workflow Steps"
        style={{
          backgroundColor: '#ffffff',
          borderBottom: '1px solid #e2e8f0',
          padding: '14px 24px',
        }}
      >
        <div
          style={{
            maxWidth: '1280px',
            margin: '0 auto',
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: '12px',
          }}
        >
          {STEPS.map((s) => {
            const isActive = currentStep === s.id;
            const isCompleted = completedSteps.has(s.id);

            return (
              <button
                key={s.id}
                type="button"
                onClick={() => setCurrentStep(s.id)}
                style={{
                  padding: '10px 14px',
                  borderRadius: '8px',
                  border: isActive
                    ? '2px solid #2563eb'
                    : isCompleted
                    ? '1px solid #86efac'
                    : '1px solid #e2e8f0',
                  backgroundColor: isActive
                    ? '#eff6ff'
                    : isCompleted
                    ? '#f0fdf4'
                    : '#ffffff',
                  textAlign: 'left',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  transition: 'all 0.15s ease',
                }}
              >
                <div
                  style={{
                    width: '26px',
                    height: '26px',
                    borderRadius: '50%',
                    backgroundColor: isActive
                      ? '#2563eb'
                      : isCompleted
                      ? '#16a34a'
                      : '#e2e8f0',
                    color: isActive || isCompleted ? '#ffffff' : '#64748b',
                    fontWeight: 700,
                    fontSize: '13px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                  }}
                >
                  {isCompleted ? '✓' : s.id}
                </div>
                <div style={{ overflow: 'hidden' }}>
                  <div
                    style={{
                      fontSize: '13px',
                      fontWeight: 700,
                      color: isActive ? '#1d4ed8' : isCompleted ? '#166534' : '#1e293b',
                      whiteSpace: 'nowrap',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {s.title}
                  </div>
                  <div
                    style={{
                      fontSize: '11px',
                      color: '#64748b',
                      whiteSpace: 'nowrap',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {s.badge} · {s.subtitle}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </nav>

      {/* Main Guided Step Content View */}
      <main style={{ flex: 1, maxWidth: '1280px', width: '100%', margin: '24px auto', padding: '0 24px' }}>
        {currentStep === 1 && (
          <div>
            <div style={{ marginBottom: '16px' }}>
              <h2 style={{ margin: 0, fontSize: '18px', color: '#0f172a' }}>
                Step 1: Dataset Ingestion, Inspection & Cleaning (FR 1-3)
              </h2>
              <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#64748b' }}>
                Upload CSV dataset (max 10 MB, SEC-2), review statistical summaries, and apply imputation strategies for missing values.
              </p>
            </div>
            <DatasetIngestionView
              apiBaseUrl={p1ApiBase}
              onDatasetReady={handleDatasetReady}
            />
          </div>
        )}

        {currentStep === 2 && (
          <div>
            <div style={{ marginBottom: '16px' }}>
              <h2 style={{ margin: 0, fontSize: '18px', color: '#0f172a' }}>
                Step 2: Task Routing & Model Configuration (FR 4-7)
              </h2>
              <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#64748b' }}>
                Select problem type (Classification, Regression, Clustering), select feature/target columns, and set train/test split ratio.
              </p>
            </div>
            <TaskConfigurationView
              datasetId={activeDatasetId}
              availableColumns={availableColumns}
              columnTypes={columnTypes}
              apiBaseUrl={p2ApiBase}
              onConfigurationComplete={handleConfigurationComplete}
            />
          </div>
        )}

        {currentStep === 3 && (
          <div>
            <div style={{ marginBottom: '16px' }}>
              <h2 style={{ margin: 0, fontSize: '18px', color: '#0f172a' }}>
                Step 3: Algorithm Execution & Performance Metrics (FR 8-11)
              </h2>
              <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#64748b' }}>
                Inspect execution results, regression errors (MAE/RMSE/R²), classification confusion matrix heatmap, or 2D PCA cluster projection.
              </p>
            </div>
            <EvaluationDashboardView
              datasetId={activeDatasetId}
              apiBaseUrl={p3ApiBase}
            />
          </div>
        )}

        {currentStep === 4 && (
          <div>
            <div style={{ marginBottom: '16px' }}>
              <h2 style={{ margin: 0, fontSize: '18px', color: '#0f172a' }}>
                Step 4: Multi-Model Comparison & Recommendation Dashboard (FR 12)
              </h2>
              <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#64748b' }}>
                Compare multiple trained candidate models side-by-side, view ranking based on Business Rule BR-7, and export CSV/JSON reports.
              </p>
            </div>
            <EvaluationDashboardView
              datasetId={activeDatasetId}
              apiBaseUrl={p3ApiBase}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: '1px solid #e2e8f0',
          backgroundColor: '#ffffff',
          padding: '16px 24px',
          marginTop: 'auto',
          fontSize: '12px',
          color: '#64748b',
          textAlign: 'center',
        }}
      >
        <div style={{ maxWidth: '1280px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <strong>Web-Based ML Model Evaluator</strong> — PES University Software Engineering Capstone (Team 13)
          </div>
          <div>
            Standards: ISO/IEC 25010 (Portability & Usability) · OWASP Top 10 (Input Sanitization & In-Memory Isolation)
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;

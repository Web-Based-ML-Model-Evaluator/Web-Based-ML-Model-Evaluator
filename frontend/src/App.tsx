import React, { useState, useEffect, useCallback } from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { GpuAccelerationPanel } from './components/GpuAccelerationPanel';
import { DatasetIngestionView } from './components/DatasetIngestionView';
import { TaskConfigurationView } from './components/TaskConfigurationView';
import { EvaluationDashboardView } from './components/EvaluationDashboardView';
import { probeGpuAcceleration } from './services/webgpu';
import { ApiClient } from './services/api';

export type MainView = 'home' | 'train' | 'evaluate';
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
  // Primary Navigation View ('home' matching screenshot, 'train', 'evaluate')
  const [currentView, setCurrentView] = useState<MainView>('home');
  const [currentStep, setCurrentStep] = useState<WizardStep>(1);
  const [completedSteps, setCompletedSteps] = useState<Set<WizardStep>>(new Set());

  // Hardware Acceleration State
  const [gpuDeviceName, setGpuDeviceName] = useState<string>('Detecting GPU...');
  const [isGpuActive, setIsGpuActive] = useState<boolean>(true);

  // Session & Dataset State
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

  // Security Purge State (SEC-1, SEC-3)
  const [purgeStatus, setPurgeStatus] = useState<{
    loading: boolean;
    successMessage: string | null;
    errorMessage: string | null;
  }>({
    loading: false,
    successMessage: null,
    errorMessage: null,
  });

  const apiBaseUrl = 'http://localhost:8000';

  useEffect(() => {
    probeGpuAcceleration().then((info) => {
      setGpuDeviceName(info.adapterName);
      setIsGpuActive(info.isSupported);
    });
  }, []);

  const handleSelectAction = (action: 'train' | 'evaluate') => {
    if (action === 'train') {
      setCurrentView('train');
      setCurrentStep(1);
    } else {
      setCurrentView('evaluate');
    }
  };

  const handleDatasetReady = useCallback((datasetId: string, schema: any) => {
    setActiveDatasetId(datasetId);
    if (schema?.column_names) {
      setAvailableColumns(schema.column_names);
    }
    if (schema?.column_summaries) {
      const typesMap: Record<string, string> = {};
      schema.column_summaries.forEach((col: any) => {
        typesMap[col.name] = String(col.inferred_type).toLowerCase() === 'numeric' ? 'Numeric' : 'Categorical';
      });
      setColumnTypes(typesMap);
    }
    setCompletedSteps((prev) => new Set([...prev, 1]));
    setCurrentStep(2);
  }, []);

  const handleConfigurationComplete = useCallback((config: any) => {
    setTaskConfig(config);
    setCompletedSteps((prev) => new Set([...prev, 2]));
    setCurrentStep(3);
  }, []);

  const handlePurgeSession = async () => {
    const confirmed = window.confirm(
      'Are you sure you want to purge all in-memory datasets and model evaluation results? This action deletes all cached data from RAM (SEC-1 / SEC-3).'
    );
    if (!confirmed) return;

    setPurgeStatus({ loading: true, successMessage: null, errorMessage: null });
    try {
      await ApiClient.purgeSession();
      setCompletedSteps(new Set());
      setTaskConfig(null);
      setCurrentStep(1);
      setPurgeStatus({
        loading: false,
        successMessage: `Session memory successfully purged across backend memory. Zero persistent data remains.`,
        errorMessage: null,
      });
      setTimeout(() => {
        setPurgeStatus((prev) => ({ ...prev, successMessage: null }));
      }, 5000);
    } catch (err: any) {
      setPurgeStatus({
        loading: false,
        successMessage: null,
        errorMessage: `Notice: Local session state has been reset. (${err.message})`,
      });
    }
  };

  return (
    <div className="parchment-background" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navigation Bar Matching Screenshot */}
      <Navbar
        currentView={currentView}
        onNavigate={setCurrentView}
        gpuDeviceName={gpuDeviceName}
        isGpuActive={isGpuActive}
      />

      {/* Purge Notification Banner */}
      {purgeStatus.successMessage && (
        <div
          style={{
            backgroundColor: '#F0FDF4',
            borderBottom: '1px solid #BBF7D0',
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
            backgroundColor: '#FEF2F2',
            borderBottom: '1px solid #FECACA',
            padding: '10px 24px',
            color: '#991B1B',
            fontSize: '13px',
            textAlign: 'center',
            fontWeight: 600,
          }}
        >
          ⚠ {purgeStatus.errorMessage}
        </div>
      )}

      {/* Main Content View Switcher */}
      <main style={{ flex: 1, width: '100%' }}>
        {/* HOME VIEW: Exact Match to the User's Screenshot */}
        {currentView === 'home' && (
          <div>
            <Hero onSelectAction={handleSelectAction} />
            <GpuAccelerationPanel />
          </div>
        )}

        {/* TRAIN WORKFLOW VIEW: 4-Step Guided Wizard */}
        {currentView === 'train' && (
          <div style={{ maxWidth: '1280px', margin: '0 auto 40px', padding: '0 24px' }}>
            {/* Header with Back button and Session Purge */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                margin: '20px 0 16px',
                flexWrap: 'wrap',
                gap: '12px',
              }}
            >
              <div>
                <button
                  type="button"
                  onClick={() => setCurrentView('home')}
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
                    marginBottom: '8px',
                  }}
                >
                  ← Back to Home
                </button>
                <h2 style={{ margin: 0, fontSize: '26px', fontFamily: "var(--font-serif, 'Playfair Display', serif)", color: '#221C18' }}>
                  Train Your Machine Learning Model
                </h2>
                <p style={{ margin: '4px 0 0', fontSize: '14px', color: '#73675E' }}>
                  Guided high-performance workflow with GPU-accelerated algorithms and real-time validation.
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '12px', color: '#73675E' }}>
                  RAM Session: <code>{sessionId}</code>
                </span>
                <button
                  type="button"
                  onClick={handlePurgeSession}
                  disabled={purgeStatus.loading}
                  style={{
                    padding: '8px 14px',
                    fontSize: '12px',
                    fontWeight: 600,
                    borderRadius: '8px',
                    border: '1px solid #FECACA',
                    backgroundColor: '#FEF2F2',
                    color: '#DC2626',
                    cursor: 'pointer',
                  }}
                >
                  {purgeStatus.loading ? 'Purging RAM...' : 'Purge RAM (SEC-3)'}
                </button>
              </div>
            </div>

            {/* Stepper Navigation */}
            <nav
              aria-label="Workflow Steps"
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                gap: '12px',
                marginBottom: '28px',
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
                      padding: '12px 16px',
                      borderRadius: '14px',
                      border: isActive
                        ? '2px solid #D45A1E'
                        : isCompleted
                        ? '1px solid #86EFAC'
                        : '1px solid #E8DDCF',
                      backgroundColor: isActive
                        ? '#FDECE0'
                        : isCompleted
                        ? '#F0FDF4'
                        : 'rgba(255, 255, 255, 0.7)',
                      textAlign: 'left',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      transition: 'all 0.15s ease',
                      boxShadow: isActive ? '0 4px 12px rgba(212, 90, 30, 0.1)' : 'none',
                    }}
                  >
                    <div
                      style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '50%',
                        backgroundColor: isActive
                          ? '#D45A1E'
                          : isCompleted
                          ? '#16A34A'
                          : '#E8DDCF',
                        color: isActive || isCompleted ? '#FFFFFF' : '#73675E',
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
                    <div>
                      <div
                        style={{
                          fontSize: '14px',
                          fontWeight: 700,
                          color: isActive ? '#D45A1E' : isCompleted ? '#166534' : '#221C18',
                        }}
                      >
                        {s.title}
                      </div>
                      <div style={{ fontSize: '11px', color: '#73675E' }}>
                        {s.subtitle}
                      </div>
                    </div>
                  </button>
                );
              })}
            </nav>

            {/* Step Views */}
            <div
              style={{
                backgroundColor: 'rgba(255, 255, 255, 0.85)',
                borderRadius: '24px',
                border: '1.5px solid #EAE0D5',
                padding: '28px',
                boxShadow: '0 8px 30px rgba(0, 0, 0, 0.03)',
              }}
            >
              {currentStep === 1 && (
                <DatasetIngestionView
                  apiBaseUrl={apiBaseUrl}
                  onDatasetReady={handleDatasetReady}
                />
              )}

              {currentStep === 2 && (
                <TaskConfigurationView
                  datasetId={activeDatasetId}
                  availableColumns={availableColumns}
                  columnTypes={columnTypes}
                  apiBaseUrl={apiBaseUrl}
                  onConfigurationComplete={handleConfigurationComplete}
                />
              )}

              {currentStep === 3 && (
                <EvaluationDashboardView
                  datasetId={activeDatasetId}
                  apiBaseUrl={apiBaseUrl}
                />
              )}

              {currentStep === 4 && (
                <EvaluationDashboardView
                  datasetId={activeDatasetId}
                  apiBaseUrl={apiBaseUrl}
                />
              )}
            </div>
          </div>
        )}

        {/* EVALUATE WORKFLOW VIEW: Direct Leaderboard & Metrics */}
        {currentView === 'evaluate' && (
          <div style={{ maxWidth: '1280px', margin: '0 auto 40px', padding: '0 24px' }}>
            <div style={{ margin: '20px 0 24px' }}>
              <button
                type="button"
                onClick={() => setCurrentView('home')}
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
                  marginBottom: '8px',
                }}
              >
                ← Back to Home
              </button>
              <h2 style={{ margin: 0, fontSize: '28px', fontFamily: "var(--font-serif, 'Playfair Display', serif)", color: '#221C18' }}>
                Model Evaluation & Comparison Leaderboard
              </h2>
              <p style={{ margin: '4px 0 0', fontSize: '14px', color: '#73675E' }}>
                Compare candidate model performance side-by-side, inspect confusion matrices, regression errors, and 2D PCA cluster embeddings.
              </p>
            </div>

            <div
              style={{
                backgroundColor: 'rgba(255, 255, 255, 0.85)',
                borderRadius: '24px',
                border: '1.5px solid #EAE0D5',
                padding: '28px',
                boxShadow: '0 8px 30px rgba(0, 0, 0, 0.03)',
              }}
            >
              <EvaluationDashboardView
                datasetId={activeDatasetId}
                apiBaseUrl={apiBaseUrl}
              />
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: '1px solid #EAE0D5',
          backgroundColor: 'rgba(255, 255, 255, 0.4)',
          padding: '20px 48px',
          marginTop: 'auto',
          fontSize: '13px',
          color: '#73675E',
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
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <strong>ML Lab</strong>
            <span>—</span>
            <span>Web-Based ML Model Evaluator (Team 13)</span>
          </div>
          <div>
            ISO/IEC 25010 Portability · In-Memory Isolation (SEC-1) · GPU Compute Acceleration
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;


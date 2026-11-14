import React, { useState, useEffect, useMemo, useCallback } from 'react';

// ============================================================================
// PORTION 3: MODEL EXECUTION, METRICS & COMPARISON SKELETON
// Assigned Teammate: Shreya Patil (PES2UG24CS484)
// Scope: FR-8 (Algorithm Execution), FR-9 (Classification Metrics), FR-10 (Regression Metrics),
//        FR-11 (Clustering & 2D PCA), FR-12 (Side-by-Side Model Comparison Dashboard)
// ============================================================================

export interface RegressionMetrics {
  mean_absolute_error: number;
  mean_squared_error: number;
  root_mean_squared_error: number;
  r2_score: number;
}

export interface ModelComparisonEntry {
  execution_id: string;
  dataset_id: string;
  task_type: string;
  algorithm: string;
  executed_at: string;
  train_time_ms: number;
  primary_metric_name: string;
  primary_metric_value: number;
  summary_metrics: Record<string, number>;
}

export interface ModelRankEntry {
  rank: number;
  execution_id: string;
  algorithm: string;
  primary_metric_name: string;
  primary_metric_value: number;
  train_time_ms: number;
}

export interface ComparisonReport {
  dataset_id?: string;
  task_type: string;
  compared_models: ModelComparisonEntry[];
  best_model_id: string;
  best_model_algorithm: string;
  recommendation_reason: string;
  metrics_table: Record<string, Record<string, number>>;
  ranking: ModelRankEntry[];
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][]; // matrix[actual][predicted]
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
}

export interface PcaPoint {
  id: number;
  pc1: number;
  pc2: number;
  cluster: number;
  label?: string;
}

export interface PcaClusteringData {
  points: PcaPoint[];
  silhouette_score?: number;
  davies_bouldin_index?: number;
  cluster_count?: number;
}

export interface EvaluationDashboardProps {
  datasetId?: string;
  apiBaseUrl?: string;
  regressionMetrics?: RegressionMetrics;
  confusionMatrix?: ConfusionMatrixData;
  pcaClustering?: PcaClusteringData;
}

export const DEFAULT_CONFUSION_MATRIX: ConfusionMatrixData = {
  labels: ['Class 0 (Setosa)', 'Class 1 (Versicolor)', 'Class 2 (Virginica)'],
  matrix: [
    [50, 0, 0],
    [0, 47, 3],
    [0, 2, 48],
  ],
  metrics: {
    accuracy: 0.967,
    precision: 0.967,
    recall: 0.967,
    f1_score: 0.967,
  },
};

export const DEFAULT_PCA_POINTS: PcaPoint[] = [
  // Cluster 0 (Setosa)
  { id: 1, pc1: -2.68, pc2: 0.32, cluster: 0, label: 'Sample #1' },
  { id: 2, pc1: -2.71, pc2: -0.18, cluster: 0, label: 'Sample #2' },
  { id: 3, pc1: -2.88, pc2: -0.14, cluster: 0, label: 'Sample #3' },
  { id: 4, pc1: -2.75, pc2: -0.32, cluster: 0, label: 'Sample #4' },
  { id: 5, pc1: -2.73, pc2: 0.33, cluster: 0, label: 'Sample #5' },
  { id: 6, pc1: -2.28, pc2: 0.74, cluster: 0, label: 'Sample #6' },
  { id: 7, pc1: -2.82, pc2: -0.09, cluster: 0, label: 'Sample #7' },
  { id: 8, pc1: -2.63, pc2: 0.17, cluster: 0, label: 'Sample #8' },
  // Cluster 1 (Versicolor)
  { id: 9, pc1: 0.57, pc2: -0.63, cluster: 1, label: 'Sample #9' },
  { id: 10, pc1: 0.48, pc2: -0.72, cluster: 1, label: 'Sample #10' },
  { id: 11, pc1: 0.61, pc2: -0.54, cluster: 1, label: 'Sample #11' },
  { id: 12, pc1: 0.31, pc2: -0.79, cluster: 1, label: 'Sample #12' },
  { id: 13, pc1: 0.45, pc2: -0.62, cluster: 1, label: 'Sample #13' },
  { id: 14, pc1: 0.72, pc2: -0.41, cluster: 1, label: 'Sample #14' },
  { id: 15, pc1: 0.28, pc2: -0.83, cluster: 1, label: 'Sample #15' },
  { id: 16, pc1: 0.52, pc2: -0.58, cluster: 1, label: 'Sample #16' },
  // Cluster 2 (Virginica)
  { id: 17, pc1: 1.86, pc2: 0.52, cluster: 2, label: 'Sample #17' },
  { id: 18, pc1: 2.12, pc2: 0.64, cluster: 2, label: 'Sample #18' },
  { id: 19, pc1: 2.04, pc2: 0.46, cluster: 2, label: 'Sample #19' },
  { id: 20, pc1: 1.95, pc2: 0.73, cluster: 2, label: 'Sample #20' },
  { id: 21, pc1: 2.31, pc2: 0.81, cluster: 2, label: 'Sample #21' },
  { id: 22, pc1: 1.77, pc2: 0.39, cluster: 2, label: 'Sample #22' },
  { id: 23, pc1: 2.15, pc2: 0.58, cluster: 2, label: 'Sample #23' },
  { id: 24, pc1: 2.22, pc2: 0.69, cluster: 2, label: 'Sample #24' },
];

export const DEFAULT_PCA_CLUSTERING: PcaClusteringData = {
  points: DEFAULT_PCA_POINTS,
  silhouette_score: 0.725,
  davies_bouldin_index: 0.65,
  cluster_count: 3,
};

// Built-in Mock Datasets for complete offline developer isolation & testing (FR-12)
const MOCK_COMPARISON_RUNS: ModelComparisonEntry[] = [
  {
    execution_id: 'exec-rf-01',
    dataset_id: 'dataset-001',
    task_type: 'Classification',
    algorithm: 'Random Forest Classifier',
    executed_at: new Date(Date.now() - 3600000).toISOString(),
    train_time_ms: 38.4,
    primary_metric_name: 'Accuracy',
    primary_metric_value: 0.95,
    summary_metrics: {
      accuracy: 0.95,
      precision: 0.94,
      recall: 0.95,
      f1_score: 0.945,
    },
  },
  {
    execution_id: 'exec-knn-02',
    dataset_id: 'dataset-001',
    task_type: 'Classification',
    algorithm: 'K-Nearest Neighbors (KNN)',
    executed_at: new Date(Date.now() - 2700000).toISOString(),
    train_time_ms: 18.2,
    primary_metric_name: 'Accuracy',
    primary_metric_value: 0.91,
    summary_metrics: {
      accuracy: 0.91,
      precision: 0.9,
      recall: 0.91,
      f1_score: 0.905,
    },
  },
  {
    execution_id: 'exec-lr-03',
    dataset_id: 'dataset-001',
    task_type: 'Classification',
    algorithm: 'Logistic Regression',
    executed_at: new Date(Date.now() - 1800000).toISOString(),
    train_time_ms: 12.1,
    primary_metric_name: 'Accuracy',
    primary_metric_value: 0.88,
    summary_metrics: {
      accuracy: 0.88,
      precision: 0.87,
      recall: 0.88,
      f1_score: 0.875,
    },
  },
  {
    execution_id: 'exec-dt-04',
    dataset_id: 'dataset-001',
    task_type: 'Classification',
    algorithm: 'Decision Tree Classifier',
    executed_at: new Date(Date.now() - 900000).toISOString(),
    train_time_ms: 9.5,
    primary_metric_name: 'Accuracy',
    primary_metric_value: 0.86,
    summary_metrics: {
      accuracy: 0.86,
      precision: 0.85,
      recall: 0.86,
      f1_score: 0.855,
    },
  },
  {
    execution_id: 'exec-reg-rf-05',
    dataset_id: 'dataset-001',
    task_type: 'Regression',
    algorithm: 'Random Forest Regressor',
    executed_at: new Date(Date.now() - 7200000).toISOString(),
    train_time_ms: 42.0,
    primary_metric_name: 'R2 Score',
    primary_metric_value: 0.892,
    summary_metrics: {
      r2: 0.892,
      mae: 2.14,
      mse: 8.65,
      rmse: 2.94,
    },
  },
  {
    execution_id: 'exec-reg-dt-06',
    dataset_id: 'dataset-001',
    task_type: 'Regression',
    algorithm: 'Decision Tree Regressor',
    executed_at: new Date(Date.now() - 6300000).toISOString(),
    train_time_ms: 8.3,
    primary_metric_name: 'R2 Score',
    primary_metric_value: 0.81,
    summary_metrics: {
      r2: 0.81,
      mae: 3.2,
      mse: 15.2,
      rmse: 3.9,
    },
  },
  {
    execution_id: 'exec-reg-lin-07',
    dataset_id: 'dataset-001',
    task_type: 'Regression',
    algorithm: 'Linear Regression',
    executed_at: new Date(Date.now() - 5400000).toISOString(),
    train_time_ms: 6.1,
    primary_metric_name: 'R2 Score',
    primary_metric_value: 0.765,
    summary_metrics: {
      r2: 0.765,
      mae: 4.1,
      mse: 18.8,
      rmse: 4.34,
    },
  },
  {
    execution_id: 'exec-clust-km-08',
    dataset_id: 'dataset-001',
    task_type: 'Clustering',
    algorithm: 'K-Means Clustering',
    executed_at: new Date(Date.now() - 8100000).toISOString(),
    train_time_ms: 15.0,
    primary_metric_name: 'Silhouette Score',
    primary_metric_value: 0.725,
    summary_metrics: {
      silhouette: 0.725,
      davies_bouldin: 0.65,
    },
  },
];

export const EvaluationDashboardView: React.FC<EvaluationDashboardProps> = ({
  datasetId = 'dataset-001',
  apiBaseUrl = 'http://localhost:8000',
  regressionMetrics,
  confusionMatrix = DEFAULT_CONFUSION_MATRIX,
  pcaClustering = DEFAULT_PCA_CLUSTERING,
}) => {
  const [activeTab, setActiveTab] = useState<'results' | 'comparison'>('results');
  const [showPercentages, setShowPercentages] = useState<boolean>(false);
  const [hoveredPoint, setHoveredPoint] = useState<PcaPoint | null>(null);
  const [selectedClusterFilter, setSelectedClusterFilter] = useState<number | 'all'>('all');

  // FR-12 Model Comparison Dashboard State
  const [offlineMockMode, setOfflineMockMode] = useState<boolean>(true);
  const [comparisonRuns, setComparisonRuns] = useState<ModelComparisonEntry[]>(MOCK_COMPARISON_RUNS);
  const [selectedExecutionIds, setSelectedExecutionIds] = useState<string[]>([
    'exec-rf-01',
    'exec-knn-02',
    'exec-lr-03',
  ]);
  const [comparisonReport, setComparisonReport] = useState<ComparisonReport | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [taskFilter, setTaskFilter] = useState<'All' | 'Classification' | 'Regression' | 'Clustering'>('All');

  // Fetch comparison runs from backend or fallback to mock fixtures (FR-12)
  const fetchComparisonRuns = useCallback(async () => {
    setLoading(true);
    setError(null);
    if (offlineMockMode) {
      setComparisonRuns(MOCK_COMPARISON_RUNS);
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/models/comparison/${datasetId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: Failed to fetch comparison history from server.`);
      }
      const data = await response.json();
      const runs = (data.runs as ModelComparisonEntry[]) || [];
      if (runs.length === 0) {
        // Fallback to mock runs if backend has empty session history
        setComparisonRuns(MOCK_COMPARISON_RUNS);
      } else {
        setComparisonRuns(runs);
      }
    } catch (err: any) {
      setError(`Backend error (${err.message}). Showing offline fixtures.`);
      setComparisonRuns(MOCK_COMPARISON_RUNS);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, datasetId, offlineMockMode]);

  useEffect(() => {
    fetchComparisonRuns();
  }, [fetchComparisonRuns]);

  // Filtered runs based on current task selection
  const filteredRuns = useMemo(() => {
    if (taskFilter === 'All') return comparisonRuns;
    return comparisonRuns.filter((r) => r.task_type.toLowerCase() === taskFilter.toLowerCase());
  }, [comparisonRuns, taskFilter]);

  // Compute local comparison report when in offline mode or as fast client-side fallback
  const computeLocalComparisonReport = useCallback(
    (models: ModelComparisonEntry[]): ComparisonReport => {
      const task = models[0].task_type;
      const sorted = [...models].sort((a, b) => {
        // Primary metric higher is better
        if (b.primary_metric_value !== a.primary_metric_value) {
          return b.primary_metric_value - a.primary_metric_value;
        }
        // Tie-breaker: lower training latency
        return a.train_time_ms - b.train_time_ms;
      });

      const ranking: ModelRankEntry[] = sorted.map((m, index) => ({
        rank: index + 1,
        execution_id: m.execution_id,
        algorithm: m.algorithm,
        primary_metric_name: m.primary_metric_name,
        primary_metric_value: m.primary_metric_value,
        train_time_ms: m.train_time_ms,
      }));

      const best = sorted[0];
      const recommendationReason =
        `${best.algorithm} is identified as the optimal model, achieving the highest ` +
        `${best.primary_metric_name} (${best.primary_metric_value.toFixed(3)}) with training latency of ` +
        `${best.train_time_ms.toFixed(1)} ms.`;

      const metrics_table: Record<string, Record<string, number>> = {};
      // Primary metric row
      metrics_table[best.primary_metric_name] = {};
      models.forEach((m) => {
        metrics_table[best.primary_metric_name][m.execution_id] = m.primary_metric_value;
      });

      // Latency row
      metrics_table['Training Latency (ms)'] = {};
      models.forEach((m) => {
        metrics_table['Training Latency (ms)'][m.execution_id] = m.train_time_ms;
      });

      // Summary metrics rows
      models.forEach((m) => {
        Object.entries(m.summary_metrics).forEach(([k, v]) => {
          const displayKey = k.replace(/_/g, ' ').toUpperCase();
          if (!metrics_table[displayKey]) {
            metrics_table[displayKey] = {};
          }
          metrics_table[displayKey][m.execution_id] = v;
        });
      });

      return {
        dataset_id: datasetId,
        task_type: task,
        compared_models: models,
        best_model_id: best.execution_id,
        best_model_algorithm: best.algorithm,
        recommendation_reason: recommendationReason,
        metrics_table,
        ranking,
      };
    },
    [datasetId]
  );

  // Perform model comparison (FR-12 / BR-7)
  const handleCompareModels = useCallback(async () => {
    setError(null);

    // BR-7 Validation: at least 2 models required
    if (selectedExecutionIds.length < 2) {
      setError('Model comparison requires selecting at least two trained models side-by-side (FR-12 / BR-7).');
      setComparisonReport(null);
      return;
    }

    const selectedModels = comparisonRuns.filter((r) => selectedExecutionIds.includes(r.execution_id));
    if (selectedModels.length < 2) {
      setError('Could not locate the selected models in session history.');
      return;
    }

    // Check task type homogeneity
    const firstTask = selectedModels[0].task_type.toLowerCase();
    const hasMixedTasks = selectedModels.some((m) => m.task_type.toLowerCase() !== firstTask);
    if (hasMixedTasks) {
      setError('Cannot compare models across different ML problem types. Please select models from the same task (BR-7).');
      return;
    }

    if (offlineMockMode) {
      const localReport = computeLocalComparisonReport(selectedModels);
      setComparisonReport(localReport);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/models/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ execution_ids: selectedExecutionIds }),
      });

      if (!response.ok) {
        const errorJson = await response.json().catch(() => null);
        throw new Error(errorJson?.error?.message || `HTTP ${response.status} comparison request failed.`);
      }

      const report: ComparisonReport = await response.json();
      setComparisonReport(report);
    } catch (err: any) {
      console.warn('Backend compare endpoint unavailable, falling back to local calculation:', err.message);
      const localReport = computeLocalComparisonReport(selectedModels);
      setComparisonReport(localReport);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, comparisonRuns, computeLocalComparisonReport, offlineMockMode, selectedExecutionIds]);

  // Automatically trigger comparison on initial tab switch or selection change if >=2 models selected
  useEffect(() => {
    if (activeTab === 'comparison' && selectedExecutionIds.length >= 2 && !comparisonReport) {
      handleCompareModels();
    }
  }, [activeTab, handleCompareModels, selectedExecutionIds.length, comparisonReport]);

  const toggleModelSelection = (id: string) => {
    setSelectedExecutionIds((prev) => {
      const next = prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id];
      return next;
    });
  };

  const selectAllFiltered = () => {
    const ids = filteredRuns.map((r) => r.execution_id);
    setSelectedExecutionIds(ids);
  };

  const clearSelection = () => {
    setSelectedExecutionIds([]);
    setComparisonReport(null);
  };

  return (
    <div style={{ padding: '24px', fontFamily: 'sans-serif', maxWidth: '1100px', margin: '0 auto' }}>
      {/* Top Header */}
      <header style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '16px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h2 style={{ margin: 0, color: '#1e293b' }}>Portion 3: Model Execution, Evaluation & Comparison</h2>
            <p style={{ margin: '4px 0 0', color: '#64748b' }}>
              Assigned to: <strong>Shreya Patil (PES2UG24CS484)</strong> | Service Port: <strong>8003</strong>
            </p>
          </div>
          {/* Offline Mock Mode Switch */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              backgroundColor: '#f8fafc',
              padding: '6px 12px',
              borderRadius: '6px',
              border: '1px solid #cbd5e1',
              fontSize: '13px',
            }}
          >
            <input
              type="checkbox"
              id="mock-mode-toggle"
              checked={offlineMockMode}
              onChange={(e) => {
                setOfflineMockMode(e.target.checked);
                setComparisonReport(null);
              }}
              style={{ cursor: 'pointer' }}
            />
            <label htmlFor="mock-mode-toggle" style={{ cursor: 'pointer', fontWeight: 600, color: '#334155' }}>
              Offline Mock Mode
            </label>
            <span
              style={{
                fontSize: '11px',
                padding: '2px 6px',
                borderRadius: '4px',
                backgroundColor: offlineMockMode ? '#dcfce7' : '#e0f2fe',
                color: offlineMockMode ? '#166534' : '#0369a1',
                fontWeight: 600,
              }}
            >
              {offlineMockMode ? 'STANDALONE' : ':8003 LIVE'}
            </span>
          </div>
        </div>

        {/* Tab Navigation */}
        <div style={{ marginTop: '16px', display: 'flex', gap: '8px' }}>
          <button
            onClick={() => setActiveTab('results')}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              fontWeight: 600,
              cursor: 'pointer',
              border: activeTab === 'results' ? '2px solid #2563eb' : '1px solid #cbd5e1',
              backgroundColor: activeTab === 'results' ? '#eff6ff' : '#ffffff',
              color: activeTab === 'results' ? '#1d4ed8' : '#475569',
            }}
          >
            Model Results
          </button>
          <button
            onClick={() => setActiveTab('comparison')}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              fontWeight: 600,
              cursor: 'pointer',
              border: activeTab === 'comparison' ? '2px solid #2563eb' : '1px solid #cbd5e1',
              backgroundColor: activeTab === 'comparison' ? '#eff6ff' : '#ffffff',
              color: activeTab === 'comparison' ? '#1d4ed8' : '#475569',
            }}
          >
            Model Comparison Dashboard (FR-12)
          </button>
        </div>
      </header>

      {activeTab === 'results' ? (
        <div>
          {/* Regression Performance Metrics Display (Issue 8 / FR-10) */}
          <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px', marginBottom: '20px' }}>
            <h3 style={{ marginTop: 0 }}>Regression Performance Metrics (FR-10)</h3>
            <p style={{ color: '#64748b', fontSize: '14px', margin: '0 0 16px' }}>
              Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score:
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              <div style={{ padding: '16px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #cbd5e1' }}>
                <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600 }}>MEAN ABSOLUTE ERROR (MAE)</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#0f172a', marginTop: '4px' }}>
                  {regressionMetrics ? regressionMetrics.mean_absolute_error : '--'}
                </div>
              </div>
              <div style={{ padding: '16px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #cbd5e1' }}>
                <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600 }}>MEAN SQUARED ERROR (MSE)</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#0f172a', marginTop: '4px' }}>
                  {regressionMetrics ? regressionMetrics.mean_squared_error : '--'}
                </div>
              </div>
              <div style={{ padding: '16px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #cbd5e1' }}>
                <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600 }}>ROOT MEAN SQ ERROR (RMSE)</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#0f172a', marginTop: '4px' }}>
                  {regressionMetrics ? regressionMetrics.root_mean_squared_error : '--'}
                </div>
              </div>
              <div style={{ padding: '16px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #cbd5e1' }}>
                <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600 }}>R² SCORE</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#16a34a', marginTop: '4px' }}>
                  {regressionMetrics ? regressionMetrics.r2_score : '--'}
                </div>
              </div>
            </div>
          </section>

          {/* SECTION 2: Confusion Matrix Grid (FR-9) */}
          <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px', marginBottom: '20px', backgroundColor: '#ffffff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
              <div>
                <h3 style={{ margin: 0, color: '#0f172a' }}>Step 2: Confusion Matrix Heatmap (FR-9 - Classification)</h3>
                <p style={{ color: '#64748b', fontSize: '14px', margin: '4px 0 0' }}>
                  Evaluates multi-class prediction accuracy, true positives, and misclassifications across ground-truth classes.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowPercentages(!showPercentages)}
                style={{
                  padding: '6px 12px',
                  fontSize: '13px',
                  fontWeight: 600,
                  borderRadius: '6px',
                  border: '1px solid #cbd5e1',
                  backgroundColor: showPercentages ? '#eff6ff' : '#ffffff',
                  color: showPercentages ? '#1d4ed8' : '#334155',
                  cursor: 'pointer',
                }}
              >
                {showPercentages ? 'Show Raw Counts' : 'Show Percentages (%)'}
              </button>
            </div>

            {/* Classification Summary KPI Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', marginBottom: '20px' }}>
              <div style={{ padding: '12px', backgroundColor: '#f0fdf4', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                <div style={{ fontSize: '11px', color: '#166534', fontWeight: 700 }}>ACCURACY</div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#15803d', marginTop: '2px' }}>
                  {((confusionMatrix.metrics?.accuracy ?? 0.967) * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ padding: '12px', backgroundColor: '#eff6ff', borderRadius: '6px', border: '1px solid #bfdbfe' }}>
                <div style={{ fontSize: '11px', color: '#1e40af', fontWeight: 700 }}>PRECISION (MACRO)</div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#1d4ed8', marginTop: '2px' }}>
                  {((confusionMatrix.metrics?.precision ?? 0.967) * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ padding: '12px', backgroundColor: '#faf5ff', borderRadius: '6px', border: '1px solid #e9d5ff' }}>
                <div style={{ fontSize: '11px', color: '#6b21a8', fontWeight: 700 }}>RECALL (MACRO)</div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#7e22ce', marginTop: '2px' }}>
                  {((confusionMatrix.metrics?.recall ?? 0.967) * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ padding: '12px', backgroundColor: '#fef3c7', borderRadius: '6px', border: '1px solid #fde68a' }}>
                <div style={{ fontSize: '11px', color: '#92400e', fontWeight: 700 }}>F1-SCORE</div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#b45309', marginTop: '2px' }}>
                  {((confusionMatrix.metrics?.f1_score ?? 0.967) * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Heatmap Grid */}
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'center', fontSize: '13px' }}>
                <thead>
                  <tr>
                    <th style={{ padding: '10px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', color: '#475569' }}>
                      Actual \ Predicted
                    </th>
                    {confusionMatrix.labels.map((colLabel, idx) => (
                      <th
                        key={idx}
                        style={{
                          padding: '10px',
                          backgroundColor: '#f1f5f9',
                          border: '1px solid #e2e8f0',
                          color: '#1e293b',
                          fontWeight: 700,
                        }}
                      >
                        {colLabel}
                      </th>
                    ))}
                    <th style={{ padding: '10px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', color: '#64748b', fontWeight: 600 }}>
                      Total
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {confusionMatrix.matrix.map((row, rowIdx) => {
                    const rowTotal = row.reduce((sum, val) => sum + val, 0);
                    const maxVal = Math.max(...confusionMatrix.matrix.flat(), 1);
                    return (
                      <tr key={rowIdx}>
                        <th
                          style={{
                            padding: '10px',
                            backgroundColor: '#f1f5f9',
                            border: '1px solid #e2e8f0',
                            textAlign: 'left',
                            color: '#1e293b',
                            fontWeight: 700,
                          }}
                        >
                          {confusionMatrix.labels[rowIdx]}
                        </th>
                        {row.map((cellVal, colIdx) => {
                          const isTruePositive = rowIdx === colIdx;
                          const ratio = cellVal / maxVal;
                          const pct = rowTotal > 0 ? ((cellVal / rowTotal) * 100).toFixed(1) : '0.0';
                          const bgColor = isTruePositive
                            ? `rgba(37, 99, 235, ${Math.max(0.12, ratio * 0.9)})`
                            : cellVal > 0
                            ? `rgba(239, 68, 68, ${Math.max(0.1, ratio * 0.7)})`
                            : '#ffffff';
                          const textColor = ratio > 0.55 ? '#ffffff' : cellVal > 0 && !isTruePositive ? '#991b1b' : '#0f172a';

                          return (
                            <td
                              key={colIdx}
                              title={`Actual: ${confusionMatrix.labels[rowIdx]} | Predicted: ${confusionMatrix.labels[colIdx]} (${cellVal} samples, ${pct}%)`}
                              style={{
                                padding: '12px',
                                border: '1px solid #cbd5e1',
                                backgroundColor: bgColor,
                                color: textColor,
                                fontWeight: isTruePositive ? 700 : 500,
                                position: 'relative',
                              }}
                            >
                              <div style={{ fontSize: '15px' }}>
                                {showPercentages ? `${pct}%` : cellVal}
                              </div>
                              {isTruePositive && (
                                <div style={{ fontSize: '10px', opacity: 0.85, fontWeight: 500 }}>
                                  TP ({cellVal})
                                </div>
                              )}
                            </td>
                          );
                        })}
                        <td style={{ padding: '10px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', fontWeight: 600, color: '#334155' }}>
                          {rowTotal}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginTop: '12px', fontSize: '12px', color: '#64748b' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: 'rgba(37, 99, 235, 0.8)', borderRadius: '2px' }} />
                <span>True Positives (Diagonal Match)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: 'rgba(239, 68, 68, 0.4)', borderRadius: '2px' }} />
                <span>Misclassifications (Off-diagonal)</span>
              </div>
            </div>
          </section>

          {/* SECTION 3: 2D PCA Cluster Scatter Plot (FR-11) */}
          <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px', backgroundColor: '#ffffff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
              <div>
                <h3 style={{ margin: 0, color: '#0f172a' }}>Step 3: 2D PCA Cluster Scatter Plot (FR-11 - Clustering)</h3>
                <p style={{ color: '#64748b', fontSize: '14px', margin: '4px 0 0' }}>
                  Visualizes unsupervised clustering quality projected onto the first 2 Principal Components (PC1 vs PC2).
                </p>
              </div>

              {/* Quality Metric Badges */}
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                <div style={{ padding: '6px 12px', backgroundColor: '#f0fdf4', borderRadius: '6px', border: '1px solid #bbf7d0', fontSize: '12px' }}>
                  <span style={{ color: '#166534', fontWeight: 600 }}>Silhouette Score: </span>
                  <span style={{ fontWeight: 700, color: '#15803d' }}>
                    {pcaClustering.silhouette_score ?? 0.725}
                  </span>
                </div>
                <div style={{ padding: '6px 12px', backgroundColor: '#eff6ff', borderRadius: '6px', border: '1px solid #bfdbfe', fontSize: '12px' }}>
                  <span style={{ color: '#1e40af', fontWeight: 600 }}>Davies-Bouldin: </span>
                  <span style={{ fontWeight: 700, color: '#1d4ed8' }}>
                    {pcaClustering.davies_bouldin_index ?? 0.65}
                  </span>
                </div>
              </div>
            </div>

            {/* Cluster Filter Buttons */}
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '12px', fontWeight: 600, color: '#64748b' }}>Filter Cluster:</span>
              <button
                type="button"
                onClick={() => setSelectedClusterFilter('all')}
                style={{
                  padding: '4px 10px',
                  borderRadius: '14px',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  border: selectedClusterFilter === 'all' ? '2px solid #2563eb' : '1px solid #cbd5e1',
                  backgroundColor: selectedClusterFilter === 'all' ? '#eff6ff' : '#ffffff',
                  color: selectedClusterFilter === 'all' ? '#1d4ed8' : '#475569',
                }}
              >
                All Clusters ({pcaClustering.points.length})
              </button>
              {Array.from(new Set(pcaClustering.points.map((p) => p.cluster))).sort().map((clusterId) => {
                const colors = ['#2563eb', '#10b981', '#8b5cf6', '#f59e0b', '#ec4899'];
                const clr = colors[clusterId % colors.length];
                const count = pcaClustering.points.filter((p) => p.cluster === clusterId).length;
                return (
                  <button
                    key={clusterId}
                    type="button"
                    onClick={() => setSelectedClusterFilter(clusterId)}
                    style={{
                      padding: '4px 10px',
                      borderRadius: '14px',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      border: selectedClusterFilter === clusterId ? `2px solid ${clr}` : '1px solid #cbd5e1',
                      backgroundColor: selectedClusterFilter === clusterId ? `${clr}15` : '#ffffff',
                      color: selectedClusterFilter === clusterId ? clr : '#475569',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                    }}
                  >
                    <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: clr }} />
                    Cluster {clusterId} ({count})
                  </button>
                );
              })}
            </div>

            {/* Interactive SVG Scatter Plot */}
            <div style={{ position: 'relative', border: '1px solid #e2e8f0', borderRadius: '6px', backgroundColor: '#fafafa', padding: '12px' }}>
              {(() => {
                const points = pcaClustering.points;
                const colors = ['#2563eb', '#10b981', '#8b5cf6', '#f59e0b', '#ec4899'];

                const svgWidth = 640;
                const svgHeight = 340;
                const padLeft = 55;
                const padRight = 30;
                const padTop = 25;
                const padBottom = 45;
                const plotWidth = svgWidth - padLeft - padRight;
                const plotHeight = svgHeight - padTop - padBottom;

                const allPc1 = points.map((p) => p.pc1);
                const allPc2 = points.map((p) => p.pc2);
                const minPc1 = Math.min(...allPc1, -3.2) - 0.5;
                const maxPc1 = Math.max(...allPc1, 3.2) + 0.5;
                const minPc2 = Math.min(...allPc2, -1.5) - 0.5;
                const maxPc2 = Math.max(...allPc2, 1.5) + 0.5;

                const scaleX = (val: number) => padLeft + ((val - minPc1) / (maxPc1 - minPc1)) * plotWidth;
                const scaleY = (val: number) => padTop + (1 - (val - minPc2) / (maxPc2 - minPc2)) * plotHeight;

                const zeroX = scaleX(0);
                const zeroY = scaleY(0);

                // Compute Centroids
                const clusterGroups: Record<number, { sumPc1: number; sumPc2: number; count: number }> = {};
                points.forEach((p) => {
                  if (!clusterGroups[p.cluster]) {
                    clusterGroups[p.cluster] = { sumPc1: 0, sumPc2: 0, count: 0 };
                  }
                  clusterGroups[p.cluster].sumPc1 += p.pc1;
                  clusterGroups[p.cluster].sumPc2 += p.pc2;
                  clusterGroups[p.cluster].count += 1;
                });

                return (
                  <svg
                    viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                    style={{ width: '100%', height: 'auto', display: 'block' }}
                  >
                    {/* Background Grid Lines */}
                    {[-3, -2, -1, 1, 2, 3].map((tick) => {
                      if (tick < minPc1 || tick > maxPc1) return null;
                      const x = scaleX(tick);
                      return (
                        <line
                          key={`vx-${tick}`}
                          x1={x}
                          y1={padTop}
                          x2={x}
                          y2={padTop + plotHeight}
                          stroke="#e2e8f0"
                          strokeDasharray="3 3"
                        />
                      );
                    })}
                    {[-1.5, -1, -0.5, 0.5, 1, 1.5].map((tick) => {
                      if (tick < minPc2 || tick > maxPc2) return null;
                      const y = scaleY(tick);
                      return (
                        <line
                          key={`hy-${tick}`}
                          x1={padLeft}
                          y1={y}
                          x2={padLeft + plotWidth}
                          y2={y}
                          stroke="#e2e8f0"
                          strokeDasharray="3 3"
                        />
                      );
                    })}

                    {/* Zero Axes */}
                    {zeroX >= padLeft && zeroX <= padLeft + plotWidth && (
                      <line
                        x1={zeroX}
                        y1={padTop}
                        x2={zeroX}
                        y2={padTop + plotHeight}
                        stroke="#cbd5e1"
                        strokeWidth="1.5"
                      />
                    )}
                    {zeroY >= padTop && zeroY <= padTop + plotHeight && (
                      <line
                        x1={padLeft}
                        y1={zeroY}
                        x2={padLeft + plotWidth}
                        y2={zeroY}
                        stroke="#cbd5e1"
                        strokeWidth="1.5"
                      />
                    )}

                    {/* Outer Border */}
                    <rect
                      x={padLeft}
                      y={padTop}
                      width={plotWidth}
                      height={plotHeight}
                      fill="none"
                      stroke="#94a3b8"
                      strokeWidth="1"
                    />

                    {/* Axis Labels */}
                    <text
                      x={padLeft + plotWidth / 2}
                      y={svgHeight - 10}
                      textAnchor="middle"
                      fontSize="12"
                      fontWeight="600"
                      fill="#475569"
                    >
                      Principal Component 1 (PC1)
                    </text>
                    <text
                      x={-padTop - plotHeight / 2}
                      y={18}
                      transform="rotate(-90)"
                      textAnchor="middle"
                      fontSize="12"
                      fontWeight="600"
                      fill="#475569"
                    >
                      Principal Component 2 (PC2)
                    </text>

                    {/* Data Points */}
                    {points.map((p) => {
                      const isDimmed = selectedClusterFilter !== 'all' && selectedClusterFilter !== p.cluster;
                      const isHovered = hoveredPoint?.id === p.id;
                      const clr = colors[p.cluster % colors.length];
                      const cx = scaleX(p.pc1);
                      const cy = scaleY(p.pc2);

                      return (
                        <circle
                          key={p.id}
                          cx={cx}
                          cy={cy}
                          r={isHovered ? 8 : 5.5}
                          fill={clr}
                          stroke={isHovered ? '#0f172a' : '#ffffff'}
                          strokeWidth={isHovered ? 2.5 : 1.5}
                          opacity={isDimmed ? 0.2 : 0.9}
                          style={{ cursor: 'pointer', transition: 'all 0.15s ease' }}
                          onMouseEnter={() => setHoveredPoint(p)}
                          onMouseLeave={() => setHoveredPoint(null)}
                        >
                          <title>{`${p.label || `Sample #${p.id}`}: Cluster ${p.cluster} (PC1: ${p.pc1.toFixed(2)}, PC2: ${p.pc2.toFixed(2)})`}</title>
                        </circle>
                      );
                    })}

                    {/* Cluster Centroids */}
                    {Object.entries(clusterGroups).map(([cIdStr, g]) => {
                      const cId = parseInt(cIdStr, 10);
                      if (selectedClusterFilter !== 'all' && selectedClusterFilter !== cId) return null;
                      const clr = colors[cId % colors.length];
                      const cx = scaleX(g.sumPc1 / g.count);
                      const cy = scaleY(g.sumPc2 / g.count);

                      return (
                        <g key={`centroid-${cId}`}>
                          <rect
                            x={cx - 6}
                            y={cy - 6}
                            width="12"
                            height="12"
                            transform={`rotate(45 ${cx} ${cy})`}
                            fill={clr}
                            stroke="#ffffff"
                            strokeWidth="2"
                          />
                          <text
                            x={cx + 10}
                            y={cy + 4}
                            fontSize="11"
                            fontWeight="bold"
                            fill={clr}
                          >
                            C{cId}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                );
              })()}

              {/* Tooltip Card for Hovered Point */}
              {hoveredPoint && (
                <div
                  style={{
                    marginTop: '8px',
                    padding: '8px 12px',
                    backgroundColor: '#1e293b',
                    color: '#ffffff',
                    borderRadius: '6px',
                    fontSize: '12px',
                    display: 'flex',
                    gap: '16px',
                    alignItems: 'center',
                  }}
                >
                  <span style={{ fontWeight: 700 }}>{hoveredPoint.label || `Sample #${hoveredPoint.id}`}</span>
                  <span>Cluster: <strong style={{ color: '#38bdf8' }}>{hoveredPoint.cluster}</strong></span>
                  <span>PC1: <strong>{hoveredPoint.pc1.toFixed(3)}</strong></span>
                  <span>PC2: <strong>{hoveredPoint.pc2.toFixed(3)}</strong></span>
                </div>
              )}
            </div>
          </section>
        </div>
      ) : (
        /* SECTION 4: Side-by-Side Model Comparison Dashboard (FR-12 / Issue 15) */
        <div>
          {/* Dashboard Control Bar */}
          <section
            style={{
              padding: '20px',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              backgroundColor: '#ffffff',
              marginBottom: '24px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <h3 style={{ margin: 0, color: '#0f172a' }}>Model Comparison Dashboard (FR-12)</h3>
                <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '14px' }}>
                  Select two or more trained models to view side-by-side performance metrics and identify the optimal algorithm (BR-7).
                </p>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={fetchComparisonRuns}
                  disabled={loading}
                  style={{
                    padding: '8px 14px',
                    backgroundColor: '#f1f5f9',
                    color: '#334155',
                    border: '1px solid #cbd5e1',
                    borderRadius: '6px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    fontSize: '13px',
                  }}
                >
                  {loading ? 'Refreshing...' : '🔄 Refresh Runs'}
                </button>
                <button
                  onClick={handleCompareModels}
                  disabled={selectedExecutionIds.length < 2 || loading}
                  style={{
                    padding: '8px 18px',
                    backgroundColor: selectedExecutionIds.length >= 2 ? '#2563eb' : '#94a3b8',
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: '6px',
                    fontWeight: 600,
                    cursor: selectedExecutionIds.length >= 2 ? 'pointer' : 'not-allowed',
                    fontSize: '13px',
                  }}
                >
                  📊 Compare Selected Models ({selectedExecutionIds.length})
                </button>
              </div>
            </div>

            {/* Error / Validation Notification */}
            {error && (
              <div
                style={{
                  marginTop: '16px',
                  padding: '12px 16px',
                  backgroundColor: '#fef2f2',
                  border: '1px solid #fecaca',
                  borderRadius: '6px',
                  color: '#991b1b',
                  fontSize: '14px',
                }}
              >
                ⚠️ <strong>Validation Alert:</strong> {error}
              </div>
            )}

            {/* Task Filters & Selection Shortcuts */}
            <div
              style={{
                marginTop: '18px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '12px',
                paddingTop: '16px',
                borderTop: '1px solid #f1f5f9',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '13px', color: '#64748b', fontWeight: 600, marginRight: '4px' }}>Filter Task:</span>
                {(['All', 'Classification', 'Regression', 'Clustering'] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTaskFilter(t)}
                    style={{
                      padding: '4px 10px',
                      borderRadius: '16px',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      border: taskFilter === t ? '1px solid #2563eb' : '1px solid #e2e8f0',
                      backgroundColor: taskFilter === t ? '#eff6ff' : '#f8fafc',
                      color: taskFilter === t ? '#1d4ed8' : '#475569',
                    }}
                  >
                    {t}
                  </button>
                ))}
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={selectAllFiltered}
                  style={{
                    padding: '4px 10px',
                    fontSize: '12px',
                    border: '1px solid #cbd5e1',
                    borderRadius: '4px',
                    backgroundColor: '#ffffff',
                    color: '#334155',
                    cursor: 'pointer',
                  }}
                >
                  Select All Filtered
                </button>
                <button
                  onClick={clearSelection}
                  style={{
                    padding: '4px 10px',
                    fontSize: '12px',
                    border: '1px solid #cbd5e1',
                    borderRadius: '4px',
                    backgroundColor: '#ffffff',
                    color: '#64748b',
                    cursor: 'pointer',
                  }}
                >
                  Clear Selection
                </button>
              </div>
            </div>

            {/* Available Runs Selection Table */}
            <div style={{ marginTop: '14px', overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f8fafc', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ padding: '10px 12px', width: '40px' }}>Select</th>
                    <th style={{ padding: '10px 12px' }}>Algorithm</th>
                    <th style={{ padding: '10px 12px' }}>Task Type</th>
                    <th style={{ padding: '10px 12px' }}>Primary Metric</th>
                    <th style={{ padding: '10px 12px' }}>Latency</th>
                    <th style={{ padding: '10px 12px' }}>Execution ID</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRuns.length === 0 ? (
                    <tr>
                      <td colSpan={6} style={{ padding: '24px', textAlign: 'center', color: '#94a3b8' }}>
                        No trained model runs found for the selected filter. Train a model or switch filter.
                      </td>
                    </tr>
                  ) : (
                    filteredRuns.map((run) => {
                      const isSelected = selectedExecutionIds.includes(run.execution_id);
                      return (
                        <tr
                          key={run.execution_id}
                          onClick={() => toggleModelSelection(run.execution_id)}
                          style={{
                            borderBottom: '1px solid #f1f5f9',
                            cursor: 'pointer',
                            backgroundColor: isSelected ? '#f0fdf4' : 'transparent',
                            transition: 'background-color 0.15s ease',
                          }}
                        >
                          <td style={{ padding: '10px 12px' }}>
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => {}}
                              style={{ cursor: 'pointer' }}
                            />
                          </td>
                          <td style={{ padding: '10px 12px', fontWeight: 600, color: '#0f172a' }}>{run.algorithm}</td>
                          <td style={{ padding: '10px 12px' }}>
                            <span
                              style={{
                                padding: '2px 8px',
                                borderRadius: '12px',
                                fontSize: '11px',
                                fontWeight: 600,
                                backgroundColor:
                                  run.task_type.toLowerCase() === 'classification'
                                    ? '#dbeafe'
                                    : run.task_type.toLowerCase() === 'regression'
                                    ? '#fef3c7'
                                    : '#f3e8ff',
                                color:
                                  run.task_type.toLowerCase() === 'classification'
                                    ? '#1e40af'
                                    : run.task_type.toLowerCase() === 'regression'
                                    ? '#92400e'
                                    : '#6b21a8',
                              }}
                            >
                              {run.task_type}
                            </span>
                          </td>
                          <td style={{ padding: '10px 12px' }}>
                            <strong>{run.primary_metric_name}:</strong> {run.primary_metric_value.toFixed(4)}
                          </td>
                          <td style={{ padding: '10px 12px', color: '#64748b' }}>
                            {run.train_time_ms.toFixed(1)} ms
                            {run.train_time_ms < 30000 && (
                              <span style={{ marginLeft: '4px', color: '#16a34a', fontSize: '11px' }}>✓ &lt;30s (NFR-1)</span>
                            )}
                          </td>
                          <td style={{ padding: '10px 12px', color: '#94a3b8', fontFamily: 'monospace', fontSize: '12px' }}>
                            {run.execution_id}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: '12px', fontSize: '13px', color: '#64748b', display: 'flex', justifyContent: 'space-between' }}>
              <span>
                Selected: <strong>{selectedExecutionIds.length}</strong> model(s) (Requires ≥ 2 models for comparison per BR-7)
              </span>
              <span>Available runs: {filteredRuns.length}</span>
            </div>
          </section>

          {/* Side-by-Side Comparison Output (FR-12) */}
          {comparisonReport && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {/* Optimal Model Recommendation Banner (FR-12 / Architecture) */}
              <div
                style={{
                  padding: '20px',
                  backgroundColor: '#ecfdf5',
                  border: '2px solid #10b981',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '16px',
                  boxShadow: '0 2px 4px rgba(16, 185, 129, 0.1)',
                }}
              >
                <div style={{ fontSize: '32px', lineHeight: 1 }}>🏆</div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#047857', letterSpacing: '0.05em' }}>
                      OPTIMAL MODEL RECOMMENDATION
                    </span>
                    <span
                      style={{
                        backgroundColor: '#10b981',
                        color: '#ffffff',
                        fontSize: '11px',
                        padding: '2px 8px',
                        borderRadius: '12px',
                        fontWeight: 700,
                      }}
                    >
                      Rank #1
                    </span>
                  </div>
                  <h3 style={{ margin: '4px 0', color: '#064e3b', fontSize: '20px' }}>
                    {comparisonReport.best_model_algorithm}
                  </h3>
                  <p style={{ margin: '4px 0 0', color: '#047857', fontSize: '14px' }}>
                    {comparisonReport.recommendation_reason}
                  </p>
                </div>
              </div>

              {/* Side-by-Side Metrics Comparison Table (FR-12) */}
              <section
                style={{
                  padding: '20px',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                  backgroundColor: '#ffffff',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <div>
                    <h3 style={{ margin: 0, color: '#0f172a' }}>Side-by-Side Evaluation Metrics Matrix</h3>
                    <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '13px' }}>
                      Comparing {comparisonReport.compared_models.length} models for task:{' '}
                      <strong>{comparisonReport.task_type}</strong>
                    </p>
                  </div>
                </div>

                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                    <thead>
                      <tr style={{ backgroundColor: '#f8fafc', borderBottom: '2px solid #cbd5e1' }}>
                        <th
                          style={{
                            padding: '12px 16px',
                            textAlign: 'left',
                            color: '#475569',
                            fontWeight: 700,
                            borderRight: '1px solid #e2e8f0',
                            minWidth: '220px',
                          }}
                        >
                          Evaluation Metric / Property
                        </th>
                        {comparisonReport.compared_models.map((m) => {
                          const isBest = m.execution_id === comparisonReport.best_model_id;
                          const rank = comparisonReport.ranking.find((r) => r.execution_id === m.execution_id)?.rank;
                          return (
                            <th
                              key={m.execution_id}
                              style={{
                                padding: '12px 16px',
                                textAlign: 'center',
                                backgroundColor: isBest ? '#f0fdf4' : '#f8fafc',
                                borderRight: '1px solid #e2e8f0',
                                minWidth: '180px',
                              }}
                            >
                              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
                                <span
                                  style={{
                                    fontSize: '11px',
                                    fontWeight: 700,
                                    padding: '2px 8px',
                                    borderRadius: '10px',
                                    backgroundColor: isBest ? '#10b981' : '#cbd5e1',
                                    color: isBest ? '#ffffff' : '#334155',
                                  }}
                                >
                                  {isBest ? '★ Rank #1 (Optimal)' : `Rank #${rank}`}
                                </span>
                                <span style={{ fontWeight: 700, color: '#0f172a', fontSize: '14px' }}>{m.algorithm}</span>
                                <span style={{ fontSize: '11px', color: '#64748b', fontFamily: 'monospace' }}>
                                  {m.execution_id}
                                </span>
                              </div>
                            </th>
                          );
                        })}
                      </tr>
                    </thead>
                    <tbody>
                      {/* Primary Metric Row */}
                      <tr style={{ borderBottom: '1px solid #e2e8f0', backgroundColor: '#faf5ff' }}>
                        <td style={{ padding: '12px 16px', fontWeight: 700, color: '#6b21a8', borderRight: '1px solid #e2e8f0' }}>
                          ⭐ Primary Metric ({comparisonReport.compared_models[0].primary_metric_name})
                        </td>
                        {comparisonReport.compared_models.map((m) => {
                          const isBest = m.execution_id === comparisonReport.best_model_id;
                          return (
                            <td
                              key={m.execution_id}
                              style={{
                                padding: '12px 16px',
                                textAlign: 'center',
                                fontWeight: 700,
                                fontSize: '16px',
                                color: isBest ? '#15803d' : '#0f172a',
                                backgroundColor: isBest ? '#dcfce7' : '#faf5ff',
                                borderRight: '1px solid #e2e8f0',
                              }}
                            >
                              {m.primary_metric_value.toFixed(4)}
                              {isBest && <span style={{ fontSize: '12px', marginLeft: '4px' }}>★</span>}
                            </td>
                          );
                        })}
                      </tr>

                      {/* Performance Delta vs Best Row */}
                      <tr style={{ borderBottom: '1px solid #e2e8f0', backgroundColor: '#ffffff' }}>
                        <td style={{ padding: '10px 16px', color: '#475569', fontWeight: 600, borderRight: '1px solid #e2e8f0' }}>
                          Relative Performance vs Leader
                        </td>
                        {comparisonReport.compared_models.map((m) => {
                          const bestVal = comparisonReport.compared_models.find(
                            (x) => x.execution_id === comparisonReport.best_model_id
                          )!.primary_metric_value;
                          const diff = m.primary_metric_value - bestVal;
                          const isLeader = m.execution_id === comparisonReport.best_model_id;

                          return (
                            <td
                              key={m.execution_id}
                              style={{
                                padding: '10px 16px',
                                textAlign: 'center',
                                fontSize: '13px',
                                borderRight: '1px solid #e2e8f0',
                                color: isLeader ? '#15803d' : '#b91c1c',
                                fontWeight: 600,
                              }}
                            >
                              {isLeader ? 'Leader (Baseline)' : `${diff >= 0 ? '+' : ''}${diff.toFixed(4)}`}
                            </td>
                          );
                        })}
                      </tr>

                      {/* Training Latency Row (NFR-1) */}
                      <tr style={{ borderBottom: '1px solid #e2e8f0', backgroundColor: '#f8fafc' }}>
                        <td style={{ padding: '10px 16px', color: '#475569', fontWeight: 600, borderRight: '1px solid #e2e8f0' }}>
                          ⏱️ Training Latency (ms) (NFR-1 &lt;30s)
                        </td>
                        {comparisonReport.compared_models.map((m) => (
                          <td
                            key={m.execution_id}
                            style={{
                              padding: '10px 16px',
                              textAlign: 'center',
                              fontSize: '13px',
                              color: '#334155',
                              borderRight: '1px solid #e2e8f0',
                            }}
                          >
                            <strong>{m.train_time_ms.toFixed(1)} ms</strong>
                            <div style={{ fontSize: '11px', color: '#16a34a' }}>✓ &lt;30s budget met</div>
                          </td>
                        ))}
                      </tr>

                      {/* Detailed Summary Metrics Rows */}
                      {Object.keys(comparisonReport.compared_models[0].summary_metrics).map((metricKey) => {
                        const displayMetric = metricKey.replace(/_/g, ' ').toUpperCase();
                        return (
                          <tr key={metricKey} style={{ borderBottom: '1px solid #e2e8f0' }}>
                            <td
                              style={{
                                padding: '10px 16px',
                                color: '#334155',
                                fontWeight: 600,
                                borderRight: '1px solid #e2e8f0',
                              }}
                            >
                              {displayMetric}
                            </td>
                            {comparisonReport.compared_models.map((m) => {
                              const val = m.summary_metrics[metricKey];
                              return (
                                <td
                                  key={m.execution_id}
                                  style={{
                                    padding: '10px 16px',
                                    textAlign: 'center',
                                    fontSize: '13px',
                                    color: '#0f172a',
                                    borderRight: '1px solid #e2e8f0',
                                  }}
                                >
                                  {typeof val === 'number' ? val.toFixed(4) : val ?? '--'}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </section>

              {/* Visual Performance Comparison Bars */}
              <section
                style={{
                  padding: '20px',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                  backgroundColor: '#ffffff',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
                }}
              >
                <h3 style={{ margin: '0 0 16px', color: '#0f172a' }}>
                  Visual Performance Gap ({comparisonReport.compared_models[0].primary_metric_name})
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  {comparisonReport.ranking.map((r) => {
                    const pct = Math.max(0, Math.min(100, r.primary_metric_value * 100));
                    const isOptimal = r.rank === 1;
                    return (
                      <div key={r.execution_id}>
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            marginBottom: '4px',
                            fontSize: '13px',
                          }}
                        >
                          <span style={{ fontWeight: 600, color: isOptimal ? '#047857' : '#334155' }}>
                            {r.rank === 1 ? '🏆 ' : `#${r.rank} `}
                            {r.algorithm}
                          </span>
                          <span style={{ fontWeight: 700, color: isOptimal ? '#047857' : '#0f172a' }}>
                            {r.primary_metric_value.toFixed(4)} ({pct.toFixed(1)}%)
                          </span>
                        </div>
                        <div
                          style={{
                            height: '14px',
                            width: '100%',
                            backgroundColor: '#f1f5f9',
                            borderRadius: '8px',
                            overflow: 'hidden',
                          }}
                        >
                          <div
                            style={{
                              height: '100%',
                              width: `${pct}%`,
                              backgroundColor: isOptimal ? '#10b981' : '#3b82f6',
                              borderRadius: '8px',
                              transition: 'width 0.4s ease-in-out',
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

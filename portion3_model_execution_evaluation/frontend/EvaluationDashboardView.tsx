import React, { useState } from 'react';

// ============================================================================
// PORTION 3: MODEL EXECUTION, METRICS & COMPARISON SKELETON
// Assigned Teammate: Shreya Patil (PES2UG24CS484)
// Scope: FR-8 (Algorithm Execution), FR-9 (Classification Metrics), FR-10 (Regression Metrics),
//        FR-11 (Clustering & 2D PCA), FR-12 (Side-by-Side Model Comparison)
// ============================================================================

export interface EvaluationDashboardProps {
  datasetId?: string;
  apiBaseUrl?: string;
}

export const EvaluationDashboardView: React.FC<EvaluationDashboardProps> = ({
  datasetId = 'dataset-001',
  apiBaseUrl = 'http://localhost:8003',
}) => {
  const [activeTab, setActiveTab] = useState<'results' | 'comparison'>('results');

  // TODO: Shreya Patil - Implement fetch to GET /api/v1/models/comparison/{datasetId} (FR-12)
  const handleFetchComparison = async () => {
    console.log('Fetch comparison for dataset:', datasetId);
    // TODO: Shreya Patil - Call `${apiBaseUrl}/api/v1/models/comparison/${datasetId}`
  };

  return (
    <div style={{ padding: '24px', fontFamily: 'sans-serif', maxWidth: '1000px', margin: '0 auto' }}>
      <header style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '16px', marginBottom: '24px' }}>
        <h2 style={{ margin: 0, color: '#1e293b' }}>Portion 3: Model Execution, Evaluation & Comparison</h2>
        <p style={{ margin: '4px 0 0', color: '#64748b' }}>
          Assigned to: <strong>Shreya Patil (PES2UG24CS484)</strong> | Service Port: <strong>8003</strong>
        </p>
        <div style={{ marginTop: '12px', display: 'flex', gap: '8px' }}>
          <button onClick={() => setActiveTab('results')} style={{ padding: '6px 14px', borderRadius: '4px', border: '1px solid #cbd5e1' }}>
            Model Results
          </button>
          <button onClick={() => setActiveTab('comparison')} style={{ padding: '6px 14px', borderRadius: '4px', border: '1px solid #cbd5e1' }}>
            Model Comparison Dashboard (FR-12)
          </button>
        </div>
      </header>

      {activeTab === 'results' ? (
        <div>
          {/* SKELETON SECTION 1: Metrics Summary Cards (FR-9, FR-10, FR-11) */}
          <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px', marginBottom: '20px' }}>
            <h3>Step 1: Evaluation Metrics (FR-9, FR-10, FR-11)</h3>
            <p style={{ color: '#64748b', fontSize: '14px' }}>
              {/* TODO: Shreya Patil - Render metrics summary cards for Accuracy, F1, MAE, R2, or Silhouette Score */}
              [TODO: Shreya Patil - Render performance metric summary cards here]
            </p>
          </section>

          {/* SKELETON SECTION 2: Confusion Matrix Grid (FR-9) */}
          <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px', marginBottom: '20px' }}>
            <h3>Step 2: Confusion Matrix (FR-9 - Classification)</h3>
            <p style={{ color: '#64748b', fontSize: '14px' }}>
              {/* TODO: Shreya Patil - Render interactive K x K confusion matrix table */}
              [TODO: Shreya Patil - Render confusion matrix heatmap grid here]
            </p>
          </section>

          {/* SKELETON SECTION 3: 2D PCA Cluster Scatter Plot (FR-11) */}
          <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
            <h3>Step 3: 2D PCA Cluster Scatter Plot (FR-11 - Clustering)</h3>
            <p style={{ color: '#64748b', fontSize: '14px' }}>
              {/* TODO: Shreya Patil - Render 2D scatter plot using PC1 and PC2 coordinates with cluster coloring */}
              [TODO: Shreya Patil - Render 2D PCA cluster scatter plot using Recharts or Plotly.js here]
            </p>
          </section>
        </div>
      ) : (
        /* SKELETON SECTION 4: Side-by-Side Model Comparison Dashboard (FR-12) */
        <section style={{ padding: '20px', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
          <h3>Model Comparison Dashboard (FR-12)</h3>
          <p style={{ color: '#64748b', fontSize: '14px' }}>
            {/* TODO: Shreya Patil - Render side-by-side comparison table of all trained models */}
            [TODO: Shreya Patil - Render side-by-side comparison table across multiple model runs here]
          </p>
          <button onClick={handleFetchComparison} style={{ padding: '8px 16px', backgroundColor: '#2563eb', color: '#fff', border: 'none', borderRadius: '6px' }}>
            Refresh Comparison Table
          </button>
        </section>
      )}
    </div>
  );
};

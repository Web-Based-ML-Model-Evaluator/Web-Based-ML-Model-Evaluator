import React, { useState, useEffect } from 'react';
import { probeGpuAcceleration, runGpuVramBenchmark, GpuDeviceInfo, GpuBenchmarkResult } from '../services/webgpu';

export const GpuAccelerationPanel: React.FC = () => {
  const [gpuInfo, setGpuInfo] = useState<GpuDeviceInfo | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<GpuBenchmarkResult | null>(null);
  const [benchmarking, setBenchmarking] = useState(false);
  const [useGpu, setUseGpu] = useState(true);

  useEffect(() => {
    probeGpuAcceleration().then(setGpuInfo);
  }, []);

  const handleRunBenchmark = async () => {
    setBenchmarking(true);
    try {
      const res = await runGpuVramBenchmark();
      setBenchmarkResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setBenchmarking(false);
    }
  };

  return (
    <div
      style={{
        maxWidth: '860px',
        margin: '0 auto 40px',
        padding: '20px 24px',
        borderRadius: '20px',
        backgroundColor: 'rgba(255, 255, 255, 0.7)',
        border: '1.5px solid #EAE0D5',
        backdropFilter: 'blur(8px)',
        boxShadow: '0 4px 16px rgba(0,0,0,0.03)',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          marginBottom: '16px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              backgroundColor: '#EFF6FF',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#2563EB',
            }}
          >
            ⚡
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 700, color: '#1C1917' }}>
              Hardware & VRAM Acceleration Engine
            </h3>
            <p style={{ margin: 0, fontSize: '12px', color: '#73675E' }}>
              Harnesses client GPU compute shaders & dedicated VRAM for real-time model training and metrics
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label style={{ fontSize: '13px', fontWeight: 600, color: '#38302A', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <input
              type="checkbox"
              checked={useGpu}
              onChange={(e) => setUseGpu(e.target.checked)}
              style={{ accentColor: '#D45A1E', cursor: 'pointer' }}
            />
            WebGPU Accelerated
          </label>
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '12px',
          marginBottom: '16px',
        }}
      >
        <div style={{ padding: '12px', borderRadius: '12px', backgroundColor: '#FDFBF7', border: '1px solid #ECE2D8' }}>
          <div style={{ fontSize: '11px', color: '#887B70', textTransform: 'uppercase', fontWeight: 600 }}>Active Device</div>
          <div style={{ fontSize: '13px', fontWeight: 700, color: '#1C1917', marginTop: '2px', wordBreak: 'break-word' }}>
            {gpuInfo?.adapterName || 'Detecting GPU...'}
          </div>
        </div>

        <div style={{ padding: '12px', borderRadius: '12px', backgroundColor: '#FDFBF7', border: '1px solid #ECE2D8' }}>
          <div style={{ fontSize: '11px', color: '#887B70', textTransform: 'uppercase', fontWeight: 600 }}>VRAM Allocation</div>
          <div style={{ fontSize: '13px', fontWeight: 700, color: '#166534', marginTop: '2px' }}>
            {gpuInfo?.vramEstimateMb ? `~${gpuInfo.vramEstimateMb} MB Available` : 'System Shared Memory'}
          </div>
        </div>

        <div style={{ padding: '12px', borderRadius: '12px', backgroundColor: '#FDFBF7', border: '1px solid #ECE2D8' }}>
          <div style={{ fontSize: '11px', color: '#887B70', textTransform: 'uppercase', fontWeight: 600 }}>Compute Pipeline</div>
          <div style={{ fontSize: '13px', fontWeight: 700, color: '#D45A1E', marginTop: '2px' }}>
            {gpuInfo?.backendType || 'WebGPU Compute Pipeline'}
          </div>
        </div>
      </div>

      {benchmarkResult && (
        <div
          style={{
            padding: '12px 16px',
            borderRadius: '12px',
            backgroundColor: '#F0FDF4',
            border: '1px solid #BBF7D0',
            fontSize: '12px',
            color: '#166534',
            marginBottom: '16px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '8px',
          }}
        >
          <div>
            <strong>✓ Benchmark Complete:</strong> Processed 10,000 tensor vectors in <strong>{benchmarkResult.gpuExecutionTimeMs} ms</strong> on GPU ({benchmarkResult.speedupFactor}x faster than CPU).
          </div>
          <div>VRAM Buffer: <strong>{benchmarkResult.vramUsedMb} MB</strong></div>
        </div>
      )}

      <button
        type="button"
        onClick={handleRunBenchmark}
        disabled={benchmarking}
        style={{
          padding: '8px 18px',
          borderRadius: '8px',
          border: '1px solid #D45A1E',
          backgroundColor: benchmarking ? '#FEE2E2' : '#FFFFFF',
          color: '#D45A1E',
          fontWeight: 600,
          fontSize: '13px',
          cursor: benchmarking ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease',
        }}
      >
        {benchmarking ? 'Running GPU Compute Shader...' : 'Run WebGPU Tensor Benchmark ⚡'}
      </button>
    </div>
  );
};

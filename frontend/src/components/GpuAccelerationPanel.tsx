import React, { useEffect, useState } from 'react';
import { DeviceInfo, GpuBenchmarkResult, gpuLabel, probeDevice, runGpuBenchmark } from '../services/webgpu';
import { InfoTooltip } from './InfoTooltip';

const NOT_REPORTED = 'Not reported by this browser';

const card: React.CSSProperties = {
  padding: '12px',
  borderRadius: '12px',
  backgroundColor: '#FDFBF7',
  border: '1px solid #ECE2D8',
};
const cardLabel: React.CSSProperties = {
  fontSize: '11px',
  color: '#887B70',
  textTransform: 'uppercase',
  fontWeight: 600,
};
const cardValue: React.CSSProperties = {
  fontSize: '13px',
  fontWeight: 700,
  color: '#1C1917',
  marginTop: '2px',
  wordBreak: 'break-word',
};

/** The visitor's own device: everything shown here is read from the browser. */
export const GpuAccelerationPanel: React.FC = () => {
  const [info, setInfo] = useState<DeviceInfo | null>(null);
  const [benchmark, setBenchmark] = useState<GpuBenchmarkResult | null>(null);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
  const [benchmarking, setBenchmarking] = useState(false);

  useEffect(() => {
    probeDevice().then(setInfo);
  }, []);

  const handleRunBenchmark = async () => {
    setBenchmarking(true);
    setBenchmark(null);
    setBenchmarkError(null);
    try {
      setBenchmark(await runGpuBenchmark());
    } catch (err: any) {
      setBenchmarkError(err?.message ?? String(err));
    } finally {
      setBenchmarking(false);
    }
  };

  const gpu = info?.webgpu;

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
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
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
          <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 700, color: '#1C1917', display: 'flex', alignItems: 'center' }}>
            Your Device
            <InfoTooltip
              title="Runs on your device"
              content="This site does all its work in your browser. Built-in models train on your CPU (WebAssembly); uploaded ONNX models run on your GPU through WebGPU when it is available. The values below are what your browser reports."
            />
          </h3>
          <p style={{ margin: 0, fontSize: '12px', color: '#73675E' }}>
            Your data and models stay in this browser tab; the server only delivered the app.
          </p>
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
        <div style={card}>
          <div style={cardLabel}>CPU threads</div>
          <div style={cardValue}>{info ? (info.cpuThreads ?? NOT_REPORTED) : 'Detecting...'}</div>
        </div>
        <div style={card}>
          <div style={cardLabel}>Memory (approx.)</div>
          <div style={cardValue}>
            {info ? (info.deviceMemoryGb != null ? `about ${info.deviceMemoryGb} GB` : NOT_REPORTED) : 'Detecting...'}
          </div>
        </div>
        <div style={card}>
          <div style={cardLabel}>GPU (WebGPU)</div>
          <div style={{ ...cardValue, color: gpu?.supported ? '#166534' : '#73675E' }}>
            {info ? (gpu?.supported ? gpuLabel(info) : 'Not available') : 'Detecting...'}
          </div>
          {gpu && !gpu.supported && gpu.reason && (
            <div style={{ fontSize: '11px', color: '#887B70', marginTop: '4px' }}>{gpu.reason}</div>
          )}
        </div>
        <div style={card}>
          <div style={cardLabel}>Largest GPU buffer</div>
          <div style={cardValue}>
            {gpu?.supported ? (gpu.maxBufferMb != null ? `${gpu.maxBufferMb} MB` : NOT_REPORTED) : '—'}
          </div>
          <div style={{ fontSize: '11px', color: '#887B70', marginTop: '4px' }}>
            Browsers don't reveal total VRAM.
          </div>
        </div>
      </div>

      {benchmark && (
        <div
          style={{
            padding: '12px 16px',
            borderRadius: '12px',
            backgroundColor: '#F0FDF4',
            border: '1px solid #BBF7D0',
            fontSize: '12px',
            color: '#166534',
            marginBottom: '16px',
          }}
        >
          <strong>Measured on this device:</strong> distances from {benchmark.points.toLocaleString()} points to{' '}
          {benchmark.centers} centers ({benchmark.features} features) took <strong>{benchmark.gpuMs} ms</strong> on the GPU
          (including copying results back) and <strong>{benchmark.cpuMs} ms</strong> in JavaScript on the CPU
          {benchmark.gpuMs > 0 && <> ({(benchmark.cpuMs / benchmark.gpuMs).toFixed(1)}× ratio)</>}. Results agree to within{' '}
          {benchmark.maxDifference.toExponential(1)}; GPU buffers used: {benchmark.gpuBufferMb} MB.
        </div>
      )}
      {benchmarkError && (
        <div
          role="alert"
          style={{
            padding: '12px 16px',
            borderRadius: '12px',
            backgroundColor: '#F8FAFC',
            border: '1px solid #E2E8F0',
            fontSize: '12px',
            color: '#475569',
            marginBottom: '16px',
          }}
        >
          {benchmarkError}
        </div>
      )}

      <button
        type="button"
        onClick={handleRunBenchmark}
        disabled={benchmarking || !gpu?.supported}
        title={gpu?.supported ? undefined : 'Needs WebGPU'}
        style={{
          padding: '8px 18px',
          borderRadius: '8px',
          border: '1px solid #D45A1E',
          backgroundColor: benchmarking ? '#FEE2E2' : '#FFFFFF',
          color: '#D45A1E',
          fontWeight: 600,
          fontSize: '13px',
          cursor: benchmarking || !gpu?.supported ? 'not-allowed' : 'pointer',
          opacity: gpu?.supported ? 1 : 0.6,
        }}
      >
        {benchmarking ? 'Measuring GPU and CPU...' : 'Measure GPU vs CPU on this device'}
      </button>
    </div>
  );
};

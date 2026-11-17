/**
 * What this device offers for running ML, as reported by the browser.
 *
 * Every value here is read from browser APIs. When the browser doesn't report something
 * (for example, no browser exposes how much VRAM a GPU has), it is left out rather than
 * guessed.
 *
 * Where work actually runs:
 *   - built-in models train in the engine worker on the CPU (WebAssembly);
 *   - uploaded ONNX models run on the GPU through WebGPU when it is available.
 */

// WebGPU buffer usage flags (GPUBufferUsage), spelled out so this file has no DOM type dependency
const BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};
const MAP_MODE_READ = 0x0001;

export interface DeviceInfo {
  /** Logical CPU threads (navigator.hardwareConcurrency). */
  cpuThreads: number | null;
  /** Approximate RAM in GB, rounded and capped by the browser (navigator.deviceMemory; Chromium only). */
  deviceMemoryGb: number | null;
  webgpu: {
    supported: boolean;
    /** Why WebGPU isn't usable, when it isn't. */
    reason?: string;
    vendor?: string;
    architecture?: string;
    description?: string;
    /** Largest single GPU buffer the browser allows, in MB (a limit, not the GPU's VRAM). */
    maxBufferMb?: number;
    maxStorageBufferMb?: number;
  };
}

export interface GpuBenchmarkResult {
  device: string;
  points: number;
  centers: number;
  features: number;
  /** Upload, compute and read back on the GPU. */
  gpuMs: number;
  /** The same distance computation in plain JavaScript on the CPU. */
  cpuMs: number;
  /** Largest absolute difference between GPU and CPU results (float32 rounding). */
  maxDifference: number;
  gpuBufferMb: number;
}

let cachedDevice: any = null;
let cachedInfo: Promise<DeviceInfo> | null = null;

const toMb = (bytes: unknown) =>
  typeof bytes === 'number' && bytes > 0 ? Math.round(bytes / (1024 * 1024)) : undefined;

async function probe(): Promise<DeviceInfo> {
  const nav = typeof navigator === 'undefined' ? ({} as any) : (navigator as any);
  const info: DeviceInfo = {
    cpuThreads: typeof nav.hardwareConcurrency === 'number' ? nav.hardwareConcurrency : null,
    deviceMemoryGb: typeof nav.deviceMemory === 'number' ? nav.deviceMemory : null,
    webgpu: { supported: false },
  };

  if (!nav.gpu) {
    info.webgpu.reason = 'This browser does not provide WebGPU.';
    return info;
  }
  try {
    const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      info.webgpu.reason = 'WebGPU is present, but no usable GPU adapter was found.';
      return info;
    }
    const device = await adapter.requestDevice();
    cachedDevice = device;
    const details = adapter.info ?? (await adapter.requestAdapterInfo?.()) ?? {};
    info.webgpu = {
      supported: true,
      vendor: details.vendor || undefined,
      architecture: details.architecture || undefined,
      description: details.description || details.device || undefined,
      maxBufferMb: toMb(device.limits?.maxBufferSize),
      maxStorageBufferMb: toMb(device.limits?.maxStorageBufferBindingSize),
    };
  } catch (err) {
    info.webgpu.reason = `WebGPU could not start: ${String((err as Error)?.message ?? err)}`;
  }
  return info;
}

/** Reads what the browser reports about this device (cached after the first call). */
export function probeDevice(): Promise<DeviceInfo> {
  cachedInfo ??= probe();
  return cachedInfo;
}

/** A short name for the GPU, from whatever the browser reported. */
export function gpuLabel(info: DeviceInfo): string {
  const w = info.webgpu;
  if (!w.supported) return 'No WebGPU';
  return w.description || [w.vendor, w.architecture].filter(Boolean).join(' ') || 'WebGPU GPU';
}

/** WGSL compute shader: Euclidean distance from every point to every center. */
const DISTANCE_MATRIX_WGSL = `
struct Uniforms {
  num_points: u32,
  num_centers: u32,
  num_features: u32,
  _padding: u32,
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<f32>;
@group(0) @binding(2) var<storage, read> centers: array<f32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let point_idx = global_id.x;
  if (point_idx >= params.num_points) {
    return;
  }
  let feat_count = params.num_features;
  for (var c: u32 = 0u; c < params.num_centers; c = c + 1u) {
    var sum: f32 = 0.0;
    for (var f: u32 = 0u; f < feat_count; f = f + 1u) {
      let diff = points[point_idx * feat_count + f] - centers[c * feat_count + f];
      sum = sum + diff * diff;
    }
    distances[point_idx * params.num_centers + c] = sqrt(sum);
  }
}
`;

function cpuDistances(points: Float32Array, centers: Float32Array, n: number, k: number, f: number) {
  const out = new Float32Array(n * k);
  for (let p = 0; p < n; p++) {
    for (let c = 0; c < k; c++) {
      let sum = 0;
      for (let j = 0; j < f; j++) {
        const d = points[p * f + j] - centers[c * f + j];
        sum += d * d;
      }
      out[p * k + c] = Math.sqrt(sum);
    }
  }
  return out;
}

/**
 * Computes a point-to-center distance matrix (the core step of K-Means / KNN) on the GPU
 * and on the CPU, checks that both agree, and reports both measured times.
 * Rejects when WebGPU isn't available; nothing is estimated.
 */
export async function runGpuBenchmark(points = 200_000, centers = 16, features = 16): Promise<GpuBenchmarkResult> {
  const info = await probeDevice();
  if (!info.webgpu.supported || !cachedDevice) {
    throw new Error(info.webgpu.reason ?? 'WebGPU is not available, so there is no GPU to benchmark.');
  }
  const device = cachedDevice;

  const pointsData = new Float32Array(points * features);
  for (let i = 0; i < pointsData.length; i++) pointsData[i] = Math.random();
  const centersData = new Float32Array(centers * features);
  for (let i = 0; i < centersData.length; i++) centersData[i] = Math.random();
  const outBytes = points * centers * 4;

  const gpuStart = performance.now();
  const make = (size: number, usage: number) => device.createBuffer({ size, usage });
  const pointsBuffer = make(pointsData.byteLength, BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST);
  const centersBuffer = make(centersData.byteLength, BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST);
  const distancesBuffer = make(outBytes, BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC);
  const readBuffer = make(outBytes, BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST);
  const uniformBuffer = make(16, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST);
  device.queue.writeBuffer(pointsBuffer, 0, pointsData);
  device.queue.writeBuffer(centersBuffer, 0, centersData);
  device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([points, centers, features, 0]));

  try {
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: DISTANCE_MATRIX_WGSL }), entryPoint: 'main' },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: pointsBuffer } },
        { binding: 2, resource: { buffer: centersBuffer } },
        { binding: 3, resource: { buffer: distancesBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(points / 64));
    pass.end();
    encoder.copyBufferToBuffer(distancesBuffer, 0, readBuffer, 0, outBytes);
    device.queue.submit([encoder.finish()]);

    // Timing includes reading the results back, so it is comparable with the CPU run
    await readBuffer.mapAsync(MAP_MODE_READ);
    const gpuResult = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    const gpuMs = performance.now() - gpuStart;

    const cpuStart = performance.now();
    const cpuResult = cpuDistances(pointsData, centersData, points, centers, features);
    const cpuMs = performance.now() - cpuStart;

    let maxDifference = 0;
    for (let i = 0; i < cpuResult.length; i++) {
      maxDifference = Math.max(maxDifference, Math.abs(cpuResult[i] - gpuResult[i]));
    }

    return {
      device: gpuLabel(info),
      points,
      centers,
      features,
      gpuMs: Math.round(gpuMs * 100) / 100,
      cpuMs: Math.round(cpuMs * 100) / 100,
      maxDifference,
      gpuBufferMb: Math.round(((pointsData.byteLength + centersData.byteLength + 2 * outBytes) / 1048576) * 10) / 10,
    };
  } finally {
    for (const b of [pointsBuffer, centersBuffer, distancesBuffer, readBuffer, uniformBuffer]) b.destroy();
  }
}

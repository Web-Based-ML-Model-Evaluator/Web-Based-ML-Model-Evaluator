/**
 * Runs uploaded ONNX models on this device with ONNX Runtime Web (inside the engine worker).
 *
 * WebGPU is used when the browser offers it, so the model runs on the GPU and its tensors
 * live in VRAM; otherwise ONNX Runtime's WebAssembly backend runs it on the CPU. The model
 * file never leaves the device. ONNX Runtime is loaded only when a model is first used.
 */
import type * as OrtTypes from 'onnxruntime-web';

type Ort = typeof OrtTypes;
type Session = OrtTypes.InferenceSession;

export type ExecutionProvider = 'webgpu' | 'wasm';

export interface LoadedModel {
  model_id: string;
  filename: string;
  file_format: string;
  file_size_bytes: number;
  task_type: 'classification' | 'regression';
  input_name: string;
  /** Expected input shape; null entries are dynamic (e.g. the batch size). */
  input_shape: Array<number | null> | null;
  output_names: string[];
  execution_provider: ExecutionProvider;
}

export interface ModelPredictions {
  predictions: Array<string | number>;
  /** "class_index" when outputs were scores and the highest one was taken. */
  kind: 'label' | 'class_index';
  inference_ms: number;
}

let ortPromise: Promise<Ort> | null = null;
const models = new Map<string, { session: Session; info: LoadedModel }>();

function loadOrt(): Promise<Ort> {
  ortPromise ??= import('onnxruntime-web/webgpu').then((ort) => {
    // Multi-threaded WebAssembly needs cross-origin isolation, which static hosts like
    // GitHub Pages can't enable, so use one thread unless the page is isolated.
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1))
      : 1;
    return ort as unknown as Ort;
  });
  return ortPromise;
}

export class ModelError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

/** Formats that need Python (or another runtime) to execute and can't run in a browser. */
const NEEDS_EXPORT = ['.pt', '.pth', '.pkl', '.pickle', '.joblib', '.h5', '.hdf5', '.keras'];

export async function loadModel(
  filename: string,
  bytes: Uint8Array,
  taskType: string,
): Promise<LoadedModel> {
  const lower = filename.toLowerCase();
  if (!lower.endsWith('.onnx')) {
    const ext = lower.slice(lower.lastIndexOf('.'));
    throw new ModelError(
      'UNSUPPORTED_MODEL_FORMAT',
      NEEDS_EXPORT.includes(ext)
        ? `${ext} files need Python to run, so they can't be executed in the browser. Export the model to ONNX first (e.g. torch.onnx.export, skl2onnx or tf2onnx) and upload the .onnx file.`
        : 'Only ONNX models (.onnx) can be evaluated. Export your model to ONNX and upload that file.',
    );
  }
  if (taskType !== 'classification' && taskType !== 'regression') {
    throw new ModelError('UNSUPPORTED_TASK_TYPE', 'Choose classification or regression for an uploaded model.');
  }

  const ort = await loadOrt();
  const providers: ExecutionProvider[] = 'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm'];
  let session: Session | null = null;
  let used: ExecutionProvider = 'wasm';
  let lastError: unknown = null;
  for (const provider of providers) {
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: [provider] });
      used = provider;
      break;
    } catch (e) {
      lastError = e; // e.g. no usable GPU adapter: fall back to the CPU
    }
  }
  if (!session) {
    throw new ModelError(
      'MODEL_LOAD_FAILED',
      `The ONNX model could not be loaded: ${String((lastError as Error)?.message ?? lastError)}`,
    );
  }
  if (session.inputNames.length !== 1) {
    await session.release();
    throw new ModelError(
      'UNSUPPORTED_MODEL_INPUTS',
      `This model has ${session.inputNames.length} inputs. Only models with a single [rows, features] input are supported.`,
    );
  }

  const meta = (session as any).inputMetadata?.[0];
  const shape: Array<number | null> | null = Array.isArray(meta?.shape)
    ? meta.shape.map((d: unknown) => (typeof d === 'number' && d > 0 ? d : null))
    : null;

  const info: LoadedModel = {
    model_id: `onnx-model-${crypto.randomUUID()}`,
    filename,
    file_format: 'ONNX (.onnx)',
    file_size_bytes: bytes.byteLength,
    task_type: taskType as LoadedModel['task_type'],
    input_name: session.inputNames[0],
    input_shape: shape,
    output_names: [...session.outputNames],
    execution_provider: used,
  };
  models.set(info.model_id, { session, info });
  return info;
}

export function listModels(): LoadedModel[] {
  return [...models.values()].map((m) => m.info);
}

export async function releaseAll(): Promise<void> {
  const all = [...models.values()];
  models.clear();
  await Promise.all(all.map((m) => m.session.release().catch(() => undefined)));
}

export function getModel(modelId: string): LoadedModel {
  const entry = models.get(modelId);
  if (!entry) throw new ModelError('MODEL_NOT_FOUND', 'That model is no longer loaded. Upload it again.');
  return entry.info;
}

/** Runs the model on every row and returns one prediction per row. */
export async function predict(modelId: string, features: number[][]): Promise<ModelPredictions> {
  const entry = models.get(modelId);
  if (!entry) throw new ModelError('MODEL_NOT_FOUND', 'That model is no longer loaded. Upload it again.');
  const { session, info } = entry;
  const ort = await loadOrt();

  const rows = features.length;
  const width = features[0]?.length ?? 0;
  const expected = info.input_shape?.[1];
  if (expected != null && expected !== width) {
    throw new ModelError(
      'FEATURE_COUNT_MISMATCH',
      `The model expects ${expected} features per row, but ${width} feature column(s) were selected.`,
    );
  }
  // Models exported with a fixed batch size (often 1) are run in chunks of that size
  const batch = info.input_shape?.[0] ?? rows;
  const inputType = (session as any).inputMetadata?.[0]?.type === 'float64' ? 'float64' : 'float32';

  const outputs: OrtTypes.Tensor[][] = [];
  const started = performance.now();
  for (let start = 0; start < rows; start += batch) {
    const chunk = features.slice(start, start + batch);
    const flat = chunk.flat();
    const data = inputType === 'float64' ? Float64Array.from(flat) : Float32Array.from(flat);
    const tensor = new ort.Tensor(inputType, data, [chunk.length, width]);
    let result: OrtTypes.InferenceSession.ReturnType;
    try {
      result = await session.run({ [info.input_name]: tensor });
    } catch (e) {
      throw new ModelError('MODEL_RUN_FAILED', `The model failed while predicting: ${String((e as Error)?.message ?? e)}`);
    }
    outputs.push(session.outputNames.map((name) => result[name]));
  }
  const inferenceMs = performance.now() - started;

  // Pick the output that holds one prediction per row
  for (let o = 0; o < session.outputNames.length; o++) {
    const parts = outputs.map((p) => p[o]);
    if (!parts.every((t) => t && typeof (t as any).dims !== 'undefined')) continue; // not a tensor
    const dims = parts[0].dims;
    const perRow = dims.length >= 2 ? dims.slice(1).reduce((a, b) => a * b, 1) : 1;
    const values = parts.flatMap((t) => Array.from(t.data as ArrayLike<number | bigint | string>));
    if (values.length !== rows * perRow) continue;

    if (info.task_type === 'regression') {
      if (perRow !== 1 || typeof values[0] === 'string') continue;
      return { predictions: values.map(Number), kind: 'label', inference_ms: inferenceMs };
    }
    if (perRow === 1) {
      // Labels (text or numbers), e.g. scikit-learn's "label" output
      const labels = values.map((v) => (typeof v === 'string' ? v : Number(v)));
      return { predictions: labels, kind: 'label', inference_ms: inferenceMs };
    }
    // Scores / probabilities per class: take the highest
    const picked: number[] = [];
    for (let r = 0; r < rows; r++) {
      let best = 0;
      for (let c = 1; c < perRow; c++) {
        if (Number(values[r * perRow + c]) > Number(values[r * perRow + best])) best = c;
      }
      picked.push(best);
    }
    return { predictions: picked, kind: 'class_index', inference_ms: inferenceMs };
  }
  throw new ModelError(
    'UNSUPPORTED_MODEL_OUTPUTS',
    'None of the model outputs has one prediction per row. Export the model so it outputs labels, class scores or values as a tensor.',
  );
}

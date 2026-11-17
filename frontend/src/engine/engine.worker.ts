/// <reference lib="webworker" />
/**
 * Runs the WebAssembly ML engine off the main thread, so training never freezes the page.
 *
 * The engine (Rust compiled to WebAssembly) is loaded once into this worker. Every dataset
 * and result it holds lives in this worker's memory, on the visitor's own device. Uploaded
 * ONNX models also run here (see ./onnx.ts), next to the data they are evaluated on.
 */
import init, { Engine } from './pkg/ml_engine.js';
import { getModel, listModels, loadModel, ModelError, predict, releaseAll } from './onnx';
import type { WorkerReply, WorkerRequest } from './protocol';

let engine: Engine | null = null;
const ready = init().then(() => {
  engine = new Engine();
});

/** Calls the Rust engine; throws its structured error string on failure. */
function callRust<T>(method: string, args: unknown, bytes?: Uint8Array): T {
  return JSON.parse(engine!.call(method, JSON.stringify(args ?? {}), bytes)) as T;
}

const str = (args: any, key: string): string => String(args?.[key] ?? '');

/** Methods handled in JavaScript; everything else goes to the Rust engine. */
async function handle(method: string, args: any, bytes?: Uint8Array): Promise<unknown> {
  switch (method) {
    case 'load_model':
      return loadModel(str(args, 'filename'), bytes ?? new Uint8Array(), str(args, 'task_type'));
    case 'list_models':
      return listModels();
    case 'evaluate_model': {
      const model = getModel(str(args, 'model_id'));
      const common = {
        dataset_id: str(args, 'dataset_id'),
        task_type: model.task_type,
        target_column: str(args, 'target_column'),
      };
      const inputs = callRust<{ features: number[][] }>('model_inputs', {
        ...common,
        feature_columns: args?.feature_columns ?? [],
      });
      const output = await predict(model.model_id, inputs.features);
      const device = model.execution_provider === 'webgpu' ? 'GPU via WebGPU' : 'CPU via WebAssembly';
      const result = callRust<Record<string, unknown>>('record_external_predictions', {
        ...common,
        model_name: `Uploaded ONNX: ${model.filename}`,
        predictions: output.predictions,
        kind: output.kind,
        inference_ms: output.inference_ms,
      });
      return { ...result, execution_provider: model.execution_provider, device };
    }
    case 'purge_session': {
      await releaseAll();
      return callRust(method, args);
    }
    default:
      return callRust(method, args, bytes);
  }
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { id, method, args, bytes } = event.data;
  let reply: WorkerReply;
  try {
    await ready;
    reply = { id, ok: true, result: await handle(method, args, bytes) };
  } catch (thrown) {
    if (typeof thrown === 'string') {
      // A structured engine error (FR-13), e.g. {"error":{"code":"DATASET_NOT_FOUND",...}}
      reply = { id, ok: false, error: JSON.parse(thrown).error, fatal: false };
    } else if (thrown instanceof ModelError) {
      reply = { id, ok: false, error: { code: thrown.code, message: thrown.message }, fatal: false };
    } else if (!(thrown instanceof WebAssembly.RuntimeError)) {
      const message = String((thrown as Error)?.message ?? thrown);
      reply = { id, ok: false, error: { code: 'ENGINE_ERROR', message }, fatal: false };
    } else {
      // A WebAssembly trap (e.g. a panic inside the ML library). The engine's memory can't be
      // trusted after this, so the page restarts the worker.
      reply = {
        id,
        ok: false,
        fatal: true,
        error: {
          code: 'ENGINE_CRASHED',
          message: `The engine stopped unexpectedly (${String((thrown as Error)?.message ?? thrown)}). It was restarted; please upload your dataset again.`,
        },
      };
    }
  }
  self.postMessage(reply);
};

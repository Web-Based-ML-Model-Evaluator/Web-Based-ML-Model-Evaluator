/// <reference lib="webworker" />
/**
 * Runs the WebAssembly ML engine off the main thread, so training never freezes the page.
 *
 * The engine (Rust compiled to WebAssembly) is loaded once into this worker. Every dataset
 * and result it holds lives in this worker's memory, on the visitor's own device.
 */
import init, { Engine } from './pkg/ml_engine.js';
import type { WorkerReply, WorkerRequest } from './protocol';

let engine: Engine | null = null;
const ready = init().then(() => {
  engine = new Engine();
});

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { id, method, args, bytes } = event.data;
  let reply: WorkerReply;
  try {
    await ready;
    const json = engine!.call(method, JSON.stringify(args ?? {}), bytes ?? undefined);
    reply = { id, ok: true, result: JSON.parse(json) };
  } catch (thrown) {
    if (typeof thrown === 'string') {
      // A structured engine error (FR-13), e.g. {"error":{"code":"DATASET_NOT_FOUND",...}}
      reply = { id, ok: false, error: JSON.parse(thrown).error, fatal: false };
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

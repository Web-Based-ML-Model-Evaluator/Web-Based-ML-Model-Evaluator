/**
 * Page-side handle to the on-device engine.
 *
 * Starts the engine worker on first use and turns each call into a promise. If the engine
 * crashes, the worker is replaced with a fresh one (its in-memory session is lost, and the
 * caller gets an ENGINE_CRASHED error explaining that).
 */
import type { EngineErrorDetail, WorkerReply, WorkerRequest } from './protocol';

export class EngineError extends Error {
  readonly code: string;
  readonly details?: unknown;

  constructor(detail: EngineErrorDetail) {
    super(detail.message);
    this.name = 'EngineError';
    this.code = detail.code;
    this.details = detail.details;
  }
}

type Pending = { resolve: (value: unknown) => void; reject: (error: Error) => void };

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();

function failAll(error: Error) {
  for (const p of pending.values()) p.reject(error);
  pending.clear();
}

function startWorker(): Worker {
  const w = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' });
  w.onmessage = (event: MessageEvent<WorkerReply>) => {
    const reply = event.data;
    const p = pending.get(reply.id);
    if (!p) return;
    pending.delete(reply.id);
    if (reply.ok) {
      p.resolve(reply.result);
    } else {
      p.reject(new EngineError(reply.error));
      if (reply.fatal) restartWorker();
    }
  };
  w.onerror = (event) => {
    event.preventDefault();
    failAll(
      new EngineError({
        code: 'ENGINE_LOAD_FAILED',
        message:
          'The ML engine could not start in this browser. A browser with WebAssembly and module Web Worker support is required.',
      }),
    );
    restartWorker();
  };
  return w;
}

function restartWorker() {
  worker?.terminate();
  worker = null;
}

/** Calls an engine method; rejects with an EngineError on failure. */
export function callEngine<T>(method: string, args?: unknown, bytes?: Uint8Array): Promise<T> {
  if (!worker) worker = startWorker();
  const id = nextId++;
  const request: WorkerRequest = { id, method, args, bytes };
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    // Transfer the file buffer instead of copying it
    worker!.postMessage(request, bytes ? [bytes.buffer] : []);
  });
}

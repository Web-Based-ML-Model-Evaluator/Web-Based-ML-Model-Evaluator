/** Messages between the page and the engine worker. */

export interface WorkerRequest {
  id: number;
  method: string;
  args?: unknown;
  /** File contents (e.g. an uploaded CSV). */
  bytes?: Uint8Array;
}

export interface EngineErrorDetail {
  code: string;
  message: string;
  details?: unknown;
}

export type WorkerReply =
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: EngineErrorDetail; fatal: boolean };

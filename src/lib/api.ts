// Thin client for the local qls-service API. Images are referenced by URL (with the
// token as a query parameter) so the webview streams them directly — no base64 copies.

import type {
  AppConfig, AppErrorBody, GenerationRecord, GenerationRequest, InputRecord, JobView, Plan, ServiceEvent, SetupReport, StatusResponse,
} from "./types";

export class ApiError extends Error {
  constructor(public body: AppErrorBody, public status: number) {
    super(body.message);
  }
}

let base = "";
let token = "";

export function configure(baseUrl: string, tok: string) {
  base = baseUrl;
  token = tok;
}

export const isConfigured = () => base !== "";

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(base + path, {
      ...init,
      headers: { Authorization: `Bearer ${token}`, ...(init.body && typeof init.body === "string" ? { "Content-Type": "application/json" } : {}), ...(init.headers || {}) },
    });
  } catch (e) {
    throw new ApiError({ code: "server_unavailable", message: "The local service is not reachable.", detail: String(e) }, 0);
  }
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    throw new ApiError(data?.error ?? { code: "internal", message: `HTTP ${res.status}` }, res.status);
  }
  return data as T;
}

const json = (body: unknown) => JSON.stringify(body);

export const api = {
  status: () => call<StatusResponse>("/status"),
  setup: () => call<SetupReport>("/setup"),
  config: () => call<AppConfig>("/config"),
  saveConfig: (c: AppConfig) => call<{ saved: boolean; backend_restart_needed: boolean }>("/config", { method: "PUT", body: json(c) }),
  startBackend: () => call<{ ready: boolean }>("/backend/start", { method: "POST" }),
  stopBackend: () => call<{ stopped: boolean }>("/backend/stop", { method: "POST" }),
  restartBackend: () => call<{ ready: boolean }>("/backend/restart", { method: "POST" }),
  diagnostics: () => call<{ text: string }>("/diagnostics"),
  plan: (r: GenerationRequest) => call<Plan>("/plan", { method: "POST", body: json(r) }),
  submit: (r: GenerationRequest) => call<JobView>("/jobs", { method: "POST", body: json(r) }),
  job: (id: string) => call<JobView>(`/jobs/${id}`),
  jobs: () => call<JobView[]>("/jobs"),
  cancel: (id: string) => call<JobView>(`/jobs/${id}/cancel`, { method: "POST" }),
  upload: (bytes: Blob | ArrayBuffer | Uint8Array, name?: string) =>
    call<InputRecord>("/inputs", { method: "POST", body: bytes as BodyInit, headers: { "Content-Type": "application/octet-stream", ...(name ? { "x-filename": encodeURIComponent(name) } : {}) } }),
  importPath: (path: string) => call<InputRecord>("/inputs/import", { method: "POST", body: json({ path }) }),
  input: (id: string) => call<InputRecord>(`/inputs/${id}`),
  gallery: (q: { q?: string; favorites?: boolean; limit?: number; offset?: number; sort?: string }) => {
    const p = new URLSearchParams();
    if (q.q) p.set("q", q.q);
    if (q.favorites) p.set("favorites", "true");
    if (q.limit) p.set("limit", String(q.limit));
    if (q.offset) p.set("offset", String(q.offset));
    if (q.sort) p.set("sort", q.sort);
    return call<{ items: GenerationRecord[]; total: number }>(`/gallery?${p}`);
  },
  generation: (id: string) => call<GenerationRecord>(`/gallery/${id}`),
  setFavorite: (id: string, favorite: boolean) => call<GenerationRecord>(`/gallery/${id}`, { method: "PATCH", body: json({ favorite }) }),
  deleteGeneration: (id: string) => call<{ deleted: string }>(`/gallery/${id}`, { method: "DELETE" }),
  exportGeneration: (id: string, path: string) => call<{ path: string }>(`/gallery/${id}/export`, { method: "POST", body: json({ path, overwrite: true }) }),
};

export const urls = {
  input: (id: string) => `${base}/inputs/${id}/image?token=${token}`,
  inputThumb: (id: string) => `${base}/inputs/${id}/thumb?token=${token}`,
  gen: (id: string) => `${base}/gallery/${id}/image?token=${token}`,
  genThumb: (id: string) => `${base}/gallery/${id}/thumb?token=${token}`,
};

/** Server-sent events with automatic reconnect. Returns an unsubscribe function. */
export function subscribe(onEvent: (e: ServiceEvent) => void, onState?: (connected: boolean) => void): () => void {
  let es: EventSource | null = null;
  let closed = false;
  let retry: number | undefined;
  const open = () => {
    es = new EventSource(`${base}/events?token=${token}`);
    es.onopen = () => onState?.(true);
    es.onmessage = (m) => {
      try {
        onEvent(JSON.parse(m.data));
      } catch {
        /* ignore malformed */
      }
    };
    es.onerror = () => {
      onState?.(false);
      es?.close();
      if (!closed) retry = window.setTimeout(open, 1500);
    };
  };
  open();
  return () => {
    closed = true;
    window.clearTimeout(retry);
    es?.close();
  };
}

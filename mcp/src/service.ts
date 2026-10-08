// Client for the local kokon-service. Finds it via service.json, starts it if needed
// (the service in turn starts/reuses sd-server), and holds an "mcp" lease while alive.

import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export interface ServiceInfo {
  port: number;
  token: string;
  pid: number;
  version: string;
}

export class ServiceError extends Error {
  constructor(
    message: string,
    public code = "service_error",
    public suggestion?: string,
    public detail?: string,
  ) {
    super(message);
  }
}

export function dataDir(): string {
  if (process.env.QLS_DATA_DIR) return process.env.QLS_DATA_DIR;
  const base = process.env.LOCALAPPDATA ?? path.join(process.env.USERPROFILE ?? "C:\\", "AppData", "Local");
  // The service moves the folder of the app's old name (QwenLocalStudio) to KokonData on its
  // first start; until then the old one is read.
  const [next, old] = [path.join(base, "KokonData"), path.join(base, "QwenLocalStudio")];
  return !fs.existsSync(next) && fs.existsSync(old) ? old : next;
}

function readInfo(): ServiceInfo | null {
  try {
    return JSON.parse(fs.readFileSync(path.join(dataDir(), "service.json"), "utf8"));
  } catch {
    return null;
  }
}

async function ping(port: number): Promise<boolean> {
  try {
    const r = await fetch(`http://127.0.0.1:${port}/api/v1/ping`, { signal: AbortSignal.timeout(1500) });
    return r.ok;
  } catch {
    return false;
  }
}

export function findServiceExe(): string | null {
  const local = process.env.LOCALAPPDATA ?? "";
  const candidates = [
    process.env.QLS_SERVICE_EXE,
    path.join(path.dirname(process.execPath), "kokon-service.exe"),
    path.join(local, "Kokon", "kokon-service.exe"),
    path.join(process.env.ProgramFiles ?? "C:\\Program Files", "Kokon", "kokon-service.exe"),
    // Development checkout (mcp/build -> target/...)
    path.resolve(__dirname, "..", "..", "target", "release", "kokon-service.exe"),
    path.resolve(__dirname, "..", "..", "target", "debug", "kokon-service.exe"),
  ];
  for (const c of candidates) {
    if (c && fs.existsSync(c)) return c;
  }
  return null;
}

export class ServiceClient {
  private info: ServiceInfo | null = null;
  private starting: Promise<ServiceInfo> | null = null;
  private heartbeat: NodeJS.Timeout | null = null;
  readonly clientId = `mcp-${process.pid}`;

  async ensure(): Promise<ServiceInfo> {
    if (this.info && (await ping(this.info.port))) return this.info;
    if (!this.starting) {
      this.starting = this.start().finally(() => (this.starting = null));
    }
    this.info = await this.starting;
    this.startHeartbeat();
    return this.info;
  }

  private async start(): Promise<ServiceInfo> {
    const existing = readInfo();
    if (existing && (await ping(existing.port))) return existing;
    const exe = findServiceExe();
    if (!exe) {
      throw new ServiceError(
        "Kokon is not installed (kokon-service.exe not found).",
        "not_installed",
        "Install Kokon, or set QLS_SERVICE_EXE to the path of kokon-service.exe.",
      );
    }
    // Detached: the service outlives this MCP process and exits on its own when idle.
    const child = spawn(exe, [], { detached: true, stdio: "ignore", windowsHide: true });
    child.unref();
    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 300));
      const i = readInfo();
      if (i && (await ping(i.port))) return i;
    }
    throw new ServiceError("The Kokon service did not start.", "service_unavailable", `See ${path.join(dataDir(), "logs", "service.log")}`);
  }

  private startHeartbeat() {
    if (this.heartbeat) return;
    const beat = () => this.call("POST", "/session/heartbeat", { client_id: this.clientId, kind: "mcp", pid: process.pid }).catch(() => {});
    beat();
    this.heartbeat = setInterval(beat, 15_000);
    this.heartbeat.unref();
  }

  async release() {
    if (this.heartbeat) clearInterval(this.heartbeat);
    if (!this.info) return;
    try {
      await fetch(`http://127.0.0.1:${this.info.port}/api/v1/session/release`, {
        method: "POST",
        headers: { Authorization: `Bearer ${this.info.token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ client_id: this.clientId }),
        signal: AbortSignal.timeout(2000),
      });
    } catch {
      /* service already gone */
    }
  }

  async call<T = unknown>(method: string, route: string, body?: unknown, retry = true): Promise<T> {
    const info = await this.ensure();
    let res: Response;
    try {
      res = await fetch(`http://127.0.0.1:${info.port}/api/v1${route}`, {
        method,
        headers: { Authorization: `Bearer ${info.token}`, ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      if (retry) {
        this.info = null;
        return this.call(method, route, body, false);
      }
      throw new ServiceError("The Kokon service is not reachable.", "service_unavailable", undefined, String(e));
    }
    const text = await res.text();
    const data = text ? JSON.parse(text) : null;
    if (!res.ok) {
      const err = data?.error ?? {};
      throw new ServiceError(err.message ?? `HTTP ${res.status}`, err.code ?? "error", err.suggestion, err.detail);
    }
    return data as T;
  }
}

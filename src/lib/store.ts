// Minimal global store (useSyncExternalStore) — no extra dependencies.

import { useSyncExternalStore } from "react";
import { api, ApiError, subscribe } from "./api";
import { buildRequest, emptyComposer, newKey, persistentAdvanced, refFromGen, refFromInput, type CompareMode, type Composer } from "./logic";
import type { AppErrorBody, GenerationRecord, GenerationRequest, InputRecord, JobView, ServerState, StatusResponse } from "./types";

export type Overlay = null | "gallery" | "settings" | "inpaint" | "expand";

export interface SessionItem {
  key: string;
  kind: "gen" | "input" | "job";
  gen?: GenerationRecord;
  input?: InputRecord;
  jobId?: string;
  /** Request that produced a pending job (for "Try again"). */
  request?: GenerationRequest;
}

export interface Toast {
  id: string;
  message: string;
  suggestion?: string;
  detail?: string;
  kind: "error" | "info";
}

export interface State {
  connected: boolean;
  connectError: string | null;
  server: { state: ServerState; owned: boolean; error: AppErrorBody | null };
  status: StatusResponse | null;
  session: SessionItem[];
  selected: string | null;
  jobs: Record<string, JobView>;
  composer: Composer;
  overlay: Overlay;
  toasts: Toast[];
  galleryVersion: number;
  needsSetup: boolean;
  /** Before/after view for results made from another image; kept while switching results. */
  compare: CompareMode;
  /** Flicker period in ms. */
  flickerMs: number;
}

const PREFS_KEY = "qls.prefs.v1";

function loadPrefs(): Partial<Composer> {
  try {
    const p: Partial<Composer> = JSON.parse(localStorage.getItem(PREFS_KEY) || "{}");
    // Older versions also saved CFG/steps/strength, which then overrode every preset.
    return { ...p, advanced: persistentAdvanced(p.advanced) };
  } catch {
    return {};
  }
}

function savePrefs(c: Composer) {
  try {
    const { quality, resolution, aspect, customWidth, customHeight } = c;
    const advanced = persistentAdvanced(c.advanced);
    localStorage.setItem(PREFS_KEY, JSON.stringify({ quality, resolution, aspect, customWidth, customHeight, advanced }));
  } catch {
    /* storage unavailable */
  }
}

let state: State = {
  connected: false,
  connectError: null,
  server: { state: "stopped", owned: false, error: null },
  status: null,
  session: [],
  selected: null,
  jobs: {},
  composer: emptyComposer(loadPrefs()),
  overlay: null,
  toasts: [],
  galleryVersion: 0,
  needsSetup: false,
  compare: "off",
  flickerMs: 500,
};

const listeners = new Set<() => void>();

export function getState() {
  return state;
}

export function setState(patch: Partial<State> | ((s: State) => Partial<State>)) {
  const p = typeof patch === "function" ? patch(state) : patch;
  state = { ...state, ...p };
  if (p.composer) savePrefs(state.composer);
  listeners.forEach((l) => l());
}

export function useStore<T>(sel: (s: State) => T): T {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => sel(state),
  );
}

export function updateComposer(patch: Partial<Composer>) {
  setState((s) => ({ composer: { ...s.composer, ...patch } }));
}

// ---------------------------------------------------------------------------------------
// Toasts
// ---------------------------------------------------------------------------------------

export function toastError(e: unknown) {
  const body: AppErrorBody = e instanceof ApiError ? e.body : { code: "internal", message: e instanceof Error ? e.message : String(e) };
  pushToast({ id: newKey(), message: body.message, suggestion: body.suggestion, detail: body.detail, kind: "error" });
}

export function toastInfo(message: string) {
  pushToast({ id: newKey(), message, kind: "info" });
}

function pushToast(t: Toast) {
  setState((s) => ({ toasts: [...s.toasts.slice(-3), t] }));
  if (t.kind === "info") window.setTimeout(() => dismissToast(t.id), 3500);
}

export function dismissToast(id: string) {
  setState((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
}

// ---------------------------------------------------------------------------------------
// Connection and events
// ---------------------------------------------------------------------------------------

let unsub: (() => void) | null = null;

export async function refreshStatus() {
  try {
    const st = await api.status();
    setState({ status: st, server: { state: st.server.state, owned: st.server.owned, error: st.server.error } });
    return st;
  } catch {
    return null;
  }
}

export function startEvents() {
  unsub?.();
  unsub = subscribe(
    (ev) => {
      if (ev.type === "job") onJob(ev.job);
      else if (ev.type === "server") setState({ server: { state: ev.state, owned: ev.owned, error: ev.error } });
      else if (ev.type === "gallery") setState((s) => ({ galleryVersion: s.galleryVersion + 1 }));
    },
    (ok) => {
      setState({ connected: ok });
      if (ok) {
        // Catch up on anything missed while disconnected.
        api.jobs().then((js) => js.forEach(onJob)).catch(() => {});
        refreshStatus();
      }
    },
  );
}

function onJob(job: JobView) {
  setState((s) => {
    if (!s.session.some((i) => i.jobId === job.id) && !s.jobs[job.id]) {
      // A job from another client (e.g. MCP): track it but do not add it to this session.
      return { jobs: { ...s.jobs, [job.id]: job } };
    }
    const jobs = { ...s.jobs, [job.id]: job };
    if (job.status !== "done") return { jobs };
    // Replace the pending item with one item per result.
    const idx = s.session.findIndex((i) => i.jobId === job.id && i.kind === "job");
    if (idx < 0) return { jobs };
    const results: SessionItem[] = job.results.map((g) => ({ key: newKey(), kind: "gen", gen: g, jobId: job.id }));
    const session = [...s.session.slice(0, idx), ...results, ...s.session.slice(idx + 1)];
    const wasSelected = s.selected === s.session[idx].key;
    return { jobs, session, selected: wasSelected && results[0] ? results[0].key : s.selected };
  });
  if (job.status === "failed" && job.error) toastError(new ApiError(job.error, 500));
}

// ---------------------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------------------

export function addSessionItem(item: SessionItem, select = true) {
  setState((s) => ({ session: [...s.session, item], selected: select ? item.key : s.selected }));
}

export function addInputs(records: InputRecord[]) {
  if (!records.length) return;
  setState((s) => {
    const refs = [...s.composer.refs, ...records.map(refFromInput)].slice(0, 10);
    const items: SessionItem[] = records.map((r) => ({ key: newKey(), kind: "input", input: r }));
    return {
      // The first image sets the shape again (a ratio picked for text-to-image would distort it).
      composer: { ...s.composer, refs, outpaint: null, aspect: s.composer.refs.length ? s.composer.aspect : "auto" },
      session: [...s.session, ...items],
      selected: items[items.length - 1].key,
    };
  });
  if (getState().composer.refs.length >= 10) toastInfo("Up to 10 reference images are supported.");
}

export async function submitRequest(req: GenerationRequest, opts: { clearComposer?: boolean } = {}) {
  try {
    const job = await api.submit(req);
    const item: SessionItem = { key: newKey(), kind: "job", jobId: job.id, request: req };
    setState((s) => ({
      jobs: { ...s.jobs, [job.id]: job },
      session: [...s.session, item],
      selected: item.key,
      composer: opts.clearComposer ? { ...s.composer, seed: null } : s.composer,
    }));
  } catch (e) {
    toastError(e);
  }
}

export async function submitComposer() {
  const s = getState();
  const parent = s.session.find((i) => i.key === s.selected)?.gen?.id ?? null;
  const req = buildRequest(s.composer, parent);
  if (!req.prompt) {
    toastInfo("Describe what to create or change.");
    return;
  }
  await submitRequest(req, { clearComposer: true });
}

export async function cancelJob(jobId: string) {
  try {
    const j = await api.cancel(jobId);
    onJob(j);
  } catch (e) {
    toastError(e);
  }
}

/** Use a result or input as the (only) image to edit. */
export function editItem(item: SessionItem) {
  const ref = item.gen ? refFromGen(item.gen) : item.input ? refFromInput(item.input) : null;
  if (!ref) return;
  updateComposer({ refs: [ref], mask: null, outpaint: null, mode: "auto", aspect: "auto" });
  focusPrompt();
}

/** Append as an additional reference. */
export function addAsReference(item: SessionItem) {
  const ref = item.gen ? refFromGen(item.gen) : item.input ? refFromInput(item.input) : null;
  if (!ref) return;
  setState((s) => ({ composer: { ...s.composer, refs: [...s.composer.refs, ref].slice(0, 10), outpaint: null, aspect: s.composer.refs.length ? s.composer.aspect : "auto" } }));
  toastInfo(`Added as image ${Math.min(10, getState().composer.refs.length)}`);
}

/** Open the inpaint / expand editor on an item (it becomes image 1). */
export function openEditorFor(item: SessionItem, overlay: "inpaint" | "expand") {
  const s = getState();
  const first = s.composer.refs[0];
  const src = item.gen ? `gen:${item.gen.id}` : item.input ? `input:${item.input.id}` : null;
  if (!first || first.source !== src) editItem(item);
  setState({ overlay });
}

export function focusPrompt() {
  window.setTimeout(() => document.getElementById("prompt")?.focus(), 30);
}

export function removeSessionItem(key: string) {
  setState((s) => {
    const session = s.session.filter((i) => i.key !== key);
    return { session, selected: s.selected === key ? session[session.length - 1]?.key ?? null : s.selected };
  });
}

export function selectedItem(s: State): SessionItem | null {
  return s.session.find((i) => i.key === s.selected) ?? null;
}

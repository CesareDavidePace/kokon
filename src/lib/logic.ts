// Pure UI logic (unit-tested in logic.test.ts).

import type { Advanced, GenerationRecord, GenerationRequest, InputRecord, JobView, Mode, OutpaintSpec, Plan, Quality, ResolutionMode, Size, UpscaleMethod } from "./types";

export interface Ref {
  key: string;
  /** "input:<id>" or "gen:<id>" */
  source: string;
  width: number;
  height: number;
  name: string;
  /** HEIC source with an HDR gain map. */
  hdr?: boolean;
}

export interface Composer {
  prompt: string;
  negative: string;
  refs: Ref[];
  /** Mask painted on refs[0] (input id of the mask PNG). */
  mask: { inputId: string; refKey: string } | null;
  outpaint: OutpaintSpec | null;
  mode: Mode;
  aspect: string;
  quality: Quality;
  resolution: ResolutionMode;
  customWidth: number;
  customHeight: number;
  seed: number | null;
  advanced: Advanced;
}

export const emptyComposer = (prefs?: Partial<Composer>): Composer => ({
  prompt: "",
  negative: "",
  refs: [],
  mask: null,
  outpaint: null,
  mode: "auto",
  aspect: prefs?.aspect ?? "auto",
  quality: prefs?.quality ?? "balanced",
  resolution: prefs?.resolution ?? "auto",
  customWidth: prefs?.customWidth ?? 1024,
  customHeight: prefs?.customHeight ?? 1024,
  seed: null,
  advanced: prefs?.advanced ?? {},
});

let counter = 0;
export const newKey = () => `k${Date.now().toString(36)}${(counter++).toString(36)}`;

export const refFromInput = (r: InputRecord): Ref => ({ key: newKey(), source: `input:${r.id}`, width: r.width, height: r.height, name: r.original_name ? decodeSafe(r.original_name) : "Image", hdr: r.hdr || undefined });
export const refFromGen = (g: GenerationRecord): Ref => ({ key: newKey(), source: `gen:${g.id}`, width: g.width, height: g.height, name: "Result" });

function decodeSafe(s: string) {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

/** Moves a reference, keeping all others in order. */
export function moveRef(refs: Ref[], from: number, to: number): Ref[] {
  if (from === to || from < 0 || from >= refs.length) return refs;
  const out = refs.slice();
  const [r] = out.splice(from, 1);
  out.splice(Math.max(0, Math.min(to, out.length)), 0, r);
  return out;
}

export function hasOutpaint(o: OutpaintSpec | null): boolean {
  return !!o && (o.left > 0 || o.right > 0 || o.top > 0 || o.bottom > 0);
}

/** Composer state -> stable service request. */
export function buildRequest(c: Composer, parentId?: string | null): GenerationRequest {
  const maskValid = c.mask && c.refs[0] && c.mask.refKey === c.refs[0].key;
  const custom = c.resolution === "custom";
  // Inpaint and expand keep the source's shape; create, edit and combine take the chosen ratio.
  const shaped = !maskValid && !hasOutpaint(c.outpaint);
  return {
    prompt: c.prompt.trim(),
    negative_prompt: c.negative.trim() || null,
    mode: c.mode,
    images: c.mode === "generate" ? [] : c.refs.map((r) => r.source),
    mask: maskValid ? `input:${c.mask!.inputId}` : null,
    outpaint: hasOutpaint(c.outpaint) ? c.outpaint : null,
    quality: c.quality,
    resolution: c.resolution,
    aspect_ratio: shaped && c.aspect !== "auto" && !custom ? c.aspect : null,
    width: custom ? c.customWidth : null,
    height: custom ? c.customHeight : null,
    seed: c.seed,
    preserve_dimensions: true,
    advanced: cleanAdvanced(c.advanced),
    client: "ui",
    parent_id: parentId ?? null,
  };
}

/** Advanced fields that change how the model samples. They last for the session only: a value
 * typed once (e.g. CFG 4 during a test) must not silently override the presets after a restart.
 * `flow_shift` is no longer in the UI but may still sit in old saved preferences. */
const SAMPLING_OVERRIDES = ["steps", "cfg", "sampler", "scheduler", "strength", "flow_shift"];

/** Labels of the sampling overrides currently set, e.g. ["CFG 4", "Steps 20"]. */
export function activeOverrides(a: Advanced): string[] {
  const label: Record<string, string> = { steps: "Steps", cfg: "CFG", sampler: "Sampler", scheduler: "Scheduler", strength: "Strength", flow_shift: "Flow shift" };
  return Object.entries(cleanAdvanced(a))
    .filter(([k]) => SAMPLING_OVERRIDES.includes(k))
    .map(([k, v]) => `${label[k]} ${v}`);
}

/** `a` without the session-only sampling overrides (what is kept in saved preferences). */
export function persistentAdvanced(a: Advanced | undefined): Advanced {
  const out = cleanAdvanced(a ?? {});
  for (const k of SAMPLING_OVERRIDES) delete (out as Record<string, unknown>)[k];
  return out;
}

function cleanAdvanced(a: Advanced): Advanced {
  const out: Advanced = {};
  for (const [k, v] of Object.entries(a)) {
    if (v !== null && v !== undefined && v !== "") (out as Record<string, unknown>)[k] = v;
  }
  return out;
}

/** Restores composer fields from a stored generation (gallery "reopen"). */
export function composerFromRecord(g: GenerationRecord, base: Composer): Composer {
  const req = g.request;
  const refs: Ref[] = g.references.map((r) => ({ key: newKey(), source: `input:${r.input_id}`, width: r.width, height: r.height, name: `Image ${r.index + 1}` }));
  const c: Composer = {
    ...base,
    prompt: g.prompt,
    negative: g.negative_prompt ?? "",
    refs,
    mask: g.mask_input_id && refs[0] ? { inputId: g.mask_input_id, refKey: refs[0].key } : null,
    outpaint: g.outpaint ? { ...g.outpaint, units: "pixels" } : null,
    // Upscales are started from the image toolbar, not the composer.
    mode: req?.mode && req.mode !== "upscale" ? req.mode : "auto",
    quality: g.quality ?? base.quality,
    resolution: req?.resolution ?? base.resolution,
    aspect: req?.aspect_ratio ?? "auto",
    seed: null,
    advanced: req?.advanced ?? {},
  };
  if (req?.width && req?.height) {
    c.customWidth = req.width;
    c.customHeight = req.height;
  }
  return c;
}

/** A variation: same recipe, new random seed. */
export function variationRequest(g: GenerationRecord): GenerationRequest | null {
  if (!g.request) return null;
  return { ...g.request, seed: null, client: "ui", parent_id: g.id };
}

export const MODE_LABEL: Record<Mode, string> = {
  auto: "Auto",
  generate: "Create",
  edit: "Edit",
  inpaint: "Inpaint",
  outpaint: "Expand",
  multi_reference: "Combine",
  upscale: "Upscale",
};

/** Short truthful status text for a job. */
export function jobLabel(j: JobView): string {
  switch (j.status) {
    case "queued":
      return j.queue_position > 1 ? `Queued · ${j.queue_position}` : "Queued";
    case "starting_backend":
      return "Loading model";
    case "preparing":
      return "Preparing";
    case "saving":
      return "Saving";
    case "done":
      return "Done";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    case "running": {
      const batch = j.image && j.image[1] > 1 ? ` · ${j.image[0]}/${j.image[1]}` : "";
      switch (j.stage) {
        case "preparing":
          return "Preparing";
        case "encoding_images":
          return "Encoding references";
        case "encoding_prompt":
          return "Reading prompt";
        case "generating":
          return j.step ? `Generating · ${j.step[0]}/${j.step[1]}${batch}` : `Generating${batch}`;
        case "upscaling":
          return "Upscaling";
        case "refining":
          return j.step ? `Refining · ${j.step[0]}/${j.step[1]}${batch}` : `Refining${batch}`;
        case "decoding":
          return "Decoding";
        case "finishing":
          return "Finishing";
        default:
          return "Working";
      }
    }
  }
}

/** 0..1 when the backend reports steps, else null (indeterminate). */
export function jobFraction(j: JobView): number | null {
  if (j.status !== "running") return null;
  if (j.stage === "generating" && j.step) return j.step[0] / Math.max(1, j.step[1]);
  if (j.stage === "decoding" || j.stage === "finishing") return 1;
  return null;
}

export function remainingSeconds(j: JobView): number | null {
  if (j.status !== "running" || !j.step || !j.seconds_per_step) return null;
  if (j.stage !== "generating") return null;
  return Math.max(0, (j.step[1] - j.step[0]) * j.seconds_per_step);
}

// ---------------------------------------------------------------------------------------
// Upscaling
// ---------------------------------------------------------------------------------------

/** Server limits for upscale results (8192 px per side, ~40 MP). */
export const MAX_UPSCALE_SIDE = 8192;
export const MAX_UPSCALE_AREA = 40_000_000;

export interface UpscaleTarget { key: string; label: string; width: number; height: number }

const fits = (s: Size) => s.width >= 64 && s.height >= 64 && s.width <= MAX_UPSCALE_SIDE && s.height <= MAX_UPSCALE_SIDE && s.width * s.height <= MAX_UPSCALE_AREA;

/** Size with the given width and the source's aspect ratio. */
export function sizeForWidth(w: number, h: number, width: number): Size {
  return { width: Math.round(width), height: Math.max(1, Math.round((width * h) / w)) };
}

/** Upscale presets for a w×h image: 2x, 4K (3840 long side) and the original source
 * size of an edit, when each is a real enlargement within the limits. */
export function upscaleTargets(w: number, h: number, source?: Size | null): UpscaleTarget[] {
  const out: UpscaleTarget[] = [];
  const add = (key: string, label: string, s: Size) => {
    if (fits(s) && s.width * s.height > w * h * 1.05 && !out.some((o) => o.width === s.width && o.height === s.height)) out.push({ key, label, ...s });
  };
  add("2x", "2×", { width: w * 2, height: h * 2 });
  const k4 = w >= h ? sizeForWidth(w, h, 3840) : { width: Math.round((3840 * w) / h), height: 3840 };
  add("4k", "4K", k4);
  if (source) add("source", "Source size", source);
  return out;
}

export function upscaleRequest(source: string, target: Size, method: UpscaleMethod, parentId?: string | null): GenerationRequest {
  return {
    prompt: "",
    mode: "upscale",
    images: [source],
    quality: "balanced",
    resolution: "auto",
    preserve_dimensions: true,
    advanced: {},
    client: "ui",
    parent_id: parentId ?? null,
    upscale: { method, width: target.width, height: target.height },
  };
}

const dims = (s: Size) => `${s.width}×${s.height}`;
/** ESRGAN models offered for enlargement: [file stem, label]. Mirrors UPSCALER_CHOICES in config.rs. */
export const UPSCALERS: [string, string][] = [
  ["4x-UltraSharp", "UltraSharp"],
  ["4x_NMKD-Superscale-SP_178000_G", "Superscale-SP"],
];

/** Short name of the upscaler that ran or will run: "UltraSharp", "Superscale-SP", "Lanczos"... */
export const upscalerLabel = (name: string | null | undefined) => (!name || name === "Lanczos" ? "Lanczos" : UPSCALERS.find(([n]) => n === name)?.[1] ?? name);

/** "1376×768 → 5120×2880 (UltraSharp)" for the Details panel; two-pass records from
 * older versions read "1376×768 → refined 2048×1152 (Fast, 6 steps, UltraSharp) → ...". */
export function pipelineSummary(g: GenerationRecord): string | null {
  const p = g.pipeline;
  if (!p) return null;
  const parts: string[] = [];
  if (p.base) {
    const mode = p.refine === "best" ? "Best" : "Fast";
    parts.push(dims(p.base), `refined ${g.working_width}×${g.working_height} (${mode}, ${p.refine_steps ?? "?"} steps, ${upscalerLabel(p.hires_upscaler)})`);
  }
  if (p.enlarged_with && p.enlarged_from) {
    if (!parts.length) parts.push(dims(p.enlarged_from));
    parts.push(`${g.width}×${g.height} (${upscalerLabel(p.enlarged_with)})`);
  }
  return parts.length ? parts.join(" → ") : null;
}

/** Short description of what a plan will do, for the Advanced panel. */
export function planPipeline(plan: Plan): string {
  const out = dims(plan.output);
  const enlarge = plan.output.width * plan.output.height > plan.content.width * plan.content.height * 1.21;
  const via = upscalerLabel(plan.upscaler);
  if (plan.upscale?.method === "quick") return `${dims(plan.upscale.source)} → ${out} (${via})`;
  const work = dims(plan.working);
  return enlarge ? `${work} render → ${out} output (${via})` : `${work} render → ${out} output`;
}

export function formatDuration(s: number): string {
  if (s < 60) return `${Math.max(1, Math.round(s))}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return r ? `${m}m ${r}s` : `${m}m`;
}

export const ASPECTS = ["auto", "1:1", "4:3", "3:4", "16:9", "9:16", "custom"] as const;

/** Expansion presets in source pixels. */
export function expansionPreset(kind: "left" | "right" | "top" | "bottom" | "landscape" | "portrait" | "square", w: number, h: number, cur: OutpaintSpec): OutpaintSpec {
  const base: OutpaintSpec = { left: cur.left, right: cur.right, top: cur.top, bottom: cur.bottom, units: "pixels" };
  const qx = Math.round(w * 0.25);
  const qy = Math.round(h * 0.25);
  const toAspect = (a: number): OutpaintSpec => {
    if (w / h < a) {
      const extra = Math.max(0, Math.round(h * a) - w);
      return { left: Math.floor(extra / 2), right: extra - Math.floor(extra / 2), top: 0, bottom: 0, units: "pixels" };
    }
    const extra = Math.max(0, Math.round(w / a) - h);
    return { left: 0, right: 0, top: Math.floor(extra / 2), bottom: extra - Math.floor(extra / 2), units: "pixels" };
  };
  switch (kind) {
    case "left":
      return { ...base, left: base.left + qx };
    case "right":
      return { ...base, right: base.right + qx };
    case "top":
      return { ...base, top: base.top + qy };
    case "bottom":
      return { ...base, bottom: base.bottom + qy };
    case "landscape":
      return toAspect(16 / 9);
    case "portrait":
      return toAspect(9 / 16);
    case "square":
      return toAspect(1);
  }
}

/** Expansion in pixels regardless of units. */
export function expansionPixels(o: OutpaintSpec, w: number, h: number) {
  const f = (v: number, b: number) => (o.units === "percent" ? Math.round((b * v) / 100) : Math.round(v));
  return { left: f(o.left, w), right: f(o.right, w), top: f(o.top, h), bottom: f(o.bottom, h) };
}

// ---------------------------------------------------------------------------------------
// Before / after comparison
// ---------------------------------------------------------------------------------------

export type CompareMode = "off" | "slider" | "flicker" | "diff";

/** The image a result was made from and where it sits inside the result, as fractions of the
 * result's width/height (an expanded image sits inside a larger canvas). */
export interface CompareBase {
  inputId: string;
  rect: { x: number; y: number; w: number; h: number };
}

export function compareBase(g: GenerationRecord): CompareBase | null {
  const r = g.references[0];
  // A new image or a combination of several has no single "before".
  if (!r || !["edit", "inpaint", "outpaint", "upscale"].includes(g.mode)) return null;
  const o = g.outpaint;
  if (g.mode === "outpaint" && o) {
    // Expansion is in source pixels; the result is the source plus the expansion.
    const W = r.width + o.left + o.right;
    const H = r.height + o.top + o.bottom;
    return { inputId: r.input_id, rect: { x: o.left / W, y: o.top / H, w: r.width / W, h: r.height / H } };
  }
  return { inputId: r.input_id, rect: { x: 0, y: 0, w: 1, h: 1 } };
}

/** Per-pixel change between two equally sized RGBA buffers: the dimmed result in grey with
 * changed pixels in red (stronger = bigger change). Pixels the "before" does not cover
 * (alpha 0, e.g. expanded areas) count as changed. Returns the changed share of the image. */
export function diffPixels(after: Uint8ClampedArray, before: Uint8ClampedArray, out: Uint8ClampedArray, threshold = 24): number {
  let changed = 0;
  const n = after.length / 4;
  for (let i = 0; i < after.length; i += 4) {
    const grey = 0.3 * (0.299 * after[i] + 0.587 * after[i + 1] + 0.114 * after[i + 2]);
    const d = before[i + 3] < 128 ? 255 : Math.max(Math.abs(after[i] - before[i]), Math.abs(after[i + 1] - before[i + 1]), Math.abs(after[i + 2] - before[i + 2]));
    const a = d <= threshold ? 0 : Math.min(1, 0.35 + (d - threshold) / 96);
    if (a > 0) changed++;
    out[i] = grey + (255 - grey) * a;
    out[i + 1] = grey + (64 - grey) * a;
    out[i + 2] = grey + (64 - grey) * a;
    out[i + 3] = 255;
  }
  return n ? changed / n : 0;
}

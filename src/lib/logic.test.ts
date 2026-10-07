import { describe, expect, it } from "vitest";
import {
  activeOverrides, buildRequest, compareBase, diffPixels, emptyComposer, expansionPixels, expansionPreset, jobFraction, jobLabel, moveRef, persistentAdvanced, pipelineSummary,
  planPipeline, remainingSeconds, sizeForWidth, upscaleRequest, upscaleTargets, type Ref,
} from "./logic";
import type { GenerationRecord, JobView, Plan } from "./types";

const ref = (n: number): Ref => ({ key: `k${n}`, source: `input:${n}`, width: 100, height: 100, name: `r${n}` });

describe("references", () => {
  it("moves while preserving order of others", () => {
    const refs = [ref(1), ref(2), ref(3)];
    expect(moveRef(refs, 2, 0).map((r) => r.key)).toEqual(["k3", "k1", "k2"]);
    expect(moveRef(refs, 0, 2).map((r) => r.key)).toEqual(["k2", "k3", "k1"]);
    expect(moveRef(refs, 1, 1)).toBe(refs);
  });
});

describe("advanced overrides", () => {
  // What an older version saved after a CFG test (found in a real WebView profile).
  const saved = { batch_count: null, cfg: 4, flow_shift: null, hdr: false, output_format: "png" as const, refine: null, sampler: null, scheduler: null, steps: 20, strength: 1, vae_tiling: null };

  it("lists only the sampling overrides that are set", () => {
    expect(activeOverrides(saved)).toEqual(["CFG 4", "Steps 20", "Strength 1"]);
    expect(activeOverrides({ output_format: "webp", hdr: true, batch_count: 2 })).toEqual([]);
    expect(activeOverrides({})).toEqual([]);
  });

  it("keeps preferences but never persists sampling overrides", () => {
    expect(persistentAdvanced(saved)).toEqual({ hdr: false, output_format: "png" });
    expect(persistentAdvanced(undefined)).toEqual({});
    expect(emptyComposer({ advanced: persistentAdvanced(saved) }).advanced.cfg).toBeUndefined();
  });
});

describe("buildRequest", () => {
  it("text to image uses aspect, no images", () => {
    const c = { ...emptyComposer(), prompt: " a fox ", aspect: "16:9" };
    const r = buildRequest(c);
    expect(r.prompt).toBe("a fox");
    expect(r.images).toEqual([]);
    expect(r.aspect_ratio).toBe("16:9");
    expect(r.width).toBeNull();
  });
  it("keeps reference order and the chosen aspect for edits and combines", () => {
    const c = { ...emptyComposer(), prompt: "combine", refs: [ref(3), ref(1), ref(2)], aspect: "16:9" };
    const r = buildRequest(c);
    expect(r.images).toEqual(["input:3", "input:1", "input:2"]);
    expect(r.aspect_ratio).toBe("16:9");
  });
  it("inpaint and expand ignore the aspect", () => {
    const masked = { ...emptyComposer(), prompt: "x", refs: [ref(1)], mask: { inputId: "m", refKey: "k1" }, aspect: "16:9" };
    expect(buildRequest(masked).aspect_ratio).toBeNull();
    const expanded = { ...emptyComposer(), prompt: "x", refs: [ref(1)], outpaint: { left: 10, right: 10, top: 0, bottom: 0, units: "percent" as const }, aspect: "16:9" };
    expect(buildRequest(expanded).aspect_ratio).toBeNull();
  });
  it("mask only applies to the image it was painted on", () => {
    const c = { ...emptyComposer(), prompt: "x", refs: [ref(1)], mask: { inputId: "m", refKey: "k1" } };
    expect(buildRequest(c).mask).toBe("input:m");
    const moved = { ...c, refs: [ref(2), ref(1)] };
    expect(buildRequest(moved).mask).toBeNull();
  });
  it("custom resolution sends dimensions", () => {
    const c = { ...emptyComposer(), prompt: "x", resolution: "custom" as const, customWidth: 1280, customHeight: 720 };
    const r = buildRequest(c);
    expect([r.width, r.height]).toEqual([1280, 720]);
  });
  it("forced Create mode ignores attached images", () => {
    const c = { ...emptyComposer(), prompt: "x", refs: [ref(1)], mode: "generate" as const };
    expect(buildRequest(c).images).toEqual([]);
  });
});

describe("expansion", () => {
  const zero = { left: 0, right: 0, top: 0, bottom: 0, units: "pixels" as const };
  it("presets", () => {
    expect(expansionPreset("right", 1000, 800, zero)).toMatchObject({ right: 250 });
    const l = expansionPreset("landscape", 1024, 1024, zero);
    expect(l.left + l.right + 1024).toBe(1820);
    const s = expansionPreset("square", 1920, 1080, zero);
    expect(s.top + s.bottom + 1080).toBe(1920);
  });
  it("percent to pixels", () => {
    expect(expansionPixels({ left: 10, right: 0, top: 50, bottom: 0, units: "percent" }, 1000, 600)).toEqual({ left: 100, right: 0, top: 300, bottom: 0 });
  });
});

describe("job labels are truthful", () => {
  const base: JobView = {
    id: "j", status: "running", stage: "generating", step: [12, 30], image: null, seconds_per_step: 2, queue_position: 0, created_at: "",
    started_at: null, finished_at: null, prompt: "", plan: null, seed: 1, results: [], error: null, notes: [], detailed_progress: true,
  };
  it("shows steps only when reported", () => {
    expect(jobLabel(base)).toBe("Generating · 12/30");
    expect(jobFraction(base)).toBeCloseTo(0.4);
    const noDetail = { ...base, step: null };
    expect(jobLabel(noDetail)).toBe("Generating");
    expect(jobFraction(noDetail)).toBeNull();
    expect(jobLabel({ ...base, status: "queued", queue_position: 3 })).toBe("Queued · 3");
  });

  it("one pass fills the bar; enlarging is indeterminate", () => {
    expect(remainingSeconds({ ...base, step: [10, 20] as [number, number] })).toBeCloseTo(10 * 2);
    expect(jobLabel({ ...base, stage: "upscaling", step: null })).toBe("Upscaling");
    expect(jobFraction({ ...base, stage: "upscaling", step: null })).toBeNull();
    expect(jobFraction({ ...base, stage: "decoding", step: null })).toBe(1);
  });
});

describe("upscaling", () => {
  it("presets are real enlargements within limits", () => {
    const t = upscaleTargets(1376, 768);
    expect(t.map((x) => x.key)).toEqual(["2x", "4k"]);
    expect(t[0]).toMatchObject({ width: 2752, height: 1536 });
    expect(t[1]).toMatchObject({ width: 3840, height: 2143 });
    // Portrait 4K uses the long side.
    expect(upscaleTargets(768, 1376).find((x) => x.key === "4k")).toMatchObject({ width: 2143, height: 3840 });
    // A 1 MP edit of a 5120x2880 photo can go back to the photo's size.
    expect(upscaleTargets(1376, 768, { width: 5120, height: 2880 }).map((x) => x.key)).toEqual(["2x", "4k", "source"]);
    expect(upscaleTargets(5120, 2880, { width: 5120, height: 2880 }).map((x) => x.key), "2x is over 40 MP; 4K and source are not larger").toEqual([]);
    expect(upscaleTargets(4000, 4000).length).toBe(0);
  });
  it("custom width keeps the aspect", () => {
    expect(sizeForWidth(1376, 768, 3000)).toEqual({ width: 3000, height: 1674 });
  });
  it("builds an upscale request without a prompt", () => {
    const r = upscaleRequest("gen:abc", { width: 2752, height: 1536 }, "detailed", "abc");
    expect(r).toMatchObject({ mode: "upscale", images: ["gen:abc"], prompt: "", parent_id: "abc", upscale: { method: "detailed", width: 2752, height: 1536 } });
  });
  it("summarises the pipeline for the details panel", () => {
    const g = {
      width: 2752, height: 1536, working_width: 2048, working_height: 1152,
      pipeline: {
        base: { width: 1376, height: 768 }, refine: "fast", refine_steps: 6, refine_denoise: 0.35, hires_upscaler: "4x-UltraSharp",
        cache_threshold: 0.2, enlarged_with: "4x-UltraSharp", enlarged_from: { width: 2048, height: 1152 },
      },
    } as unknown as GenerationRecord;
    expect(pipelineSummary(g)).toBe("1376×768 → refined 2048×1152 (Fast, 6 steps, UltraSharp) → 2752×1536 (UltraSharp)");
    const edit = { ...g, width: 5120, height: 2880, pipeline: { ...g.pipeline!, base: null, enlarged_from: { width: 1376, height: 768 }, enlarged_with: "Lanczos" } } as GenerationRecord;
    expect(pipelineSummary(edit)).toBe("1376×768 → 5120×2880 (Lanczos)");
    expect(pipelineSummary({ ...g, pipeline: null } as GenerationRecord)).toBeNull();
  });
  it("describes plans", () => {
    const p = {
      output: { width: 2752, height: 1536 }, working: { width: 2080, height: 1152 }, content: { x: 12, y: 2, width: 2056, height: 1148 },
      upscaler: "4x-UltraSharp", upscale: null,
    } as unknown as Plan;
    expect(planPipeline(p)).toBe("2080×1152 render → 2752×1536 output (UltraSharp)");
    const quick = { ...p, upscaler: null, upscale: { method: "quick", source: { width: 1376, height: 768 } } } as Plan;
    expect(planPipeline(quick)).toBe("1376×768 → 2752×1536 (Lanczos)");
  });
});

describe("before / after comparison", () => {
  const rec = (mode: string, extra: Partial<GenerationRecord> = {}) =>
    ({ mode, references: [{ index: 0, input_id: "in1", width: 1000, height: 500 }], outpaint: null, ...extra }) as unknown as GenerationRecord;

  it("edits, inpaints and upscales compare against the whole source", () => {
    for (const m of ["edit", "inpaint", "upscale"]) expect(compareBase(rec(m))).toEqual({ inputId: "in1", rect: { x: 0, y: 0, w: 1, h: 1 } });
  });

  it("an expanded image places the source inside the larger result", () => {
    const b = compareBase(rec("outpaint", { outpaint: { left: 500, right: 500, top: 0, bottom: 500 } }))!;
    expect(b.rect).toEqual({ x: 0.25, y: 0, w: 0.5, h: 0.5 });
  });

  it("new images and multi-reference results have no before", () => {
    expect(compareBase(rec("generate"))).toBeNull();
    expect(compareBase(rec("multi_reference"))).toBeNull();
    expect(compareBase(rec("edit", { references: [] }))).toBeNull();
  });

  it("marks only pixels that changed beyond the threshold, and uncovered pixels", () => {
    const after = new Uint8ClampedArray([100, 100, 100, 255, 100, 100, 100, 255, 200, 0, 0, 255, 50, 50, 50, 255]);
    const before = new Uint8ClampedArray([100, 100, 100, 255, 110, 100, 100, 255, 0, 0, 0, 255, 0, 0, 0, 0]);
    const out = new Uint8ClampedArray(16);
    expect(diffPixels(after, before, out)).toBe(0.5);
    expect(out[0]).toBe(out[1]); // unchanged: grey
    expect(out[4]).toBe(out[5]); // small change below the threshold: grey
    expect(out[8]).toBeGreaterThan(out[9]); // big change: red
    expect(out[12]).toBeGreaterThan(out[13]); // not covered by the source (expanded): red
  });
});

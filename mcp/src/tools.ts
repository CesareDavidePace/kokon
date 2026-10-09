// Tool schemas and handlers. Handlers are framework-free (testable): they receive a
// context with the service client, the project folder and a progress callback.

import { z } from "zod";
import { allowedRoots, checkInput, checkOutput, PathError } from "./paths.js";
import { ServiceClient, ServiceError } from "./service.js";

const quality = z
  .enum(["turbo", "fast", "balanced", "quality", "ultra"])
  .describe("turbo = ~1 MP in 8 steps with Qwen's distilled Turbo model (the default, fastest; runs as balanced if the Turbo model is not installed); fast = ~1 MP, 20 steps; balanced = ~1.5K, 20 steps; quality = native 2K, 20 steps (slower; edits share 2K with their references); ultra = native 2K, 40 steps (slowest)");
const seed = z.number().int().min(-1).max(2_147_483_647).describe("Seed for reproducible results; omit or -1 for random");
const dim = z.number().int().min(64).max(4096);
const aspect = z
  .string()
  .regex(/^(auto|\d{1,2}:\d{1,2})$/, "Use a ratio like 16:9 or 'auto'")
  .describe("Aspect ratio such as 1:1, 4:3, 3:4, 16:9, 9:16, or auto (inferred from the prompt)");
const prompt = z.string().trim().min(1, "prompt must not be empty").max(8000);
const outputPath = z
  .string()
  .min(1)
  .describe("Optional destination (absolute, or relative to the project). Must be inside the project or the app output folder. Existing files are never overwritten; a numbered name is used instead.");
const preview = z.boolean().describe("Also return a small JPEG preview in the tool result. Default false: images stay local and are not sent to the model.");

export const schemas = {
  generate_image: {
    prompt: prompt.describe("What to generate, in natural language"),
    negative_prompt: z.string().max(2000).optional(),
    width: dim.optional().describe("Output width in pixels (needs height too). Omit to choose automatically."),
    height: dim.optional().describe("Output height in pixels (needs width too)."),
    aspect_ratio: aspect.optional(),
    quality: quality.optional(),
    seed: seed.optional(),
    output_path: outputPath.optional(),
    include_preview: preview.optional(),
  },
  edit_image: {
    image_path: z.string().min(1).describe("Image to edit (absolute, or relative to the project)"),
    prompt: prompt.describe("The change to make, e.g. 'make the background nighttime, keep the foreground unchanged'"),
    output_path: outputPath.optional(),
    quality: quality.optional(),
    preserve_dimensions: z.boolean().optional().describe("Keep the input's exact width/height (default true)"),
    seed: seed.optional(),
    negative_prompt: z.string().max(2000).optional(),
    include_preview: preview.optional(),
  },
  edit_images: {
    image_paths: z.array(z.string().min(1)).min(1).max(10).describe("Reference images in order; refer to them as image 1, image 2, ... in the prompt"),
    prompt: prompt.describe("e.g. 'Use the person from image 1, the clothing from image 2 and the lighting of image 3'"),
    width: dim.optional(),
    height: dim.optional(),
    aspect_ratio: aspect.optional(),
    quality: quality.optional(),
    seed: seed.optional(),
    output_path: outputPath.optional(),
    include_preview: preview.optional(),
  },
  outpaint_image: {
    image_path: z.string().min(1),
    prompt: prompt.describe("What the extended area should contain"),
    left: z.number().min(0).max(16384).default(0),
    right: z.number().min(0).max(16384).default(0),
    top: z.number().min(0).max(16384).default(0),
    bottom: z.number().min(0).max(16384).default(0),
    units: z.enum(["pixels", "percent"]).default("pixels").describe("Units for left/right/top/bottom; percent is relative to the image width (left/right) or height (top/bottom)"),
    seed: seed.optional(),
    quality: quality.optional(),
    output_path: outputPath.optional(),
    include_preview: preview.optional(),
  },
  inpaint_image: {
    image_path: z.string().min(1),
    mask_path: z.string().min(1).describe("Mask image: white = area to change, black = keep (or opaque strokes on a transparent layer)"),
    action: z
      .enum(["edit", "remove", "replace", "add"])
      .default("edit")
      .describe("edit: change the area as the prompt says; remove: erase what is there and fill with background (prompt optional, e.g. 'the bin'); replace: put the prompt's subject in its place; add: add the prompt's subject inside the area"),
    prompt: z.string().trim().max(8000).default("").describe("The change, or the subject to add/replace with; optional for remove"),
    seed: seed.optional(),
    quality: quality.optional(),
    output_path: outputPath.optional(),
    include_preview: preview.optional(),
  },
  upscale_image: {
    image_path: z.string().min(1).describe("Image to upscale (absolute, or relative to the project)"),
    scale: z.number().gt(1).max(8).optional().describe("Enlargement factor, e.g. 2 (default 2 when no width/height is given)"),
    width: z.number().int().min(64).max(8192).optional().describe("Target width in pixels; height follows the aspect ratio unless also given"),
    height: z.number().int().min(64).max(8192).optional().describe("Target height in pixels; width follows the aspect ratio unless also given"),
    method: z
      .enum(["quick", "detailed"])
      .optional()
      .describe("quick = the configured ESRGAN upscaler model only (UltraSharp by default; seconds, adds no new content, default); detailed = re-render fine detail at ~2K with low denoise, then the upscaler model (about a minute)"),
    prompt: z.string().max(8000).optional().describe("Optional description used by the detailed refine (defaults to the image's own prompt or a neutral one)"),
    output_path: outputPath.optional(),
    include_preview: preview.optional(),
  },
  get_image_info: {
    image_path: z.string().min(1),
  },
  get_generator_status: {},
} as const;

export interface ToolContext {
  client: ServiceClient;
  cwd: string;
  progress: (message: string) => void;
  signal?: AbortSignal;
}

export interface ToolResult {
  text: string;
  structured: Record<string, unknown>;
  previewJpeg?: Buffer;
  isError?: boolean;
}

type Args<K extends keyof typeof schemas> = z.infer<z.ZodObject<(typeof schemas)[K]>>;

interface JobView {
  id: string;
  status: string;
  stage: string | null;
  step: [number, number] | null;
  queue_position: number;
  plan: { mode: string; working: { width: number; height: number }; notes: string[] } | null;
  results: Record<string, unknown>[];
  error: { code: string; message: string; suggestion?: string; detail?: string } | null;
  notes: string[];
}

let outputDirCache: string | null = null;
async function outputDir(client: ServiceClient): Promise<string | null> {
  if (outputDirCache) return outputDirCache;
  try {
    const cfg = await client.call<{ outputDir: string }>("GET", "/config");
    outputDirCache = cfg.outputDir;
  } catch {
    /* fall back to project only */
  }
  return outputDirCache;
}

export function label(j: JobView): string {
  switch (j.status) {
    case "queued":
      return j.queue_position > 1 ? `Queued (position ${j.queue_position})` : "Queued";
    case "starting_backend":
      return "Loading the model";
    case "preparing":
      return "Preparing";
    case "saving":
      return "Saving";
    case "running":
      if (j.stage === "generating" && j.step) return `Generating step ${j.step[0]}/${j.step[1]}`;
      if (j.stage === "refining" && j.step) return `Refining step ${j.step[0]}/${j.step[1]}`;
      return (
        {
          encoding_images: "Encoding reference images",
          encoding_prompt: "Reading the prompt",
          decoding: "Decoding",
          finishing: "Finishing",
          generating: "Generating",
          upscaling: "Upscaling",
          refining: "Refining",
        }[j.stage ?? ""] ?? "Working"
      );
    default:
      return j.status;
  }
}

async function runJob(ctx: ToolContext, body: Record<string, unknown>, includePreview: boolean): Promise<ToolResult> {
  const job = await ctx.client.call<JobView>("POST", "/jobs", { ...body, workspace: ctx.cwd, client: "mcp" });
  let last = "";
  let cancelled = false;
  const onAbort = () => {
    cancelled = true;
    ctx.client.call("POST", `/jobs/${job.id}/cancel`).catch(() => {});
  };
  ctx.signal?.addEventListener("abort", onAbort, { once: true });
  try {
    for (;;) {
      const j = await ctx.client.call<JobView>("GET", `/jobs/${job.id}`);
      const l = label(j);
      if (l !== last) {
        last = l;
        ctx.progress(l);
      }
      if (j.status === "done") return formatDone(ctx, j, includePreview);
      if (j.status === "failed" || j.status === "cancelled") {
        const e = j.error ?? { code: "cancelled", message: cancelled ? "Cancelled." : "The job was cancelled." };
        return errorResult(e.message, e.code, e.suggestion, e.detail);
      }
      await new Promise((r) => setTimeout(r, 700));
    }
  } finally {
    ctx.signal?.removeEventListener("abort", onAbort);
  }
}

async function formatDone(ctx: ToolContext, j: JobView, includePreview: boolean): Promise<ToolResult> {
  const images = j.results.map((r) => {
    const extra = (r.extra_paths as string[]) ?? [];
    return {
      path: extra[0] ?? (r.file_path as string),
      library_path: r.file_path,
      id: r.id,
      width: r.width,
      height: r.height,
      seed: r.seed,
    };
  });
  const r0 = j.results[0];
  const structured: Record<string, unknown> = {
    ...images[0],
    images: images.length > 1 ? images : undefined,
    mode: r0.mode,
    prompt: r0.prompt,
    model: r0.model,
    steps: r0.steps,
    cfg: r0.cfg,
    sampler: r0.sampler,
    quality: r0.quality,
    working_size: `${r0.working_width}x${r0.working_height}`,
    pipeline: r0.pipeline ?? undefined,
    duration_seconds: Math.round(((r0.duration_ms as number) ?? 0) / 100) / 10,
    notes: [...new Set([...(j.notes ?? []), ...((r0.notes as string[]) ?? [])])],
  };
  let previewJpeg: Buffer | undefined;
  if (includePreview) {
    const info = await ctx.client.ensure();
    const res = await fetch(`http://127.0.0.1:${info.port}/api/v1/gallery/${r0.id}/thumb?token=${info.token}`);
    if (res.ok) previewJpeg = Buffer.from(await res.arrayBuffer());
  }
  const lines = [`Saved ${images.map((i) => i.path).join(", ")} (${images[0].width}x${images[0].height}, seed ${images[0].seed}).`];
  if (structured.notes && (structured.notes as string[]).length) lines.push(`Notes: ${(structured.notes as string[]).join(" ")}`);
  return { text: lines.join("\n") + "\n" + JSON.stringify(structured, null, 2), structured, previewJpeg };
}

export function errorResult(message: string, code = "error", suggestion?: string, detail?: string): ToolResult {
  const structured = { error: { code, message, suggestion, detail: detail?.slice(0, 2000) } };
  return { text: [message, suggestion].filter(Boolean).join(" "), structured, isError: true };
}

async function roots(ctx: ToolContext) {
  return allowedRoots(ctx.cwd, await outputDir(ctx.client));
}

function validateDims(w?: number, h?: number) {
  if ((w == null) !== (h == null)) throw new PathError("Provide both width and height, or neither.");
}

export const handlers = {
  async generate_image(a: Args<"generate_image">, ctx: ToolContext) {
    validateDims(a.width, a.height);
    const out = a.output_path ? checkOutput(a.output_path, ctx.cwd, await roots(ctx)) : undefined;
    return runJob(
      ctx,
      {
        prompt: a.prompt,
        negative_prompt: a.negative_prompt,
        mode: "generate",
        images: [],
        width: a.width,
        height: a.height,
        resolution: a.width ? "custom" : "auto",
        aspect_ratio: a.aspect_ratio,
        quality: a.quality ?? "turbo",
        seed: a.seed,
        output_path: out,
        advanced: {},
      },
      !!a.include_preview,
    );
  },

  async edit_image(a: Args<"edit_image">, ctx: ToolContext) {
    const r = await roots(ctx);
    const img = checkInput(a.image_path, ctx.cwd, r);
    const out = a.output_path ? checkOutput(a.output_path, ctx.cwd, r) : undefined;
    return runJob(
      ctx,
      { prompt: a.prompt, negative_prompt: a.negative_prompt, mode: "edit", images: [img], quality: a.quality ?? "turbo", seed: a.seed, preserve_dimensions: a.preserve_dimensions ?? true, output_path: out, advanced: {} },
      !!a.include_preview,
    );
  },

  async edit_images(a: Args<"edit_images">, ctx: ToolContext) {
    validateDims(a.width, a.height);
    const r = await roots(ctx);
    const imgs = a.image_paths.map((x) => checkInput(x, ctx.cwd, r));
    const out = a.output_path ? checkOutput(a.output_path, ctx.cwd, r) : undefined;
    return runJob(
      ctx,
      {
        prompt: a.prompt,
        mode: imgs.length > 1 ? "multi_reference" : "edit",
        images: imgs,
        width: a.width,
        height: a.height,
        resolution: a.width ? "custom" : "auto",
        aspect_ratio: a.aspect_ratio,
        quality: a.quality ?? "turbo",
        seed: a.seed,
        output_path: out,
        advanced: {},
      },
      !!a.include_preview,
    );
  },

  async outpaint_image(a: Args<"outpaint_image">, ctx: ToolContext) {
    if (!(a.left > 0 || a.right > 0 || a.top > 0 || a.bottom > 0)) throw new PathError("Set at least one of left, right, top or bottom.");
    const r = await roots(ctx);
    const img = checkInput(a.image_path, ctx.cwd, r);
    const out = a.output_path ? checkOutput(a.output_path, ctx.cwd, r) : undefined;
    return runJob(
      ctx,
      {
        prompt: a.prompt,
        mode: "outpaint",
        images: [img],
        outpaint: { left: a.left, right: a.right, top: a.top, bottom: a.bottom, units: a.units },
        quality: a.quality ?? "turbo",
        seed: a.seed,
        output_path: out,
        advanced: {},
      },
      !!a.include_preview,
    );
  },

  async inpaint_image(a: Args<"inpaint_image">, ctx: ToolContext) {
    const r = await roots(ctx);
    const img = checkInput(a.image_path, ctx.cwd, r);
    const mask = checkInput(a.mask_path, ctx.cwd, r);
    const out = a.output_path ? checkOutput(a.output_path, ctx.cwd, r) : undefined;
    if (!a.prompt && a.action !== "remove") throw new PathError("prompt must not be empty (only action 'remove' works without one).");
    return runJob(
      ctx,
      { prompt: a.prompt, mode: "inpaint", inpaint_action: a.action, images: [img], mask, quality: a.quality ?? "turbo", seed: a.seed, output_path: out, advanced: {} },
      !!a.include_preview,
    );
  },

  async upscale_image(a: Args<"upscale_image">, ctx: ToolContext) {
    if (a.scale != null && (a.width != null || a.height != null)) throw new PathError("Give either scale or width/height, not both.");
    const r = await roots(ctx);
    const img = checkInput(a.image_path, ctx.cwd, r);
    const out = a.output_path ? checkOutput(a.output_path, ctx.cwd, r) : undefined;
    return runJob(
      ctx,
      {
        prompt: a.prompt ?? "",
        mode: "upscale",
        images: [img],
        upscale: { method: a.method ?? "quick", scale: a.scale, width: a.width, height: a.height },
        output_path: out,
        advanced: {},
      },
      !!a.include_preview,
    );
  },

  async get_image_info(a: Args<"get_image_info">, ctx: ToolContext) {
    const img = checkInput(a.image_path, ctx.cwd, await roots(ctx));
    const info = await ctx.client.call<Record<string, unknown>>("POST", "/image-info", { path: img });
    const g = info.generation as Record<string, unknown> | null;
    const structured = {
      path: info.path,
      width: info.width,
      height: info.height,
      aspect_ratio: info.aspect_ratio,
      file_type: info.file_type,
      file_size: info.file_size,
      has_transparency: info.has_transparency,
      exif_orientation: info.exif_orientation,
      generation: g
        ? { prompt: g.prompt, negative_prompt: g.negative_prompt, seed: g.seed, mode: g.mode, model: g.model, steps: g.steps, cfg: g.cfg, sampler: g.sampler, created_at: g.created_at }
        : info.parameters ?? null,
    };
    return { text: JSON.stringify(structured, null, 2), structured } as ToolResult;
  },

  async get_generator_status(_a: Args<"get_generator_status">, ctx: ToolContext) {
    const st = await ctx.client.call<Record<string, any>>("GET", "/status");
    const active = st.active_job as JobView | null;
    const structured = {
      server_running: st.server.state === "ready" || st.server.state === "starting",
      model_loaded: !!st.server.model_loaded,
      server_state: st.server.state,
      model: st.server.model ?? st.profile?.display_name,
      gpu: st.gpu?.name ?? null,
      vram_total_mib: st.gpu?.memory_total_mib ?? null,
      vram_free_mib: st.gpu?.memory_free_mib ?? null,
      active_job: active ? { id: active.id, status: active.status, progress: label(active) } : null,
      queued_jobs: st.queue?.queued ?? 0,
      service_version: st.service?.version,
    };
    return { text: JSON.stringify(structured, null, 2), structured } as ToolResult;
  },
};

export function toErrorResult(e: unknown): ToolResult {
  if (e instanceof PathError) return errorResult(e.message, "invalid_path");
  if (e instanceof ServiceError) return errorResult(e.message, e.code, e.suggestion, e.detail);
  if (e instanceof z.ZodError) return errorResult(e.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; "), "invalid_input");
  return errorResult(e instanceof Error ? e.message : String(e), "internal");
}

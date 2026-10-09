import { describe, expect, it } from "vitest";
import { z } from "zod";
import { handlers, label, schemas, toErrorResult } from "../src/tools";

const obj = <K extends keyof typeof schemas>(k: K) => z.object(schemas[k]);

describe("input validation", () => {
  it("prompt is required and trimmed", () => {
    expect(obj("generate_image").safeParse({ prompt: "  " }).success).toBe(false);
    expect(obj("generate_image").safeParse({ prompt: " a fox " }).data?.prompt).toBe("a fox");
  });
  it("dimensions, aspect, quality, seed", () => {
    expect(obj("generate_image").safeParse({ prompt: "x", width: 32, height: 512 }).success).toBe(false);
    expect(obj("generate_image").safeParse({ prompt: "x", width: 5000, height: 512 }).success).toBe(false);
    expect(obj("generate_image").safeParse({ prompt: "x", aspect_ratio: "16:9" }).success).toBe(true);
    expect(obj("generate_image").safeParse({ prompt: "x", aspect_ratio: "wide" }).success).toBe(false);
    expect(obj("generate_image").safeParse({ prompt: "x", quality: "ultra" }).success).toBe(true);
    expect(obj("generate_image").safeParse({ prompt: "x", quality: "max" }).success).toBe(false);
    expect(obj("generate_image").safeParse({ prompt: "x", seed: -2 }).success).toBe(false);
  });
  it("edit_images limits and order", () => {
    expect(obj("edit_images").safeParse({ prompt: "x", image_paths: [] }).success).toBe(false);
    expect(obj("edit_images").safeParse({ prompt: "x", image_paths: Array(11).fill("a.png") }).success).toBe(false);
    expect(obj("edit_images").safeParse({ prompt: "x", image_paths: ["b.png", "a.png"] }).data?.image_paths).toEqual(["b.png", "a.png"]);
  });
  it("outpaint defaults", () => {
    const r = obj("outpaint_image").parse({ image_path: "a.png", prompt: "x", left: 256 });
    expect(r).toMatchObject({ left: 256, right: 0, top: 0, bottom: 0, units: "pixels" });
  });
  it("width without height is rejected before any service call", async () => {
    const ctx = { client: {} as never, cwd: process.cwd(), progress: () => {} };
    const res = await handlers.generate_image({ prompt: "x", width: 512 } as never, ctx).catch(toErrorResult);
    expect(res.isError).toBe(true);
    expect(res.text).toMatch(/both width and height/);
  });
});

describe("upscale_image", () => {
  it("validates factor, size and method", () => {
    expect(obj("upscale_image").safeParse({ image_path: "a.png" }).success).toBe(true);
    expect(obj("upscale_image").safeParse({ image_path: "a.png", scale: 1 }).success).toBe(false);
    expect(obj("upscale_image").safeParse({ image_path: "a.png", scale: 2, method: "detailed" }).success).toBe(true);
    expect(obj("upscale_image").safeParse({ image_path: "a.png", width: 9000 }).success).toBe(false);
    expect(obj("upscale_image").safeParse({ image_path: "a.png", method: "ultra" }).success).toBe(false);
  });
  it("scale and width together are rejected before any service call", async () => {
    const ctx = { client: {} as never, cwd: process.cwd(), progress: () => {} };
    const res = await handlers.upscale_image({ image_path: "a.png", scale: 2, width: 4000 } as never, ctx).catch(toErrorResult);
    expect(res.isError).toBe(true);
    expect(res.text).toMatch(/either scale or width/);
  });
});

describe("progress labels are truthful", () => {
  const j = { id: "j", status: "running", stage: "generating", step: [3, 30] as [number, number], queue_position: 0, plan: null, results: [], error: null, notes: [] };
  it("only shows steps when reported", () => {
    expect(label(j)).toBe("Generating step 3/30");
    expect(label({ ...j, step: null })).toBe("Generating");
    expect(label({ ...j, status: "starting_backend" })).toBe("Loading the model");
    expect(label({ ...j, stage: "upscaling", step: null })).toBe("Upscaling");
    expect(label({ ...j, stage: "refining", step: [2, 6] })).toBe("Refining step 2/6");
  });
});

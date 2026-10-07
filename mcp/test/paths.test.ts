import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { checkInput, checkOutput, isWithin, normalize, PathError } from "../src/paths";

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "qls-mcp-"));
const proj = path.join(tmp, "proj");
fs.mkdirSync(path.join(proj, "assets"), { recursive: true });
fs.writeFileSync(path.join(proj, "assets", "header.png"), "x");
fs.writeFileSync(path.join(proj, "notes.txt"), "x");
const outside = path.join(tmp, "proj2");
fs.mkdirSync(outside);
fs.writeFileSync(path.join(outside, "x.png"), "x");
const roots = [proj];

describe("normalize", () => {
  it.each([["\\\\server\\share\\a.png"], ["\\\\?\\C:\\a.png"], ["//server/a.png"], ["C:\\a.png:ads"], ["C:\\x\\CON.png"], ["C:rel.png"], [""], ["a\u0000b"]])("rejects %s", (p) => {
    expect(() => normalize(p, proj)).toThrow(PathError);
  });
  it("resolves relative paths against the project", () => {
    expect(normalize("assets/header.png", proj)).toBe(path.join(proj, "assets", "header.png"));
  });
});

describe("inputs", () => {
  it("accepts a relative image in the project", () => {
    expect(checkInput("assets\\header.png", proj, roots)).toBe(path.join(proj, "assets", "header.png"));
  });
  it("rejects traversal outside the project", () => {
    expect(() => checkInput("..\\proj2\\x.png", proj, roots)).toThrow(/outside/);
    expect(() => checkInput(path.join(outside, "x.png"), proj, roots)).toThrow(/outside/);
  });
  it("rejects non-images and missing files", () => {
    expect(() => checkInput("notes.txt", proj, roots)).toThrow(/Only PNG/);
    expect(() => checkInput("missing.png", proj, roots)).toThrow(/not found/);
  });
  it("sibling prefix is not inside", () => {
    expect(isWithin(path.join(outside, "x.png"), proj)).toBe(false);
  });
});

describe("outputs", () => {
  it("adds .png and stays inside roots", () => {
    expect(checkOutput("assets/hero", proj, roots)).toBe(path.join(proj, "assets", "hero.png"));
    expect(() => checkOutput("../escape.png", proj, roots)).toThrow(/outside/);
    expect(() => checkOutput("x.exe", proj, roots)).toThrow(/png/);
  });
});

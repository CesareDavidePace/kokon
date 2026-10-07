// Path policy for MCP clients. An agent may only read images from — and write images
// to — the project folder (MCP working directory), the app's output folder, and any
// folders listed in QLS_MCP_ALLOWED_DIRS (semicolon separated). The service validates
// again on its side (UNC/device paths, traversal, no overwrite).

import fs from "node:fs";
import path from "node:path";

const p = path.win32;

export const INPUT_EXT = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"];
export const OUTPUT_EXT = [".png", ".jpg", ".jpeg", ".webp"];
export const MAX_INPUT_BYTES = 200 * 1024 * 1024;

export class PathError extends Error {}

function realish(target: string): string {
  // Resolve symlinks/junctions on the deepest existing ancestor.
  let cur = target;
  const rest: string[] = [];
  while (!fs.existsSync(cur)) {
    const parent = p.dirname(cur);
    if (parent === cur) return target;
    rest.unshift(p.basename(cur));
    cur = parent;
  }
  let real = cur;
  try {
    real = fs.realpathSync.native(cur);
  } catch {
    /* keep lexical */
  }
  return p.join(real, ...rest);
}

export function isWithin(target: string, root: string): boolean {
  const t = realish(p.resolve(target)).toLowerCase();
  const r = realish(p.resolve(root)).toLowerCase().replace(/\\+$/, "");
  return t === r || t.startsWith(r + "\\");
}

/** Lexical checks shared by inputs and outputs; returns an absolute normalized path. */
export function normalize(raw: string, cwd: string): string {
  const s = (raw ?? "").trim();
  if (!s) throw new PathError("The path is empty.");
  if (/[\u0000-\u001f]/.test(s)) throw new PathError("The path contains invalid characters.");
  if (s.startsWith("\\\\") || s.startsWith("//")) throw new PathError("Network and device paths are not allowed.");
  if (/^[a-zA-Z]:(?![\\/])/.test(s)) throw new PathError("Drive-relative paths (like C:file.png) are not allowed.");
  const abs = p.resolve(cwd, s.replace(/\//g, "\\"));
  const parts = abs.slice(3).split("\\");
  for (const part of parts) {
    if (part.includes(":")) throw new PathError("Alternate data streams are not allowed.");
    if (/^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$/i.test(part)) throw new PathError(`'${part}' is a reserved Windows name.`);
  }
  return abs;
}

export function checkInput(raw: string, cwd: string, roots: string[]): string {
  const abs = normalize(raw, cwd);
  if (!INPUT_EXT.includes(p.extname(abs).toLowerCase())) throw new PathError("Only PNG, JPEG, WebP, BMP, GIF, TIFF and HEIC images are supported.");
  if (!roots.some((r) => isWithin(abs, r))) throw new PathError(`${abs} is outside the allowed folders (${roots.join("; ")}).`);
  let st: fs.Stats;
  try {
    st = fs.statSync(abs);
  } catch {
    throw new PathError(`File not found: ${abs}`);
  }
  if (!st.isFile()) throw new PathError(`${abs} is not a file.`);
  if (st.size > MAX_INPUT_BYTES) throw new PathError("The image file is larger than 200 MB.");
  return abs;
}

export function checkOutput(raw: string, cwd: string, roots: string[]): string {
  let abs = normalize(raw, cwd);
  const ext = p.extname(abs).toLowerCase();
  if (!ext) abs += ".png";
  else if (!OUTPUT_EXT.includes(ext)) throw new PathError("output_path must end in .png, .jpg or .webp.");
  if (!roots.some((r) => isWithin(abs, r))) throw new PathError(`${abs} is outside the allowed output folders (${roots.join("; ")}).`);
  return abs;
}

export function allowedRoots(cwd: string, outputDir: string | null): string[] {
  const extra = (process.env.QLS_MCP_ALLOWED_DIRS ?? "")
    .split(";")
    .map((x) => x.trim())
    .filter(Boolean);
  return [cwd, ...(outputDir ? [outputDir] : []), ...extra].map((r) => p.resolve(r));
}

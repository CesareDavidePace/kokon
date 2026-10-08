// Builds build/kokon-mcp.exe: the bundled server injected into a copy of node.exe
// (Node "single executable application"). No separate Node install is needed to run it.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const build = path.join(root, "build");
const run = (cmd, args) => execFileSync(cmd, args, { cwd: root, stdio: "inherit", shell: process.platform === "win32" && cmd === "npx" });

run("npx", ["esbuild", "src/index.ts", "--bundle", "--platform=node", "--target=node22", "--format=cjs", "--outfile=build/kokon-mcp.cjs", "--minify"]);
fs.writeFileSync(
  path.join(build, "sea-config.json"),
  JSON.stringify({ main: "build/kokon-mcp.cjs", output: "build/sea-prep.blob", disableExperimentalSEAWarning: true, useSnapshot: false, useCodeCache: true }),
);
run(process.execPath, ["--experimental-sea-config", "build/sea-config.json"]);
const exe = path.join(build, "kokon-mcp.exe");
fs.copyFileSync(process.execPath, exe);
run("npx", ["postject", exe, "NODE_SEA_BLOB", path.join(build, "sea-prep.blob"), "--sentinel-fuse", "NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2"]);

const sidecar = path.resolve(root, "..", "src-tauri", "binaries", "kokon-mcp-x86_64-pc-windows-msvc.exe");
fs.copyFileSync(exe, sidecar);
console.log(`built ${exe} (${(fs.statSync(exe).size / 1e6).toFixed(1)} MB) -> ${sidecar}`);

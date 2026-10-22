#!/usr/bin/env node
// Prints Kokon's status pages for a coding agent at the start of a session.
// Claude Code: add it as a SessionStart hook in .claude/settings.local.json, e.g.
//   {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "node scripts/dev/agent-context.mjs"}]}]}}
// Usage: node scripts/dev/agent-context.mjs [path to the kokon clone]  (default: this script's repo)

import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(process.argv[2] ?? join(dirname(fileURLToPath(import.meta.url)), "..", ".."));

function show(title, file, pick) {
  const path = join(root, file);
  if (!existsSync(path)) return;
  const text = readFileSync(path, "utf8");
  console.log(`===== ${title} (${path}) =====\n${(pick ? pick(text) : text).trim()}\n`);
}

// Only the "## Rules" section of the maintainer entry point: the rest is read on demand.
function rulesSection(text) {
  const start = text.indexOf("## Rules");
  if (start < 0) return text;
  const end = text.indexOf("\n## ", start + 1);
  return text.slice(start, end < 0 ? undefined : end);
}

console.log(`Kokon repo: ${root}. Read AGENTS.md there before working on Kokon.\n`);
show("Kokon public status", "docs/STATUS.md");
show("Maintainer rules", "private/AGENTS.md", rulesSection);
show("Maintainer status", "private/STATUS.md");

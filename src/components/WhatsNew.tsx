import { CircleAlert, Images, LayoutGrid, Maximize, SlidersHorizontal, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { formatBytes, installProgress, installStep, installTimeLeft } from "../lib/logic";
import { refreshStatus, setState, toastError, useStore } from "../lib/store";
import type { GpuBackend, InstallItem, InstallStatus } from "../lib/types";
import { KokonMark } from "./Welcome";

/** The release these notes describe. Shown once to people updating from an earlier version. */
export const WHATS_NEW_VERSION = "0.9.1";

const NEWS = [
  {
    icon: Sparkles,
    title: "Superscale-SP by default",
    text: "Results are enlarged with a sharper, more natural upscaler. The previous one, UltraSharp, was great on some images but smoothed skin and textures into an artificial look on others. You can still download it.",
  },
  {
    icon: SlidersHorizontal,
    title: "Choose your upscaler",
    text: "In the Advanced menu next to the prompt: Superscale-SP, the experimental MoSR (very fast, included), UltraSharp or a plain resize, a strength slider, and a side-by-side comparison.",
  },
  {
    icon: LayoutGrid,
    title: "Gallery preview",
    text: "Click an image in the gallery for a large preview and browse with the arrow keys; F or a double-click for full screen. Enter opens it in its project.",
  },
  {
    icon: Maximize,
    title: "Full screen and renaming",
    text: "Full screen follows the project's timeline: the arrow keys step through versions without leaving it. Renaming now starts from the current name.",
  },
];

/** Remembers that these notes were read and, if the engine update was postponed, for which
 * release, so neither is offered again at the next start. */
async function markSeen(engineSkipped: string | null) {
  try {
    const cfg = await api.config();
    const ui: Record<string, unknown> = { ...((cfg.ui ?? {}) as Record<string, unknown>), whatsNewSeen: WHATS_NEW_VERSION };
    if (engineSkipped) ui.engineUpdateSkipped = engineSkipped;
    await api.saveConfig({ ...cfg, ui });
  } catch (e) {
    toastError(e);
  }
}

/** Whether to show the notes at start: once per release after an update (not on a new install,
 * which gets the welcome screen), and again while a needed engine update was not postponed. */
export function whatsNewDue(ui: Record<string, unknown>, engine: { release: string; update_needed: boolean } | undefined): boolean {
  if (!ui.welcomeSeen) return false;
  return ui.whatsNewSeen !== WHATS_NEW_VERSION || (!!engine?.update_needed && ui.engineUpdateSkipped !== engine.release);
}

/** What's new in this release, plus the engine download when the release needs a newer
 * stable-diffusion.cpp (only the files that changed; models and settings stay as they are). */
export function WhatsNew() {
  const engine = useStore((s) => s.status?.engine);
  const needsEngine = !!engine?.update_needed;
  const [missing, setMissing] = useState<InstallItem[] | null>(null);
  const [target, setTarget] = useState<{ preset: string; backend: GpuBackend } | null>(null);
  const [inst, setInst] = useState<InstallStatus | null>(null);

  useEffect(() => {
    if (!needsEngine) return;
    api.hardware().then((h) => {
      const preset = h.presets.find((p) => p.id === h.current) ?? h.presets.find((p) => p.id === h.recommended);
      setTarget(preset ? { preset: preset.id, backend: h.backend } : null);
      setMissing((preset?.files ?? []).filter((f) => f.state !== "present"));
    }, toastError);
    api.installStatus().then((i) => i.running && setInst(i), () => {});
  }, [needsEngine]);

  const running = inst?.running ?? false;
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(() => api.installStatus().then(setInst, () => {}), 500);
    return () => window.clearInterval(t);
  }, [running]);

  const items = inst?.items ?? missing ?? [];
  const failed = items.filter((i) => i.state === "failed");
  const done = !!inst && !inst.running && inst.complete;
  const size = (missing ?? []).reduce((a, i) => a + Math.max(0, i.size - i.received), 0);
  const prog = installProgress(items);
  const showEngine = needsEngine || !!inst;

  const update = async () => {
    if (!target) return;
    try {
      setInst(await api.install(target.preset, target.backend, false));
    } catch (e) {
      toastError(e);
    }
  };
  const close = (postponed: boolean) => {
    if (running) api.cancelInstall().catch(() => {});
    setState({ overlay: null });
    markSeen(postponed && engine ? engine.release : null);
    refreshStatus();
    api.startBackend().then(refreshStatus, () => refreshStatus());
  };

  return (
    <div className="modal-scrim">
      <div className="welcome" role="dialog" aria-label={`What's new in Kokon ${WHATS_NEW_VERSION}`}>
        <KokonMark size={44} />
        <h1>What's new in Kokon {WHATS_NEW_VERSION}</h1>
        <div className="welcome-grid">
          {NEWS.map(({ icon: Icon, title, text }) => (
            <div className="welcome-item" key={title}>
              <Icon size={20} />
              <strong>{title}</strong>
              <span>{text}</span>
            </div>
          ))}
        </div>
        <button className="btn" onClick={() => setState({ compareUpscalers: true })}>
          <Images size={14} /> Compare the upscalers
        </button>

        {showEngine && !done && (
          <p className="welcome-lead">
            This version needs a newer stable-diffusion.cpp, the engine that runs the models on your graphics chip, and the Superscale-SP model. Your images, models and settings stay
            as they are.
          </p>
        )}
        {(running || (inst && !done && !failed.length)) && (
          <div className="setup-card setup-progress" style={{ width: "100%", textAlign: "left" }}>
            <div className="setup-progress-head">
              <span className="pct">{Math.floor(prog.frac * 100)}%</span>
              <span className="muted">
                {formatBytes(prog.received)} of {formatBytes(prog.total)}
                {installTimeLeft(prog.total - prog.received, inst?.bytes_per_sec ?? 0) ? ` · ${installTimeLeft(prog.total - prog.received, inst?.bytes_per_sec ?? 0)}` : ""}
              </span>
            </div>
            <div className="progress thick">
              <i style={{ width: `${prog.frac * 100}%` }} />
            </div>
            <span className="muted">{installStep(items) ?? "Starting…"}</span>
          </div>
        )}
        {failed.length > 0 && !running && (
          <div className="setup-note" style={{ width: "100%", textAlign: "left" }}>
            <CircleAlert size={14} /> {failed.map((f) => `${f.label}: ${f.error}`).join(" · ")}
            <div className="muted">Try again, or open Settings → Models → Set up models… to download it by hand.</div>
          </div>
        )}

        {!showEngine || done ? (
          <button className="btn primary big" onClick={() => close(false)}>
            {done ? "Start creating" : "Got it"}
          </button>
        ) : (
          <>
            <div style={{ display: "flex", gap: 10 }}>
              <button className="btn primary big" disabled={running || !target || missing === null} onClick={update}>
                {failed.length ? "Try again" : running ? "Updating…" : `Update now${size ? ` · ${formatBytes(size)}` : ""}`}
              </button>
              <button className="btn big" onClick={() => close(true)}>
                Later
              </button>
            </div>
            <span className="muted welcome-foot">Until you update, enlargements use a plain resize. You can update later from Settings → Models → Set up models…</span>
          </>
        )}
      </div>
    </div>
  );
}

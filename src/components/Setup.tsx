import { Bug, Check, ChevronDown, CircleAlert, Download, ExternalLink, FolderOpen, Loader2, Pause, Play, RefreshCw, Settings2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { formatBytes, installItemShort, installProgress, installStep, installTimeLeft } from "../lib/logic";
import { native } from "../lib/native";
import { refreshStatus, setState, toastError, useStore } from "../lib/store";
import type { GpuBackend, HardwareReport, InstallItem, InstallStatus, SetupReport } from "../lib/types";
import { Seg } from "./ui";
import { ReadyTips, Showcase } from "./Welcome";

const gb = (mib: number) => `${Math.round(mib / 1024)} GB`;

function engineOptions(vendor: string | null): { value: GpuBackend; label: string }[] {
  if (vendor === "apple") return [{ value: "metal", label: "Metal (beta)" }];
  const vulkan = { value: "vulkan" as GpuBackend, label: vendor === "nvidia" ? "Vulkan" : "Vulkan (beta)" };
  return vendor === "nvidia" ? [{ value: "cuda", label: "CUDA" }, vulkan] : [vulkan];
}

function engineNote(b: GpuBackend, nvidia: boolean): string {
  if (b === "cuda") return "NVIDIA's CUDA build: the fastest and tested path. Recommended.";
  if (b === "metal") return "Beta. Runs on the Mac's GPU through Metal. Needs macOS 26 or newer.";
  return nvidia
    ? "The same model through Vulkan: works on NVIDIA, but slower than CUDA (no SageAttention) and the first image takes longer. Useful to try the AMD / Intel path; both engines can stay installed."
    : "Beta. AMD and Intel cards run the model through Vulkan, with the graphics driver you already have. Not yet tested on a real AMD or Intel card: please report how it goes.";
}

function ItemIcon({ it }: { it: InstallItem }) {
  switch (it.state) {
    case "present":
    case "done":
      return <Check size={16} />;
    case "failed":
      return <CircleAlert size={16} />;
    case "downloading":
    case "verifying":
    case "extracting":
      return <Loader2 className="spin" size={16} />;
    default:
      return <Download size={16} style={{ opacity: 0.4 }} />;
  }
}

/** A file that could not be downloaded: what happened and how to get it by hand. */
function Failure({ it, issuesUrl }: { it: InstallItem; issuesUrl: string }) {
  const folder = it.kind === "engine" ? it.path : it.path.replace(/[\\/][^\\/]+$/, "");
  return (
    <div className="setup-failure">
      <strong>{it.label}</strong>
      <span>{it.error}</span>
      <span>{it.hint}</span>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button className="btn" onClick={() => native.openLink(it.page).catch(toastError)}>
          <ExternalLink size={14} /> Download page
        </button>
        <button className="btn" onClick={() => native.openFolder(folder).catch(toastError)}>
          <FolderOpen size={14} /> Open folder
        </button>
        <button className="btn" onClick={() => native.openLink(issuesUrl).catch(toastError)}>
          <Bug size={14} /> Report the problem
        </button>
      </div>
    </div>
  );
}

function FileList({ items }: { items: InstallItem[] }) {
  return (
    <div className="setup-files">
      {items.map((it) => (
        <div className={`setup-file ${it.state}`} key={it.id}>
          <ItemIcon it={it} />
          <span className="name">{it.label}</span>
          <span className="state">{installItemShort(it)}</span>
        </div>
      ))}
    </div>
  );
}

/** First launch (and Settings → Models → Set up models): detect the GPU, pick a preset,
 * download stable-diffusion.cpp and the official Qwen-Image 2.1 weights, then start. */
export function Setup() {
  const [hw, setHw] = useState<HardwareReport | null>(null);
  const [report, setReport] = useState<SetupReport | null>(null);
  const [inst, setInst] = useState<InstallStatus | null>(null);
  const [preset, setPreset] = useState<string | null>(null);
  const [backend, setBackend] = useState<GpuBackend | null>(null);
  const [showFiles, setShowFiles] = useState(false);
  const [checking, setChecking] = useState(false);
  const wasRunning = useRef(false);
  const setupDone = useStore((s) => s.status?.setup_complete ?? false);
  const engineUpdate = useStore((s) => s.status?.engine);

  const refresh = async (engine: GpuBackend | null = backend) => {
    setChecking(true);
    try {
      const [h, r, i] = await Promise.all([api.hardware(engine ?? undefined), api.setup(), api.installStatus()]);
      setHw(h);
      setBackend(h.backend);
      setReport(r);
      setInst(i);
      setPreset((p) => p ?? (i.running && i.preset ? i.preset : h.recommended));
    } catch (e) {
      toastError(e);
    } finally {
      setChecking(false);
    }
  };
  useEffect(() => {
    refresh();
  }, []);

  // Follow a running installation; re-check everything once it ends.
  useEffect(() => {
    if (!inst?.running) {
      if (wasRunning.current) {
        wasRunning.current = false;
        refresh();
      }
      return;
    }
    wasRunning.current = true;
    const t = window.setInterval(() => api.installStatus().then(setInst, () => {}), 500);
    return () => window.clearInterval(t);
  }, [inst?.running]);

  const chooseFolder = async () => {
    const dir = await native.pickFolder(hw?.sdcpp_dir);
    if (!dir) return;
    try {
      const cfg = await api.config();
      await api.saveConfig({ ...cfg, sdcppDir: dir });
      refresh();
    } catch (e) {
      toastError(e);
    }
  };

  const install = async () => {
    if (!preset) return;
    try {
      setInst(await api.install(preset, backend ?? hw?.backend ?? "cuda"));
    } catch (e) {
      toastError(e);
    }
  };

  const start = async () => {
    try {
      const [cfg, st] = await Promise.all([api.config(), refreshStatus()]);
      // An engine update the user chose not to install now: don't open Setup for it again.
      const ui = { ...((cfg.ui ?? {}) as Record<string, unknown>) };
      if (st?.engine?.update_needed) ui.engineUpdateSkipped = st.engine.release;
      await api.saveConfig({ ...cfg, setupComplete: true, ui });
      setState({ needsSetup: false });
      // Load the model in the background; the status indicator shows progress.
      api.startBackend().then(refreshStatus, toastError);
      refreshStatus();
    } catch (e) {
      toastError(e);
    }
  };

  const chosen = hw?.presets.find((p) => p.id === preset) ?? null;
  const running = inst?.running ?? false;
  // While (or after) installing this preset, show the live list; otherwise what is on disk.
  const live = !!inst && inst.preset === preset && inst.items.length > 0 && (running || inst.items.some((i) => i.state !== "pending" && i.state !== "present"));
  const items = live ? inst!.items : chosen?.files ?? [];
  const missing = items.filter((i) => i.state !== "present" && i.state !== "done");
  const failed = items.filter((i) => i.state === "failed");
  const paused = !running && items.some((i) => i.state === "cancelled" || (i.state === "pending" && i.received > 0));
  const toDownload = missing.reduce((a, i) => a + Math.max(0, i.size - i.received), 0);
  const prog = installProgress(items);
  const lowSpace = hw?.free_bytes != null && chosen != null && hw.free_bytes < chosen.disk_bytes;
  const presetInstalled = missing.length === 0;
  const ready = !!report?.ready && !running;
  const rate = inst?.bytes_per_sec ?? 0;
  const vram = hw?.vram_mib ?? null;
  const engineMissing = missing.filter((i) => i.kind === "engine");

  return (
    <div className="setup">
      <div className="setup-inner">
        <div>
          <h1>{setupDone ? "Models" : "Welcome to Kokon"}</h1>
          <p className="setup-lead">
            Everything runs on this PC. Setup downloads the image model once ({chosen ? formatBytes(chosen.files.reduce((a, f) => a + f.size, 0)) : "about 12 GB"}); after that, nothing
            leaves your computer.
          </p>
        </div>

        {setupDone && engineUpdate?.update_needed && engineMissing.length > 0 && !running && (
          <div className="setup-note">
            This version of Kokon enlarges images with Superscale-SP by default, adds an upscaler strength slider and the experimental MoSR model. It needs the updated engine:
            press Install to download it ({formatBytes(engineMissing.reduce((a, i) => a + Math.max(0, i.size - i.received), 0))}). Until then, images are enlarged with Lanczos.
          </div>
        )}

        {!hw ? (
          <div className="setup-card">
            <Loader2 className="spin" size={18} /> Checking this PC…
          </div>
        ) : (
          <>
            <div className="setup-card setup-gpu">
              <strong>{hw.gpu ? hw.gpu.name : "No graphics card found"}</strong>
              <span className="muted">
                {hw.gpu?.unified
                  ? `${gb(hw.gpu.memory_total_mib)} unified memory${hw.vram_simulated ? ` (planning as ${gb(vram ?? 0)})` : ""}`
                  : `${vram ? `${gb(vram)} VRAM${hw.vram_simulated ? " (simulated)" : ""} · ` : ""}${gb(hw.ram_total_mib)} RAM`}{hw.gpu?.driver_version ? ` · driver ${hw.gpu.driver_version}` : ""}
              </span>
            </div>
            {!hw.gpu && (
              <div className="setup-note">
                Kokon needs a graphics card with at least 12 GB of VRAM: NVIDIA (RTX 20 series or newer), or AMD / Intel through Vulkan (beta), or a Mac with Apple Silicon. Make sure its driver is installed.
              </div>
            )}
            {hw.gpu?.unified ? (
              hw.gpu.memory_total_mib < 23 * 1024 ? (
                <div className="setup-note">
                  With {gb(hw.gpu.memory_total_mib)} of unified memory Kokon works, but slowly: model weights are read from disk as needed and images are made at about 1 megapixel, which takes about 4 to 5 minutes on an M2 Pro. 32 GB or more and a Pro, Max or Ultra chip are recommended. Close other apps while generating.
                </div>
              ) : (
                hw.gpu.memory_total_mib < 31 * 1024 && (
                  <div className="setup-note">
                    With {gb(hw.gpu.memory_total_mib)} of unified memory expect several minutes per image. 32 GB or more and a Pro, Max or Ultra chip are recommended.
                  </div>
                )
              )
            ) : (
              <>
                {vram !== null && vram < 11 * 1024 && <div className="setup-note">Your GPU has {gb(vram)} of VRAM. 12 GB is the smallest size this app is made for; it may fail or run slowly.</div>}
                {hw.ram_total_mib < 15 * 1024 && <div className="setup-note">The text encoder runs from system RAM; with {gb(hw.ram_total_mib)} of RAM, 16 GB or more is recommended.</div>}
              </>
            )}

            {running || paused ? (
              <div className="muted">
                {chosen?.label} preset · {hw.sdcpp_dir}
              </div>
            ) : (
              <>
                {hw.gpu && (
                  <div className="field">
                    <span>Engine</span>
                    <Seg<GpuBackend>
                      value={backend ?? hw.backend}
                      options={engineOptions(hw.vendor)}
                      onChange={(b) => {
                        setBackend(b);
                        refresh(b);
                      }}
                    />
                    <span className="muted">{engineNote(backend ?? hw.backend, hw.nvidia)}</span>
                  </div>
                )}
                <div className="field">
                  <span>{hw.gpu?.unified ? "Preset for your Mac's memory" : "Preset for your graphics card"}</span>
                  <Seg<string> value={preset ?? hw.recommended} options={hw.presets.map((p) => ({ value: p.id, label: p.label }))} onChange={setPreset} />
                  {chosen && (
                    <span className="muted">
                      {chosen.summary}.{chosen.id === hw.recommended ? " Recommended for your GPU." : ""}
                      {hw.gpu?.unified ? " Not yet tested on a Mac." : chosen.verified ? "" : " Not yet tested on a real card of this size."}
                    </span>
                  )}
                  {hw.gpu?.unified && (
                    <span className="muted">
                      Presets are named by total unified memory. macOS lets the graphics part use about two thirds of it, and the text encoder shares the same memory, so the image model gets less than the number on your Mac: a 16 GB Mac has room for the lightest preset only. Pick a larger one only on a Mac with that much memory.
                    </span>
                  )}
                </div>
                <div className="field">
                  <span>Install folder</span>
                  <div className="path-row">
                    <input className="input" type="text" value={hw.sdcpp_dir} readOnly spellCheck={false} />
                    <button className="btn" onClick={chooseFolder}>
                      Change
                    </button>
                  </div>
                  <span className="muted">
                    {hw.free_bytes != null ? `${formatBytes(hw.free_bytes)} free` : ""}
                    {chosen && chosen.disk_bytes > 0 ? ` · needs about ${formatBytes(chosen.disk_bytes)}` : ""}
                    {lowSpace ? " — not enough space on this drive; choose another folder." : ""}
                  </span>
                </div>
              </>
            )}

            {/* Progress panel while installing or paused. */}
            {(running || paused) && (
              <div className="setup-card setup-progress">
                <div className="setup-progress-head">
                  <span className="pct">{Math.floor(prog.frac * 100)}%</span>
                  <span className="muted">
                    {formatBytes(prog.received)} of {formatBytes(prog.total)}
                    {running && rate > 0 ? ` · ${formatBytes(rate)}/s` : ""}
                    {running ? (installTimeLeft(prog.total - prog.received, rate) ? ` · ${installTimeLeft(prog.total - prog.received, rate)}` : "") : " · paused"}
                  </span>
                </div>
                <div className="progress thick">
                  <i style={{ width: `${prog.frac * 100}%` }} />
                </div>
                <span className="muted">{running ? installStep(items) ?? "Starting…" : "Downloads continue where they stopped, also after closing the app."}</span>
                <div>
                  {running ? (
                    <button className="btn" onClick={() => api.cancelInstall().then(setInst, toastError)}>
                      <Pause size={14} /> Pause
                    </button>
                  ) : (
                    <button className="btn primary" onClick={install} disabled={lowSpace}>
                      <Play size={14} /> Resume
                    </button>
                  )}
                </div>
              </div>
            )}

            {failed.length > 0 && !running && (
              <div className="setup-note">
                <div style={{ marginBottom: 8 }}>
                  {failed.length === 1 ? "One file" : `${failed.length} files`} could not be downloaded. Try again, or get {failed.length === 1 ? "it" : "them"} by hand and press Check again.
                  Please report a broken link so it can be fixed for everyone.
                </div>
                {failed.map((it) => (
                  <Failure key={it.id} it={it} issuesUrl={hw.issues_url} />
                ))}
              </div>
            )}

            {presetInstalled && !running && (
              <div className="setup-card setup-ready">
                <Check size={18} />
                <span>{hw.current === chosen?.id || !chosen ? "Everything is installed." : `The ${chosen.label} files are already here.`}</span>
              </div>
            )}
            {presetInstalled && !running && !setupDone && <ReadyTips />}
            {(running || paused) && <Showcase />}

            {/* What gets installed. */}
            <div className="field">
              <button className="setup-toggle" onClick={() => setShowFiles((v) => !v)} aria-expanded={showFiles || running}>
                <ChevronDown size={14} style={{ transform: showFiles || running ? "none" : "rotate(-90deg)", transition: "transform 0.15s" }} />
                {items.length} files · {missing.length ? `${formatBytes(toDownload)} to download` : "all installed"}
              </button>
              {(showFiles || running) && <FileList items={items} />}
            </div>
          </>
        )}

        <div className="setup-actions">
          {hw && !running && !paused && !presetInstalled && (
            <button className="btn primary big" disabled={!chosen || lowSpace} onClick={install}>
              <Download size={16} /> {failed.length ? "Try again" : `Install · ${formatBytes(toDownload)}`}
            </button>
          )}
          {hw && presetInstalled && !running && chosen && (hw.current !== chosen.id || hw.current_backend !== hw.backend) && (
            // Files of another preset or engine are already here: switch without downloading.
            <button className="btn" onClick={install}>
              {hw.current_backend !== hw.backend ? `Use ${{ cuda: "CUDA", vulkan: "Vulkan", metal: "Metal" }[hw.backend]} with the ${chosen.label} preset` : `Use the ${chosen.label} preset`}
            </button>
          )}
          {(ready || presetInstalled) && !running && (
            <button className={`btn big ${presetInstalled ? "primary" : ""}`} disabled={!ready} onClick={start}>
              {setupDone ? "Done" : "Start"}
            </button>
          )}
          <span style={{ flex: 1 }} />
          <button className="btn ghost" onClick={() => setState({ overlay: "settings" })} disabled={running}>
            <Settings2 size={14} /> Settings
          </button>
          <button className="btn ghost" onClick={() => refresh()} disabled={checking || running}>
            <RefreshCw size={14} /> Check again
          </button>
        </div>

        {report && !report.ready && !running && !paused && missing.length === 0 && (
          <div className="setup-note">
            {report.checks
              .filter((c) => !c.ok)
              .map((c, i) => (
                <div key={i}>
                  <strong>{c.label}</strong>: {c.detail}
                </div>
              ))}
          </div>
        )}
        <p className="muted setup-foot">
          Qwen-Image 2.1 Turbo is released by Alibaba's Qwen team under the Qwen Research License: research and evaluation only, no commercial use. By downloading it you accept that license. The upscalers are Superscale-SP by Nmkd
          (WTFPL) and the experimental 4xNomos2_hq_mosr by Philip Hofmann (CC BY 4.0). Downloads come only from GitHub (stable-diffusion.cpp, upscaler) and Hugging Face, and every file is verified with its SHA-256 checksum.
        </p>
      </div>
    </div>
  );
}

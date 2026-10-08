import { Bug, Check, ChevronDown, CircleAlert, Download, ExternalLink, FolderOpen, Info, Loader2, Pause, Play, RefreshCw, Settings2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { formatBytes, installItemShort, installProgress, installStep, installTimeLeft } from "../lib/logic";
import { native } from "../lib/native";
import { refreshStatus, setState, toastError, useStore } from "../lib/store";
import type { HardwareReport, InstallItem, InstallStatus, SetupReport } from "../lib/types";
import { Seg, Switch } from "./ui";

const gb = (mib: number) => `${Math.round(mib / 1024)} GB`;

function ItemIcon({ it }: { it: InstallItem }) {
  switch (it.state) {
    case "present":
    case "done":
      return <Check size={16} />;
    case "failed":
      return it.kind === "optional" ? <Info size={16} /> : <CircleAlert size={16} />;
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
          <span className="name">
            {it.label}
            {it.kind === "optional" ? <span className="muted"> · optional</span> : null}
          </span>
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
  const [upscaler, setUpscaler] = useState(true);
  const [showFiles, setShowFiles] = useState(false);
  const [checking, setChecking] = useState(false);
  const wasRunning = useRef(false);
  const setupDone = useStore((s) => s.status?.setup_complete ?? false);

  const refresh = async () => {
    setChecking(true);
    try {
      const [h, r, i] = await Promise.all([api.hardware(), api.setup(), api.installStatus()]);
      setHw(h);
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
      setInst(await api.install(preset, upscaler));
    } catch (e) {
      toastError(e);
    }
  };

  const start = async () => {
    try {
      const cfg = await api.config();
      await api.saveConfig({ ...cfg, setupComplete: true });
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
  const items = (live ? inst!.items : chosen?.files ?? []).filter((i) => upscaler || i.kind !== "optional" || live);
  const missing = items.filter((i) => i.state !== "present" && i.state !== "done");
  const failed = items.filter((i) => i.state === "failed");
  const paused = !running && items.some((i) => i.state === "cancelled" || (i.state === "pending" && i.received > 0));
  const toDownload = missing.reduce((a, i) => a + Math.max(0, i.size - i.received), 0);
  const prog = installProgress(items);
  const lowSpace = hw?.free_bytes != null && chosen != null && hw.free_bytes < chosen.disk_bytes;
  const presetInstalled = missing.filter((i) => i.kind !== "optional").length === 0;
  const ready = !!report?.ready && !running;
  const rate = inst?.bytes_per_sec ?? 0;
  const vram = hw?.vram_mib ?? null;
  const optionalMissing = chosen?.files.some((f) => f.kind === "optional" && f.state !== "present");

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

        {!hw ? (
          <div className="setup-card">
            <Loader2 className="spin" size={18} /> Checking this PC…
          </div>
        ) : (
          <>
            <div className="setup-card setup-gpu">
              <strong>{hw.gpu ? hw.gpu.name : "No NVIDIA GPU found"}</strong>
              <span className="muted">
                {vram ? `${gb(vram)} VRAM${hw.vram_simulated ? " (simulated)" : ""} · ` : ""}
                {gb(hw.ram_total_mib)} RAM{hw.gpu ? ` · driver ${hw.gpu.driver_version}` : ""}
              </span>
            </div>
            {!hw.nvidia && (
              <div className="setup-note">
                This version needs an NVIDIA GPU (RTX 20 series or newer) with at least 12 GB of VRAM and a recent driver. Support for AMD and Intel cards (Vulkan) is planned.
              </div>
            )}
            {vram !== null && vram < 11 * 1024 && <div className="setup-note">Your GPU has {gb(vram)} of VRAM. 12 GB is the smallest size this app is made for; it may fail or run slowly.</div>}
            {hw.ram_total_mib < 15 * 1024 && <div className="setup-note">The text encoder runs from system RAM; with {gb(hw.ram_total_mib)} of RAM, 16 GB or more is recommended.</div>}

            {running || paused ? (
              <div className="muted">
                {chosen?.label} preset · {hw.sdcpp_dir}
              </div>
            ) : (
              <>
                <div className="field">
                  <span>Preset for your graphics card</span>
                  <Seg<string> value={preset ?? hw.recommended} options={hw.presets.map((p) => ({ value: p.id, label: p.label }))} onChange={setPreset} />
                  {chosen && (
                    <span className="muted">
                      {chosen.summary}.{chosen.id === hw.recommended ? " Recommended for your GPU." : ""}
                      {chosen.verified ? "" : " Not yet tested on a real card of this size."}
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

            {/* What gets installed. */}
            <div className="field">
              <button className="setup-toggle" onClick={() => setShowFiles((v) => !v)} aria-expanded={showFiles || running}>
                <ChevronDown size={14} style={{ transform: showFiles || running ? "none" : "rotate(-90deg)", transition: "transform 0.15s" }} />
                {items.length} files · {missing.length ? `${formatBytes(toDownload)} to download` : "all installed"}
              </button>
              {(showFiles || running) && <FileList items={items} />}
              {optionalMissing && !running && !paused && (
                <div className="field-row" style={{ paddingTop: 4 }}>
                  <span>Include the 4x-UltraSharp upscaler for sharper enlargements (67 MB, CC BY-NC-SA 4.0, non-commercial)</span>
                  <Switch on={upscaler} onChange={setUpscaler} label="Include the upscaler" />
                </div>
              )}
            </div>
          </>
        )}

        <div className="setup-actions">
          {hw && !running && !paused && !presetInstalled && (
            <button className="btn primary big" disabled={!chosen || lowSpace} onClick={install}>
              <Download size={16} /> {failed.length ? "Try again" : `Install · ${formatBytes(toDownload)}`}
            </button>
          )}
          {hw && presetInstalled && !running && chosen && hw.current !== chosen.id && (
            // Files of another preset are already here: switch to it without downloading.
            <button className="btn" onClick={install}>
              Use the {chosen.label} preset
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
          <button className="btn ghost" onClick={refresh} disabled={checking || running}>
            <RefreshCw size={14} /> Check again
          </button>
        </div>

        {report && !report.ready && !running && !paused && missing.length === 0 && (
          <div className="setup-note">
            {report.checks
              .filter((c) => !c.ok && c.id !== "upscaler")
              .map((c, i) => (
                <div key={i}>
                  <strong>{c.label}</strong>: {c.detail}
                </div>
              ))}
          </div>
        )}
        <p className="muted setup-foot">
          Qwen-Image 2.1 and Qwen3-VL are released by Alibaba's Qwen team under their own licenses; check them before commercial use. Downloads come only from GitHub
          (stable-diffusion.cpp) and Hugging Face, and every file is verified with its SHA-256 checksum.
        </p>
      </div>
    </div>
  );
}

import { Check, CircleAlert, FolderOpen, Info, Loader2, RefreshCw, Settings2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { native } from "../lib/native";
import { refreshStatus, setState, toastError } from "../lib/store";
import type { SetupReport } from "../lib/types";
import { defaultProfilePaths } from "./Settings";

/** First launch: verify folder, executable, the four model files and CUDA. */
export function Setup() {
  const [report, setReport] = useState<SetupReport | null>(null);
  const [checking, setChecking] = useState(false);

  const run = async () => {
    setChecking(true);
    try {
      setReport(await api.setup());
    } catch (e) {
      toastError(e);
    } finally {
      setChecking(false);
    }
  };
  useEffect(() => {
    run();
  }, []);

  const chooseFolder = async () => {
    const dir = await native.pickFolder("C:\\AI\\sdcpp");
    if (!dir) return;
    try {
      const cfg = await api.config();
      const profiles = cfg.profiles.map((p) => (p.id === cfg.activeProfile ? defaultProfilePaths(dir, p) : p));
      await api.saveConfig({ ...cfg, sdcppDir: dir, profiles });
      run();
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

  return (
    <div className="setup">
      <div className="setup-inner">
        <h1>Welcome</h1>
        <p className="muted" style={{ fontSize: 14, margin: 0 }}>
          Qwen Local Studio runs entirely on this PC using stable-diffusion.cpp. Checking your installation…
        </p>
        <div className="check-list">
          {!report && (
            <div className="check-item">
              <Loader2 className="spin" size={18} /> <span className="what">Checking</span>
            </div>
          )}
          {report?.checks.map((c, i) => (
            <div className="check-item" key={i}>
              {c.ok ? <Check size={18} /> : c.id === "upscaler" ? <Info size={18} /> : <CircleAlert size={18} />}
              <div className="what">
                <div>{c.label}</div>
                <div className="detail">{c.detail}</div>
              </div>
            </div>
          ))}
          {report?.backend_version && <div className="muted" style={{ paddingTop: 8 }}>stable-diffusion.cpp {report.backend_version}</div>}
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button className="btn primary" disabled={!report?.ready} onClick={start}>
            Start
          </button>
          <button className="btn" onClick={chooseFolder}>
            <FolderOpen size={15} /> Choose stable-diffusion.cpp folder
          </button>
          <button className="btn" onClick={() => setState({ overlay: "settings" })}>
            <Settings2 size={15} /> Settings
          </button>
          <button className="btn" onClick={run} disabled={checking}>
            <RefreshCw size={15} /> Check again
          </button>
        </div>
        {report && !report.ready && <p className="muted">Fix the items marked above, then check again. Model files are not downloaded by this app.</p>}
      </div>
    </div>
  );
}

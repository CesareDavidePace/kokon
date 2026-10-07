import { getCurrentWebview } from "@tauri-apps/api/webview";
import { Images, Loader2, Settings, SquarePen, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api, configure } from "./lib/api";
import { emptyComposer } from "./lib/logic";
import { native } from "./lib/native";
import {
  addInputs, dismissToast, getState, refreshStatus, setState, startEvents, submitComposer, toastError, toastInfo, useStore,
} from "./lib/store";
import { Composer, importPaths } from "./components/Composer";
import { ExpandEditor } from "./components/ExpandEditor";
import { Gallery } from "./components/Gallery";
import { InpaintEditor } from "./components/InpaintEditor";
import { Settings as SettingsDialog } from "./components/Settings";
import { Setup } from "./components/Setup";
import { HistoryRail, Stage } from "./components/Stage";
import { IconButton } from "./components/ui";

const IMAGE_EXT = /\.(png|jpe?g|webp|bmp|gif|tiff?|heic|heif)$/i;

function StatusIndicator() {
  const connected = useStore((s) => s.connected);
  const server = useStore((s) => s.server);
  const busy = useStore((s) => Object.values(s.jobs).some((j) => !["done", "failed", "cancelled"].includes(j.status)));
  let cls = "dot";
  let text = "Connecting";
  let title = "";
  if (connected) {
    switch (server.state) {
      case "ready":
        cls = "dot ready";
        text = busy ? "Working" : "Ready";
        title = server.owned ? "Model loaded" : "Using an inference server that was already running";
        break;
      case "starting":
        cls = "dot busy";
        text = "Loading model";
        break;
      case "error":
        cls = "dot error";
        text = server.error?.message ?? "Inference server error";
        title = "Open Diagnostics";
        break;
      default:
        text = "Model not loaded";
        title = "Click to load the model now";
    }
  }
  return (
    <button
      className="status"
      title={title}
      onClick={() => {
        if (server.state === "error") setState({ overlay: "settings" });
        else if (server.state === "stopped" && connected) api.startBackend().then(refreshStatus, toastError);
      }}
    >
      <span className={cls} />
      <span style={{ maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{text}</span>
    </button>
  );
}

function Toasts() {
  const toasts = useStore((s) => s.toasts);
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="toasts" role="status">
      {toasts.map((t) => (
        <div key={t.id} className="toast">
          <div className="body">
            <div>{t.message}</div>
            {t.suggestion && <div className="sugg">{t.suggestion}</div>}
            {t.detail && (
              <button onClick={() => setOpen(open === t.id ? null : t.id)} style={{ marginTop: 4 }}>
                {open === t.id ? "Hide details" : "Details"}
              </button>
            )}
            {open === t.id && <pre>{t.detail}</pre>}
          </div>
          <button onClick={() => dismissToast(t.id)} aria-label="Dismiss">
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}

async function uploadBlobs(files: File[]) {
  const recs = [];
  for (const f of files) {
    try {
      recs.push(await api.upload(f, f.name || "pasted.png"));
    } catch (e) {
      toastError(e);
    }
  }
  addInputs(recs);
}

export default function App() {
  const overlay = useStore((s) => s.overlay);
  const needsSetup = useStore((s) => s.needsSetup);
  const [boot, setBoot] = useState<"connecting" | "ok" | string>("connecting");
  const [dragging, setDragging] = useState(false);

  // Connect to (or start) the local service.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const conn = await native.connect();
        if (cancelled) return;
        configure(conn.baseUrl, conn.token);
        startEvents();
        const cfg = await api.config();
        const st = await refreshStatus();
        setState({ needsSetup: !cfg.setupComplete });
        setBoot("ok");
        // Preload the model so the first generation starts immediately.
        if (cfg.setupComplete && st && st.server.state === "stopped") api.startBackend().then(refreshStatus, () => refreshStatus());
      } catch (e) {
        setBoot(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Native file drops (paths) — efficient: the service reads the files itself.
  useEffect(() => {
    let un: (() => void) | undefined;
    getCurrentWebview()
      .onDragDropEvent((ev) => {
        const p = ev.payload;
        if (p.type === "enter" || p.type === "over") setDragging(true);
        else if (p.type === "leave") setDragging(false);
        else if (p.type === "drop") {
          setDragging(false);
          const paths = p.paths.filter((x) => IMAGE_EXT.test(x));
          if (paths.length) importPaths(paths);
          else if (p.paths.length) toastInfo("Only image files can be added.");
        }
      })
      .then((u) => (un = u))
      .catch(() => {});
    return () => un?.();
  }, []);

  // Clipboard paste: browser image data first, native clipboard as a fallback.
  useEffect(() => {
    const onPaste = async (e: ClipboardEvent) => {
      if (getState().overlay) return;
      const files = Array.from(e.clipboardData?.files ?? []).filter((f) => f.type.startsWith("image/"));
      if (files.length) {
        e.preventDefault();
        uploadBlobs(files);
        return;
      }
      if (e.clipboardData?.getData("text")) return;
      try {
        const png = await native.readClipboardImage();
        if (png) uploadBlobs([new File([png as BlobPart], "pasted.png", { type: "image/png" })]);
      } catch {
        /* nothing to paste */
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, []);

  // Keyboard: arrows browse the session, Ctrl+Enter generates.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement)?.closest("input, textarea, select");
      const s = getState();
      if (s.overlay) return;
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        submitComposer();
        return;
      }
      if (typing) return;
      if (e.key === "ArrowLeft" || e.key === "ArrowRight" || e.key === "ArrowUp" || e.key === "ArrowDown") {
        const i = s.session.findIndex((x) => x.key === s.selected);
        const d = e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1;
        const n = s.session[Math.max(0, Math.min(s.session.length - 1, i + d))];
        if (n) setState({ selected: n.key });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (boot !== "ok") {
    return (
      <div className="app" style={{ display: "grid", placeItems: "center" }}>
        {boot === "connecting" ? (
          <Loader2 className="spin" size={22} />
        ) : (
          <div className="failed">
            <strong>Could not start the local service</strong>
            <span>{boot}</span>
            <button className="btn" onClick={() => location.reload()}>
              Try again
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          <StatusIndicator />
        </div>
        <div className="toolbar">
          <IconButton
            title="New"
            onClick={() => {
              const c = getState().composer;
              setState({ composer: emptyComposer(c), selected: null });
            }}
          >
            <SquarePen size={18} />
          </IconButton>
          <IconButton title="Gallery" active={overlay === "gallery"} onClick={() => setState({ overlay: overlay === "gallery" ? null : "gallery" })}>
            <Images size={18} />
          </IconButton>
          <IconButton title="Settings" onClick={() => setState({ overlay: "settings" })}>
            <Settings size={18} />
          </IconButton>
        </div>
      </div>
      <div className="main">
        {needsSetup ? (
          <Setup />
        ) : (
          <>
            <Stage />
            <HistoryRail />
          </>
        )}
        {overlay === "gallery" && <Gallery />}
        {overlay === "inpaint" && <InpaintEditor />}
        {overlay === "expand" && <ExpandEditor />}
        {dragging && <div className="drop-hint">Drop images to add them</div>}
      </div>
      {!needsSetup && overlay !== "inpaint" && overlay !== "expand" && overlay !== "gallery" && <Composer />}
      {overlay === "settings" && <SettingsDialog />}
      <Toasts />
    </div>
  );
}

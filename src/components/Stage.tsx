import {
  Brush, ClipboardPaste, Combine, Copy, Download, Ellipsis, Expand, FolderOpen, Heart, Info, Loader2, MousePointerClick, Pencil, RotateCcw, Shuffle, Square, TextCursorInput, Trash2, Wand2, X, ZoomIn,
} from "lucide-react";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { api, urls } from "../lib/api";
import {
  compareBase, composerFromRecord, formatDuration, jobFraction, jobLabel, MAX_UPSCALE_AREA, MAX_UPSCALE_SIDE, MODE_LABEL, pipelineSummary, remainingSeconds, sizeForWidth, upscaleRequest, upscalerLabel,
  upscaleTargets, variationRequest, versionLabels,
} from "../lib/logic";
import { native } from "../lib/native";
import {
  cancelJob, focusPrompt, getState, itemSource, removeSessionItem, selectedItem, selectItem, setState, submitRequest, toastError, toastInfo, useStore, type SessionItem,
} from "../lib/store";
import type { GenerationRecord, JobView, Mode, Size, UpscaleMethod } from "../lib/types";
import { CompareBar, CompareFrame } from "./Compare";
import { IconButton, Popover, Seg, useFit } from "./ui";
import { FullscreenButton, ZoomImage } from "./Viewer";

export const MODE_ICON: Record<Mode, React.ReactNode> = {
  auto: <Wand2 size={11} />,
  generate: <Wand2 size={11} />,
  edit: <Pencil size={11} />,
  multi_reference: <Combine size={11} />,
  inpaint: <Brush size={11} />,
  outpaint: <Expand size={11} />,
  upscale: <ZoomIn size={11} />,
};

const MODE_DOING: Record<Mode, string> = {
  auto: "Working",
  generate: "Creating",
  edit: "Editing",
  multi_reference: "Combining images",
  inpaint: "Inpainting",
  outpaint: "Expanding",
  upscale: "Upscaling",
};

function Empty() {
  return (
    <div className="empty">
      <h1>What should we make?</h1>
      <p>Describe a new image below, or bring one in to edit it.</p>
      <div className="hint">
        <span>
          <MousePointerClick size={14} /> Drop images here
        </span>
        <span>
          <ClipboardPaste size={14} /> Paste with Ctrl+V
        </span>
      </div>
    </div>
  );
}

/** A version in progress: the image it starts from, dimmed, with live progress on top. */
function Pending({ job, item, labels }: { job: JobView; item: SessionItem; labels: Record<string, string> }) {
  const o = job.plan?.output;
  const { ref, box } = useFit(o ? o.width / o.height : 1);
  const session = useStore((s) => s.session);
  const frac = jobFraction(job);
  const remain = remainingSeconds(job);
  const canCancel = !["done", "failed", "cancelled"].includes(job.status);
  const src = item.request?.mode === "generate" ? undefined : item.request?.images?.[0];
  const from = src ? session.find((i) => itemSource(i) === src) : undefined;
  const mode = job.plan?.mode;
  // "v3 · Editing v2", "v1 · Creating"
  const title = [labels[item.key], mode && `${MODE_DOING[mode]}${from && mode !== "multi_reference" ? ` ${labels[from.key]}` : ""}`].filter(Boolean).join(" · ");
  // The source keeps its own shape inside an expansion; elsewhere it fills the frame.
  const outpaint = item.request?.outpaint;
  return (
    <div className="stage-frame" ref={ref}>
      <div className="pending" style={{ width: box.w, height: box.h }}>
        {src && !outpaint && <img className="pending-src" src={urls.source(src)} alt="" draggable={false} />}
        <div className="pending-info">
          <span className="pending-title">{title}</span>
          <div className={`progress ${frac === null ? "indeterminate" : ""}`}>
            <i style={frac !== null ? { width: `${Math.round(frac * 100)}%` } : undefined} />
          </div>
          <span>
            {jobLabel(job)}
            {o ? ` · ${o.width}×${o.height}` : ""}
            {remain !== null ? ` · ${formatDuration(remain)} left` : ""}
          </span>
          {job.notes.length > 0 && <span className="notes">{job.notes.join(" ")}</span>}
          {canCancel && (
            <button className="btn" onClick={() => cancelJob(job.id)} title="Cancel">
              <Square size={13} /> Stop
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function Failed({ job, item }: { job: JobView; item: SessionItem }) {
  const [details, setDetails] = useState(false);
  const req = item.request;
  return (
    <div className="stage-frame">
      <div className="failed">
        <strong>{job.status === "cancelled" ? "Cancelled" : job.error?.message ?? "Generation failed"}</strong>
        {job.error?.suggestion && <span>{job.error.suggestion}</span>}
        <div style={{ display: "flex", gap: 8 }}>
          {req && (
            <button className="btn" onClick={() => (removeSessionItem(item.key), submitRequest(req))}>
              <RotateCcw size={14} /> Try again
            </button>
          )}
          <button className="btn" onClick={() => removeSessionItem(item.key)}>
            <X size={14} /> Dismiss
          </button>
          {job.error?.detail && (
            <button className="btn" onClick={() => setDetails((d) => !d)}>
              <Info size={14} /> Details
            </button>
          )}
        </div>
        {details && <pre className="diag" style={{ maxHeight: 220, textAlign: "left", width: "100%" }}>{job.error?.detail}</pre>}
      </div>
    </div>
  );
}

function InfoPanel({ g }: { g: GenerationRecord }) {
  const pipeline = pipelineSummary(g);
  const rows: [string, string][] = [
    ["Mode", MODE_LABEL[g.mode]],
    ["Size", `${g.width}×${g.height}${g.working_width !== g.width || g.working_height !== g.height ? ` (worked at ${g.working_width}×${g.working_height})` : ""}`],
    ...(pipeline ? ([["Pipeline", pipeline]] as [string, string][]) : []),
    ["Seed", String(g.seed)],
    ["Steps", String(g.steps)],
    ["CFG", String(g.cfg)],
    ["Sampler", `${g.sampler}${g.scheduler ? ` / ${g.scheduler}` : ""}`],
    ...(g.strength != null ? ([["Strength", String(g.strength)]] as [string, string][]) : []),
    ["References", String(g.references.length)],
    ["Model", g.model],
    ["Time", formatDuration(g.duration_ms / 1000)],
    ["Created", new Date(g.created_at).toLocaleString()],
  ];
  return (
    <div style={{ width: 340, padding: 8, display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ fontSize: 13, userSelect: "text", maxHeight: 120, overflow: "auto" }}>{g.prompt}</div>
      {g.negative_prompt && <div className="muted">Negative: {g.negative_prompt}</div>}
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "4px 12px", fontSize: 12 }}>
        {rows.map(([k, v]) => (
          <Fragment key={k}>
            <span className="muted">{k}</span>
            <span style={{ userSelect: "text", wordBreak: "break-all" }}>{v}</span>
          </Fragment>
        ))}
      </div>
      {g.notes.length > 0 && <div className="muted">{g.notes.join(" ")}</div>}
      <div className="muted" style={{ wordBreak: "break-all", userSelect: "text" }}>{g.file_path}</div>
    </div>
  );
}

/** Upscale menu: Quick (upscaler model only) or Detailed (refine at ~2K, then the upscaler model). */
function UpscaleMenu(props: { source: string; size: Size; original?: Size | null; parentId?: string | null; close: () => void }) {
  const { width: w, height: h } = props.size;
  const [method, setMethod] = useState<UpscaleMethod>("quick");
  const up = useStore((s) => s.status?.upscaler);
  const via = up?.available ? upscalerLabel(up.model) : "Lanczos";
  const [customW, setCustomW] = useState(Math.min(MAX_UPSCALE_SIDE, w * 2));
  const targets = upscaleTargets(w, h, props.original);
  const custom = sizeForWidth(w, h, customW);
  const customOk = custom.width > w && custom.width <= MAX_UPSCALE_SIDE && custom.height <= MAX_UPSCALE_SIDE && custom.width * custom.height <= MAX_UPSCALE_AREA;
  const go = (t: Size) => {
    submitRequest(upscaleRequest(props.source, t, method, props.parentId));
    props.close();
  };
  return (
    <div style={{ width: 300, padding: 8, display: "flex", flexDirection: "column", gap: 10 }}>
      <Seg<UpscaleMethod>
        value={method}
        options={[
          { value: "quick", label: "Quick" },
          { value: "detailed", label: "Detailed" },
        ]}
        onChange={setMethod}
      />
      <span className="muted">
        {method === "quick" ? `${via} only: a few seconds, adds no new content.` : `Re-renders fine detail at about 2K (low denoise), then ${via}. About a minute.`}
      </span>
      {targets.map((t) => (
        <button key={t.key} className="btn" onClick={() => go(t)}>
          {t.label} · {t.width}×{t.height}
        </button>
      ))}
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <input
          className="input"
          type="number"
          min={w + 1}
          max={MAX_UPSCALE_SIDE}
          style={{ width: 90 }}
          value={customW}
          onChange={(e) => setCustomW(Math.max(64, Math.min(MAX_UPSCALE_SIDE, Math.round(Number(e.target.value) || 0))))}
          aria-label="Custom width"
        />
        <span className="muted" style={{ flex: 1 }}>
          × {custom.height}
        </span>
        <button className="btn" disabled={!customOk} onClick={() => go(custom)}>
          Upscale
        </button>
      </div>
    </div>
  );
}

function UpscaleButton(props: { source: string; size: Size; original?: Size | null; parentId?: string | null }) {
  return (
    <Popover
      placement="top-end"
      trigger={(open, toggle) => (
        <IconButton title="Upscale" onClick={toggle} active={open}>
          <ZoomIn size={17} />
        </IconButton>
      )}
    >
      {(close) => <UpscaleMenu {...props} close={close} />}
    </Popover>
  );
}

function MoreMenu({ item, path }: { item: SessionItem; path: string | null }) {
  const g = item.gen;
  return (
    <Popover
      placement="top-end"
      trigger={(open, toggle) => (
        <IconButton title="More" onClick={toggle} active={open}>
          <Ellipsis size={17} />
        </IconButton>
      )}
    >
      {(close) => (
        <>
          {g && (
            <button
              className="opt"
              onClick={() => {
                close();
                const c = getState().composer;
                setState({ composer: { ...composerFromRecord(g, c), mode: "auto" } });
                focusPrompt();
              }}
            >
              <span className="check">
                <TextCursorInput size={15} />
              </span>
              <span>Reuse prompt and settings</span>
            </button>
          )}
          {path && (
            <button className="opt" onClick={() => (close(), native.reveal(path).catch(toastError))}>
              <span className="check">
                <FolderOpen size={15} />
              </span>
              <span>Show in Explorer</span>
            </button>
          )}
          <button className="opt" onClick={() => (close(), removeSessionItem(item.key))}>
            <span className="check">
              <Trash2 size={15} />
            </span>
            <span>Remove from project</span>
          </button>
        </>
      )}
    </Popover>
  );
}

function GenActions({ item, g }: { item: SessionItem; g: GenerationRecord }) {
  const original = g.references[0] && g.mode !== "generate" && g.mode !== "upscale" ? { width: g.references[0].width, height: g.references[0].height } : null;
  const [fav, setFav] = useState(g.favorite);
  return (
    <div className="actions">
      <IconButton
        title="Variation (same prompt, new seed)"
        onClick={() => {
          const r = variationRequest(g);
          if (r) submitRequest(r);
        }}
      >
        <Shuffle size={17} />
      </IconButton>
      <UpscaleButton source={`gen:${g.id}`} size={{ width: g.width, height: g.height }} original={original} parentId={g.id} />
      <span className="sep" />
      <IconButton
        title={fav ? "Remove from favorites" : "Favorite"}
        active={fav}
        onClick={async () => {
          try {
            await api.setFavorite(g.id, !fav);
            setFav(!fav);
          } catch (e) {
            toastError(e);
          }
        }}
      >
        <Heart size={17} fill={fav ? "currentColor" : "none"} />
      </IconButton>
      <IconButton
        title="Save as…"
        onClick={async () => {
          const name = g.file_path.split(/[\\/]/).pop() ?? "image.png";
          const path = await native.saveAs(name);
          if (!path) return;
          try {
            const r = await api.exportGeneration(g.id, path);
            toastInfo(`Saved ${r.path.split(/[\\/]/).pop()}`);
          } catch (e) {
            toastError(e);
          }
        }}
      >
        <Download size={17} />
      </IconButton>
      <IconButton
        title="Copy image"
        onClick={async () => {
          try {
            await native.copyImage(g.file_path);
            toastInfo("Image copied");
          } catch (e) {
            toastError(e);
          }
        }}
      >
        <Copy size={17} />
      </IconButton>
      <Popover
        placement="top-end"
        trigger={(open, toggle) => (
          <IconButton title="Details" onClick={toggle} active={open}>
            <Info size={17} />
          </IconButton>
        )}
      >
        {() => <InfoPanel g={g} />}
      </Popover>
      <MoreMenu item={item} path={g.file_path} />
    </div>
  );
}

function InputActions({ item }: { item: SessionItem }) {
  const input = item.input!;
  return (
    <div className="actions">
      <UpscaleButton source={`input:${input.id}`} size={{ width: input.width, height: input.height }} />
      <span className="sep" />
      <MoreMenu item={item} path={input.original_path ?? input.path} />
    </div>
  );
}

/** A result or input fitted to the stage (never enlarged past its own size), with the
 * full-screen button in its top-right corner. */
function PlainFrame({ src, alt, w, h }: { src: string; alt: string; w: number; h: number }) {
  const { ref, box } = useFit(w / h);
  const bw = Math.min(box.w, w);
  return (
    <div className="stage-frame" ref={ref}>
      <div className="image-box" style={{ width: bw, height: bw * (h / w) }}>
        <img className="stage-img fill" src={src} alt={alt} draggable={false} />
        <FullscreenButton hint="Click to zoom to 100 %" content={<ZoomImage src={src} alt={alt} />} />
      </div>
    </div>
  );
}

/** One line under the image: which version this is, what made it, and what you can do with it. */
function StageBar({ item, label }: { item: SessionItem; label: string }) {
  const g = item.gen;
  const base = g ? compareBase(g) : null;
  const w = g?.width ?? item.input!.width;
  const h = g?.height ?? item.input!.height;
  const name = item.input?.original_name ? safeDecode(item.input.original_name) : "Imported image";
  return (
    <div className="stage-bar">
      <div className="stage-meta">
        <span className="ver">{label}</span>
        {g && <span className="mode">{MODE_LABEL[g.mode]}</span>}
        <span className="stage-prompt" title={g?.prompt ?? name}>
          {g ? g.prompt || (g.mode === "upscale" ? `${g.references[0]?.width ?? "?"}×${g.references[0]?.height ?? "?"} → ${w}×${h}` : "") : name}
        </span>
        <span className="dims">
          {w}×{h}
        </span>
      </div>
      {base && <CompareBar />}
      {g ? <GenActions key={`actions-${g.id}`} item={item} g={g} /> : <InputActions item={item} />}
    </div>
  );
}

function safeDecode(s: string) {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

/** The project's versions in order: Original → v1 → v2 … The image the prompt edits carries a pencil. */
function Timeline({ labels }: { labels: Record<string, string> }) {
  const session = useStore((s) => s.session);
  const selected = useStore((s) => s.selected);
  const jobs = useStore((s) => s.jobs);
  const target = useStore((s) => s.composer.refs[0]?.source);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    boxRef.current?.querySelector(".frame.sel")?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [selected, session.length]);

  if (!session.length) return null;
  return (
    <div className="timeline" ref={boxRef} aria-label="Versions">
      {session.map((it, i) => {
        const job = it.jobId ? jobs[it.jobId] : undefined;
        const src = itemSource(it);
        const pending = it.kind === "job";
        const failed = pending && job && ["failed", "cancelled"].includes(job.status);
        const frac = pending && job ? jobFraction(job) : null;
        const mode: Mode | undefined = it.gen?.mode ?? job?.plan?.mode;
        const from = it.gen?.parent_id ? session.find((x) => x.gen?.id === it.gen!.parent_id) : undefined;
        const pendingSrc = pending && it.request?.mode !== "generate" ? it.request?.images?.[0] : undefined;
        const prompt = it.gen?.prompt || it.request?.prompt || "";
        const tip = [labels[it.key], mode ? MODE_LABEL[mode] : it.input ? "Imported" : "", from && session[i - 1] !== from ? `from ${labels[from.key]}` : "", prompt ? `— ${prompt}` : ""].filter(Boolean).join(" ");
        return (
          <Fragment key={it.key}>
            {i > 0 && <span className="link" />}
            <button className={`frame ${selected === it.key ? "sel" : ""}`} onClick={() => selectItem(it.key)} title={tip}>
              <span className="frame-img">
                {src && <img src={urls.sourceThumb(src)} alt="" draggable={false} />}
                {pendingSrc && <img className="dim" src={urls.sourceThumb(pendingSrc)} alt="" draggable={false} />}
                {pending && !failed && (
                  <span className="frame-state">
                    {frac !== null ? <Ring frac={frac} /> : <Loader2 className="spin" size={16} />}
                  </span>
                )}
                {failed && (
                  <span className="frame-state">
                    <X size={16} />
                  </span>
                )}
                {src && src === target && (
                  <span className="frame-badge" title="The prompt edits this image">
                    <Pencil size={10} />
                  </span>
                )}
              </span>
              <span className="frame-label">
                {mode && MODE_ICON[mode]}
                {labels[it.key]}
              </span>
            </button>
          </Fragment>
        );
      })}
    </div>
  );
}

function Ring({ frac }: { frac: number }) {
  const r = 8;
  const c = 2 * Math.PI * r;
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden>
      <circle cx="10" cy="10" r={r} fill="none" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2.5" />
      <circle cx="10" cy="10" r={r} fill="none" stroke="currentColor" strokeWidth="2.5" strokeDasharray={c} strokeDashoffset={c * (1 - frac)} strokeLinecap="round" transform="rotate(-90 10 10)" />
    </svg>
  );
}

export function Stage() {
  const item = useStore(selectedItem);
  const session = useStore((s) => s.session);
  const job = useStore((s) => (item?.jobId ? s.jobs[item.jobId] : undefined));
  const labels = useMemo(() => versionLabels(session), [session]);

  if (!item) {
    return (
      <div className="stage">
        <Empty />
        <Timeline labels={labels} />
      </div>
    );
  }

  let view: React.ReactNode;
  if (item.kind === "job") {
    view = !job ? (
      <div className="stage-frame">
        <Loader2 className="spin" size={22} />
      </div>
    ) : job.status === "failed" || job.status === "cancelled" ? (
      <Failed job={job} item={item} />
    ) : (
      <Pending job={job} item={item} labels={labels} />
    );
  } else {
    const src = item.gen ? urls.gen(item.gen.id) : urls.input(item.input!.id);
    const w = item.gen?.width ?? item.input!.width;
    const h = item.gen?.height ?? item.input!.height;
    const base = item.gen ? compareBase(item.gen) : null;
    view = (
      <>
        {item.gen && base ? (
          <CompareFrame key={`compare-${item.gen.id}`} g={item.gen} base={base} />
        ) : (
          <PlainFrame key={src} src={src} alt={item.gen?.prompt ?? item.input?.original_name ?? ""} w={w} h={h} />
        )}
        <StageBar item={item} label={labels[item.key]} />
      </>
    );
  }

  return (
    <div className="stage">
      {view}
      <Timeline labels={labels} />
    </div>
  );
}

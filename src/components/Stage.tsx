import {
  Brush, Copy, Download, Expand, FolderOpen, Heart, ImagePlus, Info, Loader2, MousePointerClick, Pencil, RotateCcw, Shuffle, Square, X, ClipboardPaste, ZoomIn,
} from "lucide-react";
import { Fragment, useState } from "react";
import { api, urls } from "../lib/api";
import {
  compareBase, formatDuration, jobFraction, jobLabel, MAX_UPSCALE_AREA, MAX_UPSCALE_SIDE, MODE_LABEL, pipelineSummary, remainingSeconds, sizeForWidth, upscaleRequest, upscalerLabel, upscaleTargets,
  variationRequest,
} from "../lib/logic";
import { native } from "../lib/native";
import {
  addAsReference, cancelJob, editItem, openEditorFor, removeSessionItem, selectedItem, setState, submitRequest, toastError, toastInfo, useStore, type SessionItem,
} from "../lib/store";
import type { GenerationRecord, JobView, Size, UpscaleMethod } from "../lib/types";
import { CompareBar, CompareFrame } from "./Compare";
import { IconButton, Popover, Seg, useFit } from "./ui";
import { FullscreenButton, ZoomImage } from "./Viewer";

function Empty() {
  return (
    <div className="empty">
      <h1>What should we make?</h1>
      <p>Describe an image, or bring one to edit.</p>
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

function Pending({ job, aspect }: { job: JobView; aspect: number }) {
  const { ref, box } = useFit(aspect);
  const frac = jobFraction(job);
  const remain = remainingSeconds(job);
  const canCancel = !["done", "failed", "cancelled"].includes(job.status);
  return (
    <div className="stage-frame" ref={ref}>
      <div className="pending" style={{ width: box.w, height: box.h }}>
        <div className="pending-info">
          <span style={{ color: "var(--text)", fontWeight: 500 }}>{jobLabel(job)}</span>
          <div className={`progress ${frac === null ? "indeterminate" : ""}`}>
            <i style={frac !== null ? { width: `${Math.round(frac * 100)}%` } : undefined} />
          </div>
          <span className="muted">
            {job.plan ? `${MODE_LABEL[job.plan.mode]} · ${job.plan.output.width}×${job.plan.output.height}` : ""}
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
      placement="top-start"
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

function GenActions({ item, g }: { item: SessionItem; g: GenerationRecord }) {
  const original = g.references[0] && g.mode !== "generate" && g.mode !== "upscale" ? { width: g.references[0].width, height: g.references[0].height } : null;
  const [fav, setFav] = useState(g.favorite);
  return (
    <div className="actions">
      <IconButton title="Edit" onClick={() => editItem(item)}>
        <Pencil size={17} />
      </IconButton>
      <IconButton
        title="Create variation"
        onClick={() => {
          const r = variationRequest(g);
          if (r) submitRequest(r);
        }}
      >
        <Shuffle size={17} />
      </IconButton>
      <IconButton title="Use as reference" onClick={() => addAsReference(item)}>
        <ImagePlus size={17} />
      </IconButton>
      <IconButton title="Inpaint" onClick={() => openEditorFor(item, "inpaint")}>
        <Brush size={17} />
      </IconButton>
      <IconButton title="Expand" onClick={() => openEditorFor(item, "expand")}>
        <Expand size={17} />
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
      <IconButton title="Show in Explorer" onClick={() => native.reveal(g.file_path).catch(toastError)}>
        <FolderOpen size={17} />
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
    </div>
  );
}

function InputActions({ item }: { item: SessionItem }) {
  return (
    <div className="actions">
      <IconButton title="Edit" onClick={() => editItem(item)}>
        <Pencil size={17} />
      </IconButton>
      <IconButton title="Use as reference" onClick={() => addAsReference(item)}>
        <ImagePlus size={17} />
      </IconButton>
      <IconButton title="Inpaint" onClick={() => openEditorFor(item, "inpaint")}>
        <Brush size={17} />
      </IconButton>
      <IconButton title="Expand" onClick={() => openEditorFor(item, "expand")}>
        <Expand size={17} />
      </IconButton>
      {item.input && <UpscaleButton source={`input:${item.input.id}`} size={{ width: item.input.width, height: item.input.height }} />}
      <span className="sep" />
      <IconButton title="Remove from session" onClick={() => removeSessionItem(item.key)}>
        <X size={17} />
      </IconButton>
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
        <img className="stage-img fill" src={src} alt={alt} draggable={false} onDoubleClick={() => setState({ overlay: null })} />
        <FullscreenButton hint="Click to zoom to 100 %" content={<ZoomImage src={src} alt={alt} />} />
      </div>
    </div>
  );
}

export function Stage() {
  const item = useStore(selectedItem);
  const job = useStore((s) => (item?.jobId ? s.jobs[item.jobId] : undefined));

  if (!item) {
    return (
      <div className="stage">
        <Empty />
      </div>
    );
  }
  if (item.kind === "job" && job) {
    if (job.status === "failed" || job.status === "cancelled") {
      return (
        <div className="stage">
          <Failed job={job} item={item} />
        </div>
      );
    }
    const o = job.plan?.output;
    return (
      <div className="stage">
        <Pending job={job} aspect={o ? o.width / o.height : 1} />
        <div className="stage-caption">
          <span className="prompt">{job.prompt}</span>
        </div>
      </div>
    );
  }
  if (item.kind === "job") {
    return (
      <div className="stage">
        <Loader2 className="spin" size={22} />
      </div>
    );
  }
  const src = item.gen ? urls.gen(item.gen.id) : urls.input(item.input!.id);
  const w = item.gen?.width ?? item.input!.width;
  const h = item.gen?.height ?? item.input!.height;
  const base = item.gen ? compareBase(item.gen) : null;
  return (
    <div className="stage">
      {item.gen && base ? (
        <CompareFrame key={`compare-${item.gen.id}`} g={item.gen} base={base} />
      ) : (
        <PlainFrame key={src} src={src} alt={item.gen?.prompt ?? item.input?.original_name ?? ""} w={w} h={h} />
      )}
      {base && <CompareBar />}
      <div className="stage-caption">
        {item.gen ? <span className="prompt" title={item.gen.prompt}>{item.gen.prompt}</span> : <span>{item.input?.original_name ? decodeURIComponent(item.input.original_name) : "Image"}</span>}
        <span>
          {w}×{h}
        </span>
      </div>
      {item.gen ? <GenActions key={`actions-${item.gen.id}`} item={item} g={item.gen} /> : <InputActions item={item} />}
    </div>
  );
}

export function HistoryRail() {
  const session = useStore((s) => s.session);
  const selected = useStore((s) => s.selected);
  const jobs = useStore((s) => s.jobs);
  if (!session.length) return null;
  return (
    <div className="rail" aria-label="Session history">
      {session.map((it) => {
        const job = it.jobId ? jobs[it.jobId] : undefined;
        const pending = it.kind === "job";
        return (
          <button key={it.key} className={`thumb ${selected === it.key ? "sel" : ""}`} onClick={() => setState({ selected: it.key })} title={it.gen?.prompt ?? (job ? jobLabel(job) : "Image")}>
            {it.gen && <img src={urls.genThumb(it.gen.id)} alt="" draggable={false} />}
            {it.input && <img src={urls.inputThumb(it.input.id)} alt="" draggable={false} />}
            {pending && job && !["failed", "cancelled"].includes(job.status) && <Loader2 className="spin" size={18} />}
            {pending && job && ["failed", "cancelled"].includes(job.status) && <X size={18} />}
            {it.input && <span className="tag">in</span>}
          </button>
        );
      })}
    </div>
  );
}

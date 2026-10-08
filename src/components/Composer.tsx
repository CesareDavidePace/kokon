import { ArrowUp, Brush, ChevronDown, Expand, FolderOpen, Gauge, ImagePlus, RectangleHorizontal, RectangleVertical, SlidersHorizontal, Sparkles, Square, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, urls } from "../lib/api";
import { activeOverrides, buildRequest, expansionPixels, formatDuration, hasOutpaint, moveRef, needsPrompt, versionLabels, type Ref } from "../lib/logic";
import { native } from "../lib/native";
import {
  addAsReference, addInputs, getState, itemSource, selectedItem, selectItem, setState, submitComposer, toastError, updateComposer, useStore, type SessionItem,
} from "../lib/store";
import type { InpaintAction, Plan, Quality } from "../lib/types";
import { AdvancedPanel } from "./Advanced";
import { Option, Popover, Seg } from "./ui";

const QUALITY: { value: Quality; label: string; sub: string }[] = [
  { value: "fast", label: "Fast", sub: "~1 MP, 20 steps" },
  { value: "balanced", label: "Balanced", sub: "~1 MP, 40 steps · recommended" },
  { value: "quality", label: "Quality", sub: "native 2K, 40 steps · slower" },
];

const INPAINT_ACTIONS: { value: InpaintAction; label: string; placeholder: string }[] = [
  { value: "edit", label: "Change", placeholder: "Describe the change in the painted area…" },
  { value: "remove", label: "Remove", placeholder: "What to remove (optional)…" },
  { value: "replace", label: "Replace", placeholder: "What to put there instead…" },
  { value: "add", label: "Add", placeholder: "What to add in the painted area…" },
];

const INPAINT_VERB: Record<InpaintAction, string> = { edit: "Changing", remove: "Clearing", replace: "Replacing", add: "Adding to" };

const ASPECT_ICON: Record<string, React.ReactNode> = {
  "1:1": <Square size={15} />,
  "4:3": <RectangleHorizontal size={15} />,
  "16:9": <RectangleHorizontal size={15} />,
  "3:4": <RectangleVertical size={15} />,
  "9:16": <RectangleVertical size={15} />,
};

export async function importPaths(paths: string[]) {
  const recs = [];
  for (const p of paths) {
    try {
      recs.push(await api.importPath(p));
    } catch (e) {
      toastError(e);
    }
  }
  addInputs(recs);
}

/** Version name of a reference when it is part of the project ("v2", "Original"). */
function useRefNames() {
  const session = useStore((s) => s.session);
  return useMemo(() => {
    const labels = versionLabels(session);
    const bySource = new Map<string, string>();
    for (const i of session) {
      const s = itemSource(i);
      if (s) bySource.set(s, labels[i.key]);
    }
    return (r: Ref) => bySource.get(r.source) ?? null;
  }, [session]);
}

/** Thumbnails of the images the prompt works on; drag to reorder when there are several. */
function RefStrip() {
  const refs = useStore((s) => s.composer.refs);
  const nameOf = useRefNames();
  const [drag, setDrag] = useState<{ from: number; to: number; active: boolean } | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const start = useRef<{ x: number; idx: number } | null>(null);

  const indexAt = (clientX: number) => {
    const els = Array.from(boxRef.current?.querySelectorAll<HTMLElement>(".ref") ?? []);
    for (let i = 0; i < els.length; i++) {
      const r = els[i].getBoundingClientRect();
      if (clientX < r.left + r.width / 2) return i;
    }
    return els.length;
  };

  return (
    <div
      className="refs"
      ref={boxRef}
      onPointerMove={(e) => {
        if (!start.current) return;
        if (!drag?.active && Math.abs(e.clientX - start.current.x) < 5) return;
        setDrag({ from: start.current.idx, to: indexAt(e.clientX), active: true });
      }}
      onPointerUp={() => {
        if (drag?.active) {
          const to = drag.to > drag.from ? drag.to - 1 : drag.to;
          updateComposer({ refs: moveRef(getState().composer.refs, drag.from, to), mask: null, outpaint: null });
        }
        start.current = null;
        setDrag(null);
      }}
      onPointerLeave={() => {
        start.current = null;
        setDrag(null);
      }}
    >
      {refs.map((r, i) => (
        <div
          key={r.key}
          className={`ref ${drag?.active && drag.from === i ? "dragging" : ""} ${drag?.active && drag.to === i && drag.from !== i ? "drop-before" : ""}`}
          title={`${refs.length > 1 ? `Image ${i + 1}: ` : ""}${nameOf(r) ?? r.name} · ${r.width}×${r.height}${refs.length > 1 ? " — drag to reorder" : ""}`}
          onPointerDown={(e) => {
            if ((e.target as HTMLElement).closest(".rm")) return;
            (e.currentTarget.parentElement as HTMLElement).setPointerCapture?.(e.pointerId);
            start.current = { x: e.clientX, idx: i };
          }}
        >
          <img src={urls.sourceThumb(r.source)} alt="" draggable={false} />
          {refs.length > 1 && <span className="num">{i + 1}</span>}
          <button
            className="rm"
            title="Remove"
            aria-label={`Remove image ${i + 1}`}
            onClick={() => {
              const c = getState().composer;
              updateComposer({ refs: c.refs.filter((x) => x.key !== r.key), mask: i === 0 ? null : c.mask, outpaint: i === 0 ? null : c.outpaint });
            }}
          >
            <X size={11} />
          </button>
        </div>
      ))}
    </div>
  );
}

/** Says, in words, what the prompt is going to do, and offers the tools for that image. */
function TargetBar() {
  const c = useStore((s) => s.composer);
  const sel = useStore(selectedItem);
  const session = useStore((s) => s.session);
  const nameOf = useRefNames();

  if (!c.refs.length) {
    const img = sel && (sel.gen || sel.input) ? sel : [...session].reverse().find((i) => i.gen || i.input);
    if (!img) return null;
    return (
      <div className="target create">
        <span className="target-icon">
          <Sparkles size={15} />
        </span>
        <span className="target-text">
          <span className="what">New image</span>
          <span className="sub">Made from the prompt alone</span>
        </span>
        <span className="spacer" />
        <button className="tool" onClick={() => selectItem(img.key)}>
          Edit {versionLabels(session)[img.key]} instead
        </button>
      </div>
    );
  }

  const first = c.refs[0];
  const name = nameOf(first) ?? (first.source.startsWith("gen:") ? "this result" : first.name);
  const outPx = hasOutpaint(c.outpaint) ? expansionPixels(c.outpaint!, first.width, first.height) : null;
  const what =
    c.refs.length > 1 ? `Combining ${c.refs.length} images` : c.mask ? `${INPAINT_VERB[c.inpaintAction]} the painted area of ${name}` : outPx ? `Expanding ${name}` : `Editing ${name}`;
  const sub =
    c.refs.length > 1
      ? "Refer to them as image 1, image 2… in the prompt"
      : outPx
        ? `${first.width}×${first.height} → ${first.width + outPx.left + outPx.right}×${first.height + outPx.top + outPx.bottom}`
        : `${first.width}×${first.height}`;

  return (
    <div className="target">
      <RefStrip />
      <span className="target-text">
        <span className="what">{what}</span>
        <span className="sub">{sub}</span>
      </span>
      <span className="spacer" />
      <span className="tool-group">
        <button className={`tool ${c.mask ? "on" : ""}`} onClick={() => setState({ overlay: "inpaint" })} title="Paint the area to change; the rest stays as it is">
          <Brush size={14} /> {c.mask ? "Area painted" : "Paint area"}
        </button>
        {c.mask && (
          <button className="tool-x" title="Remove the painted area" onClick={() => updateComposer({ mask: null })}>
            <X size={13} />
          </button>
        )}
      </span>
      <span className="tool-group">
        <button className={`tool ${outPx ? "on" : ""}`} onClick={() => setState({ overlay: "expand" })} title="Extend the image beyond its borders">
          <Expand size={14} /> {outPx ? "Expanded" : "Expand"}
        </button>
        {outPx && (
          <button className="tool-x" title="Remove the expansion" onClick={() => updateComposer({ outpaint: null })}>
            <X size={13} />
          </button>
        )}
      </span>
      <button className="tool" title="Create a new image from the prompt instead" onClick={() => updateComposer({ refs: [], mask: null, outpaint: null })}>
        New image
      </button>
    </div>
  );
}

/** Adds a reference: a version from this project, or files. */
function AddImage() {
  const session = useStore((s) => s.session);
  const refs = useStore((s) => s.composer.refs);
  const labels = useMemo(() => versionLabels(session), [session]);
  const used = new Set(refs.map((r) => r.source));
  const candidates = session.filter((i) => {
    const s = itemSource(i);
    return s && !used.has(s);
  });
  const browse = async () => importPaths(await native.pickImages());
  const title = refs.length ? "Add a reference image" : "Add an image to edit";

  if (!candidates.length) {
    return (
      <button className="icon-btn" title={`${title} (or drop / paste)`} aria-label={title} onClick={browse}>
        <ImagePlus size={19} />
      </button>
    );
  }
  return (
    <Popover
      placement="top-start"
      trigger={(open, toggle) => (
        <button className={`icon-btn ${open ? "active" : ""}`} title={title} aria-label={title} onClick={toggle}>
          <ImagePlus size={19} />
        </button>
      )}
    >
      {(close) => (
        <div className="add-menu">
          <div className="label">{refs.length ? "Add a version as reference" : "Edit a version"}</div>
          <div className="add-grid">
            {candidates.map((it: SessionItem) => (
              <button key={it.key} title={it.gen?.prompt ?? labels[it.key]} onClick={() => (addAsReference(it), close())}>
                <img src={urls.sourceThumb(itemSource(it)!)} alt="" draggable={false} />
                <span>{labels[it.key]}</span>
              </button>
            ))}
          </div>
          <button className="opt" onClick={() => (close(), browse())}>
            <span className="check">
              <FolderOpen size={15} />
            </span>
            <span>Browse files…</span>
          </button>
          <div className="muted" style={{ padding: "2px 10px 6px" }}>
            You can also drop or paste images.
          </div>
        </div>
      )}
    </Popover>
  );
}

function usePlanPreview(): { plan: Plan | null; error: string | null } {
  const c = useStore((s) => s.composer);
  const connected = useStore((s) => s.connected);
  const [res, setRes] = useState<{ plan: Plan | null; error: string | null }>({ plan: null, error: null });
  useEffect(() => {
    if (!connected) return;
    const prompt = c.prompt.trim() || "image";
    const req = { ...buildRequest(c), prompt };
    let cancelled = false;
    const t = window.setTimeout(() => {
      api.plan(req).then(
        (p) => !cancelled && setRes({ plan: p, error: null }),
        (e) => !cancelled && setRes({ plan: null, error: e?.message ?? null }),
      );
    }, 350);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [connected, c.prompt, c.refs, c.mode, c.quality, c.resolution, c.aspect, c.outpaint, c.mask, c.inpaintAction, c.customWidth, c.customHeight, c.advanced]);
  return res;
}

export function Composer() {
  const c = useStore((s) => s.composer);
  const hasProject = useStore((s) => s.session.length > 0);
  const { plan, error } = usePlanPreview();
  const taRef = useRef<HTMLTextAreaElement>(null);
  const hasImages = c.refs.length > 0;
  const overrides = activeOverrides(c.advanced);

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(200, ta.scrollHeight) + "px";
  }, [c.prompt]);

  const outPx = hasOutpaint(c.outpaint) && c.refs[0] ? expansionPixels(c.outpaint!, c.refs[0].width, c.refs[0].height) : null;

  const placeholder = !hasImages
    ? hasProject
      ? "Describe a new image…"
      : "Describe an image to create, or drop one in to edit it…"
    : c.refs.length > 1
      ? "Describe how to combine image 1, image 2…"
      : c.mask
        ? INPAINT_ACTIONS.find((a) => a.value === c.inpaintAction)!.placeholder
        : outPx
          ? "Describe the extended scene (optional details)…"
          : "Describe the change…";

  const hint = (() => {
    if (error) return error;
    if (!plan) return "";
    return `${plan.output.width}×${plan.output.height} · ~${formatDuration(plan.estimated_seconds)}`;
  })();
  const hintTitle = plan
    ? [
        plan.working.width !== plan.output.width || plan.working.height !== plan.output.height ? `Works at ${plan.working.width}×${plan.working.height}` : "",
        plan.optimized_for_vram ? "Optimized for VRAM" : "",
        ...plan.notes,
      ]
        .filter(Boolean)
        .join("\n")
    : "";
  const shaped = c.resolution !== "custom" && !c.mask && !outPx;

  return (
    <div className="composer-wrap">
      <div className="composer">
        <TargetBar />
        <textarea
          id="prompt"
          ref={taRef}
          className="prompt"
          rows={1}
          value={c.prompt}
          placeholder={placeholder}
          onChange={(e) => updateComposer({ prompt: e.target.value })}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submitComposer();
            } else if (e.key === "ArrowUp" && !c.prompt) {
              const last = getState().lastPrompt;
              if (last) {
                e.preventDefault();
                updateComposer({ prompt: last });
              }
            }
          }}
          spellCheck
        />
        <div className="controls">
          <AddImage />

          {c.mask && (
            <Seg value={c.inpaintAction} options={INPAINT_ACTIONS} onChange={(v) => updateComposer({ inpaintAction: v })} />
          )}

          {shaped && (
            <Popover
              trigger={(open, toggle) => (
                <button className={`pill ${open ? "on" : ""}`} onClick={toggle} title="Aspect ratio">
                  {ASPECT_ICON[c.aspect] ?? <RectangleHorizontal size={15} />}
                  {c.aspect === "auto" ? (plan ? plan.aspect_label : "Auto") : c.aspect}
                  <ChevronDown size={13} />
                </button>
              )}
            >
              {(close) => (
                <>
                  {["auto", "1:1", "4:3", "3:4", "16:9", "9:16"].map((a) => (
                    <Option key={a} selected={c.aspect === a} label={a === "auto" ? "Auto" : a} sub={a === "auto" ? (hasImages ? "Same as image 1" : "From the prompt") : undefined} icon={ASPECT_ICON[a]} onClick={() => (updateComposer({ aspect: a }), close())} />
                  ))}
                  <Option label="Custom size…" icon={<SlidersHorizontal size={15} />} onClick={() => (updateComposer({ resolution: "custom" }), close())} />
                </>
              )}
            </Popover>
          )}

          <Popover
            trigger={(open, toggle) => (
              <button className={`pill ${open ? "on" : ""}`} onClick={toggle} title="Quality">
                <Gauge size={15} />
                {QUALITY.find((q) => q.value === c.quality)?.label}
                <ChevronDown size={13} />
              </button>
            )}
          >
            {(close) =>
              QUALITY.map((q) => <Option key={q.value} selected={c.quality === q.value} label={q.label} sub={q.sub} onClick={() => (updateComposer({ quality: q.value }), close())} />)
            }
          </Popover>

          <Popover
            placement="top-start"
            className="panel"
            trigger={(open, toggle) => (
              <button
                className={`icon-btn has-badge ${open ? "active" : ""}`}
                onClick={toggle}
                title={overrides.length ? `Advanced: overriding the preset (${overrides.join(", ")})` : "Advanced"}
                aria-label={overrides.length ? `Advanced settings, ${overrides.length} overrides active` : "Advanced settings"}
              >
                <SlidersHorizontal size={17} />
                {overrides.length > 0 && <span className="override-dot" />}
              </button>
            )}
          >
            {() => <AdvancedPanel plan={plan} />}
          </Popover>

          <span className="spacer" />
          <span className="plan-hint" title={hintTitle}>
            {hint}
          </span>
          <button className="send" title={hasImages ? "Apply (Enter)" : "Create (Enter)"} aria-label="Generate" disabled={!c.prompt.trim() && needsPrompt(c)} onClick={() => submitComposer()}>
            <ArrowUp size={18} strokeWidth={2.4} />
          </button>
        </div>
      </div>
    </div>
  );
}

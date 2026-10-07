import {
  ArrowUp, Brush, ChevronDown, Combine, Expand, Gauge, ImagePlus, Pencil, RectangleHorizontal, RectangleVertical, SlidersHorizontal, Sparkles, Square, Wand2, X, ZoomIn,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, urls } from "../lib/api";
import { activeOverrides, buildRequest, expansionPixels, formatDuration, hasOutpaint, MODE_LABEL, moveRef, type Ref } from "../lib/logic";
import { native } from "../lib/native";
import { addInputs, getState, setState, submitComposer, toastError, updateComposer, useStore } from "../lib/store";
import type { Mode, Plan, Quality } from "../lib/types";
import { AdvancedPanel } from "./Advanced";
import { Option, Popover } from "./ui";

const thumbUrl = (r: Ref) => (r.source.startsWith("gen:") ? urls.genThumb(r.source.slice(4)) : urls.inputThumb(r.source.slice(6)));

const MODE_ICON: Record<Mode, React.ReactNode> = {
  auto: <Sparkles size={15} />,
  generate: <Wand2 size={15} />,
  edit: <Pencil size={15} />,
  multi_reference: <Combine size={15} />,
  inpaint: <Brush size={15} />,
  outpaint: <Expand size={15} />,
  upscale: <ZoomIn size={15} />,
};

const QUALITY: { value: Quality; label: string; sub: string }[] = [
  { value: "fast", label: "Fast", sub: "~1 MP, 20 steps" },
  { value: "balanced", label: "Balanced", sub: "~1 MP, 40 steps · recommended" },
  { value: "quality", label: "Quality", sub: "native 2K, 40 steps · slower" },
];

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

function RefStrip() {
  const refs = useStore((s) => s.composer.refs);
  const mask = useStore((s) => s.composer.mask);
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

  if (!refs.length) return null;
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
          updateComposer({ refs: moveRef(getState().composer.refs, drag.from, to) });
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
          title={`Image ${i + 1} · ${r.width}×${r.height}${refs.length > 1 ? " — drag to reorder" : ""}`}
          onPointerDown={(e) => {
            if ((e.target as HTMLElement).closest(".rm")) return;
            (e.currentTarget.parentElement as HTMLElement).setPointerCapture?.(e.pointerId);
            start.current = { x: e.clientX, idx: i };
          }}
        >
          <img src={thumbUrl(r)} alt="" draggable={false} />
          {refs.length > 1 && <span className="num">{i + 1}</span>}
          {mask && mask.refKey === r.key && (
            <span className="badge" title="Mask painted">
              <Brush size={11} />
            </span>
          )}
          <button
            className="rm"
            title="Remove"
            aria-label={`Remove image ${i + 1}`}
            onClick={() => {
              const c = getState().composer;
              updateComposer({ refs: c.refs.filter((x) => x.key !== r.key), mask: c.mask?.refKey === r.key ? null : c.mask, outpaint: i === 0 ? null : c.outpaint });
            }}
          >
            <X size={12} />
          </button>
        </div>
      ))}
    </div>
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
  }, [connected, c.prompt, c.refs, c.mode, c.quality, c.resolution, c.aspect, c.outpaint, c.mask, c.customWidth, c.customHeight, c.advanced]);
  return res;
}

export function Composer() {
  const c = useStore((s) => s.composer);
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

  const detected = plan ? (plan.mode as Mode) : null;
  const modeShown: Mode = c.mode === "auto" ? detected ?? "auto" : c.mode;
  const outPx = hasOutpaint(c.outpaint) && c.refs[0] ? expansionPixels(c.outpaint!, c.refs[0].width, c.refs[0].height) : null;

  const placeholder = !hasImages
    ? "Describe an image…"
    : c.refs.length > 1
      ? "Describe how to combine image 1, image 2…"
      : c.mask
        ? "Describe what goes in the painted area…"
        : outPx
          ? "Describe the extended scene (optional details)…"
          : "Describe the change…";

  const openEditor = (overlay: "inpaint" | "expand") => {
    const first = getState().composer.refs[0];
    if (!first) return;
    setState({ overlay });
  };

  const hint = (() => {
    if (error) return error;
    if (!plan) return "";
    const parts = [`${plan.output.width}×${plan.output.height}`];
    if (plan.working.width !== plan.output.width || plan.working.height !== plan.output.height) parts.push(`works at ${plan.working.width}×${plan.working.height}`);
    parts.push(`~${formatDuration(plan.estimated_seconds)}`);
    if (plan.optimized_for_vram) parts.push("optimized for VRAM");
    return parts.join(" · ");
  })();

  return (
    <div className="composer-wrap">
      <div className="composer">
        <RefStrip />
        {(outPx || c.mask) && (
          <div style={{ display: "flex", gap: 6, paddingBottom: 6 }}>
            {outPx && (
              <span className="chip">
                <Expand size={13} /> Expand {outPx.left ? `←${outPx.left} ` : ""}
                {outPx.right ? `→${outPx.right} ` : ""}
                {outPx.top ? `↑${outPx.top} ` : ""}
                {outPx.bottom ? `↓${outPx.bottom}` : ""}
                <button onClick={() => openEditor("expand")} title="Adjust">
                  <Pencil size={12} />
                </button>
                <button onClick={() => updateComposer({ outpaint: null })} title="Remove expansion">
                  <X size={13} />
                </button>
              </span>
            )}
            {c.mask && (
              <span className="chip">
                <Brush size={13} /> Mask on image 1
                <button onClick={() => openEditor("inpaint")} title="Edit mask">
                  <Pencil size={12} />
                </button>
                <button onClick={() => updateComposer({ mask: null })} title="Remove mask">
                  <X size={13} />
                </button>
              </span>
            )}
          </div>
        )}
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
            }
          }}
          spellCheck
        />
        <div className="controls">
          <button
            className="icon-btn"
            title="Add images (or drop / paste)"
            aria-label="Add images"
            onClick={async () => importPaths(await native.pickImages())}
          >
            <ImagePlus size={19} />
          </button>

          <Popover
            trigger={(open, toggle) => (
              <button className={`pill ${open ? "on" : ""}`} onClick={toggle} title={plan?.mode_reason ? `Auto: ${plan.mode_reason}` : "Mode"}>
                {MODE_ICON[modeShown]}
                {c.mode === "auto" ? (detected ? MODE_LABEL[detected] : "Auto") : MODE_LABEL[c.mode]}
                <ChevronDown size={13} />
              </button>
            )}
          >
            {(close) => (
              <>
                <Option selected={c.mode === "auto"} label="Auto" sub={detected ? MODE_LABEL[detected] : undefined} icon={MODE_ICON.auto} onClick={() => (updateComposer({ mode: "auto" }), close())} />
                <Option selected={c.mode === "generate"} label="Create" sub="New image from text" icon={MODE_ICON.generate} onClick={() => (updateComposer({ mode: "generate" }), close())} />
                {hasImages && <Option selected={c.mode === "edit"} label="Edit" sub="Change the image" icon={MODE_ICON.edit} onClick={() => (updateComposer({ mode: "edit" }), close())} />}
                {c.refs.length > 1 && <Option selected={c.mode === "multi_reference"} label="Combine" sub="Use several images" icon={MODE_ICON.multi_reference} onClick={() => (updateComposer({ mode: "multi_reference" }), close())} />}
                {hasImages && (
                  <Option
                    selected={c.mode === "inpaint"}
                    label="Inpaint"
                    sub="Paint an area"
                    icon={MODE_ICON.inpaint}
                    onClick={() => {
                      close();
                      updateComposer({ mode: "auto" });
                      openEditor("inpaint");
                    }}
                  />
                )}
                {hasImages && (
                  <Option
                    selected={c.mode === "outpaint"}
                    label="Expand"
                    sub="Extend the borders"
                    icon={MODE_ICON.outpaint}
                    onClick={() => {
                      close();
                      updateComposer({ mode: "auto" });
                      openEditor("expand");
                    }}
                  />
                )}
              </>
            )}
          </Popover>

          {c.resolution !== "custom" && !c.mask && !outPx && (!hasImages || modeShown === "generate" || modeShown === "edit" || modeShown === "multi_reference" || modeShown === "auto") && (
            <Popover
              trigger={(open, toggle) => (
                <button className={`pill ${open ? "on" : ""}`} onClick={toggle} title="Aspect ratio">
                  {ASPECT_ICON[c.aspect] ?? <RectangleHorizontal size={15} />}
                  {c.aspect === "auto" ? (plan && (modeShown === "generate" || hasImages) ? plan.aspect_label : "Auto") : c.aspect}
                  <ChevronDown size={13} />
                </button>
              )}
            >
              {(close) => (
                <>
                  {["auto", "1:1", "4:3", "3:4", "16:9", "9:16"].map((a) => (
                    <Option key={a} selected={c.aspect === a} label={a === "auto" ? "Auto" : a} sub={a === "auto" ? (hasImages && c.mode !== "generate" ? "Same as image 1" : "From the prompt") : undefined} icon={ASPECT_ICON[a]} onClick={() => (updateComposer({ aspect: a }), close())} />
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
          <span className="plan-hint" title={plan?.notes.join("\n") || plan?.mode_reason || ""}>
            {hint}
          </span>
          <button className="send" title="Generate (Enter)" aria-label="Generate" disabled={!c.prompt.trim()} onClick={() => submitComposer()}>
            <ArrowUp size={18} strokeWidth={2.4} />
          </button>
        </div>
      </div>
    </div>
  );
}

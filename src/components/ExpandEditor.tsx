import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Check, RectangleHorizontal, RectangleVertical, RotateCcw, Square, X } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { urls } from "../lib/api";
import { expansionPixels, expansionPreset } from "../lib/logic";
import { focusPrompt, setState, updateComposer, useStore } from "../lib/store";
import type { OutpaintSpec, Units } from "../lib/types";
import { IconButton, Seg } from "./ui";

type Side = "left" | "right" | "top" | "bottom";
const MAX_FINAL = 16384;

export function ExpandEditor() {
  const ref0 = useStore((s) => s.composer.refs[0]);
  const initial = useStore((s) => s.composer.outpaint);
  const [spec, setSpec] = useState<OutpaintSpec>(initial ?? { left: 0, right: 0, top: 0, bottom: 0, units: "pixels" });
  const stageRef = useRef<HTMLDivElement>(null);
  const [stage, setStage] = useState({ w: 800, h: 600 });
  const [frozenScale, setFrozenScale] = useState<number | null>(null);
  const drag = useRef<{ side: Side; start: number; startVal: number } | null>(null);

  useLayoutEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setStage({ w: el.clientWidth - 64, h: el.clientHeight - 48 }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!ref0) setState({ overlay: null });
  }, [ref0]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setState({ overlay: null });
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  if (!ref0) return null;

  const W = ref0.width;
  const H = ref0.height;
  const px = expansionPixels(spec, W, H);
  const finalW = W + px.left + px.right;
  const finalH = H + px.top + px.bottom;
  const fitScale = Math.min(stage.w / finalW, stage.h / finalH, 1.5);
  const scale = frozenScale ?? fitScale;
  const src = ref0.source.startsWith("gen:") ? urls.gen(ref0.source.slice(4)) : urls.input(ref0.source.slice(6));

  const toUnits = (pixels: number, side: Side) => {
    if (spec.units === "pixels") return Math.max(0, Math.round(pixels));
    const base = side === "left" || side === "right" ? W : H;
    return Math.max(0, Math.round((pixels / base) * 1000) / 10);
  };
  const setSide = (side: Side, v: number) => setSpec((s) => ({ ...s, [side]: Math.max(0, Number.isFinite(v) ? v : 0) }));

  const onHandleDown = (side: Side) => (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setFrozenScale(scale);
    drag.current = { side, start: side === "left" || side === "right" ? e.clientX : e.clientY, startVal: px[side] };
  };
  const onHandleMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const pos = d.side === "left" || d.side === "right" ? e.clientX : e.clientY;
    const delta = (pos - d.start) / (frozenScale ?? scale);
    const outward = d.side === "left" || d.side === "top" ? -delta : delta;
    // Snap to 8 px for tidy numbers.
    const pixels = Math.max(0, Math.round((d.startVal + outward) / 8) * 8);
    setSide(d.side, toUnits(pixels, d.side));
  };
  const onHandleUp = () => {
    drag.current = null;
    setFrozenScale(null);
  };

  const preset = (k: Parameters<typeof expansionPreset>[0]) => {
    const cur = { ...px, units: "pixels" as Units };
    const next = expansionPreset(k, W, H, cur);
    if (spec.units === "percent") {
      setSpec({ left: toPct(next.left, W), right: toPct(next.right, W), top: toPct(next.top, H), bottom: toPct(next.bottom, H), units: "percent" });
    } else setSpec(next);
  };
  const toPct = (p: number, b: number) => Math.round((p / b) * 1000) / 10;

  const switchUnits = (u: Units) => {
    if (u === spec.units) return;
    if (u === "percent") setSpec({ left: toPct(px.left, W), right: toPct(px.right, W), top: toPct(px.top, H), bottom: toPct(px.bottom, H), units: "percent" });
    else setSpec({ ...px, units: "pixels" });
  };

  const tooBig = finalW > MAX_FINAL || finalH > MAX_FINAL;
  const any = px.left + px.right + px.top + px.bottom > 0;
  const cw = finalW * scale;
  const ch = finalH * scale;

  return (
    <div className="overlay">
      <div className="overlay-head">
        <h2>Expand the image</h2>
        <span className="muted">
          {W}×{H} → {finalW}×{finalH}
        </span>
        <span className="spacer" />
        <IconButton title="Cancel (Esc)" onClick={() => setState({ overlay: null })}>
          <X size={18} />
        </IconButton>
      </div>
      <div className="editor-stage" ref={stageRef} onPointerMove={onHandleMove} onPointerUp={onHandleUp}>
        <div className="expand-area" style={{ width: cw, height: ch }}>
          <div className="expand-canvas" style={{ left: 0, top: 0, width: cw, height: ch }} />
          <div className="expand-src" style={{ left: px.left * scale, top: px.top * scale, width: W * scale, height: H * scale }}>
            <img src={src} alt="" draggable={false} />
          </div>
          <div className="handle h" style={{ left: 0, top: ch / 2 }} onPointerDown={onHandleDown("left")} title="Drag to extend left" />
          <div className="handle h" style={{ left: cw, top: ch / 2 }} onPointerDown={onHandleDown("right")} title="Drag to extend right" />
          <div className="handle v" style={{ left: cw / 2, top: 0 }} onPointerDown={onHandleDown("top")} title="Drag to extend up" />
          <div className="handle v" style={{ left: cw / 2, top: ch }} onPointerDown={onHandleDown("bottom")} title="Drag to extend down" />
        </div>
      </div>
      <div className="editor-tools" style={{ flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "center" }}>
          <div className="group">
            <IconButton title="Extend left" onClick={() => preset("left")}>
              <ArrowLeft size={17} />
            </IconButton>
            <IconButton title="Extend right" onClick={() => preset("right")}>
              <ArrowRight size={17} />
            </IconButton>
            <IconButton title="Extend up" onClick={() => preset("top")}>
              <ArrowUp size={17} />
            </IconButton>
            <IconButton title="Extend down" onClick={() => preset("bottom")}>
              <ArrowDown size={17} />
            </IconButton>
          </div>
          <div className="group">
            <IconButton title="Landscape 16:9" onClick={() => preset("landscape")}>
              <RectangleHorizontal size={17} />
            </IconButton>
            <IconButton title="Portrait 9:16" onClick={() => preset("portrait")}>
              <RectangleVertical size={17} />
            </IconButton>
            <IconButton title="Square" onClick={() => preset("square")}>
              <Square size={17} />
            </IconButton>
            <IconButton title="Reset" onClick={() => setSpec({ left: 0, right: 0, top: 0, bottom: 0, units: spec.units })}>
              <RotateCcw size={17} />
            </IconButton>
          </div>
        </div>
        <div className="expand-fields">
          {(["left", "right", "top", "bottom"] as Side[]).map((s) => (
            <label key={s}>
              {s[0].toUpperCase() + s.slice(1)}
              <input className="num-input" type="number" min={0} step={spec.units === "pixels" ? 8 : 1} value={spec[s]} onChange={(e) => setSide(s, Number(e.target.value))} />
            </label>
          ))}
          <div style={{ width: 120 }}>
            <Seg<Units>
              value={spec.units}
              options={[
                { value: "pixels", label: "px" },
                { value: "percent", label: "%" },
              ]}
              onChange={switchUnits}
            />
          </div>
          <button
            className="btn primary"
            disabled={!any || tooBig}
            title={tooBig ? `Max ${MAX_FINAL}px per side` : undefined}
            onClick={() => {
              updateComposer({ outpaint: spec, mask: null, mode: "auto" });
              setState({ overlay: null });
              focusPrompt();
            }}
          >
            <Check size={15} /> Done
          </button>
        </div>
      </div>
    </div>
  );
}

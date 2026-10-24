// Before / after views for results made from another image (edit, inpaint, expand, upscale).

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { urls } from "../lib/api";
import { diffPixels, type CompareBase, type CompareMode } from "../lib/logic";
import { inBrowser } from "../lib/platform";
import { setState, stageImage, useStore } from "../lib/store";
import type { GenerationRecord } from "../lib/types";
import { Seg, useFit } from "./ui";
import { FullscreenButton } from "./Viewer";
import { ZoomPan } from "./ZoomPan";

const MODES: { value: CompareMode; label: string }[] = [
  { value: "off", label: "Result" },
  { value: "slider", label: "Slider" },
  { value: "flicker", label: "Flicker" },
  { value: "diff", label: "Changes" },
];
const SPEEDS: { value: string; label: string }[] = [
  { value: "250", label: "Fast" },
  { value: "500", label: "Normal" },
  { value: "1000", label: "Slow" },
];
/** Long side of the difference map; downscaling also averages out the VAE re-encode noise. */
const DIFF_SIDE = 1024;

const isTyping = (t: EventTarget | null) => t instanceof HTMLElement && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));

/** The result with the selected comparison view, sized to fit the stage (or the screen). */
export function CompareFrame({ g, base, fullscreen = false, maskSrc, onFullscreen }: { g: GenerationRecord; base: CompareBase; fullscreen?: boolean; maskSrc?: string; onFullscreen?: () => void }) {
  const mode = useStore((s) => s.compare);
  const flickerMs = useStore((s) => s.flickerMs);
  const { ref, box } = useFit(g.width / g.height);
  const [split, setSplit] = useState(0.5);
  const [holding, setHolding] = useState(false);
  const [phase, setPhase] = useState<"before" | "after">("after");
  const dragging = useRef(false);
  const boxRef = useRef<HTMLDivElement>(null);
  const contentSize = useMemo(() => ({ width: g.width, height: g.height }), [g.width, g.height]);
  // In a browser the Result view loads only the result, as a preview when it is as sharp
  // (stageImage); the original loads the first time a view shows it, then stays. Comparisons
  // and full screen use full-size images. The desktop app loads both at full size at once.
  const result = mode === "off" && !fullscreen;
  const after = result ? stageImage(`gen:${g.id}`, box.w, box.h) : urls.gen(g.id);
  const beforeWanted = useRef(!inBrowser);
  if (!result || holding) beforeWanted.current = true;
  const before = beforeWanted.current ? urls.input(base.inputId) : undefined;

  useEffect(() => {
    if (mode !== "flicker") return;
    setPhase("before");
    const t = window.setInterval(() => setPhase((p) => (p === "before" ? "after" : "before")), flickerMs);
    return () => window.clearInterval(t);
  }, [mode, flickerMs, g.id]);

  // Hold B to peek at the original from any view.
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "b" && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey && !isTyping(e.target)) setHolding(true);
    };
    const up = (e: KeyboardEvent) => e.key.toLowerCase() === "b" && setHolding(false);
    const blur = () => setHolding(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, []);

  const setFromPointer = (clientX: number) => {
    const r = boxRef.current?.getBoundingClientRect();
    if (r && r.width > 0) setSplit(Math.min(1, Math.max(0, (clientX - r.left) / r.width)));
  };

  const showBefore = holding || (mode === "flicker" && phase === "before");
  const label = holding || mode === "flicker" ? (showBefore ? "Before" : "After") : null;
  const pct = (v: number) => `${v * 100}%`;
  const { x, y, w, h } = base.rect;

  // In full screen the box keeps the image's own size inside ZoomPan, which scales it; `k`
  // undoes that scale for the labels and the slider handle so they stay at screen size.
  // While zoomed in a drag pans the image; the handle still moves the split.
  const frame = (scale: number, zoomed: boolean) => {
    const k = 1 / scale;
    const tag = (side: "left" | "right") => ({ transform: `scale(${k})`, transformOrigin: `top ${side}` });
    return (
      <div
        ref={boxRef}
        className={`compare-box ${mode === "slider" ? "slider" : ""} ${zoomed ? "zoomed" : ""}`}
        style={fullscreen ? { width: g.width, height: g.height } : { width: box.w, height: box.h }}
        onPointerDown={(e) => {
          if (e.button !== 0 || zoomed) return;
          e.stopPropagation();
          e.currentTarget.setPointerCapture(e.pointerId);
          if (mode === "slider") {
            dragging.current = true;
            setFromPointer(e.clientX);
          } else if (mode === "off") setHolding(true);
        }}
        onPointerMove={(e) => {
          if (!dragging.current) return;
          e.stopPropagation();
          setFromPointer(e.clientX);
        }}
        onPointerUp={() => {
          dragging.current = false;
          if (mode === "off") setHolding(false);
        }}
        onPointerCancel={() => {
          dragging.current = false;
          setHolding(false);
        }}
        onDoubleClick={() => setState({ overlay: null })}
        title={mode === "off" && !zoomed ? "Press and hold (or hold B) to see the original" : undefined}
      >
        <img className="compare-img" src={after} alt={g.prompt} draggable={false} />
        {mode === "diff" && !holding && before && <DiffCanvas after={after} before={before} rect={base.rect} tagStyle={tag("left")} />}
        {/* The area painted for the next edit, on the result (the original covers it when shown). */}
        {maskSrc && <img className="stage-mask" src={maskSrc} alt="" draggable={false} />}
        {/* The original stays mounted (hidden) so switching is instant. */}
        <div
          className="compare-before"
          style={{
            visibility: showBefore || mode === "slider" ? "visible" : "hidden",
            clipPath: mode === "slider" && !holding ? `inset(0 ${pct(1 - split)} 0 0)` : undefined,
          }}
        >
          <img src={before} alt="Original" draggable={false} style={{ left: pct(x), top: pct(y), width: pct(w), height: pct(h) }} />
        </div>
        {mode === "slider" && !holding && (
          <>
            <div
              className="compare-handle"
              style={{ left: pct(split), width: 2 * k, marginLeft: -k }}
              onPointerDown={(e) => {
                if (e.button !== 0) return;
                e.stopPropagation();
                boxRef.current?.setPointerCapture(e.pointerId);
                dragging.current = true;
              }}
            >
              <span style={{ transform: `scale(${k})` }} />
            </div>
            <span className="compare-tag left" style={tag("left")}>Before</span>
            <span className="compare-tag right" style={tag("right")}>After</span>
          </>
        )}
        {label && <span className="compare-tag left" style={tag("left")}>{label}</span>}
        {!fullscreen && (
          <div className="compare-overlay" onPointerDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
            <CompareBar />
          </div>
        )}
        {!fullscreen && (
          <FullscreenButton hint="Hold B for the original" onOpen={onFullscreen} content={<CompareFullscreen g={g} base={base} />} />
        )}
      </div>
    );
  };

  if (fullscreen)
    return (
      <div className="viewer-frame">
        <ZoomPan contentSize={contentSize} resetKey={g.id} enableToggleZoom={false} minimapSrc={after}>
          {({ transform, isZoomed }) => frame(transform.scale, isZoomed)}
        </ZoomPan>
      </div>
    );
  return (
    <div className="stage-frame" ref={ref}>
      {frame(1, false)}
    </div>
  );
}

/** What full screen shows for a result with an original: the comparison and its controls. */
export function CompareFullscreen({ g, base }: { g: GenerationRecord; base: CompareBase }) {
  return (
    <>
      <CompareFrame g={g} base={base} fullscreen />
      <div className="viewer-compare-bar">
        <CompareBar />
      </div>
    </>
  );
}

/** Highlights what changed: dimmed result, changed pixels in red. */
function DiffCanvas({ after, before, rect, tagStyle }: { after: string; before: string; rect: CompareBase["rect"]; tagStyle?: CSSProperties }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [share, setShare] = useState<number | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let alive = true;
    setShare(null);
    setError(false);
    const load = (src: string) =>
      new Promise<HTMLImageElement>((ok, fail) => {
        const im = new Image();
        im.crossOrigin = "anonymous";
        im.onload = () => ok(im);
        im.onerror = fail;
        // A separate URL: the displayed <img> loads without CORS, and reusing that cached
        // response for a CORS request would fail.
        im.src = `${src}${src.includes("?") ? "&" : "?"}cors=1`;
      });
    Promise.all([load(after), load(before)])
      .then(([a, b]) => {
        const cv = ref.current;
        if (!alive || !cv) return;
        const k = Math.min(1, DIFF_SIDE / Math.max(a.naturalWidth, a.naturalHeight));
        const W = Math.max(1, Math.round(a.naturalWidth * k));
        const H = Math.max(1, Math.round(a.naturalHeight * k));
        cv.width = W;
        cv.height = H;
        const ctx = cv.getContext("2d", { willReadFrequently: true })!;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(a, 0, 0, W, H);
        const A = ctx.getImageData(0, 0, W, H);
        ctx.clearRect(0, 0, W, H);
        ctx.drawImage(b, rect.x * W, rect.y * H, rect.w * W, rect.h * H);
        const B = ctx.getImageData(0, 0, W, H);
        const out = ctx.createImageData(W, H);
        const s = diffPixels(A.data, B.data, out.data);
        ctx.putImageData(out, 0, 0);
        setShare(s);
      })
      .catch(() => alive && setError(true));
    return () => {
      alive = false;
    };
  }, [after, before, rect.x, rect.y, rect.w, rect.h]);

  return (
    <>
      <canvas ref={ref} className="compare-diff" />
      <span className="compare-tag left" style={tagStyle}>
        {error ? "Could not compare the images" : share === null ? "Comparing…" : share < 0.001 ? "No visible change" : `≈${share < 0.01 ? "<1" : Math.round(share * 100)}% of the image changed`}
      </span>
    </>
  );
}

/** View switch shown under a result that has an original to compare with. */
export function CompareBar() {
  const mode = useStore((s) => s.compare);
  const flickerMs = useStore((s) => s.flickerMs);
  return (
    <div className="compare-bar">
      <Seg value={mode} options={MODES} onChange={(v) => setState({ compare: v })} />
      {mode === "flicker" && <Seg value={String(flickerMs)} options={SPEEDS} onChange={(v) => setState({ flickerMs: Number(v) })} />}
    </div>
  );
}

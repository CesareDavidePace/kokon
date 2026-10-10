// The "Paint the area to change" editor (src/components/InpaintEditor.tsx markup and classes),
// with brush strokes drawn by the frame clock instead of the pointer.
import { Brush, Check, Contrast, Eraser, Redo2, Trash2, Undo2, X } from "lucide-react";
import { staticFile } from "remotion";
import { IconButton } from "../../../../src/components/ui";

/** Where the image sits in the window (see `imageBox`). */
export function imageBox(aspect: number) {
  const availW = 1180 - 48;
  const availH = 818 - 52 - 66 - 24;
  const w = Math.min(availW, availH * aspect);
  const h = w / aspect;
  return { w, h, x: 260 + (1180 - w) / 2, y: 82 + 52 + 12 + (availH - h) / 2 };
}

/** A stroke as points in image fractions; `p` (0–1) is how much of the whole path is painted. */
export function strokePoints(points: [number, number][], p: number): [number, number][] {
  if (p <= 0) return [];
  const seg: number[] = [];
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    seg.push(d);
    total += d;
  }
  let left = total * Math.min(1, p);
  const outPts: [number, number][] = [points[0]];
  for (let i = 1; i < points.length; i++) {
    if (left >= seg[i - 1]) {
      outPts.push(points[i]);
      left -= seg[i - 1];
    } else {
      const t = left / seg[i - 1];
      outPts.push([points[i - 1][0] + (points[i][0] - points[i - 1][0]) * t, points[i - 1][1] + (points[i][1] - points[i - 1][1]) * t]);
      break;
    }
  }
  return outPts;
}

export function InpaintOverlay({ id, aspect, points, p, brush, opacity = 1 }: { id: string; aspect: number; points: [number, number][]; p: number; brush: number; opacity?: number }) {
  const box = imageBox(aspect);
  const pts = strokePoints(points, p);
  const vw = 1000;
  const vh = 1000 / aspect;
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"} ${x * vw} ${y * vh}`).join(" ");
  const tip = pts[pts.length - 1];
  return (
    <div className="overlay" style={{ opacity }}>
      <div className="overlay-head">
        <h2>Paint the area to change</h2>
        <span className="spacer" />
        <IconButton title="Cancel (Esc)">
          <X size={18} />
        </IconButton>
      </div>
      <div className="editor-stage">
        <div className="mask-wrap" style={{ width: box.w, height: box.h }}>
          <img src={staticFile(`img/${id}.jpg`)} alt="" style={{ width: box.w, height: box.h }} />
          <svg viewBox={`0 0 ${vw} ${vh}`} preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", mixBlendMode: "difference", opacity: 0.75 }}>
            {d && <path d={d} fill="none" stroke="#fff" strokeWidth={(brush / box.w) * vw} strokeLinecap="round" strokeLinejoin="round" />}
          </svg>
          {tip && <div className="brush-cursor" style={{ left: tip[0] * box.w, top: tip[1] * box.h, width: brush, height: brush }} />}
        </div>
      </div>
      <div className="editor-tools">
        <div className="group">
          <IconButton title="Brush" active>
            <Brush size={17} />
          </IconButton>
          <IconButton title="Eraser (E)">
            <Eraser size={17} />
          </IconButton>
        </div>
        <div className="group">
          <label className="slider">
            Size
            <input type="range" min={4} max={400} value={brush} readOnly />
          </label>
          <label className="slider">
            Feather
            <input type="range" min={0} max={64} value={12} readOnly />
          </label>
        </div>
        <div className="group">
          <IconButton title="Undo" disabled={!d}>
            <Undo2 size={17} />
          </IconButton>
          <IconButton title="Redo" disabled>
            <Redo2 size={17} />
          </IconButton>
          <IconButton title="Invert mask">
            <Contrast size={17} />
          </IconButton>
          <IconButton title="Clear mask">
            <Trash2 size={17} />
          </IconButton>
        </div>
        <button className="btn primary">
          <Check size={15} /> Done
        </button>
      </div>
    </div>
  );
}

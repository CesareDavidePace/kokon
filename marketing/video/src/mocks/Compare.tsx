// The app's CompareFrame (src/components/Compare.tsx) with the slider position taken from the
// frame clock (`CompareSplitContext`, provided by <AppWindow>) instead of the pointer. Markup and
// classes are unchanged; a phone gets the app's one glass button for the views (PhoneCompare).
import { ChevronDown, SquareSplitHorizontal } from "lucide-react";
import { createContext, useContext } from "react";
import { compareBase as _cb, type CompareBase, type CompareMode } from "../../../../src/lib/logic";
import type { GenerationRecord } from "../../../../src/lib/types";
import { Seg, useFit } from "../../../../src/components/ui";
import { FullscreenButton } from "../../../../src/components/Viewer";
import { urls } from "./api";
import { DevelopImg } from "./Develop";
import { useNarrow } from "./platform";
import { useStore } from "./store";

void _cb;
const MODES: { value: CompareMode; label: string }[] = [
  { value: "off", label: "Result" },
  { value: "slider", label: "Slider" },
  { value: "flicker", label: "Flicker" },
  { value: "diff", label: "Changes" },
];

export const CompareSplitContext = createContext(0.5);

export function CompareFrame({ g, base, fullscreen = false }: { g: GenerationRecord; base: CompareBase; fullscreen?: boolean }) {
  const mode = useStore((s) => s.compare);
  const phone = useNarrow();
  const split = Math.min(1, Math.max(0, useContext(CompareSplitContext)));
  const { ref, box } = useFit(g.width / g.height);
  const after = urls.gen(g.id);
  const before = urls.input(base.inputId);
  const pct = (v: number) => `${v * 100}%`;
  const { x, y, w, h } = base.rect;
  return (
    <div className={fullscreen ? "viewer-frame" : "stage-frame"} ref={ref}>
      <div className={`compare-box ${mode === "slider" ? "slider" : ""}`} style={{ width: box.w, height: box.h }}>
        <DevelopImg className="compare-img" src={after} placeholder={urls.genThumb(g.id)} alt={g.prompt} />
        <div className="compare-before" style={{ visibility: mode === "slider" ? "visible" : "hidden", clipPath: mode === "slider" ? `inset(0 ${pct(1 - split)} 0 0)` : undefined }}>
          <img src={before} alt="Original" draggable={false} style={{ left: pct(x), top: pct(y), width: pct(w), height: pct(h) }} />
        </div>
        {mode === "slider" && (
          <>
            <div className="compare-handle" style={{ left: pct(split) }}>
              <span />
            </div>
            <span className="compare-tag left">Before</span>
            <span className="compare-tag right">After</span>
          </>
        )}
        {!fullscreen && phone && <PhoneCompare />}
        {!fullscreen && !phone && (
          <div className="compare-overlay">
            <CompareBar />
          </div>
        )}
        {!fullscreen && <FullscreenButton hint="Hold B for the original" content={null} />}
      </div>
    </div>
  );
}

/** The phone's switch: one glass button on the picture naming the view (its sheet is not filmed). */
function PhoneCompare() {
  const mode = useStore((s) => s.compare);
  return (
    <div className="compare-pill-wrap">
      <button className="compare-pill" title="Compare with the original">
        <SquareSplitHorizontal size={15} />
        {MODES.find((m) => m.value === mode)?.label}
        <ChevronDown size={13} />
      </button>
    </div>
  );
}

export function CompareBar() {
  const mode = useStore((s) => s.compare);
  return (
    <div className="compare-bar">
      <Seg value={mode} options={MODES} onChange={() => {}} />
    </div>
  );
}

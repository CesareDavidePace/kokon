// The Kokon window: the app's own Sidebar, top bar, Stage and Composer, fed the state of this
// frame. Kokon draws its own title bar (the top bar holds the window's buttons), so the window
// is one surface, as in the app.
import type { CSSProperties, ReactNode } from "react";
import { useCurrentFrame } from "remotion";
import { Composer } from "../../../../src/components/Composer";
import { Sidebar } from "../../../../src/components/Sidebar";
import { Stage } from "../../../../src/components/Stage";
import { TopBar } from "../../../../src/components/TopBar";
import "../../../../src/styles.css";
import { CardMotionContext, type CardMotion } from "../mocks/card";
import { CompareSplitContext } from "../mocks/Compare";
import { setWordPace } from "../mocks/jobMotion";
import { FrameStateContext, setFrameState, type State } from "../mocks/store";
import { FPS } from "./timeline";
import "./theme.css";

// The app's own dark theme tokens (styles.css), so the video never drifts from them.
if (typeof document !== "undefined") document.documentElement.dataset.theme = "dark";

/** `split` is the before/after slider position for this frame (0–1); `card` the clock of the
 * job card in the stage (cocoon, Prompt+ text, hatching). */
export function AppWindow({ state, width, height, overlay, hideComposer, split = 0.5, card }: { state: State; width: number; height: number; overlay?: ReactNode; hideComposer?: boolean; split?: number; card?: CardMotion }) {
  const f = useCurrentFrame();
  setFrameState(state);
  setWordPace(card?.cps ?? 60);
  // CSS loops (the status capsule's silk, shimmer, caret) run on the frame clock; entrances
  // have long finished (app/theme.css).
  const vt = { "--vt": (100 + f / FPS).toFixed(3) } as CSSProperties;
  return (
    <FrameStateContext.Provider value={state}>
      <CompareSplitContext.Provider value={split}>
        <CardMotionContext.Provider value={card ?? null}>
          <div className="kokon-window" style={{ width, height, ...vt }}>
            <div className="app">
              {state.sidebar && <Sidebar />}
              <div className="workspace">
                <TopBar />
                <div className="main">
                  <Stage />
                  {overlay}
                </div>
                {!hideComposer && <Composer />}
              </div>
            </div>
          </div>
        </CardMotionContext.Provider>
      </CompareSplitContext.Provider>
    </FrameStateContext.Provider>
  );
}

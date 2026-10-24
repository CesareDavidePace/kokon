// The Kokon window: the app's own Sidebar, top bar, Stage and Composer, fed the state of this
// frame. Kokon draws its own title bar (the top bar holds the window's buttons), so the window
// is one surface, as in the app.
import type { CSSProperties, ReactNode } from "react";
import { useEffect, useState } from "react";
import { continueRender, delayRender, useCurrentFrame } from "remotion";
import { Composer } from "../../../../src/components/Composer";
import { PhoneApp } from "../../../../src/components/PhoneApp";
import { Settings } from "../../../../src/components/Settings";
import { Sidebar } from "../../../../src/components/Sidebar";
import { Stage } from "../../../../src/components/Stage";
import { TopBar } from "../../../../src/components/TopBar";
import "../../../../src/styles.css";
import "../../../../src/phone.css";
import { CardMotionContext, type CardMotion } from "../mocks/card";
import { CompareSplitContext } from "../mocks/Compare";
import { type Develop, DevelopContext } from "../mocks/Develop";
import { setWordPace } from "../mocks/jobMotion";
import { setPlatform } from "../mocks/platform";
import { FrameStateContext, setFrameState, type State } from "../mocks/store";
import { FPS } from "./timeline";
import "./theme.css";

// The app's own dark theme tokens (styles.css), so the video never drifts from them.
if (typeof document !== "undefined") document.documentElement.dataset.theme = "dark";

/** `split` is the before/after slider position for this frame (0–1); `card` the clock of the
 * job card in the stage (cocoon, Prompt+ text, hatching); `className` "silk" shows the thread
 * that runs round the composer while you write (app/theme.css). `phone`: Kokon in a phone's
 * browser (the app's phone layout, loaders/phone-container.cjs); otherwise the desktop app.
 * `light`: the app's light theme (the document stays dark). `waitFor`: what Settings must show before the frame is taken (default: the dialog itself).
 * A phone window shows the app's phone layout (components/PhoneApp.tsx): `standalone` as added to
 * the home screen (its corners, the home bar's inset), `keyboard` with the keyboard up (the scene
 * draws it under the window), `develop` the image developing into view (mocks/Develop.tsx), and
 * `vars` CSS variables for the phone's motion on the scene's clock (video011.css: `--t-screen`,
 * `--t-page` seconds into an entrance, `--swipe-dx`/`--swipe-o` a finger on the picture). */
export function AppWindow({ state, width, height, overlay, hideComposer, split = 0.5, card, className, phone = false, light = false, waitFor = ".modal", standalone = false, keyboard = false, develop = null, vars }: { state: State; width: number; height: number; overlay?: ReactNode; hideComposer?: boolean; split?: number; card?: CardMotion; className?: string; phone?: boolean; light?: boolean; waitFor?: string; standalone?: boolean; keyboard?: boolean; develop?: Develop | null; vars?: Record<string, string | number> }) {
  const f = useCurrentFrame();
  setFrameState(state);
  setPlatform(phone, phone);
  setWordPace(card?.cps ?? 60);
  // CSS loops (the status capsule's silk, shimmer, caret) run on the frame clock; entrances
  // have long finished (app/theme.css).
  const vt = { "--vt": (100 + f / FPS).toFixed(3) } as CSSProperties;
  // A phone: the window is the screen (the app's height follows the visible part, PhoneApp.tsx).
  const screen = phone
    ? ({
        containerType: "inline-size",
        // .phone-app is position: fixed: the window, not the phone's frame, holds it.
        contain: "layout",
        containerName: standalone ? "kokon-standalone" : undefined,
        "--vv-h": `${height}px`,
        "--vv-top": "0px",
        "--sab": standalone && !keyboard ? "34px" : "0px",
        ...vars,
      } as CSSProperties)
    : {};
  const desktop = (
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
      {state.overlay === "settings" && (
        <>
          <Settings />
          <WaitFor selector={waitFor} />
        </>
      )}
    </div>
  );
  return (
    <FrameStateContext.Provider value={state}>
      <CompareSplitContext.Provider value={split}>
        <CardMotionContext.Provider value={card ?? null}>
          <DevelopContext.Provider value={develop}>
            <div className={["kokon-window", className, light && "kokon-light", keyboard && "keyboard"].filter(Boolean).join(" ")} style={{ width, height, ...vt, ...screen }}>
              {phone ? (
                <div className="app phone">
                  <PhoneApp />
                </div>
              ) : (
                desktop
              )}
            </div>
          </DevelopContext.Provider>
        </CardMotionContext.Provider>
      </CompareSplitContext.Provider>
    </FrameStateContext.Provider>
  );
}

/** Holds the frame until `selector` is on the page: Settings shows once its config has arrived. */
function WaitFor({ selector }: { selector: string }) {
  const [handle] = useState(() => delayRender(`Waiting for ${selector}`));
  useEffect(() => {
    let raf = 0;
    const check = () => (document.querySelector(selector) ? continueRender(handle) : (raf = requestAnimationFrame(check)));
    check();
    return () => cancelAnimationFrame(raf);
  }, [handle, selector]);
  return null;
}

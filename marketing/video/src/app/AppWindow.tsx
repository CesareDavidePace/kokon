// The Kokon window: the app's own Sidebar, Stage and Composer, fed the state of this frame.
import { EyeOff, Minus, Redo2, Settings, Square, Undo2, X } from "lucide-react";
import type { ReactNode } from "react";
import { Composer } from "../../../../src/components/Composer";
import { ProjectTitle, Sidebar } from "../../../../src/components/Sidebar";
import { Stage } from "../../../../src/components/Stage";
import { IconButton } from "../../../../src/components/ui";
import "../../../../src/styles.css";
import { setFrameState, type State } from "../mocks/store";
import { Mark } from "./Mark";
import "./theme.css";

export function AppWindow({ state, width, height, overlay, hideComposer, busy }: { state: State; width: number; height: number; overlay?: ReactNode; hideComposer?: boolean; busy?: boolean }) {
  setFrameState(state);
  const hasHistory = state.session.length > 1;
  return (
    <div className="kokon-window" style={{ width, height }}>
      <div className="win-titlebar">
        <span className="win-icon">
          <Mark size={11} color="#0d0d0d" />
        </span>
        Kokon
        <span className="spacer" />
        <span className="ctl"><Minus size={15} /></span>
        <span className="ctl"><Square size={12} /></span>
        <span className="ctl"><X size={15} /></span>
      </div>
      <div className="app">
        {state.sidebar && <Sidebar />}
        <div className="workspace">
          <div className="topbar">
            <div className="brand">
              <ProjectTitle />
            </div>
            <div className="toolbar">
              {hasHistory && (
                <>
                  <IconButton title="Undo"><Undo2 size={18} /></IconButton>
                  <IconButton title="Redo" disabled><Redo2 size={18} /></IconButton>
                </>
              )}
              <IconButton title="Private mode"><EyeOff size={18} /></IconButton>
              <span className="status">
                <span className="dot ready" />
                <span>{busy ? "Working" : "Ready"}</span>
              </span>
              <IconButton title="Settings"><Settings size={18} /></IconButton>
            </div>
          </div>
          <div className="main">
            <Stage />
            {overlay}
          </div>
          {!hideComposer && <Composer />}
        </div>
      </div>
    </div>
  );
}

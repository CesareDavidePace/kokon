// The app's real store, except that components read the state computed for the current frame
// instead of the live global state. Each <AppWindow> provides its own state through context, so
// two windows on screen at once (cross-fades) never read each other's state, even when a
// component re-renders after layout. `getState()` (event handlers, a few renders) gets the
// state of the window rendered last.
import { createContext, useContext } from "react";
import * as real from "../../../../src/lib/store";

export * from "../../../../src/lib/store";

let frameState: real.State = real.getState();

export function setFrameState(s: real.State) {
  frameState = s;
}
export function getState(): real.State {
  return frameState;
}
export const FrameStateContext = createContext<real.State | null>(null);
export function useStore<T>(sel: (s: real.State) => T): T {
  return sel(useContext(FrameStateContext) ?? frameState);
}

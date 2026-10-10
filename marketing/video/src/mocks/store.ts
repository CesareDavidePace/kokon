// The app's real store, except that components read the state computed for the current frame
// (set by <AppWindow> right before it renders) instead of the live global state.
import * as real from "../../../../src/lib/store";

export * from "../../../../src/lib/store";

let frameState: real.State = real.getState();

export function setFrameState(s: real.State) {
  frameState = s;
}
export function getState(): real.State {
  return frameState;
}
export function useStore<T>(sel: (s: real.State) => T): T {
  return sel(frameState);
}

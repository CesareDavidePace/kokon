// Light / dark appearance. public/theme.js resolves the choice into data-theme on <html> before
// the first paint; this module stores the choice and keeps the native title bar in step.
import { useEffect } from "react";
import { native } from "./native";
import { inBrowser } from "./platform";

export type ThemeChoice = "system" | "light" | "dark";

const KEY = "qls.theme.v1";

export function themeChoice(): ThemeChoice {
  try {
    const t = localStorage.getItem(KEY);
    if (t === "light" || t === "dark") return t;
  } catch {
    /* storage unavailable */
  }
  return "system";
}

/** Native window theme (title bar); null follows the OS. Outside Tauri there is no window. */
export function syncWindowTheme(t: ThemeChoice = themeChoice()) {
  native.setTheme(t === "system" ? null : t).catch(() => {});
}

/** A phone's status bar (the theme-color tags, public/theme.js) in `color` while a black screen
 * is open (full screen, the photo views); null gives it back to the theme. Tags are replaced,
 * not edited, as theme.js does (iOS seems to read a new tag but not a changed one). */
export function setStatusBar(color: string | null) {
  if (color === null) {
    window.dispatchEvent(new Event("qls-theme"));
    return;
  }
  document.querySelectorAll('meta[name="theme-color"]').forEach((m) => {
    const fresh = document.createElement("meta");
    fresh.name = "theme-color";
    fresh.content = color;
    const media = m.getAttribute("media");
    if (media) fresh.setAttribute("media", media);
    m.replaceWith(fresh);
  });
}

// Black screens open now (StatusTint shows black while there is one).
let blackScreens = 0;
const tintListeners = new Set<() => void>();
const tintChanged = () => tintListeners.forEach((f) => f());

/** The status bar black while mounted (a browser only): the theme-color tags (Android's Chrome,
 * iOS before 26) and components/StatusTint (iOS 26+). */
export function useBlackStatusBar() {
  useEffect(() => {
    if (!inBrowser) return;
    setStatusBar("#000");
    blackScreens++;
    tintChanged();
    return () => {
      blackScreens--;
      tintChanged();
      setStatusBar(null);
    };
  }, []);
}

/** Calls `f` when a black screen opens or closes; returns the unsubscribe. */
export function onBlackScreens(f: () => void): () => void {
  tintListeners.add(f);
  return () => tintListeners.delete(f);
}

/** A black screen (full screen, the photo views) is open. */
export const blackScreenOpen = () => blackScreens > 0;

export function setThemeChoice(t: ThemeChoice) {
  try {
    if (t === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, t);
  } catch {
    /* storage unavailable: applies until restart */
  }
  window.dispatchEvent(new Event("qls-theme"));
  syncWindowTheme(t);
}

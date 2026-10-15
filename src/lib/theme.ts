// Light / dark appearance. public/theme.js resolves the choice into data-theme on <html> before
// the first paint; this module stores the choice and keeps the native title bar in step.
import { native } from "./native";

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

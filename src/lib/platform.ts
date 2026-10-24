// Platform-specific wording for labels (shortcut handlers accept Ctrl and ⌘ everywhere).

import { useSyncExternalStore } from "react";

export const isMac = typeof navigator !== "undefined" && /Mac/i.test(navigator.platform || navigator.userAgent);

/** "Ctrl+Z" on Windows, "⌘Z" on macOS. */
export const shortcut = (key: string) => (isMac ? `⌘${key}` : `Ctrl+${key}`);

export const isLinux = typeof navigator !== "undefined" && !isMac && /Linux/i.test(navigator.platform || navigator.userAgent);

export const fileManager = isMac ? "Finder" : isLinux ? "the file manager" : "Explorer";

/** A phone-sized screen (Kokon in a browser on another device): the sidebar becomes a drawer. Matches styles.css. */
export const isNarrow = () => typeof window !== "undefined" && !!window.matchMedia?.("(max-width: 720px)").matches;

/** `isNarrow()` that re-renders when the screen crosses the width (a phone turned sideways). */
export function useNarrow(): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const mq = window.matchMedia?.("(max-width: 720px)");
      mq?.addEventListener("change", onChange);
      return () => mq?.removeEventListener("change", onChange);
    },
    isNarrow,
  );
}

/** The UI runs in a browser, served by kokon-service to another device, not in the desktop app. */
export const inBrowser = typeof window !== "undefined" && !("__TAURI_INTERNALS__" in window);

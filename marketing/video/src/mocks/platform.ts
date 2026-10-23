// The app's platform checks, switched per window: the desktop app (Settings → Other devices) or
// Kokon in a phone's browser (the drawer sidebar, bottom sheets). <AppWindow> sets them before
// its components render; the app reads `inBrowser` at use, so the live binding follows.
export * from "../../../../src/lib/platform";

export let inBrowser = false;
let narrow = false;
export const isNarrow = () => narrow;
export function setPlatform(browser: boolean, phone: boolean) {
  inBrowser = browser;
  narrow = phone;
}

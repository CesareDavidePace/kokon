// The app's platform checks, switched per window: the desktop app (Settings → Other devices) or
// Kokon in a phone's browser (its own layout, components/PhoneApp.tsx). <AppWindow> sets them before
// its components render; the app reads `inBrowser` at use, so the live binding follows.
export * from "../../../../src/lib/platform";

export let inBrowser = false;
let narrow = false;
export const isNarrow = () => narrow;
export const useNarrow = () => narrow;
export function setPlatform(browser: boolean, phone: boolean) {
  inBrowser = browser;
  narrow = phone;
}

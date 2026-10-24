// The status bar colour on iOS 26+ (see StatusTint).

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { blackScreenOpen, onBlackScreens } from "../lib/theme";

/** iOS 26+ ignores theme-color: Safari and home-screen apps colour the status bar from an opaque
 * fixed element at the top of the page, read when the page loads. This strip is that element, in
 * the page's ground colour (black while a black screen is open); it is created anew with its
 * colour inline at every change (theme, system appearance, full screen), which makes iOS read
 * it again without a reload (found by cole-titze/nhl-odds#101, tested on an iPhone). */
export function StatusTint() {
  const [color, setColor] = useState<string | null>(null);
  useEffect(() => {
    const read = () => setColor(blackScreenOpen() ? "#000" : getComputedStyle(document.body).backgroundColor);
    read();
    const off = onBlackScreens(read);
    // data-theme changes with the Settings choice and with the system's (public/theme.js).
    const mo = new MutationObserver(read);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      off();
      mo.disconnect();
    };
  }, []);
  if (!color) return null;
  // On the body, not in the layout: above the full-screen view (also a portal there), so that one
  // turns the bar black too.
  return createPortal(<div key={color} className="status-tint" aria-hidden style={{ backgroundColor: color }} />, document.body);
}

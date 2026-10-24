// What the device shots need from the app's styles.css and phone.css, the rules themselves unchanged:
// - the phone layout (`@media (max-width: 720px)`) as a container query, so a window rendered
//   phone-sized inside the 1920-px video gets it (AppWindow `phone`), while desktop windows keep
//   the desktop layout; phone.css's own width query (3-column library) the same way;
// - the light theme's tokens (the first `:root` block of styles.css) also on `.kokon-light`, so a
//   window can be light while the video's document is dark (AppWindow `light`);
// - phone.css: its phone tokens (`:root` inside the phone query) on the window's `.app`, the dark
//   ones only in a dark window; the home-screen app's rules (`display-mode: standalone`) for a
//   window named `kokon-standalone` (AppWindow `standalone`); the keyboard (`html.keyboard`) on a
//   window with `.keyboard`; the home bar's inset (0 in the render) from `--sab`.
module.exports = function (source) {
  const phone = /phone\.css$/.test(this.resourcePath);
  let out = source.replace(/@media \(max-width: 720px\)/g, "@container (max-width: 720px)");
  if (phone) {
    return out
      .replace(/:root\[data-theme="dark"\] \{/g, ':root[data-theme="dark"] .kokon-window:not(.kokon-light) > .app {')
      .replace(/:root \{/g, ".kokon-window > .app {")
      .replace(/@media \(min-width: 540px\)/g, "@container (min-width: 540px)")
      .replace(/@media \(display-mode: standalone\)/g, "@container kokon-standalone (min-width: 0px)")
      .replace(/html\.keyboard /g, ".kokon-window.keyboard ")
      .replace(/html:not\(\.keyboard\) /g, ".kokon-window:not(.keyboard) ")
      .replace(/env\(safe-area-inset-bottom\)/g, "var(--sab, 0px)");
  }
  const root = source.match(/:root \{([\s\S]*?)\n\}/);
  const light = root ? `\n.kokon-window.kokon-light {${root[1]}\n}\n` : "";
  return out + light;
};

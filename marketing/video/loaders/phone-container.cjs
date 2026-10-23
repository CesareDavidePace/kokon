// Two things the device shots need from the app's styles.css, the rules themselves unchanged:
// - the phone layout (`@media (max-width: 720px)`) as a container query, so a window rendered
//   phone-sized inside the 1920-px video gets it (AppWindow `phone`), while desktop windows keep
//   the desktop layout;
// - the light theme's tokens (the first `:root` block) also on `.kokon-light`, so a window can be
//   light while the video's document is dark (AppWindow `light`).
module.exports = function (source) {
  const root = source.match(/:root \{([\s\S]*?)\n\}/);
  const light = root ? `\n.kokon-window.kokon-light {${root[1]}\n}\n` : "";
  return source.replace(/@media \(max-width: 720px\)/g, "@container (max-width: 720px)") + light;
};

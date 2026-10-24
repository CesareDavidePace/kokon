// Sets data-theme="light" | "dark" on <html> before the first paint, so styles.css and the
// splash in index.html never flash the wrong colors. The choice in Settings → General is kept in
// localStorage (src/lib/theme.ts); without one it follows the OS and updates when that changes.
// (A classic script file, not inline, because the CSP only allows scripts from the app itself.)
(function () {
  // macOS gets a lighter dark palette (styles.css); set it here too so the splash matches.
  if (/Mac/i.test(navigator.platform || navigator.userAgent)) document.documentElement.dataset.platform = "mac";
  var dark = window.matchMedia("(prefers-color-scheme: dark)");
  function apply() {
    var t = null;
    try {
      t = localStorage.getItem("qls.theme.v1");
    } catch (e) {}
    var theme = t === "light" || t === "dark" ? t : dark.matches ? "dark" : "light";
    document.documentElement.dataset.theme = theme;
    // The status bar of a phone (Kokon added to the home screen) in the page's own ground colour
    // (styles.css --ground), not the manifest's light one.
    var ground = theme === "dark" ? (document.documentElement.dataset.platform === "mac" ? "#161616" : "#0a0a0a") : "#f4f4f4";
    // Replaced, not edited: iOS (a home-screen app) seems to read a new tag but not a changed one.
    document.querySelectorAll('meta[name="theme-color"]').forEach(function (m) {
      var fresh = document.createElement("meta");
      fresh.name = "theme-color";
      fresh.content = ground;
      if (m.media) fresh.media = m.media;
      m.replaceWith(fresh);
    });
  }
  apply();
  dark.addEventListener("change", apply);
  window.addEventListener("qls-theme", apply);
})();

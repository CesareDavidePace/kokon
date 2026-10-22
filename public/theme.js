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
    document.documentElement.dataset.theme = t === "light" || t === "dark" ? t : dark.matches ? "dark" : "light";
  }
  apply();
  dark.addEventListener("change", apply);
  window.addEventListener("qls-theme", apply);
})();

// The main window starts hidden (tauri.conf.json) so it never shows an empty page while the
// web view and the app load. This runs right after the splash in index.html is in the DOM:
// show the window once that splash has been painted. (A classic script file, not inline,
// because the CSP only allows scripts from the app itself.)
requestAnimationFrame(function () {
  requestAnimationFrame(function () {
    var t = window.__TAURI_INTERNALS__;
    if (t) t.invoke("plugin:window|show", { label: "main" }).catch(function () {});
  });
});

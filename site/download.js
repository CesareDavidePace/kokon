// Point the download buttons straight at the installer of the latest GitHub release.
// If the API can't be reached (offline, rate limit), they keep linking to the release page.
// Buttons marked data-dl-auto (the nav one) download the installer for the visitor's OS;
// on any other OS they keep their own link (the hero buttons or the release page).
(() => {
  const ua = navigator.userAgent;
  const os = /Windows/.test(ua) ? "windows"
    : /Macintosh/.test(ua) && navigator.maxTouchPoints < 2 ? "mac" // iPads also say Macintosh
    : null;
  if (os) document.querySelectorAll("a[data-dl-auto]").forEach((a) => (a.dataset.dl = os));
  const links = document.querySelectorAll("a[data-dl]");
  if (!links.length) return;
  const PICK = { windows: /_x64-setup\.exe$/, mac: /_aarch64\.dmg$/ };
  fetch("https://api.github.com/repos/GabrieleLozupone/kokon/releases/latest")
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((release) =>
      links.forEach((a) => {
        const asset = release.assets.find((x) => PICK[a.dataset.dl]?.test(x.name));
        if (asset) a.href = asset.browser_download_url;
      }),
    )
    .catch(() => {});
})();

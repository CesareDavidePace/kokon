// Point the download buttons straight at the installer of the latest GitHub release.
// If the API can't be reached (offline, rate limit), they keep linking to the release page.
(() => {
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

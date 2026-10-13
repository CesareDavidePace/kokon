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

  // Once the installer starts downloading, ask (once per browser) for a GitHub star.
  // Not when the button fell back to the release page: that visitor is on GitHub already.
  const ask = document.getElementById("star-ask");
  if (!ask?.showModal) return;
  const KEY = "kokon-star-asked";
  const asked = () => { try { return localStorage.getItem(KEY); } catch { return null; } };
  links.forEach((a) =>
    a.addEventListener("click", () => {
      if (asked() || !a.href.includes("/releases/download/")) return;
      try { localStorage.setItem(KEY, "1"); } catch {}
      // Let the browser show its download first.
      setTimeout(() => ask.showModal(), 700);
    }),
  );
  document.getElementById("star-ask-go").addEventListener("click", () => ask.close());
  ask.addEventListener("click", (e) => { if (e.target === ask) ask.close(); });
})();

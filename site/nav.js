// The site menu: dropdowns on wide screens, a full-screen panel on phones.
// The markup is the same on every page (see the <header class="nav">); this only opens and closes things.
(() => {
  const nav = document.getElementById("nav");
  if (!nav) return;
  const burger = nav.querySelector(".burger");
  const groups = [...nav.querySelectorAll(".dd")];
  const wide = matchMedia("(min-width: 900px)");

  const closeGroups = (except) => groups.forEach((g) => {
    if (g === except) return;
    g.classList.remove("open");
    g.querySelector("button").setAttribute("aria-expanded", "false");
  });
  groups.forEach((g) => {
    const b = g.querySelector("button");
    b.addEventListener("click", () => {
      if (!wide.matches) return; // on phones every group is already open in the panel
      const open = !g.classList.contains("open");
      closeGroups(g);
      g.classList.toggle("open", open);
      b.setAttribute("aria-expanded", String(open));
    });
  });

  const setMenu = (open) => {
    nav.classList.toggle("menu-open", open);
    document.documentElement.classList.toggle("menu-lock", open);
    burger.setAttribute("aria-expanded", String(open));
    burger.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    burger.querySelector(".lbl").textContent = open ? "Close" : "Menu";
  };
  burger.addEventListener("click", () => setMenu(!nav.classList.contains("menu-open")));
  wide.addEventListener("change", () => { setMenu(false); closeGroups(); });

  // A link closes everything (same-page anchors don't reload the page).
  nav.querySelectorAll(".nav-links a").forEach((a) => a.addEventListener("click", () => { setMenu(false); closeGroups(); }));
  document.addEventListener("click", (e) => { if (!nav.contains(e.target)) closeGroups(); });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const open = groups.find((g) => g.classList.contains("open"));
    closeGroups();
    setMenu(false);
    (open ? open.querySelector("button") : burger).focus();
  });

  // Mark the page we're on.
  const here = location.pathname.replace(/index\.html$/, "");
  nav.querySelectorAll(".nav-links a[href^='/']").forEach((a) => {
    if (a.getAttribute("href") === here) a.setAttribute("aria-current", "page");
  });
})();

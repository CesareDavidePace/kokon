# Marketing

Launch material for Kokon. Everything visual here was made by Kokon itself or by its own UI code.

| Folder | What |
| --- | --- |
| `showcase/` | The before/after pairs and the hero image, generated with Kokon (prompts, seeds and times in `PROMPTS.md`). Web-sized JPEGs; the full-size PNGs stay in `%USERPROFILE%\Pictures\Kokon\marketing`. |
| `video/` | The launch video, made with [Remotion](https://www.remotion.dev) from the app's own React components. |
| `release-notes-0.9.0.md` | Body of the v0.9.0 GitHub release (already in the draft). |
| `../site/` | The one-page website (static HTML, no build step). |

## The video

`video/` renders the real `Sidebar`, `Stage`, `Composer` and timeline from `../src`. Webpack swaps
the app's store, service API, native bridge and Tauri modules for mocks (`video/src/mocks`), so
each frame's UI comes from a state computed for that frame. Change the app's UI and the video
follows on the next render. Two parts are copies with the same markup and classes, because
their state lives inside the component: the paint-area editor (`app/InpaintOverlay.tsx`) and
the before/after slider (`mocks/Compare.tsx`).

```powershell
cd marketing/video
npm install
npm run studio                    # preview and scrub in the browser
npm run render                    # out/kokon-launch.mp4, 1920×1080, ~87 s, about 5 minutes
```

Scenes (`src/scenes`): Intro (the mark assembles) → Hook (wall of before/afters) → Create →
Region (paint, remove, add) → Versions → Montage (six edits) → Agent (MCP) → Hardware → Outro.
Each scene is also its own composition, so one can be rendered alone, for example
`npx remotion render Montage out/montage.mp4`.

There is no music. Add a track from YouTube's free Audio Library when uploading.

The website uses a lighter copy: `site/assets/kokon.mp4` (CRF 25, no audio, ~13 MB) and
`site/assets/kokon-poster.jpg`. To refresh them after a new render:

```powershell
ffmpeg -y -i marketing/video/out/kokon-launch.mp4 -c:v libx264 -preset slow -crf 25 -pix_fmt yuv420p -movflags +faststart -an site/assets/kokon.mp4
ffmpeg -y -ss 34 -i marketing/video/out/kokon-launch.mp4 -frames:v 1 -q:v 3 site/assets/kokon-poster.jpg
```

## The website

`site/index.html` plus `site/assets/`: one static page that links every download button to the
latest GitHub release. Open the file in a browser to preview it. To publish it, any static host
works; the simplest are:

- **GitHub Pages** (free once the repository is public): Settings → Pages → Deploy from a branch
  is limited to `/` or `/docs`, so either move `site/` to its own `gh-pages` branch or use a
  small Pages workflow that uploads `site/`.
- **Cloudflare Pages, Netlify or Vercel**: point them at the repository with `site` as the
  output folder and no build command.

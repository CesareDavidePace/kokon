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
npm run render                    # out/kokon-launch.mp4, 1920×1080, ~85 s, about 5 minutes
```

Scenes (`src/scenes`): Intro (the mark assembles) → Hook (wall of before/afters) → Create →
Region (paint, remove, add) → Versions → Montage (six edits, full frame) → Agent (MCP) → Hardware →
Outro. Each scene is also its own composition, so one can be rendered alone, for example
`npx remotion render Montage out/montage.mp4` (the voice and music play only in `KokonLaunch`).

`src/app/timeline.ts` is the film's clock: scene lengths, the frame where each voice-over sentence
plays, the frames of the actions the voice names (Remove, Add, the montage cuts) and the music's
beat grid. Retime there and pictures, voice and music move together.

Sound (`src/app/sound.tsx`, all levels in one table):

- **Voice-over**: `public/voice/NN.wav`, one file per line of `audio/voiceover.md`, generated
  locally with [Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT) by
  `audio/gen_voice.py final`, which also writes `src/app/voice.json` (where each sentence starts
  and ends) so sentences can be placed one by one. The music ducks about 8 dB under it.
- **Music**: `public/music.m4a`, generated locally with [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)
  (MIT, output usable commercially) by `audio/gen_music.py`: take `house-4`, 118 BPM, with two bars
  removed from its breakdown so the second drop (60.3 s) lands on the Agent scene.
- **Effects**: `public/sfx/*.wav`, synthesized from scratch by `audio/make_sfx.py` (no samples, no
  licences), kept sparse: typing only in the first demo, quiet clicks, a soft reveal in three
  variants that take turns when a result appears, whooshes only on big moves.

To regenerate music, start the ACE-Step API (`uv run acestep-api` in its folder) and run
`python audio/gen_music.py house`; takes land in `audio/takes/` (git-ignored).

The website uses a lighter copy: `site/assets/kokon.mp4` (CRF 25, no audio, ~13 MB) and
`site/assets/kokon-poster.jpg`. To refresh them after a new render:

```powershell
ffmpeg -y -i marketing/video/out/kokon-launch.mp4 -c:v libx264 -preset slow -crf 25 -pix_fmt yuv420p -movflags +faststart -an site/assets/kokon.mp4
ffmpeg -y -ss 35.5 -i marketing/video/out/kokon-launch.mp4 -frames:v 1 -q:v 3 site/assets/kokon-poster.jpg
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

// Six edits back to back, each in the app: the sentence is typed, the image is made, and the
// before/after slider sweeps across the result. All images were made by Kokon.
import { AbsoluteFill, Sequence, useCurrentFrame } from "remotion";
import { AppWindow } from "../app/AppWindow";
import { appState, gen, genItem, jobItem, PROJECTS, PROMPTS, ref } from "../app/fixtures";
import { Backdrop, Camera, camAt, Caption, kf, prog, typed, typingDone, Vignette } from "../app/motion";
import { setCompareSplit } from "../mocks/Compare";
import type { State } from "../mocks/store";
import type { ResolvedMode } from "../../../../src/lib/types";

interface Edit {
  label: string;
  before: string;
  after: string;
  prompt: string;
  mode: ResolvedMode;
  project: string;
}

const EDITS: Edit[] = [
  { label: "Relight.", before: "cabin", after: "cabin-winter", prompt: PROMPTS.winter, mode: "edit", project: "p2" },
  { label: "Change the text.", before: "bakery", after: "bakery-text", prompt: PROMPTS.text, mode: "edit", project: "p1" },
  { label: "Restyle.", before: "potter", after: "potter-watercolor", prompt: PROMPTS.restyle, mode: "edit", project: "p3" },
  { label: "Replace.", before: "room", after: "room-sofa", prompt: PROMPTS.replace, mode: "edit", project: "p4" },
  { label: "Expand.", before: "lighthouse", after: "lighthouse-wide", prompt: PROMPTS.expand, mode: "outpaint", project: "p5" },
  { label: "Product shots.", before: "sneaker", after: "sneaker-ad", prompt: PROMPTS.product, mode: "edit", project: "p6" },
];

export const EDIT_FRAMES = 100;
export const MONTAGE_FRAMES = EDIT_FRAMES * EDITS.length;

const EXPAND = { left: 614, right: 614, top: 0, bottom: 0 };

function OneEdit({ e, index }: { e: Edit; index: number }) {
  const f = useCurrentFrame();
  const base = appState();
  const before = gen(e.before, "generate", "");
  const after = gen(e.after, e.mode, e.prompt, e.before, e.mode === "outpaint" ? { outpaint: EXPAND } : {});
  const TY = 4;
  const SEND = Math.min(36, typingDone(e.prompt, TY, 75) + 4);
  const RESULT = SEND + 18;
  const projects: State["projects"] = PROJECTS.map((p) => (p.id === e.project ? { ...p, updatedAt: Date.now() - 5_000 } : p));
  const common = { projects, projectId: e.project };
  const outpaint = e.mode === "outpaint" ? { left: 40, right: 40, top: 0, bottom: 0, units: "percent" as const } : null;

  let state: State;
  let split = 0.5;
  if (f < SEND) {
    state = appState({ session: [genItem(before)], selected: e.before, composer: { ...base.composer, refs: [ref(e.before)], outpaint, prompt: typed(e.prompt, f, TY, 75) }, ...common });
  } else if (f < RESULT) {
    const j = jobItem(`j-${e.after}`, e.mode, e.prompt, e.after, e.before, Math.min(8, Math.floor((f - SEND) / 2) + 1));
    state = appState({ session: [genItem(before), j.item], selected: j.item.key, jobs: { [j.item.key]: j.job }, composer: { ...base.composer, refs: [ref(e.before)] }, ...common });
  } else {
    split = kf(f, [[RESULT, 1], [RESULT + 4, 1], [RESULT + 34, 0.04], [EDIT_FRAMES, 0.5]]);
    state = appState({ session: [genItem(before), genItem(after)], selected: e.after, compare: "slider", composer: { ...base.composer, refs: [ref(e.after)] }, ...common });
  }
  setCompareSplit(split);

  const cam = camAt(f, [
    [0, { x: 850, y: 470, s: 1.2 }],
    [SEND, { x: 850, y: 470, s: 1.22 }],
    [RESULT + 6, { x: 850, y: 370, s: 1.34 }],
    [EDIT_FRAMES, { x: 850, y: 360, s: 1.4 }],
  ]);
  const flash = 1 - prog(f, 0, 8);
  return (
    <AbsoluteFill>
      <Backdrop />
      <Camera cam={cam}>
        <AppWindow state={state} width={1440} height={900} busy={f >= SEND && f < RESULT} />
      </Camera>
      <Vignette strength={0.95} side="top" reach={0.55} />
      <div
        style={{
          position: "absolute",
          left: 96,
          top: 76,
          fontFamily: '"Inter Variable", Inter, sans-serif',
          fontSize: 22,
          fontWeight: 600,
          letterSpacing: "0.14em",
          color: "rgba(240,240,240,0.55)",
          opacity: prog(f, 0, 10),
        }}
      >
        {String(index + 1).padStart(2, "0")} / {String(EDITS.length).padStart(2, "0")}
      </div>
      <Caption title={e.label} start={0} end={EDIT_FRAMES + 20} y={112} size={72} />
      <AbsoluteFill style={{ background: "#fff", opacity: flash * 0.06, pointerEvents: "none" }} />
    </AbsoluteFill>
  );
}

export function Montage() {
  return (
    <AbsoluteFill style={{ background: "#050505" }}>
      {EDITS.map((e, i) => (
        <Sequence key={e.after} from={i * EDIT_FRAMES} durationInFrames={EDIT_FRAMES}>
          <OneEdit e={e} index={i} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
}

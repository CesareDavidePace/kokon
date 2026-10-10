// The app in use, through its own components: create an image from a sentence, paint an area
// and remove what is in it, add something, and the version timeline.
import type { ReactNode } from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { AppWindow } from "../app/AppWindow";
import { appState, gen, genItem, jobItem, PROJECTS, PROMPTS, ref } from "../app/fixtures";
import { imageBox, InpaintOverlay, strokePoints } from "../app/InpaintOverlay";
import { Backdrop, Camera, camAt, Caption, kf, Pointer, prog, typed, typingDone, Vignette } from "../app/motion";
import { setCompareSplit } from "../mocks/Compare";
import type { State } from "../mocks/store";

const inpaintReq = (prompt: string, from: string, action: "remove" | "add") => ({
  prompt,
  mode: "inpaint" as const,
  images: [`gen:${from}`],
  quality: "quality" as const,
  resolution: "auto" as const,
  advanced: {},
  inpaint_action: action,
});
const STREET = gen("street", "generate", PROMPTS.street);
const REMOVED = gen("street-removed", "inpaint", PROMPTS.remove, "street", { request: inpaintReq(PROMPTS.remove, "street", "remove") });
const CAT = gen("street-cat", "inpaint", PROMPTS.add, "street-removed", { request: inpaintReq(PROMPTS.add, "street-removed", "add") });
const STREET_PROJECT = { id: "current", title: "Italian street café", updatedAt: Date.now() - 20_000, cover: "gen:street", count: 1 };
const withProject = (count: number, cover: string): State["projects"] => [{ ...STREET_PROJECT, count, cover }, ...PROJECTS];
const proj = (n: number, cover: string) => ({ projects: withProject(n, cover), projectId: "current" });

const SEND: [number, number] = [1232, 857];

/** Steps of a job advancing from `start`, one every `every` frames. */
const stepAt = (f: number, start: number, every: number) => Math.max(0, Math.min(8, Math.floor((f - start) / every) + 1));

// ---------------------------------------------------------------------------------------
// Create: an empty project, a sentence, the image.
// ---------------------------------------------------------------------------------------
export const CREATE_FRAMES = 390;

export function Create() {
  const f = useCurrentFrame();
  const base = appState();
  const T0 = 44;
  const T1 = typingDone(PROMPTS.street, T0, 46);
  const SEND_AT = T1 + 14;
  const JOB = SEND_AT + 2;
  const DONE = JOB + 92;

  let state: State;
  if (f < JOB) {
    state = appState({ composer: { ...base.composer, prompt: typed(PROMPTS.street, f, T0, 46) } });
  } else if (f < DONE) {
    const j = jobItem("j-street", "generate", PROMPTS.street, "street", undefined, stepAt(f, JOB + 10, 10));
    state = appState({ session: [j.item], selected: "j-street", jobs: { "j-street": j.job }, ...proj(1, "gen:street") });
  } else {
    state = appState({ session: [genItem(STREET)], selected: "street", ...proj(1, "gen:street") });
  }

  const enter = prog(f, 0, 34);
  const cam = camAt(f, [
    [0, { x: 720, y: 450, s: 1.02 }],
    [34, { x: 720, y: 450, s: 1.12 }],
    [70, { x: 850, y: 700, s: 1.5 }],
    [T1, { x: 850, y: 720, s: 1.5 }],
    [JOB + 14, { x: 850, y: 430, s: 1.22 }],
    [DONE, { x: 850, y: 400, s: 1.24 }],
    [CREATE_FRAMES, { x: 850, y: 330, s: 1.42 }],
  ]);
  return (
    <AbsoluteFill>
      <Backdrop />
      <Camera cam={cam} tilt={(1 - enter) * 16} lift={(1 - enter) * 260} opacity={enter}>
        <AppWindow state={state} width={1440} height={900} busy={f >= JOB && f < DONE} />
        <Pointer
          path={[
            [20, 980, 560],
            [42, 640, 806],
            [T1, 650, 806],
            [SEND_AT - 2, SEND[0], SEND[1]],
            [SEND_AT + 30, 1260, 640],
          ]}
          clicks={[42, SEND_AT]}
          hideBefore={20}
          hideAfter={SEND_AT + 40}
        />
      </Camera>
      <Vignette strength={prog(f, 50, 12) * (1 - prog(f, T1, 12))} side="top" />
      <Vignette strength={prog(f, JOB, 12)} />
      <Caption title="Describe it." sub="Any image, in plain words." start={56} end={T1 + 10} y={84} />
      <Caption title="Made on your own GPU." sub="Qwen-Image 2.1 Turbo · 2752×1536 in 38 s on an RTX 5070 Ti (sped up here)" start={JOB + 6} end={CREATE_FRAMES} />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------
// Region: paint the scooter, Remove, result with a sweeping before/after slider; then Add.
// ---------------------------------------------------------------------------------------
export const REGION_FRAMES = 540;
const STROKE: [number, number][] = [
  [0.47, 0.52], [0.62, 0.49], [0.665, 0.58], [0.475, 0.6], [0.455, 0.7], [0.67, 0.68], [0.68, 0.78], [0.46, 0.8], [0.465, 0.9], [0.665, 0.88], [0.62, 0.965], [0.5, 0.965],
];
const AR = 2752 / 1536;
const DONE_BTN: [number, number] = [1086, 864];
const PAINT_BTN: [number, number] = [1086, 762];
const REMOVE_SEG: [number, number] = [581, 857];
const ADD_SEG: [number, number] = [689, 857];
const CAT_PROMPT = "a ginger cat sitting on the cobblestones";

export function Region() {
  const f = useCurrentFrame();
  const base = appState();
  const OPEN = 40;
  const PAINT0 = 62;
  const PAINT1 = 160;
  const DONE_AT = 182;
  const REMOVE_AT = 212;
  const TY0 = 222;
  const TY1 = typingDone(PROMPTS.remove, TY0, 26);
  const SEND_AT = TY1 + 12;
  const JOB = SEND_AT + 2;
  const RESULT = JOB + 66;
  const ADD0 = RESULT + 110; // second edit: add a cat
  const ADD_TY = ADD0 + 26;
  const ADD_TY1 = typingDone(CAT_PROMPT, ADD_TY, 60);
  const ADD_SEND = ADD_TY1 + 8;
  const ADD_RESULT = ADD_SEND + 44;

  const mask = { inputId: "mask", refKey: "ref-street" };
  let state: State;
  let overlay: ReactNode = null;
  let split = 0.5;
  const paint = prog(f, PAINT0, PAINT1 - PAINT0, (t) => t);
  if (f < DONE_AT) {
    state = appState({ session: [genItem(STREET)], selected: "street", composer: { ...base.composer, refs: [ref("street")] }, ...proj(1, "gen:street") });
    if (f >= OPEN) overlay = <InpaintOverlay id="street" aspect={AR} points={STROKE} p={paint} brush={74} opacity={prog(f, OPEN, 8)} />;
  } else if (f < JOB) {
    state = appState({
      session: [genItem(STREET)],
      selected: "street",
      composer: { ...base.composer, refs: [ref("street")], mask, inpaintAction: f >= REMOVE_AT ? "remove" : "edit", prompt: typed(PROMPTS.remove, f, TY0, 26) },
      ...proj(1, "gen:street"),
    });
  } else if (f < RESULT) {
    const j = jobItem("j-rm", "inpaint", PROMPTS.remove, "street-removed", "street", stepAt(f, JOB + 8, 7), "remove");
    state = appState({ session: [genItem(STREET), j.item], selected: "j-rm", jobs: { "j-rm": j.job }, composer: { ...base.composer, refs: [ref("street")] }, ...proj(2, "gen:street") });
  } else if (f < ADD0) {
    split = kf(f, [[RESULT, 0.97], [RESULT + 6, 0.97], [RESULT + 50, 0.06], [RESULT + 92, 0.52]]);
    state = appState({ session: [genItem(STREET), genItem(REMOVED)], selected: "street-removed", compare: "slider", composer: { ...base.composer, refs: [ref("street-removed")] }, ...proj(2, "gen:street-removed") });
  } else if (f < ADD_SEND + 2) {
    state = appState({
      session: [genItem(STREET), genItem(REMOVED)],
      selected: "street-removed",
      composer: { ...base.composer, refs: [ref("street-removed")], mask: { inputId: "mask2", refKey: "ref-street-removed" }, inpaintAction: f >= ADD0 + 12 ? "add" : "edit", prompt: typed(CAT_PROMPT, f, ADD_TY, 60) },
      ...proj(2, "gen:street-removed"),
    });
  } else if (f < ADD_RESULT) {
    const j = jobItem("j-cat", "inpaint", PROMPTS.add, "street-cat", "street-removed", stepAt(f, ADD_SEND + 4, 5), "add");
    state = appState({ session: [genItem(STREET), genItem(REMOVED), j.item], selected: "j-cat", jobs: { "j-cat": j.job }, composer: { ...base.composer, refs: [ref("street-removed")] }, ...proj(3, "gen:street-removed") });
  } else {
    split = kf(f, [[ADD_RESULT, 0.97], [ADD_RESULT + 6, 0.97], [ADD_RESULT + 40, 0.3], [REGION_FRAMES, 0.42]]);
    state = appState({ session: [genItem(STREET), genItem(REMOVED), genItem(CAT)], selected: "street-cat", compare: "slider", composer: { ...base.composer, refs: [ref("street-cat")] }, ...proj(3, "gen:street-cat") });
  }
  setCompareSplit(split);

  const box = imageBox(AR);
  const pts = strokePoints(STROKE, paint);
  const tip = pts[pts.length - 1] ?? STROKE[0];
  const tipWin: [number, number] = [box.x + tip[0] * box.w, box.y + tip[1] * box.h];
  const painting = f >= PAINT0 - 4 && f <= PAINT1 + 2;
  const cam = camAt(f, [
    [0, { x: 850, y: 400, s: 1.24 }],
    [OPEN, { x: 850, y: 470, s: 1.16 }],
    [PAINT0 + 10, { x: 860, y: 500, s: 1.55 }],
    [PAINT1, { x: 860, y: 520, s: 1.55 }],
    [DONE_AT + 10, { x: 850, y: 640, s: 1.4 }],
    [TY1, { x: 820, y: 660, s: 1.5 }],
    [JOB + 16, { x: 850, y: 380, s: 1.3 }],
    [RESULT + 100, { x: 850, y: 330, s: 1.42 }],
    [ADD0 + 14, { x: 820, y: 660, s: 1.45 }],
    [ADD_SEND + 12, { x: 850, y: 380, s: 1.3 }],
    [REGION_FRAMES, { x: 850, y: 340, s: 1.4 }],
  ]);
  return (
    <AbsoluteFill>
      <Backdrop />
      <Camera cam={cam}>
        <AppWindow state={state} width={1440} height={900} overlay={overlay} hideComposer={!!overlay} busy={(f >= JOB && f < RESULT) || (f >= ADD_SEND && f < ADD_RESULT)} />
        {!painting && (
          <Pointer
            path={[
              [0, 1100, 520],
              [34, PAINT_BTN[0], PAINT_BTN[1]],
              [PAINT0 - 4, tipWin[0], tipWin[1]],
              [PAINT1 + 2, tipWin[0], tipWin[1]],
              [DONE_AT - 2, DONE_BTN[0], DONE_BTN[1]],
              [REMOVE_AT - 2, REMOVE_SEG[0], REMOVE_SEG[1]],
              [TY1, REMOVE_SEG[0] + 40, REMOVE_SEG[1] - 30],
              [SEND_AT - 2, SEND[0], SEND[1]],
              [SEND_AT + 30, 1300, 620],
              [ADD0 + 10, ADD_SEG[0], ADD_SEG[1]],
              [ADD_SEND - 2, SEND[0], SEND[1]],
              [ADD_SEND + 30, 1300, 620],
            ]}
            clicks={[36, DONE_AT, REMOVE_AT, SEND_AT, ADD0 + 12, ADD_SEND]}
            hideBefore={6}
            hideAfter={ADD_SEND + 34}
          />
        )}
      </Camera>
      <Vignette strength={0.9} />
      <Caption title="Paint an area." sub="Brush over what you want to change." start={48} end={DONE_AT + 6} />
      <Caption title="Only that area changes." sub="Remove, replace, add or change. Everything outside the brush stays as it was." start={RESULT - 30} end={ADD0 + 4} />
      <Caption title="Add something new." sub="“a ginger cat sitting on the cobblestones”" start={ADD0 + 8} end={REGION_FRAMES} />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------
// Versions: the timeline of the project, every step kept.
// ---------------------------------------------------------------------------------------
export const VERSIONS_FRAMES = 150;

export function Versions() {
  const f = useCurrentFrame();
  const base = appState();
  const sel = f < 50 ? "street-cat" : f < 90 ? "street-removed" : "street";
  setCompareSplit(0.5);
  const state = appState({ session: [genItem(STREET), genItem(REMOVED), genItem(CAT)], selected: sel, composer: { ...base.composer, refs: [ref(sel)] }, ...proj(3, "gen:street-cat") });
  const cam = camAt(f, [
    [0, { x: 850, y: 340, s: 1.4 }],
    [36, { x: 850, y: 560, s: 1.7 }],
    [VERSIONS_FRAMES, { x: 850, y: 540, s: 1.62 }],
  ]);
  return (
    <AbsoluteFill>
      <Backdrop />
      <Camera cam={cam}>
        <AppWindow state={state} width={1440} height={900} />
        <Pointer path={[[10, 1000, 700], [44, 812, 664], [84, 776, 664], [VERSIONS_FRAMES, 760, 690]]} clicks={[48, 88]} hideBefore={10} />
      </Camera>
      <Vignette strength={0.9} side="top" />
      <Caption title="Every version kept." sub="Go back to any step, compare, or branch from it." start={20} end={VERSIONS_FRAMES} y={90} />
    </AbsoluteFill>
  );
}

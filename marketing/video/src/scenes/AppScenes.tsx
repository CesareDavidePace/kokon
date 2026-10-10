// The app in use, through its own components: create an image from a sentence, paint an area
// and remove what is in it, add something, and the version timeline. The voice leads: each
// action lands on the sentence that names it (cue frames in app/timeline.ts). Shots hold still
// and change by cutting, finished images are shown full frame.
import type { ReactNode } from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { AppWindow } from "../app/AppWindow";
import { appState, gen, genItem, jobItem, PROJECTS, PROMPTS, ref } from "../app/fixtures";
import { FullImage } from "../app/FullImage";
import { imageBox, InpaintOverlay, strokePoints } from "../app/InpaintOverlay";
import { Backdrop, type Cam, Camera, camAt, camSpeed, Caption, DepthOfField, Dissolve, inOut, kf, Pointer, prog, typed, typingDone, Vignette } from "../app/motion";
import { Send, Sfx } from "../app/sound";
import { CREATE, LENGTH, REGION, sceneCues, sentenceAt } from "../app/timeline";
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
/** The composer, close: a macro shot past the window's bottom edge, sharp on the composer only. */
const COMPOSER_CAM: Cam = { x: 850, y: 742, s: 1.8 };
const COMPOSER_FOCUS: [number, number] = [520, 830];
/** The empty project's composer is shorter and lower: centre it. */
const PROMPT_CAM: Cam = { x: 850, y: 650, s: 1.7 };
/** Composer shots look at the window from slightly below and to the side. */
const ANGLE = { x: 16, y: -14 };
const PROMPT_FOCUS: [number, number] = [780, 1000];
/** Where the stage image sits once a result is selected (window px), to fly into it. */
const STAGE_IMG = { x: 850, y: 319, w: 820 };

/** Steps of a job advancing from `start`, one every `every` frames. */
const stepAt = (f: number, start: number, every: number) => Math.max(0, Math.min(8, Math.floor((f - start) / every) + 1));

/** Big centred words that rise in with the voice and leave with the next ones. */
function Words({ items, end }: { items: { text: string; at: number }[]; end: number }) {
  const f = useCurrentFrame();
  return (
    <>
      {items.map((w, i) => {
        const to = items[i + 1]?.at ?? end;
        if (f < w.at - 2 || f > to + 2) return null;
        const a = prog(f, w.at - 2, 14);
        const b = i < items.length - 1 ? prog(f, to - 6, 8, inOut) : prog(f, end - 10, 10, inOut);
        return (
          <div key={w.text} style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", pointerEvents: "none" }}>
            <div style={{ fontFamily: '"Inter Variable", Inter, sans-serif', fontSize: 104, fontWeight: 660, color: "#f4f4f2", letterSpacing: "-0.045em", textShadow: "0 6px 40px rgba(0,0,0,0.7)", opacity: a * (1 - b), transform: `translateY(${(1 - a) * 26 - b * 18}px)`, filter: `blur(${(1 - a) * 10 + b * 8}px)` }}>
              {w.text}
            </div>
          </div>
        );
      })}
    </>
  );
}

/** A small keyword in the lower left, for a shot whose action the voice names. */
const Tag = ({ text, start, end }: { text: string; start: number; end: number }) => <Caption title={text} start={start} end={end} y={930} size={48} />;

// ---------------------------------------------------------------------------------------
// Create: the window arrives, a sentence is typed, the image is made, and the camera flies
// into it. "No cloud. No account. No subscription." over the finished picture.
// ---------------------------------------------------------------------------------------
export const CREATE_FRAMES = LENGTH.create;

export function Create() {
  const f = useCurrentFrame();
  const base = appState();
  const cues = sceneCues("create");
  const T0 = CREATE.ask;
  const T1 = typingDone(PROMPTS.street, T0, 80);
  const SEND_AT = CREATE.send;
  const JOB = SEND_AT + 2;
  const DONE = CREATE.done;
  const FLY = CREATE.flyIn;
  const FULL = FLY + 22;

  const view = (t: number) => {
    let state: State;
    if (t < JOB) {
      state = appState({ composer: { ...base.composer, prompt: typed(PROMPTS.street, t, T0, 80) } });
    } else if (t < DONE) {
      const j = jobItem("j-street", "generate", PROMPTS.street, "street", undefined, stepAt(t, JOB + 6, 6));
      state = appState({ session: [j.item], selected: "j-street", jobs: { "j-street": j.job }, ...proj(1, "gen:street") });
    } else {
      state = appState({ session: [genItem(STREET)], selected: "street", ...proj(1, "gen:street") });
    }
    return <AppWindow state={state} width={1440} height={900} busy={t >= JOB && t < DONE} />;
  };

  const enter = prog(f, 0, 34);
  const camKeys: [number, Cam][] = [
    [0, { x: 720, y: 450, s: 1.02 }],
    [40, { x: 720, y: 450, s: 1.1 }],
    [92, PROMPT_CAM],
    [SEND_AT, { ...PROMPT_CAM, s: 1.86 }],
    [SEND_AT + 30, { x: 850, y: 330, s: 1.3 }],
    [FLY, { x: 850, y: 325, s: 1.34 }],
    [FULL, { x: STAGE_IMG.x, y: STAGE_IMG.y, s: 1920 / STAGE_IMG.w }],
  ];
  const at = (t: number) => camAt(t, camKeys);
  const cam = at(f);
  // Sharp on the composer while typing; the whole window once the image is being made.
  const dof = prog(f, 70, 20) * (1 - prog(f, SEND_AT + 2, 16));
  const app = (
    <Camera cam={cam} tilt={(1 - enter) * 16} lift={(1 - enter) * 260} opacity={enter} free={f > 60 && f < SEND_AT + 30} turn={{ x: ANGLE.x * dof, y: ANGLE.y * dof }} blur={f > FLY ? camSpeed(f, at) * 0.12 : 0}>
      <Dissolve frame={f} cuts={[JOB, DONE]} view={view} dur={10} />
      <Pointer
        path={[
          [56, 980, 700],
          [88, 640, 812],
          [112, 1080, 880],
          [SEND_AT - 16, 1120, 880],
          [SEND_AT - 2, SEND[0], SEND[1]],
          [SEND_AT + 24, 1260, 700],
        ]}
        clicks={[90, SEND_AT]}
        hideBefore={56}
        hideAfter={SEND_AT + 26}
      />
    </Camera>
  );
  const full = prog(f, FULL - 6, 8, inOut);
  const words = cues.find((c) => c.from === 2)!;
  return (
    <AbsoluteFill>
      <Backdrop />
      {dof > 0.01 ? <DepthOfField focus={PROMPT_FOCUS} blur={7 * dof}>{app}</DepthOfField> : app}
      {f >= FULL - 6 && (
        <div style={{ position: "absolute", inset: 0, opacity: full }}>
          <FullImage before="street" push={1 + prog(f, FULL, LENGTH.create - FULL, (x) => x) * 0.06} dim={0.25 * prog(f, CREATE.promises - 8, 12)} />
        </div>
      )}
      <Vignette strength={prog(f, JOB, 12) * (1 - full)} />
      <Caption title="Made on your own GPU." sub="Qwen-Image 2.1 Turbo · 2752×1536 in 38 s on an RTX 5070 Ti (sped up here)" start={JOB + 8} end={FLY} size={48} />
      <Words
        items={[
          { text: "No cloud.", at: sentenceAt(words, 2) },
          { text: "No account.", at: sentenceAt(words, 3) },
          { text: "No subscription.", at: sentenceAt(words, 4) },
        ]}
        end={LENGTH.create - 4}
      />
      <Sfx name="click" at={90} />
      <Sfx name="typing" at={T0} dur={T1 - T0} />
      <Send at={SEND_AT} />
      <Sfx name="reveal" at={DONE} variant={0} />
      <Sfx name="whoosh" at={FLY - 4} gain={0.8} />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------
// Region: paint the scooter; cut to the composer, Remove; the result full frame. Cut back,
// Add a cat; full frame again, then a before/after handle against the very first image.
// ---------------------------------------------------------------------------------------
export const REGION_FRAMES = LENGTH.region;
const STROKE: [number, number][] = [
  [0.47, 0.52], [0.62, 0.49], [0.665, 0.58], [0.475, 0.6], [0.455, 0.7], [0.67, 0.68], [0.68, 0.78], [0.46, 0.8], [0.465, 0.9], [0.665, 0.88], [0.62, 0.965], [0.5, 0.965],
];
const AR = 2752 / 1536;
const REMOVE_SEG: [number, number] = [581, 857];
const ADD_SEG: [number, number] = [689, 857];
const CAT_PROMPT = "a ginger cat sitting on the cobblestones";

export function Region() {
  const f = useCurrentFrame();
  const base = appState();
  const R = REGION;
  const box = imageBox(AR);
  // The paint editor's image filling the frame, as the full-frame picture Create ended on.
  const fillCam: Cam = { x: box.x + box.w / 2, y: box.y + box.h / 2, s: 1920 / box.w };
  const paintAt = (t: number) => prog(t, R.paint, R.paint1 - R.paint, (x) => x);

  // Shot 1: the paint editor over the image.
  const paintShot = () => {
    const state = appState({ session: [genItem(STREET)], selected: "street", composer: { ...base.composer, refs: [ref("street")] }, ...proj(1, "gen:street") });
    const cam = camAt(f, [
      [0, fillCam],
      [40, { ...fillCam, s: 1.58 }],
      [R.removeCut, { ...fillCam, y: fillCam.y + 6, s: 1.63 }],
    ]);
    return (
      <Camera cam={cam} free>
        <AppWindow state={state} width={1440} height={900} hideComposer overlay={<InpaintOverlay id="street" aspect={AR} points={STROKE} p={paintAt(f)} brush={74} />} />
      </Camera>
    );
  };

  // Shots 2 and 4: the composer, close. The action is picked as the voice says it.
  const composerShot = (from: number, action: "remove" | "add") => {
    const t = f - from;
    const pick = action === "remove" ? R.remove : R.add;
    const send = action === "remove" ? R.removeSend : R.addSend;
    const seg = action === "remove" ? REMOVE_SEG : ADD_SEG;
    const prompt = action === "remove" ? PROMPTS.remove : CAT_PROMPT;
    const img = action === "remove" ? STREET : REMOVED;
    const state = appState({
      session: action === "remove" ? [genItem(STREET)] : [genItem(STREET), genItem(REMOVED)],
      selected: img.id,
      composer: {
        ...base.composer,
        refs: [ref(img.id)],
        mask: { inputId: action === "remove" ? "mask" : "mask2", refKey: `ref-${img.id}` },
        inpaintAction: f >= pick ? action : "edit",
        prompt: typed(prompt, f, pick + 6, 70),
      },
      ...proj(action === "remove" ? 1 : 2, `gen:${img.id}`),
    });
    const cam = { ...COMPOSER_CAM, s: COMPOSER_CAM.s + t * 0.0006 };
    return (
      <DepthOfField focus={COMPOSER_FOCUS}>
        <Camera cam={cam} free turn={ANGLE}>
          <AppWindow state={state} width={1440} height={900} />
          <Pointer
            path={[
              [from, seg[0] + 120, seg[1] + 50],
              [pick - 2, seg[0], seg[1]],
              [send - 10, seg[0] + 60, seg[1] - 20],
              [send - 1, SEND[0], SEND[1]],
              [send + 10, SEND[0] + 30, SEND[1] - 30],
            ]}
            clicks={[pick, send]}
          />
        </Camera>
      </DepthOfField>
    );
  };

  let shot: ReactNode;
  if (f < R.removeCut) shot = paintShot();
  else if (f < R.removeShot) shot = composerShot(R.removeCut, "remove");
  else if (f < R.addCut) {
    shot = <FullImage before="street" after="street-removed" reveal={prog(f, R.removeShot + 6, 22, inOut)} push={1 + (f - R.removeShot) * 0.0007} />;
  } else if (f < R.addShot) shot = composerShot(R.addCut, "add");
  else {
    // The cat wipes in; then the handle compares against the very first image.
    const cmp = f >= R.same + 14;
    shot = cmp ? (
      <FullImage before="street" after="street-cat" split={kf(f, [[R.same + 14, 0], [R.same + 70, 0.72], [LENGTH.region, 0.46]])} push={1.03} />
    ) : (
      <FullImage before="street-removed" after="street-cat" reveal={prog(f, R.addShot + 6, 22, inOut)} push={1 + (f - R.addShot) * 0.0007} />
    );
  }
  // Create ended on the full-frame street; it hands over to the paint editor's identical image.
  const handover = 1 - prog(f, 0, 10, inOut);
  return (
    <AbsoluteFill>
      <Backdrop />
      {shot}
      {f < 10 && (
        <div style={{ position: "absolute", inset: 0, opacity: handover }}>
          <FullImage before="street" push={1.06} dim={0.25} />
        </div>
      )}
      {(f >= R.removeCut && f < R.removeShot) || (f >= R.addCut && f < R.addShot) ? (
        <>
          <Vignette strength={0.55} side="top" reach={0.6} />
          <Vignette strength={1} reach={0.42} />
        </>
      ) : null}
      <Tag text="Paint." start={R.paint} end={R.removeCut} />
      <Tag text="Remove." start={R.removeShot + 8} end={R.addCut} />
      <Tag text="Add." start={R.addShot + 8} end={R.same + 10} />
      {f >= R.same + 14 &&
        (["Before", "After"] as const).map((t) => (
          <div key={t} style={{ position: "absolute", top: 56, [t === "Before" ? "left" : "right"]: 64, padding: "8px 18px", borderRadius: 999, background: "rgba(0,0,0,0.5)", color: "#f2f2f2", fontFamily: '"Inter Variable", Inter, sans-serif', fontSize: 26, fontWeight: 550, opacity: prog(f, R.same + 18, 12) }}>
            {t}
          </div>
        ))}
      <Sfx name="brush" at={R.paint} dur={R.paint1 - R.paint} />
      <Sfx name="click" at={R.remove} />
      <Send at={R.removeSend} />
      <Sfx name="reveal" at={R.removeShot + 6} variant={1} />
      <Sfx name="click" at={R.add} />
      <Send at={R.addSend} />
      <Sfx name="reveal" at={R.addShot + 6} variant={2} />
      <Sfx name="slider" at={R.same + 16} />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------
// Versions: the timeline of the project, every step kept; two clicks walk back through it.
// ---------------------------------------------------------------------------------------
export const VERSIONS_FRAMES = LENGTH.versions;

export function Versions() {
  const f = useCurrentFrame();
  const base = appState();
  const cues = sceneCues("versions");
  const back1 = cues[0].at + 8;
  const back2 = sentenceAt(cues[0], 1) + 14;
  const view = (t: number) => {
    const sel = t < back1 ? "street-cat" : t < back2 ? "street-removed" : "street";
    const state = appState({ session: [genItem(STREET), genItem(REMOVED), genItem(CAT)], selected: sel, composer: { ...base.composer, refs: [ref(sel)] }, ...proj(3, "gen:street-cat") });
    return <AppWindow state={state} width={1440} height={900} />;
  };
  const cam = { x: 850, y: 520, s: 1.66 + f * 0.0005 };
  return (
    <AbsoluteFill>
      <Backdrop />
      <Camera cam={cam}>
        <Dissolve frame={f} cuts={[back1, back2]} view={view} dur={8} />
        <Pointer path={[[0, 1000, 720], [back1 - 6, 812, 664], [back2 - 6, 776, 664], [VERSIONS_FRAMES, 760, 690]]} clicks={[back1, back2]} />
      </Camera>
      <Vignette strength={0.9} side="top" />
      <Caption title="Every version kept." start={12} end={VERSIONS_FRAMES} y={90} size={56} />
      <Sfx name="click" at={back1} />
      <Sfx name="click" at={back2} />
    </AbsoluteFill>
  );
}

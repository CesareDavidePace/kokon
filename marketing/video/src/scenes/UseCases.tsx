// "Use cases": an 85-second video of everyday edits (site and YouTube, English, narrated). It opens on
// why Kokon (private, unlimited); then each beat: the before photo, the prompt typed into Kokon's
// prompt box, a wipe to the after landing on the spoken word, the caption. Fun beats pop (bold
// words, a pill, a burst); serious beats breathe (soft type, darker ground, the music lighter).
// Nothing moves the camera: motion comes from cuts on the beat, wipes and type. The clock is
// app/timelineUseCases.ts. All images were made with Kokon (marketing/README.md).
import type { CSSProperties, ReactNode } from "react";
import { AbsoluteFill, Img, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { ArrowUp, Gauge, Infinity as InfinityIcon, LockKeyhole } from "lucide-react";
import "../../../../src/styles.css";
import { Mark } from "../app/Mark";
import { inOut, prog } from "../app/motion";
import { Music, Send, Sfx, Voice } from "../app/sound";
import { FPS } from "../app/timeline";
import { at, CUES_USECASES, MONTAGE, MUSIC, musicGain, type Mood, photoTimes, seg, segFrames, SEGS, sentenceFrame, START, USECASES_FRAMES, wordFrame } from "../app/timelineUseCases";
import "../app/theme.css";

export { USECASES_FRAMES };
if (typeof document !== "undefined") document.documentElement.dataset.theme = "dark";

const font = '"Inter Variable", Inter, sans-serif';
const PAPER = "#F7F6F2";
const pic = (n: string) => staticFile(`img/usecases/${n}.jpg`);
/** Every image, for <Preload extra>. */
export const USECASES_IMAGES = ["01-before", "01-after", "02-before", "02-after", "02-ref", "03-before", "03-after", "03-color", "04-before", "04-after", "04-ref", "05-before", "05-after", "10-before", "10-after", "11-before", "11-after", "12-before", "12-after", "13-before", "13-after", "14-before", "14-after", "14-ref", "15-before", "15-after", "16-before", "16-after", "17-before", "17-after"].map((n) => `img/usecases/${n}.jpg`);

interface Ref { src: string; ar: number }
interface Case {
  before: string;
  after: string;
  /** Portrait photos sit left with the words right; landscape ones fill the frame. */
  tall: boolean;
  prompt: string;
  /** The question before the change, then the answer once it lands. */
  setup?: string;
  punch: string;
  /** The end of `punch` that goes in a pill (fun beats). */
  hl?: string;
  /** The second image of a multi-image edit, shown as a small card. */
  ref?: Ref;
  /** Landscape crop: CSS object-position. */
  pos?: string;
  /** A fixed detail of the change (fractions of the image): centre and radius (of the width). */
  loupe?: { x: number; y: number; r: number };
}

const CASES: Record<string, Case> = {
  tourists: { before: "01-before", after: "01-after", tall: true, prompt: "remove the tourists", setup: "Photobombers?", punch: "Gone.", hl: "Gone." },
  blink: {
    before: "02-before",
    after: "02-after",
    tall: false,
    prompt: "open his eyes, from the other shot",
    setup: "Someone always blinks?",
    punch: "Fixed.",
    hl: "Fixed.",
    ref: { src: "02-ref", ar: 1920 / 1427 },
    pos: "50% 28%",
    loupe: { x: 460 / 1920, y: 400 / 1427, r: 0.085 },
  },
  restore: { before: "03-before", after: "03-after", tall: true, prompt: "restore this old photo", setup: "Your grandparents' wedding photo.", punch: "Creases and scratches, gone." },
  colour: { before: "03-after", after: "03-color", tall: true, prompt: "colorize it", punch: "…and back in colour." },
  grandma: { before: "04-before", after: "04-after", tall: false, prompt: "add grandma from this photo", setup: "Someone missing?", punch: "Bring them in.", ref: { src: "04-ref", ar: 1427 / 1920 }, pos: "50% 45%" },
  flamingo: { before: "17-before", after: "17-after", tall: true, prompt: "on a flamingo, surfing a wave", setup: "Your holiday,", punch: "upgraded.", hl: "upgraded." },
  hair: { before: "13-before", after: "13-after", tall: true, prompt: "copper bob", setup: "Try the haircut", punch: "before the hairdresser.", hl: "hairdresser." },
  jacket: { before: "14-before", after: "14-after", tall: true, prompt: "wear this jacket", setup: "Try it on", punch: "before you buy it.", hl: "buy it.", ref: { src: "14-ref", ar: 1 } },
  child: { before: "05-before", after: "05-after", tall: false, prompt: "tidy up the room", setup: "Photos of your kids", punch: "don't belong on someone else's server.", pos: "50% 55%" },
  cv: { before: "16-before", after: "16-after", tall: true, prompt: "professional headshot", setup: "Kitchen selfie?", punch: "Now it's a headshot.", hl: "headshot." },
  invite: { before: "15-before", after: "15-after", tall: true, prompt: "birthday invite", punch: "Text too.", hl: "Text too." },
};
const HOME = [
  { before: "10-before", after: "10-after", prompt: "paint the wall sage green" },
  { before: "11-before", after: "11-after", prompt: "a cognac leather sofa" },
  { before: "12-before", after: "12-after", prompt: "tidy up the garden" },
];

// ---------------------------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------------------------
const fill: CSSProperties = { position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" };

/** Before under after, the after wiping in from the left (`p` 0–1), a light sweeping on its edge. */
function Wipe({ before, after, p, pos, radius = 0 }: { before: string; after: string; p: number; pos?: string; radius?: number }) {
  const moving = p > 0 && p < 1;
  const glow = Math.sin(Math.PI * Math.min(1, Math.max(0, p)));
  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden", borderRadius: radius }}>
      <Img src={pic(before)} style={{ ...fill, objectPosition: pos }} />
      <Img src={pic(after)} style={{ ...fill, objectPosition: pos, clipPath: `inset(0 ${(1 - p) * 100}% 0 0)` }} />
      {moving && (
        <>
          {/* the sweep: a soft band of light trailing the edge, on the after side */}
          <div style={{ position: "absolute", top: 0, bottom: 0, left: `${p * 100}%`, width: 260, marginLeft: -260, background: `linear-gradient(to right, rgba(255,255,255,0), rgba(255,250,240,${0.22 * glow}))`, mixBlendMode: "screen" }} />
          <div style={{ position: "absolute", top: 0, bottom: 0, left: `${p * 100}%`, width: 3, marginLeft: -1.5, background: "rgba(255,255,255,0.95)", boxShadow: `0 0 ${18 + 26 * glow}px rgba(255,255,255,${0.5 + 0.3 * glow})` }} />
        </>
      )}
    </div>
  );
}

/** "Before" / "After" tag on a photo. */
function Tag({ after, style }: { after: number; style: CSSProperties }) {
  const tag = (text: string, o: number) => (
    <span style={{ position: "absolute", left: 0, top: 0, opacity: o, padding: "7px 16px 9px", borderRadius: 999, background: "rgba(10,10,10,0.62)", color: PAPER, fontFamily: font, fontSize: 22, fontWeight: 560, letterSpacing: "-0.01em", whiteSpace: "nowrap", backdropFilter: "blur(8px)" }}>{text}</span>
  );
  return (
    <div style={{ position: "absolute", width: 120, height: 44, ...style }}>
      {tag("Before", 1 - after)}
      {tag("After", after)}
    </div>
  );
}

/**
 * Kokon's prompt box (the app's own .composer markup and styles), with the text typed in, a caret,
 * the attached images as the app shows them (`refs`), and Send pressed at `send`. `w` is its width
 * in app pixels; `scale` sizes it for the frame.
 */
function PromptBox({ text, from, to, send, refs, w, scale, origin = "0 0" }: { text: string; from: number; to: number; send: number; refs?: string[]; w: number; scale: number; origin?: string }) {
  const f = useCurrentFrame();
  const n = Math.round(text.length * prog(f, from, Math.max(1, to - from), (x) => x));
  const typing = f < send;
  const caretOn = typing && (f < to || Math.floor((f - to) / 12) % 2 === 0);
  const press = Math.max(0, 1 - Math.abs(f - send) / 4);
  const vt = { "--vt": (100 + f / FPS).toFixed(3) } as CSSProperties;
  return (
    <div className="kokon-window silk" style={{ width: w, background: "transparent", boxShadow: "none", overflow: "visible", borderRadius: 0, transform: `scale(${scale})`, transformOrigin: origin, ...vt }}>
      <div className="composer" style={{ maxWidth: "none", boxShadow: "0 0 0 1px rgba(255,255,255,0.09), 0 30px 70px -20px rgba(0,0,0,0.9)" }}>
        {refs && (
          <div className="target">
            <div className="refs" style={{ maxWidth: "none" }}>
              {refs.map((r, i) => (
                <div className="ref" key={r}>
                  <img src={pic(r)} alt="" />
                  <span className="num">{i + 1}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="prompt" style={{ color: "var(--text)", minHeight: 30, whiteSpace: "pre-wrap" }}>
          {text.slice(0, n)}
          <span style={{ display: "inline-block", width: 2, height: "1.1em", marginLeft: 2, verticalAlign: -3, borderRadius: 1, background: "var(--text)", opacity: caretOn ? 1 : 0 }} />
        </div>
        <div className="controls">
          <button className="pill">
            <Gauge size={15} />
            <span className="lbl">Quality</span>
          </button>
          <div className="spacer" />
          <button className="send" style={{ transform: `scale(${1 - press * 0.14})` }} aria-label="Generate">
            <ArrowUp size={18} strokeWidth={2.4} />
          </button>
        </div>
      </div>
    </div>
  );
}

/** How a piece enters: fun pops in on a spring, serious fades in softly. */
function useEnter(start: number, mood: Mood) {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  if (mood === "fun") {
    const s = spring({ frame: f - start, fps, config: { damping: 12, mass: 0.6, stiffness: 170 } });
    return { opacity: Math.min(1, s * 1.6), transform: `translateY(${(1 - s) * 30}px) scale(${0.9 + 0.1 * s})` } as CSSProperties;
  }
  const p = prog(f, start, 22, inOut);
  return { opacity: p, transform: `translateY(${(1 - p) * 12}px)`, filter: `blur(${(1 - p) * 6}px)` } as CSSProperties;
}

/** Fun words: bold, popping in one after another; `hl` (the end of the text) lands in a paper
 * pill with a small burst. */
function Pop({ text, hl, start, size, style }: { text: string; hl?: string; start: number; size: number; style?: CSSProperties }) {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const head = hl && text.endsWith(hl) ? text.slice(0, -hl.length).trim() : text;
  const words = head ? head.split(" ") : [];
  const items: { w: string; pill: boolean }[] = [...words.map((w) => ({ w, pill: false })), ...(hl && text.endsWith(hl) ? [{ w: hl, pill: true }] : [])];
  const sp = (i: number) => spring({ frame: f - start - i * 3, fps, config: { damping: 11, mass: 0.55, stiffness: 190 } });
  const last = items.length - 1;
  const burst = prog(f, start + last * 3 + 4, 10);
  const burstGone = prog(f, start + last * 3 + 16, 12, inOut);
  return (
    <div style={{ fontFamily: font, fontSize: size, fontWeight: 800, letterSpacing: "-0.045em", lineHeight: 1.08, color: PAPER, textShadow: "0 6px 30px rgba(0,0,0,0.5)", ...style }}>
      {items.map(({ w, pill }, i) => {
        const s = sp(i);
        const base: CSSProperties = { display: "inline-block", opacity: Math.min(1, s * 1.8), transform: `translateY(${(1 - s) * 0.5 * size}px) scale(${0.6 + 0.4 * s})`, transformOrigin: "50% 80%", marginRight: "0.22em" };
        if (!pill) return <span key={i} style={base}>{w}</span>;
        return (
          <span key={i} style={{ ...base, position: "relative", marginRight: 0 }}>
            <span style={{ display: "inline-block", padding: "0.02em 0.26em 0.1em", borderRadius: "0.24em", background: PAPER, color: "#0d0d0d", textShadow: "none", boxShadow: "0 14px 40px rgba(0,0,0,0.45)" }}>{w}</span>
            <svg width={size} height={size} viewBox="0 0 100 100" style={{ position: "absolute", right: -size * 0.62, top: -size * 0.5, overflow: "visible", opacity: 1 - burstGone }}>
              {[
                [20, 70, 48, 42],
                [44, 82, 84, 66],
                [12, 42, 22, 8],
              ].map(([x1, y1, x2, y2], k) => (
                <line key={k} x1={x1} y1={y1} x2={x1 + (x2 - x1) * burst} y2={y1 + (y2 - y1) * burst} stroke={PAPER} strokeWidth={7} strokeLinecap="round" />
              ))}
            </svg>
          </span>
        );
      })}
    </div>
  );
}

/** Serious words: lighter, fading in softly one after another. */
function Soft({ text, start, size, style, weight = 470 }: { text: string; start: number; size: number; style?: CSSProperties; weight?: number }) {
  const f = useCurrentFrame();
  return (
    <div style={{ fontFamily: font, fontSize: size, fontWeight: weight, letterSpacing: "-0.03em", lineHeight: 1.18, color: "rgba(247,246,242,0.94)", textShadow: "0 4px 30px rgba(0,0,0,0.6)", ...style }}>
      {text.split(" ").map((w, i) => {
        const p = prog(f, start + i * 4, 20, inOut);
        return (
          <span key={i} style={{ display: "inline-block", opacity: p, transform: `translateY(${(1 - p) * 12}px)`, filter: `blur(${(1 - p) * 7}px)`, marginRight: "0.24em" }}>
            {w}
          </span>
        );
      })}
    </div>
  );
}

/** A caption line in the beat's mood. */
const Line = ({ mood, text, hl, start, size, style }: { mood: Mood; text: string; hl?: string; start: number; size: number; style?: CSSProperties }) =>
  mood === "fun" ? <Pop text={text} hl={hl} start={start} size={size} style={style} /> : <Soft text={text} start={start} size={size * 0.7} style={style} />;

/** The second image of a multi-image edit: a small card with its number, as in the prompt box. */
function RefCard({ r, start, end, mood, style }: { r: Ref; start: number; end: number; mood: Mood; style: CSSProperties }) {
  const f = useCurrentFrame();
  const enter = useEnter(start, mood);
  const gone = prog(f, end, 10, inOut);
  const h = r.ar >= 1 ? 250 / r.ar : 300;
  const w = h * r.ar;
  return (
    <div style={{ position: "absolute", width: w, height: h, borderRadius: 18, overflow: "hidden", boxShadow: "0 0 0 3px rgba(247,246,242,0.92), 0 30px 70px rgba(0,0,0,0.6)", ...style, ...enter, opacity: (enter.opacity as number) * (1 - gone) }}>
      <Img src={pic(r.src)} style={fill} />
      <span style={{ position: "absolute", left: 10, top: 10, minWidth: 34, height: 34, borderRadius: 17, background: PAPER, color: "#0a0a0a", fontFamily: font, fontSize: 20, fontWeight: 650, display: "grid", placeItems: "center" }}>2</span>
    </div>
  );
}

/** A fixed circular detail of the change, crossfading as the wipe passes over it. */
function Loupe({ c, k, start, wipeP, style }: { c: Case; k: NonNullable<Case["loupe"]>; start: number; wipeP: number; style: CSSProperties }) {
  const enter = useEnter(start, "fun");
  const D = 330;
  const scale = D / (2 * k.r); // image width in px inside the loupe
  const ih = (scale * 1427) / 1920;
  const layer = (n: string, o: number) => (
    <Img src={pic(n)} style={{ position: "absolute", width: scale, height: ih, left: D / 2 - k.x * scale, top: D / 2 - k.y * ih, opacity: o, maxWidth: "none" }} />
  );
  const after = Math.min(1, Math.max(0, (wipeP - k.x) / 0.06));
  return (
    <div style={{ position: "absolute", width: D, height: D, ...style, ...enter }}>
      <div style={{ position: "absolute", inset: 0, borderRadius: "50%", overflow: "hidden", boxShadow: "0 0 0 4px rgba(247,246,242,0.95), 0 30px 70px rgba(0,0,0,0.6)" }}>
        {layer(c.before, 1)}
        {layer(c.after, after)}
      </div>
    </div>
  );
}

const Disclaimer = () => (
  <div style={{ position: "absolute", right: 34, bottom: 22, fontFamily: font, fontSize: 21, fontWeight: 450, color: "rgba(247,246,242,0.66)", letterSpacing: "-0.005em", textShadow: "0 1px 10px rgba(0,0,0,0.8)" }}>All images are examples made with Kokon.</div>
);

/** A dark wash at one edge so words stay readable on a photo. */
const Wash = ({ side, o }: { side: "top" | "bottom"; o: number }) => (
  <div style={{ position: "absolute", left: 0, right: 0, [side]: 0, height: 440, opacity: o, background: `linear-gradient(${side === "top" ? "to bottom" : "to top"}, rgba(0,0,0,0.82), rgba(0,0,0,0.5) 45%, rgba(0,0,0,0))` }} />
);

// ---------------------------------------------------------------------------------------------
// A photo beat
// ---------------------------------------------------------------------------------------------
function PhotoBeat({ id }: { id: string }) {
  const f = useCurrentFrame();
  const s = seg(id);
  const c = CASES[id];
  const mood = s.mood;
  const T = photoTimes(id);
  const p = prog(f, T.wipe, T.wipeEnd - T.wipe, inOut);
  const chipIn = useEnter(T.chip, mood);
  const refs = c.ref ? [c.before, c.ref.src] : undefined;
  const fun = mood === "fun";
  if (c.tall) {
    // Portrait: the photo left, the words right; the ground is the photo itself, blurred and dark.
    const H = 960;
    const W = (H * 1427) / 1920;
    const X = 120;
    // The prompt box (173 px high, 283 with attached images) under the question, the answer under it.
    const chipTop = c.setup ? 380 : 300;
    return (
      <AbsoluteFill style={{ background: "#050505" }}>
        <div style={{ position: "absolute", inset: -80, filter: `blur(70px) brightness(${fun ? 0.34 : 0.17}) saturate(${fun ? 1.3 : 0.9})` }}>
          <Img src={pic(c.before)} style={fill} />
          <Img src={pic(c.after)} style={{ ...fill, opacity: p }} />
        </div>
        <div style={{ position: "absolute", left: X, top: 60, width: W, height: H, borderRadius: 20, overflow: "hidden", boxShadow: "0 40px 120px rgba(0,0,0,0.65)" }}>
          <Wipe before={c.before} after={c.after} p={p} />
          <Tag after={prog(f, T.wipeEnd - 4, 8)} style={{ left: 22, top: 22 }} />
        </div>
        {c.ref && <RefCard r={c.ref} start={T.chip - 4} end={T.wipe} mood={mood} style={{ left: X + W - 190, top: 640, transform: "rotate(-3deg)" }} />}
        <div style={{ position: "absolute", left: 950, top: 0, width: 880, height: 1080 }}>
          {c.setup && <Line mood={mood} text={c.setup} start={3} size={84} style={{ position: "absolute", left: 0, top: 210, width: 880 }} />}
          <div style={{ position: "absolute", left: 0, top: chipTop, ...chipIn, opacity: (chipIn.opacity as number) * (1 - 0.2 * prog(f, T.send + 6, 12)) }}>
            <PromptBox text={c.prompt} from={T.from} to={T.to} send={T.send} refs={refs} w={460} scale={1.9} />
          </div>
          <Line mood={mood} text={c.punch} hl={c.hl} start={T.punch} size={112} style={{ position: "absolute", left: 0, top: chipTop + (refs ? 283 : 173) + 70, width: 880 }} />
        </div>
        <Disclaimer />
      </AbsoluteFill>
    );
  }
  // Landscape: the photo fills the frame; the question top left, the prompt box bottom centre, then
  // the answer bottom left as the box fades.
  const boxGone = prog(f, T.wipeEnd, 10, inOut);
  return (
    <AbsoluteFill style={{ background: "#050505" }}>
      <Wipe before={c.before} after={c.after} p={p} pos={c.pos} />
      {!fun && <div style={{ position: "absolute", inset: 0, background: "rgba(0,0,0,0.16)" }} />}
      <Wash side="top" o={c.setup ? 0.9 : 0} />
      <Wash side="bottom" o={0.55 + 0.4 * prog(f, T.punch - 6, 12)} />
      <Tag after={prog(f, T.wipeEnd - 4, 8)} style={{ right: 140, top: 44 }} />
      {c.setup && <Line mood={mood} text={c.setup} start={3} size={84} style={{ position: "absolute", left: 96, top: 70, width: 1300 }} />}
      {c.ref && <RefCard r={c.ref} start={T.chip - 4} end={T.wipeEnd} mood={mood} style={c.ref.ar >= 1 ? { right: 70, top: 120 } : { right: 90, top: 110 }} />}
      {c.loupe && <Loupe c={c} k={c.loupe} start={T.wipe - 10} wipeP={p} style={{ right: 70, top: 560 }} />}
      <div style={{ position: "absolute", left: 960 - 300, bottom: 58, ...chipIn, opacity: (chipIn.opacity as number) * (1 - boxGone) }}>
        <PromptBox text={c.prompt} from={T.from} to={T.to} send={T.send} refs={refs} w={600} scale={1.8} origin="50% 100%" />
      </div>
      <Line mood={mood} text={c.punch} hl={c.hl} start={T.punch} size={112} style={{ position: "absolute", left: 96, bottom: 96, width: 1500 }} />
      <Disclaimer />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// Why Kokon? The two reasons, each landing as the voice says it, over a still wall of results.
// ---------------------------------------------------------------------------------------------
const WALL = ["01-after", "17-after", "13-after", "04-after", "12-after", "16-after", "03-color", "15-after"];

function Reason({ Icon, word, sub, start }: { Icon: typeof LockKeyhole; word: string; sub: string; start: number }) {
  const enter = useEnter(start, "fun");
  return (
    <div style={{ width: 640, display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center" }}>
      <div style={{ ...enter, width: 120, height: 120, borderRadius: 60, background: PAPER, color: "#0d0d0d", display: "grid", placeItems: "center", boxShadow: "0 20px 60px rgba(0,0,0,0.5)" }}>
        <Icon size={64} strokeWidth={2.2} />
      </div>
      <Pop text={word} start={start + 3} size={104} style={{ marginTop: 34 }} />
      <Soft text={sub} start={start + 10} size={38} style={{ marginTop: 22, color: "rgba(247,246,242,0.86)" }} />
    </div>
  );
}

function Why() {
  return (
    <AbsoluteFill style={{ background: "#050505" }}>
      <div style={{ position: "absolute", inset: 0, display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gridTemplateRows: "repeat(2, 1fr)", filter: "brightness(0.2) saturate(0.9)" }}>
        {WALL.map((n) => (
          <Img key={n} src={pic(n)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        ))}
      </div>
      <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse 70% 65% at 50% 50%, rgba(5,5,5,0.55), rgba(5,5,5,0.9))" }} />
      <Pop text="Why Kokon?" start={sentenceFrame("why", 0)} size={132} style={{ position: "absolute", top: 120, left: 0, right: 0, textAlign: "center" }} />
      <div style={{ position: "absolute", top: 400, left: 0, right: 0, display: "flex", justifyContent: "center", gap: 120 }}>
        <Reason Icon={LockKeyhole} word="Private." sub="Your photos never leave your computer." start={wordFrame("why", "private")} />
        <Reason Icon={InfinityIcon} word="Unlimited." sub="No subscription, no daily limit." start={wordFrame("why", "unlimited")} />
      </div>
      <Disclaimer />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// The privacy line, alone on dark.
// ---------------------------------------------------------------------------------------------
function Message() {
  return (
    <AbsoluteFill style={{ background: "#030303", alignItems: "center", justifyContent: "center" }}>
      <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse 55% 45% at 50% 50%, rgba(247,246,242,0.045), transparent 70%)" }} />
      <div style={{ textAlign: "center", width: 1500 }}>
        <Soft text="Your family photos are yours." start={sentenceFrame("message", 0) - 4} size={74} weight={430} style={{ color: "rgba(247,246,242,0.8)" }} />
        <Soft text="They never leave your computer." start={sentenceFrame("message", 1) - 4} size={74} weight={640} style={{ marginTop: 30 }} />
      </div>
      <Disclaimer />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// Home: three edits cut on the beat.
// ---------------------------------------------------------------------------------------------
function Home() {
  const f = useCurrentFrame();
  const shot = MONTAGE.filter((b) => f >= at("home", b)).length - 1;
  const h = HOME[shot];
  const t0 = at("home", MONTAGE[shot]);
  const T = { from: t0 + 2, to: t0 + at("home", 0.75), send: t0 + at("home", 0.9), wipe: t0 + at("home", 1), wipeEnd: t0 + at("home", 1.75) };
  const p = prog(f, T.wipe, T.wipeEnd - T.wipe, inOut);
  const last = shot === HOME.length - 1;
  const punch = wordFrame("home", "before");
  const boxGone = last ? prog(f, T.wipeEnd, 10, inOut) : 0;
  const chipIn = useEnter(0, "fun");
  return (
    <AbsoluteFill style={{ background: "#050505" }}>
      <Wipe before={h.before} after={h.after} p={p} pos="50% 60%" />
      <Wash side="top" o={0.9} />
      <Wash side="bottom" o={0.55 + 0.4 * prog(f, punch - 6, 12)} />
      <Tag after={prog(f, T.wipeEnd - 4, 8)} style={{ right: 140, top: 44 }} />
      <Pop text="Redo your home" start={3} size={84} style={{ position: "absolute", left: 96, top: 70, width: 1300 }} />
      <div style={{ position: "absolute", left: 960 - 300, bottom: 58, ...chipIn, opacity: (chipIn.opacity as number) * (1 - boxGone) }}>
        <PromptBox text={h.prompt} from={T.from} to={T.to} send={T.send} w={600} scale={1.8} origin="50% 100%" />
      </div>
      <Pop text="before you spend a cent." hl="a cent." start={punch} size={84} style={{ position: "absolute", left: 96, top: 180, width: 1600 }} />
      <Disclaimer />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// End card.
// ---------------------------------------------------------------------------------------------
function End() {
  const f = useCurrentFrame();
  const n = segFrames("end");
  // Each line comes in as the voice says it.
  const a = prog(f, sentenceFrame("end", 0) - 6, 26);
  const b = prog(f, sentenceFrame("end", 1) - 4, 24);
  const c = prog(f, sentenceFrame("end", 2) - 4, 24);
  const d = prog(f, sentenceFrame("end", 3) - 4, 24);
  const fade = prog(f, n - 18, 18, inOut);
  const rise = (p: number): CSSProperties => ({ opacity: p, transform: `translateY(${(1 - p) * 14}px)`, filter: `blur(${(1 - p) * 6}px)` });
  return (
    <AbsoluteFill style={{ background: "#050505", fontFamily: font, alignItems: "center", justifyContent: "center", opacity: 1 - fade }}>
      <div style={{ position: "absolute", inset: 0, background: "radial-gradient(circle at 50% 40%, rgba(247,246,242,0.06) 0%, transparent 45%)" }} />
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginTop: -30 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 30, ...rise(a) }}>
          <Mark size={110} color={PAPER} />
          <span style={{ fontSize: 120, fontWeight: 620, color: PAPER, letterSpacing: "-0.045em", lineHeight: 1 }}>Kokon.</span>
        </div>
        <div style={{ marginTop: 34, fontSize: 56, fontWeight: 600, color: PAPER, letterSpacing: "-0.035em", ...rise(b) }}>Edit photos by describing the change.</div>
        <div style={{ marginTop: 26, fontSize: 34, fontWeight: 450, color: "rgba(247,246,242,0.8)", letterSpacing: "-0.015em", ...rise(c) }}>On your own computer. Free for personal use, no subscription, no daily limits.</div>
        <div style={{ marginTop: 60, display: "flex", alignItems: "center", gap: 22, fontSize: 31, color: "rgba(247,246,242,0.74)", ...rise(d) }}>
          <span>12 GB GPU (NVIDIA or AMD) or Apple Silicon Mac</span>
          <span>·</span>
          <span style={{ padding: "12px 30px 14px", borderRadius: 999, background: PAPER, color: "#0d0d0d", fontSize: 32, fontWeight: 650, letterSpacing: "-0.02em" }}>kokon.app</span>
        </div>
      </div>
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// Sound: the voice-over, the music under it (ducked while the voice speaks) and the effects, kept
// light so they never cover a word.
// ---------------------------------------------------------------------------------------------
function Sounds() {
  const out: ReactNode[] = [];
  let reveal = 0;
  for (const s of SEGS) {
    if (!s.word || s.id === "home") continue;
    const t = START[s.id];
    const T = photoTimes(s.id);
    const fun = s.mood === "fun";
    out.push(
      <Sfx key={`${s.id}-chip`} name="pop" at={t + T.chip} gain={fun ? 0.35 : 0.2} />,
      <Sfx key={`${s.id}-type`} name="typing" at={t + T.from} dur={T.to - T.from} gain={0.45} />,
      <Send key={`${s.id}-send`} at={t + T.send} />,
      <Sfx key={`${s.id}-swipe`} name="swipe" at={t + T.wipe - 2} gain={fun ? 0.8 : 0.5} />,
      fun ? <Sfx key={`${s.id}-punch`} name="pop" at={t + T.punch} gain={0.6} /> : <Sfx key={`${s.id}-punch`} name="reveal" at={t + T.punch - 4} variant={reveal++} gain={0.45} />,
    );
  }
  const home = START.home;
  MONTAGE.forEach((b, i) => {
    const t = home + at("home", b);
    out.push(<Sfx key={`home-type-${i}`} name="typing" at={t + 2} dur={at("home", 0.75) - 2} gain={0.35} />, <Sfx key={`home-click-${i}`} name="click" at={t + at("home", 0.9)} />, <Sfx key={`home-swipe-${i}`} name="swipe" at={t + at("home", 1) - 2} gain={0.7} />);
  });
  const why = START.why;
  out.push(<Sfx key="why-title" name="hit" at={why + sentenceFrame("why", 0)} gain={0.35} />, <Sfx key="why-priv" name="pop" at={why + wordFrame("why", "private") + 3} gain={0.6} />, <Sfx key="why-unl" name="pop" at={why + wordFrame("why", "unlimited") + 3} gain={0.6} />);
  out.push(<Sfx key="deep" name="deep" at={START.message + sentenceFrame("message", 1)} />);
  out.push(<Sfx key="end-hit" name="hit" at={START.end} gain={0.35} />);
  return <>{out}</>;
}

/** `music`: the music file in public/ (default: the house-4 variant; "music-usecases-house4.m4a" is
 * the same cut on the launch film's own house-4). */
export function UseCases({ music = MUSIC.src }: { music?: string }) {
  return (
    <AbsoluteFill style={{ background: "#050505" }}>
      {SEGS.map((s) => (
        <Sequence key={s.id} from={START[s.id]} durationInFrames={segFrames(s.id)} name={s.id}>
          {s.id === "why" ? <Why /> : s.id === "message" ? <Message /> : s.id === "home" ? <Home /> : s.id === "end" ? <End /> : <PhotoBeat id={s.id} />}
        </Sequence>
      ))}
      <Sounds />
      <Voice cues={CUES_USECASES} />
      <Music frames={USECASES_FRAMES} cues={CUES_USECASES} src={music} from={MUSIC.from} fadeOut={10} gain={musicGain} />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// YouTube thumbnail (1280×720): the tourists before/after, the line, the prompt box.
// ---------------------------------------------------------------------------------------------
export function UseCasesThumb() {
  const H = 640;
  const W = (H * 1427) / 1920;
  return (
    <AbsoluteFill style={{ background: "#050505", fontFamily: font }}>
      <div style={{ position: "absolute", inset: -60, filter: "blur(60px) brightness(0.3) saturate(1.3)" }}>
        <Img src={pic("01-after")} style={fill} />
      </div>
      <div style={{ position: "absolute", left: 40, top: 40, width: W, height: H, borderRadius: 18, overflow: "hidden", boxShadow: "0 30px 90px rgba(0,0,0,0.7)" }}>
        {/* before on the left, after on the right */}
        <Wipe before="01-after" after="01-before" p={0.5} />
        <div style={{ position: "absolute", top: 0, bottom: 0, left: "50%", width: 4, marginLeft: -2, background: PAPER, boxShadow: "0 0 24px rgba(255,255,255,0.7)" }} />
        <span style={{ position: "absolute", left: 14, bottom: 14, padding: "5px 12px 7px", borderRadius: 999, background: "rgba(10,10,10,0.7)", color: PAPER, fontSize: 20, fontWeight: 600 }}>Before</span>
        <span style={{ position: "absolute", right: 14, bottom: 14, padding: "5px 12px 7px", borderRadius: 999, background: "rgba(10,10,10,0.7)", color: PAPER, fontSize: 20, fontWeight: 600 }}>After</span>
      </div>
      <div style={{ position: "absolute", left: W + 90, top: 60, right: 40 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14, color: PAPER, fontSize: 40, fontWeight: 640, letterSpacing: "-0.04em" }}>
          <Mark size={44} color={PAPER} /> Kokon
        </div>
        <div style={{ marginTop: 36, color: PAPER, fontSize: 82, fontWeight: 800, letterSpacing: "-0.05em", lineHeight: 1.02 }}>
          Edit photos by{" "}
          <span style={{ display: "inline-block", padding: "0 0.2em 0.06em", borderRadius: "0.2em", background: PAPER, color: "#0d0d0d" }}>describing</span> the change.
        </div>
        <div style={{ marginTop: 44 }}>
          <PromptBox text="remove the tourists" from={-12} to={-11} send={-10} w={420} scale={1.4} />
        </div>
      </div>
    </AbsoluteFill>
  );
}

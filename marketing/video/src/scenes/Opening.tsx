// 0–4 s: the mark assembles (stem, then the two wings unfold), "Kokon", the promise.
// 4–9 s: a wall of real before/after pairs made by Kokon; the three hook lines.
import { AbsoluteFill, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { MARK_PATHS } from "../app/Mark";
import { inOut, out, prog } from "../app/motion";

const PAPER = "#F7F6F2";
const font = '"Inter Variable", Inter, sans-serif';

export function Intro() {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const sp = (delay: number, damping = 14) => spring({ frame: f - delay, fps, config: { damping, mass: 0.8, stiffness: 110 } });
  const stem = sp(4, 18);
  const upper = sp(16);
  const lower = sp(24);
  const shift = prog(f, 44, 26, inOut); // mark moves left, word appears
  const word = prog(f, 50, 26);
  const tag = prog(f, 66, 24);
  const exit = prog(f, 104, 16, inOut);
  const size = 150;
  return (
    <AbsoluteFill style={{ background: "#050505", alignItems: "center", justifyContent: "center", fontFamily: font }}>
      <div style={{ position: "absolute", inset: 0, background: `radial-gradient(circle at 50% 46%, rgba(247,246,242,${0.06 * stem}) 0%, transparent 45%)` }} />
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 34, opacity: 1 - exit, transform: `scale(${1 + exit * 0.06})`, filter: `blur(${exit * 6}px)` }}>
        <div style={{ display: "flex", alignItems: "center", gap: interpolate(shift, [0, 1], [0, 40]) }}>
          <svg width={size} height={size} viewBox="0 0 1024 1024" style={{ overflow: "visible" }}>
            <path d={MARK_PATHS.stem} fill={PAPER} style={{ transformOrigin: "175px 1000px", transform: `translateY(${(1 - stem) * 160}px) scaleY(${0.5 + stem * 0.5})`, opacity: stem }} />
            <path d={MARK_PATHS.upper} fill={PAPER} style={{ transformOrigin: "390px 549px", transform: `rotate(${(1 - upper) * -70}deg) scale(${upper})`, opacity: Math.min(1, upper * 2) }} />
            <path d={MARK_PATHS.lower} fill={PAPER} style={{ transformOrigin: "390px 587px", transform: `rotate(${(1 - lower) * 60}deg) scale(${lower})`, opacity: Math.min(1, lower * 2) }} />
          </svg>
          <div style={{ overflow: "hidden", width: interpolate(shift, [0, 1], [0, 500]), height: size + 20, display: "flex", alignItems: "center" }}>
            <span style={{ fontSize: 156, fontWeight: 620, color: PAPER, letterSpacing: "-0.045em", lineHeight: 1, transform: `translateX(${(1 - word) * -60}px)`, opacity: word, whiteSpace: "nowrap" }}>Kokon</span>
          </div>
        </div>
        <div style={{ fontSize: 40, fontWeight: 450, color: "rgba(247,246,242,0.78)", letterSpacing: "-0.02em", opacity: tag, transform: `translateY(${(1 - tag) * 18}px)`, filter: `blur(${(1 - tag) * 6}px)` }}>
          Edit photos by describing the change.
        </div>
      </div>
    </AbsoluteFill>
  );
}

const PAIRS: [string, string][] = [
  ["street", "street-removed"],
  ["cabin", "cabin-winter"],
  ["bakery", "bakery-text"],
  ["potter", "potter-watercolor"],
  ["room", "room-sofa"],
  ["lighthouse", "lighthouse-wide"],
  ["sneaker", "sneaker-ad"],
  ["street-removed", "street-cat"],
];

function Tile({ pair, flipAt, w, h }: { pair: [string, string]; flipAt: number; w: number; h: number }) {
  const f = useCurrentFrame();
  const t = prog(f, flipAt, 18, inOut);
  return (
    <div style={{ position: "relative", width: w, height: h, borderRadius: 18, overflow: "hidden", flex: "none", background: "#111", boxShadow: "0 20px 60px rgba(0,0,0,0.5)" }}>
      <Img src={staticFile(`img/${pair[0]}.jpg`)} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
      <Img src={staticFile(`img/${pair[1]}.jpg`)} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", clipPath: `inset(0 0 0 ${(1 - t) * 100}%)` }} />
      {t > 0 && t < 1 && <div style={{ position: "absolute", top: 0, bottom: 0, left: `${(1 - t) * 100}%`, width: 2, background: "#fff", boxShadow: "0 0 12px rgba(255,255,255,0.8)" }} />}
    </div>
  );
}

const LINES = [
  { text: "Describe the change.", from: 0, to: 46 },
  { text: "Your computer makes it.", from: 46, to: 92 },
  { text: "No cloud. No Python. No account.", from: 92, to: 150 },
];

export function Hook() {
  const f = useCurrentFrame();
  const enter = prog(f, 0, 30);
  const exit = prog(f, 136, 14, inOut);
  const rows = [0, 1, 2];
  return (
    <AbsoluteFill style={{ background: "#050505", overflow: "hidden", fontFamily: font }}>
      <div style={{ position: "absolute", inset: -200, perspective: 1800, opacity: enter * (1 - exit) }}>
        <div style={{ position: "absolute", left: "50%", top: "50%", transform: `translate(-50%, -50%) rotateX(22deg) rotateZ(-9deg) scale(${1.08 + f * 0.0008})`, display: "flex", flexDirection: "column", gap: 28 }}>
          {rows.map((r) => {
            const dir = r % 2 ? 1 : -1;
            const dx = dir * f * 2.2 + (r % 2 ? -700 : 0);
            return (
              <div key={r} style={{ display: "flex", gap: 28, transform: `translateX(${dx}px)` }}>
                {[...PAIRS.slice(r * 3), ...PAIRS, ...PAIRS.slice(0, r * 3)].map((p, i) => (
                  <Tile key={i} pair={p} w={460} h={300} flipAt={14 + ((i * 7 + r * 11) % 23) * 4} />
                ))}
              </div>
            );
          })}
        </div>
      </div>
      <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse 60% 50% at 50% 50%, rgba(0,0,0,0.78) 0%, rgba(0,0,0,0.45) 60%, rgba(0,0,0,0.2) 100%)" }} />
      {LINES.map((l) => {
        const a = prog(f, l.from, 16);
        const b = l.to < 150 ? prog(f, l.to - 8, 10, inOut) : exit;
        if (f < l.from || f > l.to + 2) return null;
        return (
          <div key={l.text} style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}>
            <div style={{ fontSize: 96, fontWeight: 660, color: "#f4f4f2", letterSpacing: "-0.045em", textShadow: "0 6px 40px rgba(0,0,0,0.7)", opacity: a * (1 - b), transform: `translateY(${(1 - a) * 30 - b * 20}px) scale(${0.98 + a * 0.02})`, filter: `blur(${(1 - a) * 10 + b * 8}px)` }}>
              {l.text}
            </div>
          </div>
        );
      })}
    </AbsoluteFill>
  );
}

void out;

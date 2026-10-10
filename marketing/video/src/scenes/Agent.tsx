// An AI agent in a terminal asks Kokon's MCP server for an image; the image arrives.
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Backdrop, Caption, inOut, prog, typed } from "../app/motion";
import { Send, Sfx } from "../app/sound";
import { AGENT, LENGTH } from "../app/timeline";

export const AGENT_FRAMES = LENGTH.agent;
const mono = '"Cascadia Code", "Cascadia Mono", Consolas, "SF Mono", monospace';
const ASK = "Make a hero image for my landing page: a blue morpho butterfly on a silk cocoon, 16:9";

function Line({ show, children, color = "#d6d6d6", indent = 0 }: { show: number; children: React.ReactNode; color?: string; indent?: number }) {
  const f = useCurrentFrame();
  const p = prog(f, show, 8);
  if (f < show) return null;
  return <div style={{ color, opacity: p, transform: `translateY(${(1 - p) * 6}px)`, paddingLeft: indent, whiteSpace: "pre-wrap" }}>{children}</div>;
}

export function Agent() {
  const f = useCurrentFrame();
  const T0 = 22;
  const CALL = AGENT.works; // "Kokon works with it too."
  const STEPS0 = CALL + 26;
  const SAVED = STEPS0 + 48;
  const REPLY = SAVED + 16;
  const step = Math.max(0, Math.min(8, Math.floor((f - STEPS0) / 6) + 1));
  const enter = prog(f, 0, 26);
  const card = prog(f, SAVED - 4, 30);
  const termShift = interpolate(card, [0, 1], [0, -330]);
  const exit = prog(f, AGENT_FRAMES - 14, 14, inOut);
  return (
    <AbsoluteFill style={{ opacity: 1 - exit }}>
      <Backdrop glow={0.8} />
      {/* Terminal */}
      <div
        style={{
          position: "absolute",
          left: 960 - 440 + termShift,
          top: 250,
          width: 880,
          height: 560,
          borderRadius: 14,
          background: "#0e0e0e",
          boxShadow: "0 0 0 1px rgba(255,255,255,0.09), 0 40px 120px rgba(0,0,0,0.7)",
          overflow: "hidden",
          opacity: enter,
          transform: `translateY(${(1 - enter) * 60}px)`,
          fontFamily: mono,
          fontSize: 19,
          lineHeight: 1.6,
        }}
      >
        <div style={{ height: 40, display: "flex", alignItems: "center", gap: 8, padding: "0 16px", background: "#171717", color: "#8a8a8a", fontSize: 14, fontFamily: '"Inter Variable", Inter, sans-serif' }}>
          <span style={{ width: 12, height: 12, borderRadius: 6, background: "#3a3a3a" }} />
          <span style={{ width: 12, height: 12, borderRadius: 6, background: "#3a3a3a" }} />
          <span style={{ width: 12, height: 12, borderRadius: 6, background: "#3a3a3a" }} />
          <span style={{ marginLeft: 10 }}>my-website — AI agent</span>
        </div>
        <div style={{ padding: "22px 28px", display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ color: "#f2f2f2" }}>
            <span style={{ color: "#7d7d7d" }}>&gt; </span>
            {typed(ASK, f, T0, 55)}
            {f < CALL && <span style={{ opacity: Math.floor(f / 8) % 2 ? 0 : 1 }}>▍</span>}
          </div>
          <Line show={CALL}>
            <span style={{ color: "#f7f6f2" }}>● </span>
            <b style={{ color: "#f7f6f2" }}>kokon</b>
            <span style={{ color: "#8a8a8a" }}> · generate_image</span>
          </Line>
          <Line show={CALL + 6} color="#9a9a9a" indent={26}>
            {'prompt: "a blue morpho butterfly resting on a silk cocoon…"\naspect_ratio: "16:9"   quality: "quality"'}
          </Line>
          <Line show={STEPS0} color="#9a9a9a" indent={26}>
            {f < SAVED ? `Generating on your GPU… ${step}/8` : "Generated on your GPU · 8/8"}
          </Line>
          <Line show={SAVED} color="#d6d6d6" indent={26}>
            <span style={{ color: "#7d7d7d" }}>⎿ </span>Saved public/hero.png · 2752×1536 · 39 s
          </Line>
          <Line show={REPLY} color="#f2f2f2">
            <span style={{ color: "#f7f6f2" }}>● </span>Done. It is the hero background of index.html now.
          </Line>
        </div>
      </div>
      {/* The image */}
      <div
        style={{
          position: "absolute",
          left: 1080,
          top: 320,
          width: 760,
          height: 425,
          borderRadius: 14,
          overflow: "hidden",
          opacity: card,
          transform: `translateX(${(1 - card) * 120}px) scale(${0.94 + card * 0.06}) rotate(${(1 - card) * 2}deg)`,
          boxShadow: "0 0 0 1px rgba(255,255,255,0.1), 0 50px 140px rgba(0,0,0,0.75)",
        }}
      >
        <Img src={staticFile("img/hero.jpg")} style={{ width: "100%", height: "100%", objectFit: "cover", transform: `scale(${1.08 - card * 0.08 + f * 0.0002})` }} />
        <div style={{ position: "absolute", left: 16, bottom: 14, padding: "4px 10px", borderRadius: 8, background: "rgba(0,0,0,0.55)", color: "#fff", fontSize: 14, fontFamily: '"Inter Variable", Inter, sans-serif' }}>hero.png</div>
      </div>
      <Caption title="Works with your AI agent." sub="Built-in MCP server and local HTTP API, for Claude Code, Cursor or any MCP client." start={10} end={AGENT_FRAMES} y={86} />
      <Send at={CALL} />
      <Sfx name="reveal" at={SAVED} variant={2} />
    </AbsoluteFill>
  );
}

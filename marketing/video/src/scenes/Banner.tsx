// YouTube channel banner, 2560×1440 (render: npx remotion still Banner out/youtube-banner.png).
// TVs show all of it, desktops a 2560×423 band through the middle, phones only the central
// 1546×423, so the name and the line sit in that centre and the images fill the band around it.
import { AbsoluteFill, Img, staticFile } from "remotion";
import { Mark } from "../app/Mark";

const PAPER = "#F7F6F2";
const font = '"Inter Variable", Inter, sans-serif';
const LEFT = ["cabin-winter", "bakery-text", "potter-watercolor", "room-sofa"];
const RIGHT = ["street-cat", "lighthouse-wide", "sneaker-ad", "hero"];

function Strip({ ids, side }: { ids: string[]; side: "left" | "right" }) {
  return (
    <div style={{ position: "absolute", top: 720 - 170, [side]: -60, display: "flex", gap: 22, flexDirection: side === "left" ? "row" : "row-reverse" }}>
      {ids.map((id, i) => (
        <div key={id} style={{ width: 300, height: 340, borderRadius: 22, overflow: "hidden", flex: "none", opacity: 0.95 - (3 - i) * 0.0, boxShadow: "0 30px 80px rgba(0,0,0,0.6)" }}>
          <Img src={staticFile(`img/${id}.jpg`)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </div>
      ))}
    </div>
  );
}

export function Banner() {
  return (
    <AbsoluteFill style={{ background: "#050505", fontFamily: font }}>
      <Strip ids={LEFT} side="left" />
      <Strip ids={RIGHT} side="right" />
      {/* Fade the images into the dark centre where the name sits. */}
      <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to right, rgba(5,5,5,0.15) 0%, rgba(5,5,5,0.55) 22%, #050505 34%, #050505 66%, rgba(5,5,5,0.55) 78%, rgba(5,5,5,0.15) 100%)" }} />
      <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse 30% 22% at 50% 50%, rgba(247,246,242,0.07), transparent 70%)" }} />
      <div style={{ position: "absolute", left: 0, right: 0, top: 720 - 140, height: 280, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 26 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 34 }}>
          <Mark size={118} color={PAPER} />
          <span style={{ fontSize: 132, fontWeight: 620, color: PAPER, letterSpacing: "-0.045em", lineHeight: 1 }}>Kokon</span>
        </div>
        <div style={{ fontSize: 42, fontWeight: 450, color: "rgba(247,246,242,0.78)", letterSpacing: "-0.02em" }}>Edit photos by describing the change. On your own GPU.</div>
      </div>
    </AbsoluteFill>
  );
}

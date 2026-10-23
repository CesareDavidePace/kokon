// The devices of the "Kokon 0.11" video, from the kokon.app "Other devices" scene
// (site/index.html#devices, site/assets/style.css): the computer with the graphics card, a MacBook
// and an iPhone (frames from Devices.css, MIT, in devices.css), joined by dotted links. On the site
// the fans and links run on CSS animations; here everything is a function of the frame, so the
// render is the same in any order: fans turn by an angle the scene integrates (`fanAngle`), links
// flow by the frame and draw in by a progress.
import type { CSSProperties, ReactNode } from "react";
import { Img, staticFile } from "remotion";
import { MARK_PATHS } from "./Mark";
import { FPS } from "./timeline";
import "./devices.css";

const font = '"Inter Variable", Inter, sans-serif';
const WARM = "#ffd68c";

/** Degrees a fan has turned by frame `f`, at `rps(frame)` turns a second. */
export function fanAngle(f: number, rps: (frame: number) => number): number {
  let a = 0;
  for (let i = 0; i < f; i++) a += (rps(i) * 360) / FPS;
  return a;
}

function Fan({ size, angle, style }: { size: number; angle: number; style?: CSSProperties }) {
  return (
    <span
      style={{
        display: "block",
        width: size,
        height: size,
        flex: "none",
        borderRadius: "50%",
        background: "radial-gradient(circle, #2a2a2e 0 16%, transparent 17%), repeating-conic-gradient(from 0deg, #3a3a40 0 14deg, #1a1a1d 14deg 40deg)",
        boxShadow: "inset 0 0 0 3px #232326, 0 0 0 4px #18181b",
        transform: `rotate(${angle}deg)`,
        ...style,
      }}
    />
  );
}

/** The computer with the graphics card (220×430). `on` 0–1: the power light; `glow` 0–1: the
 * GPU's warm light (the site's resting look is about 0.6); `angle`: the fans. */
export function Pc({ angle, glow = 0.6, on = 1, style }: { angle: number; glow?: number; on?: number; style?: CSSProperties }) {
  const g = glow;
  return (
    <div
      style={{
        position: "absolute",
        width: 220,
        height: 430,
        borderRadius: 20,
        background: "linear-gradient(160deg, #2a2a2e, #141416 60%)",
        boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.13), inset 0 1px 0 rgba(255,255,255,0.18), 0 50px 90px rgba(0,0,0,0.7)",
        ...style,
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: "14px 14px 70px",
          borderRadius: 12,
          overflow: "hidden",
          background: `radial-gradient(ellipse at 50% 62%, rgba(255,214,140,${0.37 * g}), transparent 60%), linear-gradient(135deg, #18181b, #0c0c0e)`,
          boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.1)",
        }}
      >
        <div style={{ position: "absolute", left: 14, right: 14, top: 14, height: 140, borderRadius: 8, background: "#141416", boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.06)" }}>
          <Fan size={64} angle={angle * 0.8} style={{ position: "absolute", left: 30, top: 40 }} />
          <div style={{ position: "absolute", right: 40, top: 22, width: 8, height: 96, borderRadius: 3, background: "linear-gradient(#2c2c30, #1f1f22)", boxShadow: "14px 0 0 #232326" }} />
        </div>
        <div
          style={{
            position: "absolute",
            left: 10,
            right: 10,
            top: 190,
            height: 76,
            borderRadius: 9,
            background: "linear-gradient(180deg, #26262a, #18181b)",
            boxShadow: `inset 0 0 0 1px rgba(255,255,255,0.12), 0 0 ${50 + 40 * g}px rgba(255,214,140,${0.47 * g})`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 9,
          }}
        >
          <Fan size={46} angle={angle} />
          <Fan size={46} angle={angle * 1.14 + 20} />
          <Fan size={46} angle={angle * 0.89 + 40} />
          <div style={{ position: "absolute", left: 12, right: 12, bottom: -1, height: 3, borderRadius: 3, background: `linear-gradient(90deg, transparent, ${WARM}, transparent)`, boxShadow: `0 0 ${18 + 12 * g}px ${WARM}`, opacity: Math.min(1, g * 1.6) }} />
        </div>
        <div style={{ position: "absolute", inset: 0, background: "linear-gradient(115deg, rgba(255,255,255,0.09) 0 22%, transparent 22% 60%, rgba(255,255,255,0.04) 60% 64%, transparent 64%)" }} />
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 70, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 22px" }}>
        <svg width={22} height={22} viewBox="0 0 1024 1024" style={{ display: "block" }}>
          {Object.values(MARK_PATHS).map((d, i) => (
            <path key={i} d={d} fill="rgba(242,241,237,0.55)" />
          ))}
        </svg>
        <i style={{ display: "block", width: 12, height: 12, borderRadius: "50%", background: on > 0 ? WARM : "#3a3a3a", opacity: 0.35 + 0.65 * on, boxShadow: `0 0 ${10 * on}px ${WARM}, 0 0 0 4px rgba(255,255,255,0.06)` }} />
      </div>
    </div>
  );
}

/** The phone's status bar (9:41, signal, Wi-Fi, battery), black on light screens. */
function StatusBar({ light }: { light: boolean }) {
  const c = light ? "#fff" : "#000";
  return (
    <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 54, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 30px 0 46px", color: c, font: `600 17px/1 -apple-system, "SF Pro Text", ${font}`, letterSpacing: "-0.01em", zIndex: 5 }}>
      <span>9:41</span>
      <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <svg viewBox="0 0 18 12" height={13} fill={c}>
          <rect x="0" y="8" width="3" height="4" rx="1" />
          <rect x="5" y="5.5" width="3" height="6.5" rx="1" />
          <rect x="10" y="3" width="3" height="9" rx="1" />
          <rect x="15" y="0" width="3" height="12" rx="1" />
        </svg>
        <svg viewBox="0 0 16 12" height={13} fill={c}>
          <path d="M8 2.6c2.3 0 4.4.9 6 2.4l1.3-1.4A10.4 10.4 0 0 0 8 .6 10.4 10.4 0 0 0 .7 3.6L2 5c1.6-1.5 3.7-2.4 6-2.4zm0 3.6c1.3 0 2.5.5 3.4 1.3l1.3-1.4A6.9 6.9 0 0 0 8 4.2c-1.8 0-3.5.7-4.7 1.9l1.3 1.4C5.5 6.7 6.7 6.2 8 6.2zm0 3.5c-.6 0-1.1.2-1.5.6L8 11.9l1.5-1.6c-.4-.4-.9-.6-1.5-.6z" />
        </svg>
        <svg viewBox="0 0 27 13" height={13}>
          <rect x="0.5" y="0.5" width="23" height="12" rx="3.5" fill="none" stroke={c} opacity="0.4" />
          <rect x="2" y="2" width="20" height="9" rx="2" fill={c} />
          <path d="M25 4.5v4c.8-.3 1.3-1.1 1.3-2s-.5-1.7-1.3-2z" fill={c} opacity="0.45" />
        </svg>
      </span>
    </div>
  );
}

/** Where a phone's content goes: 390×776 under the status bar. */
export const PHONE_SCREEN = { w: 390, h: 776 };

/** An iPhone 14 Pro (428×868 at scale 1) with the Dynamic Island, at (x, y) scaled from its top
 * left. `children` fill the screen under the status bar; `full` children fill the whole screen
 * (a camera, a home screen) and `dark` makes the status bar white for them. `off` (0–1) darkens the
 * whole screen: the phone before it opens Kokon. */
export function IPhone({ x, y, scale = 1, children, full, dark = false, off = 0, ground = "#f4f4f4", style }: { x: number; y: number; scale?: number; children?: ReactNode; full?: ReactNode; dark?: boolean; off?: number; ground?: string; style?: CSSProperties }) {
  return (
    <div className="device device-iphone-14-pro" style={{ position: "absolute", left: x, top: y, transform: `scale(${scale})`, transformOrigin: "top left", filter: "drop-shadow(0 40px 60px rgba(0,0,0,0.55))", ...style }}>
      <div className="device-frame">
        <div className="device-screen" style={{ background: ground, position: "relative" }}>
          {full}
          {children && <div style={{ position: "absolute", left: 0, top: 54, width: PHONE_SCREEN.w, height: PHONE_SCREEN.h, overflow: "hidden" }}>{children}</div>}
          <StatusBar light={dark} />
          <div style={{ position: "absolute", left: "50%", bottom: 8, width: 134, height: 5, marginLeft: -67, borderRadius: 3, background: dark ? "#fff" : "#111", zIndex: 6 }} />
          {off > 0 && <div style={{ position: "absolute", inset: 0, background: "#0b0b0c", opacity: off, zIndex: 7 }} />}
        </div>
      </div>
      <div className="device-stripe" />
      <div className="device-header" />
      <div className="device-sensors" />
      <div className="device-btns" />
      <div className="device-power" />
      <div className="device-home" />
    </div>
  );
}

/** A MacBook Pro (740×434 at scale 1): a browser bar with the address, the screen image under it. */
export function MacBook({ x, y, scale = 1, src, address, on = 1 }: { x: number; y: number; scale?: number; src: string; address: string; on?: number }) {
  return (
    <div className="device device-macbook-pro" style={{ position: "absolute", left: x, top: y, transform: `scale(${scale})`, transformOrigin: "top left", filter: "drop-shadow(0 40px 60px rgba(0,0,0,0.55))" }}>
      <div className="device-frame">
        <div className="device-screen" style={{ background: "#f4f4f4" }}>
          <div style={{ opacity: on, height: "100%" }}>
            <div style={{ height: 22, display: "flex", alignItems: "center", gap: 5, padding: "0 8px", background: "#e9e9eb", borderBottom: "1px solid #d6d6d8" }}>
              {["#ff5f57", "#febc2e", "#28c840"].map((c) => (
                <i key={c} style={{ display: "block", width: 7, height: 7, borderRadius: "50%", background: c }} />
              ))}
              <span style={{ marginLeft: 14, padding: "2px 22px 2px 10px", borderRadius: 5, background: "#fff", font: `500 8.5px/1.2 -apple-system, ${font}`, color: "#444", boxShadow: "0 0 0 0.5px #d0d0d2" }}>{address}</span>
            </div>
            <Img src={staticFile(src)} style={{ display: "block", width: "100%", height: "calc(100% - 22px)", objectFit: "cover", objectPosition: "top" }} />
          </div>
          <div style={{ position: "absolute", inset: 0, background: "#0b0b0c", opacity: 1 - on }} />
        </div>
      </div>
      <div className="device-stripe" />
      <div className="device-header" />
      <div className="device-sensors" />
      <div className="device-btns" />
      <div className="device-power" />
    </div>
  );
}

/** A cubic link: the path, and the point at `t` (0–1) along it, for a packet riding it. */
export type Curve = [number, number, number, number, number, number, number, number];
export const curvePath = (c: Curve) => `M${c[0]} ${c[1]} C ${c[2]} ${c[3]}, ${c[4]} ${c[5]}, ${c[6]} ${c[7]}`;
export function curveAt(c: Curve, t: number): [number, number] {
  const u = 1 - t;
  const a = u * u * u;
  const b = 3 * u * u * t;
  const d = 3 * u * t * t;
  const e = t * t * t;
  return [a * c[0] + b * c[2] + d * c[4] + e * c[6], a * c[1] + b * c[3] + d * c[5] + e * c[7]];
}

/** A dotted, glowing link that flows from its start to its end (by frame `f`), drawn in over `p`
 * (0–1); `hot` (0–1) brightens it while something travels on it. */
export function Link({ c, f, p = 1, hot = 0, width = 2.4, id }: { c: Curve; f: number; p?: number; hot?: number; width?: number; id: string }) {
  const d = curvePath(c);
  return (
    <g>
      <defs>
        <mask id={`draw-${id}`} maskUnits="userSpaceOnUse" x={-2000} y={-2000} width={6000} height={6000}>
          <path d={d} pathLength={1} fill="none" stroke="#fff" strokeWidth={40} strokeDasharray={`${p} 2`} />
        </mask>
      </defs>
      <g mask={`url(#draw-${id})`}>
        <path d={d} fill="none" stroke={`rgba(255,214,140,${0.08 + 0.1 * hot})`} strokeWidth={width * 6} strokeLinecap="round" />
        <path d={d} fill="none" stroke={`rgba(255,214,140,${0.55 + 0.4 * hot})`} strokeWidth={width} strokeLinecap="round" strokeDasharray="2 12" strokeDashoffset={-(f * 14) / (1.6 * FPS)} />
      </g>
    </g>
  );
}

/** A photo travelling along a link (`t` 0–1), a small rounded thumbnail with a warm glow. */
export function Packet({ c, t, src, size = 54 }: { c: Curve; t: number; src: string; size?: number }) {
  if (t <= 0 || t >= 1) return null;
  const [x, y] = curveAt(c, t);
  const s = Math.min(1, t * 6, (1 - t) * 6);
  return (
    <div style={{ position: "absolute", left: x - size / 2, top: y - size / 2, width: size, height: size, borderRadius: size * 0.22, overflow: "hidden", transform: `scale(${0.6 + 0.4 * s})`, opacity: s, boxShadow: `0 0 0 2px rgba(255,255,255,0.9), 0 0 28px rgba(255,214,140,0.8)` }}>
      <Img src={staticFile(src)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
    </div>
  );
}

/** The site's pill over the scene: "Home Wi-Fi · or Tailscale from anywhere". */
export function NetPill({ x, y, children, p = 1 }: { x: number; y: number; children: ReactNode; p?: number }) {
  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        padding: "9px 16px",
        borderRadius: 999,
        background: "rgba(20,20,20,0.85)",
        boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.14), 0 10px 30px rgba(0,0,0,0.5)",
        fontFamily: font,
        fontSize: 15,
        color: "rgba(242,241,237,0.66)",
        whiteSpace: "nowrap",
        opacity: p,
        transform: `translateY(${(1 - p) * 8}px)`,
      }}
    >
      {children}
    </div>
  );
}

/** The site's links, in its 1160×650 stage: computer → laptop, computer → phone. */
export const SCENE_LINKS: Record<"mac" | "phone", Curve> = {
  mac: [240, 330, 290, 330, 300, 362, 352, 362],
  phone: [130, 150, 230, 0, 760, 10, 884, 150],
};

/** The whole site scene (1160×650 stage): the computer, the MacBook (screenshot of the gallery in
 * a browser) and the iPhone (`phone`: the app itself), the links and the pill. */
export function HomeScene({ f, angle, glow, on = 1, links = 1, mac = 1, phone = 1, pill = 1, phoneScreen, hot = 0 }: { f: number; angle: number; glow: number; on?: number; links?: number; mac?: number; phone?: number; pill?: number; phoneScreen: ReactNode; hot?: number }) {
  return (
    <div style={{ position: "absolute", left: 0, top: 0, width: 1160, height: 650 }}>
      <svg width={1160} height={650} viewBox="0 0 1160 650" style={{ position: "absolute", inset: 0, overflow: "visible" }}>
        <Link id="mac" c={SCENE_LINKS.mac} f={f} p={links} hot={hot} />
        <Link id="phone" c={SCENE_LINKS.phone} f={f} p={links} hot={hot} />
      </svg>
      <Pc angle={angle} glow={glow} on={on} style={{ left: 20, top: 150 }} />
      <MacBook x={330} y={190} src="img/remote-laptop.webp" address="192.168.1.20:39152" on={mac} />
      <IPhone x={868} y={40} scale={0.66} off={1 - phone}>
        {phoneScreen}
      </IPhone>
      <NetPill x={400} y={22} p={pill}>
        <b style={{ color: "#f2f1ed", fontWeight: 600 }}>Home Wi-Fi</b> · or Tailscale from anywhere
      </NetPill>
    </div>
  );
}

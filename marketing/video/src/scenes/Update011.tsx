// "Kokon 0.11": the update video for Kokon on other devices (Reddit, then YouTube). Built on the
// kokon.app "Other devices" scene (app/Devices011.tsx): the computer with the graphics card makes
// the images, a phone or a laptop opens Kokon in a browser. The app on screen is the app itself:
// the desktop window's Settings → Other devices, and the browser UI in the phone layout
// (loaders/phone-container.cjs). Every shot is still and changes by a hard cut on the beat
// (app/timeline011.ts): no camera moves. The on-screen words carry every message, since Reddit
// plays muted.
import { Check, CloudOff, Compass, Lock, PlusSquare, Share, UserX, X } from "lucide-react";
import { useLayoutEffect, useMemo, type ComponentProps, type CSSProperties, type ReactNode } from "react";
import { AbsoluteFill, Img, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { siDocker, siLinux, siNvidia } from "simple-icons";
import { encode } from "uqr";
import { KokonMark } from "../../../../src/components/Welcome";
import { AppWindow } from "../app/AppWindow";
import { type Curve, fanAngle, HomeScene, IPhone, Link, NetPill, Packet, Pc, PHONE_SCREEN } from "../app/Devices011";
import { appState, gen, genItem, jobItem, PROJECTS, ref, settingsState } from "../app/fixtures";
import { Mark } from "../app/Mark";
import { Backdrop, Caption, inOut, prog, Pointer, typed, typingDone } from "../app/motion";
import { Music, Send, Sfx, Voice } from "../app/sound";
import { FPS } from "../app/timeline";
import { CUES_011, MUSIC, PROMPT_011, said, SHOT, UPDATE011_FRAMES, USE } from "../app/timeline011";
import { answer } from "../mocks/api";
import type { CardMotion } from "../mocks/card";
import type { State } from "../mocks/store";
import "../app/video011.css";

export { UPDATE011_FRAMES };
const font = '"Inter Variable", Inter, sans-serif';
const mono = '"JetBrains Mono", "Cascadia Code", Consolas, monospace';
const PAPER = "#F7F6F2";
const WARM = "#ffd68c";
/** The setup guide's home-network section: what the QR codes in the video open. */
const GUIDE = "https://kokon.app/self-hosted/#on-your-home-network";
const LAN = "http://192.168.1.20:39152";

/** Shows `children` from frame `from` (inclusive) to `to` (exclusive): one shot. */
const Shot = ({ from, to, children }: { from: number; to: number; children: ReactNode }) => {
  const f = useCurrentFrame();
  return f >= from && f < to ? <>{children}</> : null;
};

// ---------------------------------------------------------------------------------------------
// The project on the phone: a beach photo, a sunset version of it, and the edit that puts him on a
// flamingo (made from the photo).
// ---------------------------------------------------------------------------------------------
const BEACH = gen("beach", "generate", "A man on the beach, arms open");
const SUNSET = gen("beach-sunset", "edit", "Make it sunset, with golden light on the water", "beach");
const FLAMINGO = gen("beach-flamingo", "edit", PROMPT_011, "beach");
/** Beach day in the library: before the flamingo (two images, the sunset its cover), and after. */
const beachDay = (done: boolean) => ({ id: "current", title: "Beach day", updatedAt: Date.now() - (done ? 60_000 : 40 * 60_000), cover: done ? "gen:beach-flamingo" : "gen:beach-sunset", count: done ? 3 : 2 });
const BEACH_DAY = { projects: [beachDay(true), ...PROJECTS], projectId: "current", sidebar: false };
/** The phone's home: the projects the computer keeps, Beach day first. */
const LIBRARY = { projects: [beachDay(false), ...PROJECTS], projectId: "current", sidebar: true };
const resultState = (patch: Partial<State> = {}) =>
  appState({ ...BEACH_DAY, session: [genItem(BEACH), genItem(SUNSET), genItem(FLAMINGO)], selected: "beach-flamingo", composer: { ...appState().composer, refs: [ref("beach-flamingo")] }, ...patch });

/** Kokon on the phone (390×776 under the status bar), light as on the site: as added to the home
 * screen (`standalone`), or in Safari over its toolbar. */
const PhoneApp = ({ state, height = PHONE_SCREEN.h, standalone = true, className, ...rest }: { state: State; height?: number; standalone?: boolean } & Omit<ComponentProps<typeof AppWindow>, "state" | "width" | "height" | "phone" | "light">) => (
  <AppWindow state={state} width={PHONE_SCREEN.w} height={height} phone light standalone={standalone} className={["in-phone", className].filter(Boolean).join(" ")} {...rest} />
);

/** The computer's settings: Other devices, before and after the password (mocks/api.ts answers
 * `remote` with this frame's status; the window is mounted again when it changes). */
const REMOTE_OFF = { enabled: false, forced: false, port: 39152, password_set: false, listening: false, url: null, error: null, min_password_chars: 8 };
const REMOTE_ON = { ...REMOTE_OFF, enabled: true, password_set: true, listening: true, url: LAN };
function DevicesSettings({ on, width = 1440, height = 900 }: { on: boolean; width?: number; height?: number }) {
  answer("remote", on ? REMOTE_ON : REMOTE_OFF);
  return <AppWindow key={on ? "on" : "off"} state={settingsState({ settingsTab: "devices" })} width={width} height={height} light waitFor={on ? ".modal .qr" : ".modal form input"} />;
}

/** Types `chars` characters into the password field of Settings (its state is the component's own,
 * which a video can't type into): set on the element after each render, the button enabled. */
function TypePassword({ chars }: { chars: number }) {
  useLayoutEffect(() => {
    const input = document.querySelector<HTMLInputElement>('.modal form input[type="password"]');
    const btn = document.querySelector<HTMLButtonElement>(".modal form .btn.primary");
    if (input) input.value = "x".repeat(chars);
    if (btn) btn.disabled = chars < 8;
  });
  return null;
}

/** A QR code, black on white with a quiet border (as RemoteAccess.tsx draws it). */
function Qr({ text, size }: { text: string; size: number }) {
  const { d, n } = useMemo(() => {
    const qr = encode(text, { border: 2 });
    let p = "";
    qr.data.forEach((row, y) => row.forEach((on, x) => on && (p += `M${x} ${y}h1v1h-1z`)));
    return { d: p, n: qr.size };
  }, [text]);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${n} ${n}`} shapeRendering="crispEdges" style={{ display: "block" }}>
      <rect width={n} height={n} fill="#fff" />
      <path d={d} fill="#000" />
    </svg>
  );
}

/** A finger on the phone's screen: a soft dot from `from`, pressed at `at`, a ring after. */
function Touch({ x, y, at, from = at - 10 }: { x: number; y: number; at: number; from?: number }) {
  const f = useCurrentFrame();
  if (f < from || f > at + 16) return null;
  const inn = prog(f, from, 6);
  const out = 1 - prog(f, at + 8, 8);
  const press = 1 - Math.min(1, Math.abs(f - at) / 4) * 1;
  const ring = f - at;
  return (
    <div style={{ position: "absolute", left: x, top: y, zIndex: 50, pointerEvents: "none", opacity: Math.min(inn, out) }}>
      <div style={{ position: "absolute", left: -22, top: -22, width: 44, height: 44, borderRadius: "50%", background: "rgba(40,40,40,0.28)", boxShadow: "0 0 0 2px rgba(255,255,255,0.85)", transform: `scale(${1 - 0.18 * press})` }} />
      {ring >= 0 && <div style={{ position: "absolute", left: -22, top: -22, width: 44, height: 44, borderRadius: "50%", border: "2px solid rgba(40,40,40,0.5)", transform: `scale(${1 + ring / 8})`, opacity: 1 - ring / 16 }} />}
    </div>
  );
}

/** Step number and what it is, top left: "1  On the computer · Settings → Other devices". */
function Step({ n, title, sub, at }: { n: number; title: string; sub?: string; at: number }) {
  const f = useCurrentFrame();
  const p = prog(f, at, 16);
  return (
    <div style={{ position: "absolute", left: 120, top: 64, display: "flex", alignItems: "center", gap: 22, fontFamily: font, color: "#f4f4f2", opacity: p, transform: `translateY(${(1 - p) * 14}px)` }}>
      <span style={{ width: 64, height: 64, borderRadius: "50%", background: PAPER, color: "#0d0d0d", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 36, fontWeight: 700 }}>{n}</span>
      <div>
        <div style={{ fontSize: 46, fontWeight: 650, letterSpacing: "-0.035em", lineHeight: 1.05 }}>{title}</div>
        {sub && <div style={{ fontSize: 22, color: "rgba(244,244,242,0.62)", marginTop: 6 }}>{sub}</div>}
      </div>
    </div>
  );
}

const rise = (p: number): CSSProperties => ({ opacity: p, transform: `translateY(${(1 - p) * 22}px)`, filter: `blur(${(1 - p) * 8}px)` });

// ---------------------------------------------------------------------------------------------
// 1. Hook: Kokon on an iPhone, the beach photo turns into the flamingo; "Kokon 0.11".
// ---------------------------------------------------------------------------------------------
function Hook() {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const wipeAt = 30;
  const split = 1 - prog(f, wipeAt, 26, inOut);
  const state = resultState({ compare: f < wipeAt + 34 ? "slider" : "off" });
  const sp = (d: number) => spring({ frame: f - d, fps, config: { damping: 14, mass: 0.8, stiffness: 110 } });
  const mark = sp(4);
  const word = prog(f, 12, 18);
  const badge = spring({ frame: f - 22, fps, config: { damping: 12, mass: 0.6, stiffness: 160 } });
  const sub = prog(f, 34, 18);
  return (
    <AbsoluteFill>
      <Backdrop glow={0.8} />
      <IPhone x={330} y={52} scale={1.12}>
        <PhoneApp state={state} split={split} />
      </IPhone>
      <div style={{ position: "absolute", left: 960, top: 360, fontFamily: font, color: PAPER }}>
        <div style={{ display: "flex", alignItems: "center", gap: 28 }}>
          <div style={{ opacity: Math.min(1, mark * 1.5), transform: `scale(${0.7 + 0.3 * mark})` }}>
            <Mark size={110} color={PAPER} />
          </div>
          <span style={{ fontSize: 128, fontWeight: 620, letterSpacing: "-0.045em", lineHeight: 1, ...rise(word) }}>Kokon</span>
          <span style={{ padding: "8px 22px 12px", borderRadius: 999, border: `3px solid ${PAPER}`, fontSize: 64, fontWeight: 600, letterSpacing: "-0.03em", lineHeight: 1, opacity: Math.min(1, badge * 1.5), transform: `scale(${0.6 + 0.4 * badge})` }}>0.11</span>
        </div>
        <div style={{ marginTop: 34, fontSize: 52, fontWeight: 600, letterSpacing: "-0.03em", color: "#f4f4f2", ...rise(sub) }}>Now on your phone.</div>
      </div>
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// 2. The idea: the site's scene. The computer powers on, the links run to the laptop and the
// phone, the pill; then "No cloud · No account".
// ---------------------------------------------------------------------------------------------
const SCENE_AT = { x: 148, y: 30, s: 1.4 };
/** Seconds into line 2's second sentence where words start (Whisper, voice-0.11-words.json). */
const IDEA_WORDS = { phone: 0.26, laptop: 1.1, browser: 2.66 };

function Idea() {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s0 = said(2, 0);
  const s1 = said(2, 1);
  const s2 = said(2, 2);
  const w = (k: keyof typeof IDEA_WORDS) => s1 + Math.round(IDEA_WORDS[k] * FPS);
  const angle = fanAngle(f, (i) => (i < s0 ? 0 : 0.42 * prog(i, s0, 40, inOut)));
  const chip = (d: number) => spring({ frame: f - d, fps, config: { damping: 13, mass: 0.6, stiffness: 150 } });
  return (
    <AbsoluteFill>
      <Backdrop glow={0.6} />
      <div style={{ position: "absolute", left: SCENE_AT.x, top: SCENE_AT.y, transform: `scale(${SCENE_AT.s})`, transformOrigin: "top left" }}>
        <HomeScene
          f={f}
          angle={angle}
          on={prog(f, s0 + 2, 6)}
          glow={0.6 * prog(f, s0 + 4, 30)}
          links={prog(f, w("phone"), 26, inOut)}
          phone={prog(f, w("phone") + 6, 10)}
          mac={prog(f, w("laptop"), 10)}
          pill={prog(f, w("browser"), 14)}
          phoneScreen={<PhoneApp state={resultState()} />}
        />
      </div>
      <Caption title="Your computer with the graphics card makes the images." start={s0 + 2} end={s1 - 2} align="center" y={960} size={46} />
      <Caption title="Your phone, tablet or laptop opens Kokon in a browser." start={s1 + 2} end={s2 - 2} align="center" y={960} size={46} />
      <div style={{ position: "absolute", left: 0, right: 0, top: 952, display: "flex", justifyContent: "center", gap: 28, fontFamily: font }}>
        {[
          { icon: <CloudOff size={34} />, text: "No cloud", at: s2 },
          { icon: <UserX size={34} />, text: "No account", at: s2 + 14 },
        ].map(({ icon, text, at }) => {
          const p = chip(at);
          return (
            <div key={text} style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px 30px 14px 24px", borderRadius: 999, background: PAPER, color: "#0d0d0d", fontSize: 38, fontWeight: 650, letterSpacing: "-0.02em", opacity: f < at ? 0 : Math.min(1, p * 1.5), transform: `scale(${0.7 + 0.3 * p})` }}>
              {icon}
              {text}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// 3. Setup in three steps. Step 1: Settings → Other devices on the computer, a password, the QR.
// ---------------------------------------------------------------------------------------------
/** The settings window on screen, and where its password field and button are (window px,
 * measured with the Probe011 composition). */
const WIN_AT = { x: 132, y: 190, s: 1.15 };
const PW_INPUT: [number, number] = [750, 308];
const PW_BUTTON: [number, number] = [1014, 308];
const PW = (() => {
  const P = SHOT.password;
  return { enter: P + 30, field: P + 60, type: P + 64, typed: P + 88, button: P + 104, on: P + 106 };
})();

function Password() {
  const f = useCurrentFrame();
  const on = f >= PW.on;
  const chars = Math.max(0, Math.min(10, Math.floor((f - PW.type) / 2.4)));
  return (
    <AbsoluteFill>
      <Backdrop glow={0.7} />
      <div style={{ position: "absolute", left: WIN_AT.x, top: WIN_AT.y, transform: `scale(${WIN_AT.s})`, transformOrigin: "top left" }}>
        <DevicesSettings on={on} />
        {!on && <TypePassword chars={f >= PW.type ? chars : 0} />}
        <Pointer
          path={[
            [PW.enter, 1180, 640],
            [PW.field - 2, PW_INPUT[0], PW_INPUT[1]],
            [PW.typed + 4, PW_INPUT[0] + 40, PW_INPUT[1] + 30],
            [PW.button - 2, PW_BUTTON[0], PW_BUTTON[1]],
            [PW.button + 30, PW_BUTTON[0] + 60, PW_BUTTON[1] + 160],
          ]}
          clicks={[PW.field, PW.button]}
          hideBefore={PW.enter}
          hideAfter={PW.button + 30}
        />
      </div>
      <Step n={1} title="On the computer: Settings → Other devices" sub="Choose a password. Then the QR code appears." at={SHOT.password + 4} />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// Step 2: the phone's camera on the code (the QR you can scan in the video opens the setup
// guide), Safari's pill, the sign-in, Kokon in the browser.
// ---------------------------------------------------------------------------------------------
const PHONE_AT = { x: 746, y: 176, s: 1 };
/** Where the QR sits in the settings window (window px, Probe011), and how the camera sees it. */
const QR_IN_WIN = { x: 646, y: 415, size: 146 };
const CAM = { s: 1.7, cx: 195, cy: 360 };

/** The camera on the code: the phone shows only Safari's pill with the address, as it does. */
function Camera({ f, s }: { f: number; s: number }) {
  const pill = prog(f, s + 30, 10);
  const qr = QR_IN_WIN.size * CAM.s;
  return (
    <div style={{ position: "absolute", inset: 0, background: "#000", overflow: "hidden" }}>
      {/* the computer's screen, through the camera: soft and a little dark */}
      <div style={{ position: "absolute", left: CAM.cx - QR_IN_WIN.x * CAM.s, top: CAM.cy - QR_IN_WIN.y * CAM.s, transform: `scale(${CAM.s})`, transformOrigin: "top left", filter: "blur(2.5px) brightness(0.82)" }}>
        <DevicesSettings on />
      </div>
      <div style={{ position: "absolute", left: CAM.cx - qr / 2, top: CAM.cy - qr / 2, boxShadow: "0 0 0 6px rgba(255,255,255,0.5)" }}>
        <Qr text={GUIDE} size={qr} />
      </div>
      {/* Safari's suggestion for the code it read */}
      <div style={{ position: "absolute", left: 0, right: 0, top: 590, display: "flex", justifyContent: "center", opacity: pill, transform: `scale(${0.85 + 0.15 * pill})` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 20px", borderRadius: 999, background: "#ffd60a", color: "#000", font: `600 17px/1 -apple-system, ${font}` }}>
          <Compass size={18} strokeWidth={2.4} />
          Safari · 192.168.1.20
        </div>
      </div>
      {/* the camera's own controls */}
      <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 170, background: "rgba(0,0,0,0.92)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16 }}>
        <span style={{ font: `600 14px/1 -apple-system, ${font}`, letterSpacing: "0.12em", color: "#ffd60a" }}>PHOTO</span>
        <span style={{ width: 72, height: 72, borderRadius: "50%", background: "#fff", boxShadow: "0 0 0 4px #000, 0 0 0 7px #fff" }} />
      </div>
    </div>
  );
}

/** Safari's toolbar at the foot of the page (before Kokon is on the home screen). */
const SAFARI_BAR = 80;
function SafariBar() {
  return (
    <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: SAFARI_BAR, background: "rgba(246,246,246,0.96)", borderTop: "1px solid #ddd", display: "flex", justifyContent: "center", paddingTop: 10 }}>
      <div style={{ width: 340, height: 40, borderRadius: 12, background: "#e3e3e6", display: "flex", alignItems: "center", justifyContent: "center", font: `500 16px/1 -apple-system, ${font}`, color: "#111" }}>192.168.1.20</div>
    </div>
  );
}

/** The sign-in page kokon-service serves before the app: the markup of Login.tsx, typed by the frame. */
function LoginPage({ chars }: { chars: number }) {
  return (
    <div className="kokon-window kokon-light in-phone login-page" style={{ width: PHONE_SCREEN.w, height: PHONE_SCREEN.h - SAFARI_BAR }}>
      <form className="login" onSubmit={(e) => e.preventDefault()}>
        <KokonMark size={48} />
        <strong>Sign in to Kokon</strong>
        <span className="muted">Enter the password set on the computer running Kokon. This browser stays signed in.</span>
        <input className="input" type="password" placeholder="Password" readOnly value={"x".repeat(chars)} />
        <button className="btn primary" type="submit" disabled={chars === 0}>
          Sign in
        </button>
      </form>
    </div>
  );
}
const LOGIN_BUTTON: [number, number] = [195, 394];

function Scan() {
  const f = useCurrentFrame();
  const S = SHOT.scan;
  return (
    <AbsoluteFill>
      <Backdrop glow={0.7} />
      <IPhone x={PHONE_AT.x} y={PHONE_AT.y} scale={PHONE_AT.s} dark full={<Camera f={f} s={S} />}>
        <Touch x={195} y={556} at={S + 84} from={S + 70} />
      </IPhone>
      <Step n={2} title="On the phone: scan the code" sub="With the camera, like any QR code" at={S + 4} />
    </AbsoluteFill>
  );
}

const LOGIN = (() => {
  const L = SHOT.login;
  return { type: L + 14, tap: L + 52, open: L + 56 };
})();
function SignIn() {
  const f = useCurrentFrame();
  const chars = Math.max(0, Math.min(10, Math.floor((f - LOGIN.type) / 2.4)));
  return (
    <AbsoluteFill>
      <Backdrop glow={0.7} />
      <IPhone x={PHONE_AT.x} y={PHONE_AT.y} scale={PHONE_AT.s}>
        {f < LOGIN.open ? <LoginPage chars={chars} /> : <PhoneApp state={appState(LIBRARY)} height={PHONE_SCREEN.h - SAFARI_BAR} standalone={false} />}
        <SafariBar />
        <Touch x={LOGIN_BUTTON[0]} y={LOGIN_BUTTON[1]} at={LOGIN.tap} />
      </IPhone>
      <Step n={2} title="Sign in once" sub="This browser stays signed in" at={SHOT.login + 2} />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// Step 3: Share → Add to Home Screen, the Kokon icon on the home screen.
// ---------------------------------------------------------------------------------------------
const HOME = (() => {
  const H = SHOT.home;
  return { sheet: H + 4, tap: H + 26, screen: H + 40, icon: H + 46, open: H + 98 };
})();
/** The share sheet's rows: the one tapped, and plain ones around it. */
function ShareSheet({ p }: { p: number }) {
  const row = (icon: ReactNode, text: string, on = false) => (
    <div key={text} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "15px 18px", background: on ? "#e6e6ea" : "transparent", borderTop: "1px solid #e2e2e6", font: `400 17px/1 -apple-system, ${font}`, color: "#111" }}>
      {text}
      {icon}
    </div>
  );
  return (
    // Above the app: the composer's send button is sticky with its own z-index.
    <div style={{ position: "absolute", inset: 0, zIndex: 20, background: `rgba(0,0,0,${0.35 * p})` }}>
      <div style={{ position: "absolute", left: 8, right: 8, bottom: 8, borderRadius: 18, background: "#f7f7f8", overflow: "hidden", transform: `translateY(${(1 - p) * 420}px)` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "16px 18px" }}>
          <Img src={staticFile("img/kokon-icon.png")} style={{ width: 44, height: 44, borderRadius: 10 }} />
          <div style={{ font: `600 16px/1.3 -apple-system, ${font}`, color: "#111" }}>
            Kokon
            <div style={{ fontWeight: 400, fontSize: 14, color: "#777" }}>192.168.1.20</div>
          </div>
        </div>
        {row(<Share size={20} />, "Copy")}
        {row(<Share size={20} />, "Add to Reading List")}
        {row(<PlusSquare size={20} />, "Add to Home Screen", true)}
        {row(<Share size={20} />, "Find on Page")}
      </div>
    </div>
  );
}
/** Plain icons around Kokon's: no real apps, only shapes. */
const TINTS = ["#5b8def", "#f2a94e", "#58c48a", "#e46f6f", "#9b7be8", "#4fc1d6", "#d9d4c7", "#7f8794"];
function HomeScreen({ f }: { f: number }) {
  const icon = spring({ frame: f - HOME.icon, fps: FPS, config: { damping: 11, mass: 0.6, stiffness: 170 } });
  const cell = (i: number, kokon = false) => (
    <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 7 }}>
      {kokon ? (
        <Img src={staticFile("img/kokon-icon.png")} style={{ width: 62, height: 62, borderRadius: 14, opacity: Math.min(1, icon * 1.6), transform: `scale(${icon})`, boxShadow: "0 6px 16px rgba(0,0,0,0.35)" }} />
      ) : (
        <span style={{ width: 62, height: 62, borderRadius: 14, background: `linear-gradient(160deg, ${TINTS[i % TINTS.length]}, ${TINTS[(i + 3) % TINTS.length]})`, opacity: 0.55 }} />
      )}
      {kokon ? <span style={{ font: `500 12px/1 -apple-system, ${font}`, color: "#fff", opacity: icon }}>Kokon</span> : <span style={{ width: 40, height: 6, marginTop: 3, borderRadius: 3, background: "rgba(255,255,255,0.35)" }} />}
    </div>
  );
  return (
    <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse at 30% 20%, #4b3a2a 0%, #1c1a22 55%, #0d0d12 100%)" }}>
      <div style={{ position: "absolute", left: 26, right: 26, top: 84, display: "grid", gridTemplateColumns: "repeat(4, 1fr)", rowGap: 26 }}>
        {Array.from({ length: 15 }, (_, i) => cell(i, i === 13))}
      </div>
      <div style={{ position: "absolute", left: 14, right: 14, bottom: 22, height: 96, borderRadius: 30, background: "rgba(255,255,255,0.16)", display: "flex", alignItems: "center", justifyContent: "space-around" }}>
        {[0, 1, 2, 3].map((i) => (
          <span key={i} style={{ width: 62, height: 62, borderRadius: 14, background: `linear-gradient(160deg, ${TINTS[(i + 4) % 8]}, ${TINTS[(i + 6) % 8]})`, opacity: 0.55 }} />
        ))}
      </div>
    </div>
  );
}
/** Kokon's icon on the home screen (cell 13 of the grid: row 4, column 2), in screen px. */
const KOKON_ICON: [number, number] = [26 + (338 / 4) * 1.5, 84 + 3 * (62 + 7 + 9 + 26) + 31];

function AddHome() {
  const f = useCurrentFrame();
  const onHome = f >= HOME.screen;
  return (
    <AbsoluteFill>
      <Backdrop glow={0.7} />
      <IPhone x={PHONE_AT.x} y={PHONE_AT.y} scale={PHONE_AT.s} dark={onHome} full={onHome ? <HomeScreen f={f} /> : undefined}>
        {!onHome && (
          <>
            <PhoneApp state={appState(LIBRARY)} height={PHONE_SCREEN.h - SAFARI_BAR} standalone={false} />
            <SafariBar />
            <ShareSheet p={prog(f, HOME.sheet, 10)} />
            {/* the sheet's second row from the bottom: Add to Home Screen (rows 48 px, the sheet 8 px up) */}
            <Touch x={195} y={PHONE_SCREEN.h - 8 - 48 - 24} at={HOME.tap} />
          </>
        )}
      </IPhone>
      {onHome && (
        <div style={{ position: "absolute", left: PHONE_AT.x + 19, top: PHONE_AT.y + 19 }}>
          <Touch x={KOKON_ICON[0]} y={KOKON_ICON[1]} at={HOME.open} from={HOME.open - 12} />
        </div>
      )}
      <Step n={3} title="Add it to your home screen" sub="Share → Add to Home Screen, on iPhone or Android" at={SHOT.home + 2} />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// 4. Using it: the library, Beach day, a swipe back to the photo, the prompt written and sent. The
// photo runs along the link to the computer (its fans speed up, the GPU glows), the result runs
// back and develops into view, Download.
// ---------------------------------------------------------------------------------------------
const USE_PHONE = { x: 200, y: 96, s: 1.02 };
const USE_PC = { x: 1440, y: 250, s: 1.5 };
/** The link from the phone's side to the computer's (frame px). */
const USE_LINK: Curve = [660, 470, 900, 300, 1180, 330, 1428, 560];
/** Where the finger goes (phone px under the status bar, measured on stills): Beach day's card in
 * the library, the picture it drags, the closed prompt, Send over the keyboard, Save in the row
 * once it has slid, and how far the row slides. */
const CARD: [number, number] = [102, 214];
const PICTURE: [number, number] = [150, 300];
const PROMPT_CARD: [number, number] = [170, 690];
const SEND: [number, number] = [352, 440];
const SAVE: [number, number] = [281, 476];
const ROW_SLIDE = 150;
/** The keyboard's height over the screen's foot (the app shrinks above it, as on an iPhone). */
const KB_H = 300;

/** What the phone shows once Download is tapped: a banner with the photo, as for any download. */
function Downloaded({ f }: { f: number }) {
  const p = prog(f, USE.download + 4, 10);
  if (p <= 0) return null;
  return (
    <div style={{ position: "absolute", left: 10, right: 10, top: 6, zIndex: 40, display: "flex", alignItems: "center", gap: 12, padding: "10px 12px", borderRadius: 18, background: "rgba(250,250,250,0.97)", boxShadow: "0 10px 30px rgba(0,0,0,0.25), 0 0 0 1px rgba(0,0,0,0.06)", font: `400 15px/1.3 -apple-system, ${font}`, color: "#111", opacity: p, transform: `translateY(${(1 - p) * -40}px)` }}>
      <Img src={staticFile("thumb/beach-flamingo.jpg")} style={{ width: 40, height: 40, borderRadius: 9, objectFit: "cover" }} />
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 600 }}>Downloaded</div>
        <div style={{ color: "#666", fontSize: 13.5 }}>beach-flamingo.png</div>
      </div>
      <span style={{ width: 26, height: 26, borderRadius: "50%", background: "#1a7f37", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <Check size={17} strokeWidth={3} />
      </span>
    </div>
  );
}

/** The iPhone's keyboard (light), rising at `at`; the key of the letter just typed pressed. */
const KEY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];
function Keyboard({ f, at, text }: { f: number; at: number; text: string }) {
  const up = prog(f, at, 8, inOut);
  const last = text.slice(-1).toLowerCase();
  const face = (on: boolean, fn = false): CSSProperties => ({ height: 42, borderRadius: 6, background: on ? "#aeb3bc" : fn ? "#abb0ba" : "#fff", boxShadow: "0 1px 0 rgba(0,0,0,0.3)", display: "flex", alignItems: "center", justifyContent: "center", color: "#000" });
  const key = (c: string) => (
    <span key={c} style={{ ...face(c === last), width: 33, font: `400 23px/1 -apple-system, ${font}` }}>
      {c}
    </span>
  );
  const fn = (label: string, w: number) => <span style={{ ...face(false, true), width: w, font: `400 16px/1 -apple-system, ${font}` }}>{label}</span>;
  const row = (children: ReactNode, pad = 3) => <div style={{ display: "flex", justifyContent: "center", gap: 6, padding: `0 ${pad}px` }}>{children}</div>;
  const bar = <span style={{ width: 1, height: 24, background: "#b5b9c0" }} />;
  return (
    <div style={{ position: "absolute", left: 0, right: 0, top: PHONE_SCREEN.h - KB_H, height: KB_H + 60, zIndex: 30, background: "#d1d4da", transform: `translateY(${(1 - up) * (KB_H + 60)}px)`, display: "flex", flexDirection: "column", gap: 11, paddingTop: 8 }}>
      <div style={{ height: 36, display: "flex", alignItems: "center", justifyContent: "space-around", font: `400 16px/1 -apple-system, ${font}`, color: "#222" }}>
        <span>“wave”</span>
        {bar}
        <span>waves</span>
        {bar}
        <span>wavy</span>
      </div>
      {row(KEY_ROWS[0].split("").map(key))}
      {row(KEY_ROWS[1].split("").map(key), 22)}
      {row(
        <>
          {fn("⇧", 44)}
          <span style={{ width: 8 }} />
          {KEY_ROWS[2].split("").map(key)}
          <span style={{ width: 8 }} />
          {fn("⌫", 44)}
        </>,
      )}
      {row(
        <>
          {fn("123", 88)}
          <span style={{ ...face(last === " "), flex: 1, font: `400 16px/1 -apple-system, ${font}` }}>space</span>
          {fn("return", 88)}
        </>,
      )}
    </div>
  );
}

/** Slides the phone's actions row to `px` (the app's own scroll, which a video can't swipe). */
function ScrollRow({ px }: { px: number }) {
  useLayoutEffect(() => {
    const row = document.querySelector<HTMLElement>(".in-phone .phone-actions");
    if (row) row.scrollLeft = px;
  });
  return null;
}

/** The computer's work: 0 idle, 1 busy (fans fast, warm glow). */
const busy = (i: number) => prog(i, USE.job - 8, 20, inOut) * (1 - prog(i, USE.back, 30, inOut));

function UsePhone() {
  const f = useCurrentFrame();
  const composer = appState().composer;
  const before = [genItem(BEACH), genItem(SUNSET)];
  const opened = { ...LIBRARY, sidebar: false };
  // Send is pressed on the writing screen; it closes a moment later.
  const sent = USE.send + 4;
  const writing = f >= USE.prompt + 2 && f < sent;
  let state: State;
  let card: CardMotion | undefined;
  // The project slides in from the tap on (before it, the library has long been there).
  let vars: Record<string, string | number> = f >= USE.open ? { "--t-screen": ((f - USE.open) / FPS).toFixed(3) } : {};
  // Each phase is a window of its own: the composer keeps "open" in its own state once focused.
  const phase = f < USE.open ? "library" : f < USE.swiped ? "swipe" : writing ? "writing" : f < sent ? "photo" : f < USE.result ? "job" : "result";
  let page: string | undefined;
  const dragEnd = USE.swipe + USE.drag;
  if (f < USE.open) state = appState(LIBRARY);
  else if (f < USE.swiped) {
    // Beach day opens on its last version; the finger drags it right and lets go, it slides off.
    state = appState({ ...opened, session: before, selected: "beach-sunset", composer: { ...composer, refs: [ref("beach-sunset")] } });
    const drag = prog(f, USE.swipe, USE.drag, inOut);
    const away = prog(f, dragEnd, USE.swiped - dragEnd);
    const dx = f < dragEnd ? 120 * drag : 120 - 30 * away;
    const o = f < dragEnd ? 1 - dx / 600 : 0.8 * (1 - away);
    vars = { ...vars, "--swipe-dx": `${dx.toFixed(1)}px`, "--swipe-o": o.toFixed(3) };
  } else if (f < sent) {
    const prompt = f < USE.type ? "" : typed(PROMPT_011, f, USE.type, USE.cps);
    state = appState({ ...opened, session: before, selected: "beach", composer: { ...composer, refs: [ref("beach")], prompt }, ...(writing ? { writing: true } : {}) } as Partial<State>);
    vars = { ...vars, "--t-page": ((f - USE.swiped) / FPS).toFixed(3) };
    page = "page-enter-left";
  } else if (f < USE.result) {
    const j = jobItem("j-phone", "edit", PROMPT_011, "beach-flamingo", "beach", Math.min(8, Math.floor((f - USE.job) / 12)));
    state = appState({ ...opened, session: [...before, j.item], selected: "j-phone", jobs: { "j-phone": j.job }, composer: { ...composer, refs: [ref("beach")] } });
    card = { t: Math.max(0, (f - USE.send) / FPS), pctAt: (t) => Math.min(99, Math.max(0, (t - 0.7) / 3.2) * 100) };
  } else state = resultState();
  const develop = f >= USE.result ? { match: "img/beach-flamingo", t: (f - USE.result) / FPS } : null;
  const typedNow = f >= USE.type && f < sent ? typed(PROMPT_011, f, USE.type, USE.cps) : "";
  // The finger on the picture follows the drag.
  const swipeX = PICTURE[0] + 120 * prog(f, USE.swipe, USE.drag, inOut);
  const slide = prog(f, USE.row, 10, inOut);
  return (
    <>
      <PhoneApp key={phase} state={state} card={card} vars={vars} className={page} develop={develop} keyboard={writing} height={writing ? PHONE_SCREEN.h - KB_H : PHONE_SCREEN.h} />
      {f >= USE.result && <ScrollRow px={ROW_SLIDE * slide} />}
      {writing && <Keyboard f={f} at={USE.prompt + 2} text={typedNow} />}
      <Touch x={CARD[0]} y={CARD[1]} at={USE.open} />
      {f < USE.swiped && <Touch x={swipeX} y={PICTURE[1]} at={USE.swipe} from={USE.swipe - 6} />}
      <Touch x={PROMPT_CARD[0]} y={PROMPT_CARD[1]} at={USE.prompt} />
      <Touch x={SEND[0]} y={SEND[1]} at={USE.send} />
      {/* the row slid left by a finger, then Save */}
      {f < USE.row + 16 && <Touch x={SAVE[0] + ROW_SLIDE * (1 - slide)} y={SAVE[1]} at={USE.row} from={USE.row - 6} />}
      <Touch x={SAVE[0]} y={SAVE[1]} at={USE.download} />
      <Downloaded f={f} />
    </>
  );
}

function Use() {
  const f = useCurrentFrame();
  const angle = fanAngle(f, (i) => 0.42 + 1.5 * busy(i));
  const b = busy(f);
  const out = (f - USE.out) / 22;
  const back = (f - USE.back) / 22;
  const hot = Math.max(out > 0 && out < 1 ? 1 : 0, back > 0 && back < 1 ? 1 : 0, b * 0.5);
  return (
    <AbsoluteFill>
      <Backdrop glow={0.6} />
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0, overflow: "visible" }}>
        <Link id="use" c={USE_LINK} f={f} hot={hot} width={3} />
      </svg>
      <Pc angle={angle} glow={0.6 + 0.4 * b} style={{ left: 0, top: 0, transform: `translate(${USE_PC.x}px, ${USE_PC.y}px) scale(${USE_PC.s})`, transformOrigin: "top left" }} />
      <IPhone x={USE_PHONE.x} y={USE_PHONE.y} scale={USE_PHONE.s}>
        <UsePhone />
      </IPhone>
      <Packet c={USE_LINK} t={out} src="thumb/beach.jpg" size={70} />
      <Packet c={USE_LINK} t={1 - back} src="thumb/beach-flamingo.jpg" size={70} />
      <NetPill x={880} y={292}>
        <b style={{ color: "#f2f1ed", fontWeight: 600 }}>Home Wi-Fi</b>
      </NetPill>
      <Caption title="Pick a photo, describe the change." start={SHOT.use + 12} end={USE.out} x={760} y={92} size={52} maxWidth={1100} />
      <Caption title="Your PC does the work and sends it back." sub="It's saved in the computer's gallery. Download it on the phone." start={USE.out + 4} end={SHOT.safe} x={760} y={92} size={52} maxWidth={1100} />
      <div style={{ position: "absolute", left: USE_PC.x - 30, top: USE_PC.y + 430 * USE_PC.s + 26, width: 220 * USE_PC.s + 60, textAlign: "center", fontFamily: font, fontSize: 24, color: "rgba(244,244,242,0.7)", opacity: prog(f, USE.job - 8, 12) * (1 - prog(f, USE.back, 12)) }}>
        Making the image…
      </div>
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// 5. Safe by default: the padlock and what it means; away from home, a VPN, never an open port.
// ---------------------------------------------------------------------------------------------
function Padlock({ closed, size = 260 }: { closed: number; size?: number }) {
  const lift = (1 - closed) * 46;
  return (
    <svg width={size} height={size * 1.2} viewBox="0 0 200 240" style={{ overflow: "visible", filter: `drop-shadow(0 0 ${40 * closed}px rgba(255,214,140,${0.35 * closed}))` }}>
      <path d={`M 50 ${110 - lift} V ${70 - lift} a 50 50 0 0 1 100 0 V ${110 - lift}`} fill="none" stroke={PAPER} strokeWidth={22} strokeLinecap="round" transform={closed < 1 ? `rotate(${-(1 - closed) * 0} 150 110)` : undefined} />
      <rect x={20} y={104} width={160} height={124} rx={26} fill={PAPER} />
      <circle cx={100} cy={156} r={14} fill="#0d0d0d" />
      <rect x={93} y={160} width={14} height={34} rx={7} fill="#0d0d0d" />
    </svg>
  );
}

function Safe() {
  const f = useCurrentFrame();
  const S = SHOT.safe;
  const lockAt = S + 26;
  const closed = prog(f, lockAt - 6, 8, inOut);
  const lines = [
    { t: "Off until you set a password", s: "Nothing listens on the network before that.", at: S + 8 },
    { t: "5 wrong tries, and that device waits", s: "Each browser signs in once.", at: S + 38 },
    { t: "Plain HTTP on your home network", s: "Not encrypted: use a network you trust.", at: S + 62 },
  ];
  return (
    <AbsoluteFill style={{ fontFamily: font }}>
      <Backdrop glow={0.6} />
      <div style={{ position: "absolute", left: 300, top: 360 }}>
        <Padlock closed={closed} />
      </div>
      <div style={{ position: "absolute", left: 720, top: 300, display: "flex", flexDirection: "column", gap: 44, color: "#f4f4f2" }}>
        {lines.map((l) => {
          const p = prog(f, l.at, 18);
          return (
            <div key={l.t} style={rise(p)}>
              <div style={{ fontSize: 58, fontWeight: 650, letterSpacing: "-0.035em", lineHeight: 1.05 }}>{l.t}</div>
              <div style={{ fontSize: 26, color: "rgba(244,244,242,0.62)", marginTop: 10 }}>{l.s}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
}

/** A house outline round the computer at home. */
function House({ x, y, w, h }: { x: number; y: number; w: number; h: number }) {
  return (
    <svg width={w + 40} height={h + 40} viewBox={`-20 -20 ${w + 40} ${h + 40}`} style={{ position: "absolute", left: x - 20, top: y - 20, overflow: "visible" }}>
      <path d={`M 0 ${h * 0.34} L ${w / 2} 0 L ${w} ${h * 0.34} V ${h} H 0 Z`} fill="rgba(255,255,255,0.03)" stroke="rgba(244,244,242,0.4)" strokeWidth={3} strokeLinejoin="round" />
    </svg>
  );
}

/** Seconds into line 5's second sentence where words start (Whisper). */
const AWAY_WORDS = { tailscale: 1.16, never: 3.02 };
function Away() {
  const f = useCurrentFrame();
  const A = SHOT.away;
  const s1 = said(5, 1);
  const tunnel = prog(f, s1 + Math.round(AWAY_WORDS.tailscale * FPS) - 10, 22, inOut);
  const never = s1 + Math.round(AWAY_WORDS.never * FPS);
  const cross = prog(f, never, 10, inOut);
  const router = prog(f, never - 22, 14);
  const angle = fanAngle(f, () => 0.42);
  const y = 470;
  const x0 = 420;
  const x1 = 1360;
  const tx = x0 + (x1 - x0) * tunnel;
  return (
    <AbsoluteFill style={{ fontFamily: font, color: "#f4f4f2" }}>
      <Backdrop glow={0.6} />
      <div style={{ position: "absolute", left: 0, right: 0, top: 70, textAlign: "center", fontSize: 64, fontWeight: 660, letterSpacing: "-0.04em", ...rise(prog(f, A + 2, 16)) }}>Away from home</div>
      {/* the phone, out */}
      <IPhone x={250} y={262} scale={0.42}>
        <PhoneApp state={resultState()} />
      </IPhone>
      <div style={{ position: "absolute", left: 200, width: 280, top: 640, textAlign: "center", fontSize: 26, color: "rgba(244,244,242,0.7)" }}>Your phone, out</div>
      {/* the encrypted tunnel */}
      <div style={{ position: "absolute", left: x0, top: y - 26, width: tx - x0, height: 52, borderRadius: 26, background: "rgba(255,214,140,0.1)", boxShadow: `inset 0 0 0 2px rgba(255,214,140,0.7), 0 0 40px rgba(255,214,140,${0.25 * tunnel})` }} />
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        <Link id="tunnel" c={[x0 + 20, y, x0 + 300, y, x1 - 300, y, x1 - 20, y]} f={f} p={tunnel} hot={0.6} />
      </svg>
      <div style={{ position: "absolute", left: (x0 + x1) / 2 - 320, width: 640, top: y + 46, textAlign: "center", ...rise(prog(f, s1 + Math.round(AWAY_WORDS.tailscale * FPS), 16)) }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 12, fontSize: 36, fontWeight: 650, letterSpacing: "-0.02em" }}>
          <Lock size={30} />
          Tailscale or another VPN
        </div>
        <div style={{ fontSize: 24, color: "rgba(244,244,242,0.62)", marginTop: 6 }}>An encrypted link to your home</div>
      </div>
      {/* home */}
      <House x={1380} y={250} w={330} h={430} />
      <Pc angle={angle} glow={0.6} style={{ left: 0, top: 0, transform: "translate(1462px, 360px) scale(0.72)", transformOrigin: "top left" }} />
      <div style={{ position: "absolute", left: 1380, width: 330, top: 700, textAlign: "center", fontSize: 26, color: "rgba(244,244,242,0.7)" }}>Kokon at home</div>
      {/* never an open port on the router */}
      <div style={{ position: "absolute", left: 0, right: 0, top: 790, display: "flex", justifyContent: "center", ...rise(router) }}>
        <div style={{ position: "relative", display: "flex", alignItems: "center", gap: 18, padding: "16px 30px", borderRadius: 18, background: "rgba(255,255,255,0.05)", boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.12)" }}>
          <svg width={58} height={40} viewBox="0 0 58 40">
            <rect x={2} y={16} width={54} height={22} rx={6} fill="none" stroke="#f4f4f2" strokeWidth={3} />
            <path d="M14 16 L10 2 M44 16 L48 2" stroke="#f4f4f2" strokeWidth={3} strokeLinecap="round" />
            <circle cx={14} cy={27} r={3} fill="#f4f4f2" />
            <circle cx={24} cy={27} r={3} fill="#f4f4f2" />
          </svg>
          <span style={{ fontSize: 36, fontWeight: 600, letterSpacing: "-0.02em" }}>An open port on the router</span>
          <span style={{ position: "absolute", left: 92, right: 24, top: "50%", height: 4, marginTop: -2, borderRadius: 2, background: "#ff6b6b", transform: `scaleX(${cross})`, transformOrigin: "left" }} />
          <span style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 44, height: 44, borderRadius: "50%", background: "#ff6b6b", color: "#1a0d0d", opacity: cross, transform: `scale(${0.6 + 0.4 * cross})` }}>
            <X size={28} strokeWidth={3} />
          </span>
        </div>
      </div>
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// 6. A home server: the box, a terminal with the Docker commands (docs/docker.md), the service's own
// messages (crates/qls-service), the GPU lighting up when the container starts.
// ---------------------------------------------------------------------------------------------
type Line = { text: string; at: number; cmd?: boolean; cps?: number; dim?: boolean };
const SERVER = (() => {
  const V = SHOT.server;
  const oneFile = said(6, 2);
  const lines: Line[] = [
    { text: "curl -fsSLO https://raw.githubusercontent.com/GabrieleLozupone/kokon/master/docker/compose.yaml", at: V + 44, cmd: true, cps: 110 },
    { text: "docker compose up -d", at: oneFile + 18, cmd: true, cps: 40 },
    { text: " ✔ Container kokon  Started", at: 0, dim: true },
    { text: "docker exec -it kokon kokon-service --set-password", at: 0, cmd: true, cps: 60 },
    { text: "Password for other devices: ", at: 0 },
    { text: "Again: ", at: 0 },
    { text: "Password saved. Browsers sign in again; a running kokon-service uses it within a few seconds.", at: 0, dim: true },
    { text: "docker logs kokon", at: 0, cmd: true, cps: 40 },
    { text: "Kokon 0.11.0 is running.", at: 0 },
    { text: "  Open in a browser: http://localhost:39152 on the computer running Docker, or http://<its address>:39152 from other devices (password required)", at: 0 },
  ];
  const done = (l: Line) => (l.cmd ? typingDone(l.text, l.at, l.cps) : l.at);
  lines[2].at = done(lines[1]) + 10;
  lines[3].at = lines[2].at + 14;
  lines[4].at = done(lines[3]) + 8;
  lines[5].at = lines[4].at + 14;
  lines[6].at = lines[5].at + 12;
  lines[7].at = lines[6].at + 16;
  lines[8].at = done(lines[7]) + 8;
  lines[9].at = lines[8].at + 4;
  return { lines, started: lines[2].at, done };
})();

function Terminal({ f }: { f: number }) {
  const shown = SERVER.lines.filter((l) => f >= l.at);
  const typing = shown.find((l) => l.cmd && f < SERVER.done(l));
  const badge = (path: string, color: string, text: string) => (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 9, padding: "6px 14px 6px 10px", borderRadius: 999, background: "rgba(255,255,255,0.07)", boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.1)", fontFamily: font, fontSize: 18, fontWeight: 600, color: "#e9e9e7" }}>
      <svg width={22} height={22} viewBox="0 0 24 24">
        <path d={path} fill={color} />
      </svg>
      {text}
    </span>
  );
  return (
    <div style={{ position: "absolute", left: 640, top: 250, width: 1140, height: 600, borderRadius: 18, background: "#0c0c0e", boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.12), 0 50px 120px rgba(0,0,0,0.6)", overflow: "hidden" }}>
      <div style={{ height: 56, display: "flex", alignItems: "center", gap: 9, padding: "0 18px", background: "#18181b", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
        {["#ff5f57", "#febc2e", "#28c840"].map((c) => (
          <i key={c} style={{ display: "block", width: 13, height: 13, borderRadius: "50%", background: c }} />
        ))}
        <span style={{ marginLeft: 16, fontFamily: font, fontSize: 18, color: "rgba(233,233,231,0.6)", flex: 1 }}>you@home-server: ~/kokon</span>
        {badge(siDocker.path, `#${siDocker.hex}`, "Docker")}
        {badge(siNvidia.path, `#${siNvidia.hex}`, "NVIDIA · CUDA")}
      </div>
      <div style={{ padding: "22px 28px", fontFamily: mono, fontSize: 18.5, lineHeight: 1.6, color: "#e9e9e7", whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
        {shown.map((l, i) => (
          <div key={i} style={{ color: l.dim ? "rgba(233,233,231,0.58)" : undefined }}>
            {l.cmd && <span style={{ color: WARM }}>$ </span>}
            {l.cmd ? typed(l.text, f, l.at, l.cps) : l.text}
            {l === typing && <span style={{ display: "inline-block", width: 10, height: 21, marginLeft: 2, verticalAlign: "-4px", background: "#e9e9e7" }} />}
          </div>
        ))}
        {!typing && Math.floor(f / 16) % 2 === 0 && (
          <div>
            <span style={{ color: WARM }}>$ </span>
            <span style={{ display: "inline-block", width: 10, height: 21, verticalAlign: "-4px", background: "#e9e9e7" }} />
          </div>
        )}
      </div>
    </div>
  );
}

function Server() {
  const f = useCurrentFrame();
  const V = SHOT.server;
  const up = (i: number) => prog(i, SERVER.started, 30, inOut);
  const angle = fanAngle(f, (i) => 0.12 + 0.5 * up(i));
  return (
    <AbsoluteFill style={{ fontFamily: font, color: "#f4f4f2" }}>
      <Backdrop glow={0.5} />
      <div style={{ position: "absolute", left: 140, top: 66, ...rise(prog(f, V + 4, 16)) }}>
        <div style={{ fontSize: 64, fontWeight: 660, letterSpacing: "-0.04em" }}>Have a home server? Run Kokon on it.</div>
        <div style={{ display: "flex", gap: 14, marginTop: 18, fontSize: 24, color: "rgba(244,244,242,0.72)" }}>
          {[
            [siLinux.path, "Linux: kokon-service"],
            [siDocker.path, "Docker: NVIDIA cards (CUDA)"],
          ].map(([p, t]) => (
            <span key={t} style={{ display: "inline-flex", alignItems: "center", gap: 10, padding: "8px 18px 8px 12px", borderRadius: 999, background: "rgba(255,255,255,0.06)", boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.1)" }}>
              <svg width={22} height={22} viewBox="0 0 24 24">
                <path d={p} fill="rgba(244,244,242,0.85)" />
              </svg>
              {t}
            </span>
          ))}
        </div>
      </div>
      <Pc angle={angle} glow={0.12 + 0.88 * up(f)} on={1} style={{ left: 0, top: 0, transform: "translate(200px, 300px) scale(1.2)", transformOrigin: "top left" }} />
      <div style={{ position: "absolute", left: 160, width: 344, top: 840, textAlign: "center", fontSize: 24, color: "rgba(244,244,242,0.62)" }}>Your home server</div>
      <Terminal f={f} />
    </AbsoluteFill>
  );
}

// ---------------------------------------------------------------------------------------------
// 7. Close: the whole scene, "Your GPU, on every screen."; the end card with the guide's QR.
// ---------------------------------------------------------------------------------------------
function Close() {
  const f = useCurrentFrame();
  const angle = fanAngle(f, () => 0.42);
  return (
    <AbsoluteFill>
      <Backdrop glow={0.6} />
      <div style={{ position: "absolute", left: 206, top: 40, transform: "scale(1.3)", transformOrigin: "top left" }}>
        <HomeScene f={f} angle={angle} glow={0.7} phoneScreen={<PhoneApp state={resultState()} />} />
      </div>
      <Caption title="Kokon 0.11" start={said(7, 0) - 2} end={said(7, 1) - 2} align="center" y={930} size={60} />
      <Caption title="Your GPU, on every screen." start={said(7, 1)} end={SHOT.end} align="center" y={930} size={60} />
    </AbsoluteFill>
  );
}

function End() {
  const f = useCurrentFrame();
  const E = SHOT.end;
  const a = prog(f, E, 26);
  const b = prog(f, E + 12, 22);
  const qr = prog(f, E + 18, 24);
  const free = prog(f, E + 30, 22);
  const fade = prog(f, UPDATE011_FRAMES - 20, 20, inOut);
  return (
    <AbsoluteFill style={{ fontFamily: font, background: "#050505", opacity: 1 - fade }}>
      <div style={{ position: "absolute", inset: 0, background: "radial-gradient(circle at 34% 44%, rgba(247,246,242,0.06) 0%, transparent 45%)" }} />
      <div style={{ position: "absolute", left: 180, top: 290, color: PAPER }}>
        <div style={{ display: "flex", alignItems: "center", gap: 30, ...rise(a) }}>
          <Mark size={104} color={PAPER} />
          <span style={{ fontSize: 116, fontWeight: 620, letterSpacing: "-0.045em", lineHeight: 1 }}>Kokon</span>
          <span style={{ padding: "8px 20px 11px", borderRadius: 999, border: `3px solid ${PAPER}`, fontSize: 56, fontWeight: 600, letterSpacing: "-0.03em", lineHeight: 1 }}>0.11</span>
        </div>
        <div style={{ marginTop: 40, fontSize: 40, color: "rgba(247,246,242,0.85)", letterSpacing: "-0.02em", ...rise(b) }}>Your GPU, on every screen.</div>
        <div style={{ marginTop: 14, fontSize: 28, color: "rgba(247,246,242,0.6)", ...rise(b) }}>On a computer with a 12 GB GPU or Apple Silicon</div>
        <div style={{ marginTop: 40, display: "inline-flex", alignItems: "center", padding: "16px 34px", borderRadius: 999, background: PAPER, color: "#0d0d0d", fontSize: 34, fontWeight: 650, letterSpacing: "-0.02em", ...rise(free) }}>Free and open source · kokon.app</div>
      </div>
      <div style={{ position: "absolute", left: 1240, top: 230, width: 500, padding: 34, borderRadius: 30, background: "#fff", color: "#0d0d0d", display: "flex", flexDirection: "column", alignItems: "center", gap: 18, boxShadow: "0 40px 100px rgba(0,0,0,0.5)", ...rise(qr) }}>
        <Qr text={GUIDE} size={360} />
        <div style={{ fontSize: 32, fontWeight: 650, letterSpacing: "-0.02em" }}>Scan for the setup guide</div>
        <div style={{ fontSize: 24, color: "#555", marginTop: -8 }}>kokon.app/self-hosted</div>
      </div>
    </AbsoluteFill>
  );
}

/** Every effect, on the main timeline (the shots unmount when they end, a sound must not). */
function Sounds() {
  const s0 = said(2, 0);
  const typeEnd = typingDone(PROMPT_011, USE.type, USE.cps);
  return (
    <>
      <Sfx name="slider" at={30} gain={0.8} />
      <Sfx name="hit" at={22} gain={0.45} />
      <Sfx name="power" at={s0 + 2} />
      <Sfx name="pop" at={said(2, 2)} />
      <Sfx name="pop" at={said(2, 2) + 14} gain={0.8} />
      {/* step 1 */}
      <Sfx name="click" at={PW.field} />
      <Sfx name="typing" at={PW.type} dur={24} />
      <Sfx name="click" at={PW.button} />
      <Sfx name="reveal" at={PW.on + 2} variant={0} gain={0.7} />
      {/* step 2 */}
      <Sfx name="click" at={SHOT.scan + 84} gain={0.7} />
      <Sfx name="typing" at={LOGIN.type} dur={24} gain={0.7} />
      <Sfx name="click" at={LOGIN.tap} gain={0.7} />
      {/* step 3 */}
      <Sfx name="click" at={HOME.tap} gain={0.7} />
      <Sfx name="pop" at={HOME.icon} />
      <Sfx name="click" at={HOME.open} gain={0.7} />
      {/* using it */}
      <Sfx name="click" at={USE.open} gain={0.7} />
      <Sfx name="swipe" at={USE.swipe + USE.drag} gain={0.5} />
      <Sfx name="click" at={USE.prompt} gain={0.7} />
      <Sfx name="typing" at={USE.type} dur={typeEnd - USE.type} gain={0.8} />
      <Send at={USE.send} />
      <Sfx name="travel" at={USE.out} />
      <Sfx name="spinup" at={USE.job - 10} />
      <Sfx name="travel" at={USE.back} />
      <Sfx name="reveal" at={USE.result} variant={1} />
      <Sfx name="click" at={USE.download} gain={0.7} />
      <Sfx name="pop" at={USE.download + 4} gain={0.7} />
      {/* safe by default */}
      <Sfx name="lock" at={SHOT.safe + 24} />
      <Sfx name="whoosh" at={SHOT.away} gain={0.5} />
      {/* server */}
      {SERVER.lines.filter((l) => l.cmd).map((l) => (
        <Sfx key={l.text} name="typing" at={l.at} dur={SERVER.done(l) - l.at} gain={0.7} />
      ))}
      <Sfx name="spinup" at={SERVER.started - 4} gain={0.8} />
      {/* close */}
      <Sfx name="hit" at={SHOT.end} gain={0.4} />
    </>
  );
}

/** The music stays at one level all through (a background that never changes), low enough
 * under the voice without ducking: about the level the ducked music had under the 0.10 voice. */
const MUSIC_GAIN = 0.45;

export function Update011() {
  return (
    <AbsoluteFill style={{ background: "#050505" }}>
      <Shot from={SHOT.hook} to={SHOT.idea}>
        <Hook />
      </Shot>
      <Shot from={SHOT.idea} to={SHOT.password}>
        <Idea />
      </Shot>
      <Shot from={SHOT.password} to={SHOT.scan}>
        <Password />
      </Shot>
      <Shot from={SHOT.scan} to={SHOT.login}>
        <Scan />
      </Shot>
      <Shot from={SHOT.login} to={SHOT.home}>
        <SignIn />
      </Shot>
      <Shot from={SHOT.home} to={SHOT.use}>
        <AddHome />
      </Shot>
      <Shot from={SHOT.use} to={SHOT.safe}>
        <Use />
      </Shot>
      <Shot from={SHOT.safe} to={SHOT.away}>
        <Safe />
      </Shot>
      <Shot from={SHOT.away} to={SHOT.server}>
        <Away />
      </Shot>
      <Shot from={SHOT.server} to={SHOT.scene}>
        <Server />
      </Shot>
      <Shot from={SHOT.scene} to={SHOT.end}>
        <Close />
      </Shot>
      <Shot from={SHOT.end} to={UPDATE011_FRAMES}>
        <End />
      </Shot>
      <Sounds />
      <Voice cues={CUES_011} />
      <Music frames={UPDATE011_FRAMES} cues={[]} gain={() => MUSIC_GAIN} src={MUSIC.src} from={MUSIC.from} fadeOut={MUSIC.fade} />
    </AbsoluteFill>
  );
}

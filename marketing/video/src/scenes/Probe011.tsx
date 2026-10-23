// Measuring composition (not in the film): the positions of the buttons the scenes tap or click,
// printed to the render log.
import { useEffect } from "react";
import { AbsoluteFill } from "remotion";
import { AppWindow } from "../app/AppWindow";
import { appState, gen, genItem, ref, settingsState } from "../app/fixtures";
import { answer } from "../mocks/api";

const BEACH = gen("beach", "generate", "A man on the beach");
const FLAMINGO = gen("beach-flamingo", "edit", "Put him on a pink flamingo, surfing a wave", "beach");
const P = { projects: [{ id: "current", title: "Beach day", updatedAt: Date.now(), cover: "gen:beach-flamingo", count: 2 }], projectId: "current", sidebar: false };
answer("remote", { enabled: true, forced: false, port: 39152, password_set: true, listening: true, url: "http://192.168.1.20:39152", error: null, min_password_chars: 8 });

function Log() {
  useEffect(() => {
    const out: string[] = [];
    document.querySelectorAll<HTMLElement>("[data-probe]").forEach((w) => {
      const o = w.getBoundingClientRect();
      // The settings window is drawn at half size: back to window px.
      const k = w.dataset.probe === "settings" ? 2 : 1;
      const q = (name: string, sel: string) => {
        const e = w.querySelector(sel);
        if (!e) return out.push(`${w.dataset.probe} ${name}: none`);
        const r = e.getBoundingClientRect();
        out.push(`${w.dataset.probe} ${name}: x ${Math.round((r.x - o.x + r.width / 2) * k)} y ${Math.round((r.y - o.y + r.height / 2) * k)} w ${Math.round(r.width * k)} h ${Math.round(r.height * k)}`);
      };
      q("add", ".composer .controls > *:first-child");
      q("send", ".composer .send");
      q("download", '[title*="Download"], [aria-label*="Download"]');
      q("pw input", '.modal input[type="password"]');
      q("pw button", ".modal form .btn.primary");
      q("qr", ".modal .qr");
    });
    console.log(["PROBE", ...out].join("\n"));
  }, []);
  return null;
}

export function Probe011() {
  return (
    <AbsoluteFill style={{ background: "#111" }}>
      <div data-probe="empty" style={{ position: "absolute", left: 0, top: 0 }}>
        <AppWindow state={appState({ ...P })} width={390} height={776} phone light />
      </div>
      <div data-probe="photo" style={{ position: "absolute", left: 400, top: 0 }}>
        <AppWindow state={appState({ ...P, composer: { ...appState().composer, refs: [ref("beach")], prompt: "Put him on a pink flamingo, surfing a wave" } })} width={390} height={776} phone light />
      </div>
      <div data-probe="result" style={{ position: "absolute", left: 800, top: 0 }}>
        <AppWindow state={appState({ ...P, session: [genItem(BEACH), genItem(FLAMINGO)], selected: "beach-flamingo" })} width={390} height={776} phone light />
      </div>
      <div data-probe="settings" key="s" style={{ position: "absolute", left: 1200, top: 0, transform: "scale(0.5)", transformOrigin: "top left" }}>
        <AppWindow state={settingsState({ settingsTab: "devices" })} width={1440} height={900} light waitFor=".modal .qr" />
      </div>
      <Log />
    </AbsoluteFill>
  );
}

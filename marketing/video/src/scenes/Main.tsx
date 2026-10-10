// The launch video: every scene in order. Scenes that continue the same app session cut
// straight into each other; the others cross-fade.
import type { ComponentType } from "react";
import { AbsoluteFill, Series, useCurrentFrame } from "remotion";
import { prog } from "../app/motion";
import { AGENT_FRAMES, Agent } from "./Agent";
import { CREATE_FRAMES, Create, REGION_FRAMES, Region, VERSIONS_FRAMES, Versions } from "./AppScenes";
import { HARDWARE_FRAMES, Hardware, OUTRO_FRAMES, Outro } from "./Closing";
import { MONTAGE_FRAMES, Montage } from "./Montage";
import { Hook, Intro } from "./Opening";

const FADE = 12;
const SCENES: { C: ComponentType; frames: number; fadeIn: boolean }[] = [
  { C: Intro, frames: 120, fadeIn: false },
  { C: Hook, frames: 150, fadeIn: true },
  { C: Create, frames: CREATE_FRAMES, fadeIn: true },
  { C: Region, frames: REGION_FRAMES, fadeIn: false },
  { C: Versions, frames: VERSIONS_FRAMES, fadeIn: false },
  { C: Montage, frames: MONTAGE_FRAMES, fadeIn: true },
  { C: Agent, frames: AGENT_FRAMES, fadeIn: true },
  { C: Hardware, frames: HARDWARE_FRAMES, fadeIn: true },
  { C: Outro, frames: OUTRO_FRAMES, fadeIn: true },
];

export const MAIN_FRAMES = SCENES.reduce((n, s) => n + s.frames - (s.fadeIn ? FADE : 0), 0) + FADE;

function FadeIn({ C, on }: { C: ComponentType; on: boolean }) {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ opacity: on ? prog(f, 0, FADE) : 1 }}>
      <C />
    </AbsoluteFill>
  );
}

export function Main() {
  return (
    <AbsoluteFill style={{ background: "#050505" }}>
      <Series>
        {SCENES.map(({ C, frames, fadeIn }, i) => (
          <Series.Sequence key={i} durationInFrames={frames} offset={fadeIn && i > 0 ? -FADE : 0}>
            <FadeIn C={C} on={fadeIn} />
          </Series.Sequence>
        ))}
      </Series>
    </AbsoluteFill>
  );
}

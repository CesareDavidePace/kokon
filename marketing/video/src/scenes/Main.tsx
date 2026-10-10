// The launch video: every scene at its start in app/timeline.ts. Scenes that continue the same
// app session cut straight into each other; the others cross-fade. The voice-over and the music
// (ducked under the voice) run on the main timeline.
import type { ComponentType } from "react";
import { AbsoluteFill, Sequence, useCurrentFrame } from "remotion";
import { inOut, prog } from "../app/motion";
import { Music, Voice } from "../app/sound";
import { FADE, FADES_IN, LENGTH, START, TOTAL, type SceneName } from "../app/timeline";
import { Agent } from "./Agent";
import { Create, Region, Versions } from "./AppScenes";
import { Hardware, Outro } from "./Closing";
import { Montage } from "./Montage";
import { Hook, Intro } from "./Opening";

const SCENES: [SceneName, ComponentType][] = [
  ["intro", Intro],
  ["hook", Hook],
  ["create", Create],
  ["region", Region],
  ["versions", Versions],
  ["montage", Montage],
  ["agent", Agent],
  ["hardware", Hardware],
  ["outro", Outro],
];

export const MAIN_FRAMES = TOTAL;

function FadeIn({ C, on }: { C: ComponentType; on: boolean }) {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ opacity: on ? prog(f, 0, FADE, inOut) : 1 }}>
      <C />
    </AbsoluteFill>
  );
}

export function Main() {
  return (
    <AbsoluteFill style={{ background: "#050505" }}>
      {SCENES.map(([name, C]) => (
        <Sequence key={name} from={START[name]} durationInFrames={LENGTH[name]} name={name}>
          <FadeIn C={C} on={FADES_IN[name] && START[name] > 0} />
        </Sequence>
      ))}
      <Voice />
      <Music frames={MAIN_FRAMES} />
    </AbsoluteFill>
  );
}

import type { ComponentType } from "react";
import { Composition } from "remotion";
import { Preload } from "./app/Preload";
import { CREATE_FRAMES, Create, REGION_FRAMES, Region, VERSIONS_FRAMES, Versions } from "./scenes/AppScenes";
import { AGENT_FRAMES, Agent } from "./scenes/Agent";
import { HARDWARE_FRAMES, Hardware, OUTRO_FRAMES, Outro } from "./scenes/Closing";
import { Layout } from "./scenes/Layout";
import { MAIN_FRAMES, Main } from "./scenes/Main";
import { MONTAGE_FRAMES, Montage } from "./scenes/Montage";
import { Hook, Intro } from "./scenes/Opening";
import { LENGTH } from "./app/timeline";

const withPreload = (C: ComponentType) => () => (
  <>
    <Preload />
    <C />
  </>
);
const S = { fps: 30, width: 1920, height: 1080 };

export function Root() {
  return (
    <>
      <Composition id="KokonLaunch" component={withPreload(Main)} durationInFrames={MAIN_FRAMES} {...S} />
      <Composition id="Montage" component={withPreload(Montage)} durationInFrames={MONTAGE_FRAMES} {...S} />
      <Composition id="Agent" component={withPreload(Agent)} durationInFrames={AGENT_FRAMES} {...S} />
      <Composition id="Hardware" component={withPreload(Hardware)} durationInFrames={HARDWARE_FRAMES} {...S} />
      <Composition id="Outro" component={withPreload(Outro)} durationInFrames={OUTRO_FRAMES} {...S} />
      <Composition id="Intro" component={withPreload(Intro)} durationInFrames={LENGTH.intro} {...S} />
      <Composition id="Hook" component={withPreload(Hook)} durationInFrames={LENGTH.hook} {...S} />
      <Composition id="Create" component={withPreload(Create)} durationInFrames={CREATE_FRAMES} {...S} />
      <Composition id="Region" component={withPreload(Region)} durationInFrames={REGION_FRAMES} {...S} />
      <Composition id="Versions" component={withPreload(Versions)} durationInFrames={VERSIONS_FRAMES} {...S} />
      <Composition id="Layout" component={Layout} durationInFrames={4} {...S} />
    </>
  );
}

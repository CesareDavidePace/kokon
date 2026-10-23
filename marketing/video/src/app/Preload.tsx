// Holds the first frame until Inter and every image are loaded, and keeps the images decoded
// in a hidden pool so an <img> mounted by a component paints on its first frame.
import { useEffect, useState } from "react";
import { continueRender, delayRender, staticFile } from "remotion";
import { SIZES } from "./fixtures";

const IDS = Object.keys(SIZES);
const SRCS = [...IDS.map((id) => staticFile(`img/${id}.jpg`)), ...IDS.map((id) => staticFile(`thumb/${id}.jpg`))];

/** `extra`: more images to hold the first frame for (paths under public/). */
export function Preload({ extra = [] }: { extra?: string[] }) {
  const srcs = [...SRCS, ...extra.map((p) => staticFile(p))];
  const [handle] = useState(() => delayRender("Loading fonts and images"));
  useEffect(() => {
    const fonts = Promise.all(["400", "500", "600", "700"].map((w) => document.fonts.load(`${w} 16px "Inter Variable"`)));
    const images = srcs.map(
      (src) =>
        new Promise<void>((resolve) => {
          const img = new Image();
          img.onload = () => img.decode().then(() => resolve(), () => resolve());
          img.onerror = () => resolve();
          img.src = src;
        }),
    );
    Promise.all([fonts, ...images]).then(() => continueRender(handle));
  }, [handle]);
  return (
    <div style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", opacity: 0.001, pointerEvents: "none" }}>
      {srcs.map((s) => (
        <img key={s} src={s} alt="" style={{ width: 64, height: 64 }} />
      ))}
    </div>
  );
}

// Images that develop into view instead of popping in over the checkerboard.

import { useLayoutEffect, useRef, useState } from "react";

/** An `<img>` that stays hidden until it is decoded, then sharpens in out of a soft blur. Until
 * then a blurred `placeholder` (the thumbnail the timeline has already loaded) holds its place,
 * or a quiet sheen when there is none; it fades off as the image sharpens under it. An image
 * already in the cache shows at once, and so does one inside a hatching result (the Hatch reveals
 * it itself). The element stays mounted from the start, so `decode()` callers (Hatch) see it;
 * once shown, a new `src` swaps in place (the browser keeps the old picture until the new one is
 * ready). */
export function DevelopImg({ src, placeholder, className = "", alt = "", lazy = false }: { src?: string; placeholder?: string; className?: string; alt?: string; lazy?: boolean }) {
  const ref = useRef<HTMLImageElement>(null);
  // "wait": hidden; "develop": playing the entrance; "shown": plain.
  const [phase, setPhase] = useState<"wait" | "develop" | "shown">("wait");
  useLayoutEffect(() => {
    const img = ref.current;
    if (!img) return;
    if ((img.complete && img.naturalWidth > 0) || img.closest("[data-hatching]")) setPhase("shown");
  }, []);
  const ready = () => {
    const img = ref.current;
    if (!img || phase !== "wait") return;
    // decode() keeps the first frame of the entrance from showing a half-painted image.
    img.decode().catch(() => {}).then(() => setPhase((p) => (p === "wait" ? "develop" : p)));
  };
  return (
    <>
      {phase !== "shown" && (
        <span className={`develop-ph ${phase === "develop" ? "out" : ""}`} aria-hidden>
          {/* Faded in once complete: on a slow line it would otherwise paint from the top down. */}
          {placeholder && <img src={placeholder} alt="" draggable={false} onLoad={(e) => e.currentTarget.classList.add("on")} />}
        </span>
      )}
      <img
        ref={ref}
        className={`${className} ${phase === "wait" ? "develop-wait" : phase === "develop" ? "develop-in" : ""}`}
        src={src}
        alt={alt}
        loading={lazy ? "lazy" : undefined}
        draggable={false}
        onLoad={ready}
        onError={() => setPhase("shown")}
        onAnimationEnd={(e) => e.target === e.currentTarget && setPhase("shown")}
      />
    </>
  );
}

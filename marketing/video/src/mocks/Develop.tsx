// The app's DevelopImg (src/components/Develop.tsx) on the frame clock: the app starts the
// entrance when the image has decoded, which a video can't wait for. <AppWindow develop={…}> names
// the image developing and the seconds since it started (before 0: still loading, the blurred
// thumbnail under the sheen); every other image is shown at once. Markup and classes are the app's.
import { createContext, useContext, type CSSProperties } from "react";

export interface Develop {
  /** Part of the image's URL (e.g. "img/beach-flamingo"). */
  match: string;
  /** Seconds since it started to develop. */
  t: number;
}
export const DevelopContext = createContext<Develop | null>(null);

/** How long the placeholder takes to fade (styles.css .develop-ph transition). */
const PH_FADE = 0.55;

export function DevelopImg({ src, placeholder, className = "", alt = "" }: { src?: string; placeholder?: string; className?: string; alt?: string; lazy?: boolean }) {
  const d = useContext(DevelopContext);
  const t = d && src?.includes(d.match) ? d.t : null;
  if (t === null || t >= 1) return <img className={className} src={src} alt={alt} draggable={false} />;
  const waiting = t < 0;
  const ph = waiting ? 1 : Math.max(0, 1 - t / PH_FADE);
  return (
    <>
      {ph > 0 && (
        <span className={`develop-ph ${waiting ? "" : "out"}`} aria-hidden style={{ opacity: ph }}>
          {placeholder && <img className="on" src={placeholder} alt="" draggable={false} />}
        </span>
      )}
      <img className={`${className} ${waiting ? "develop-wait" : "develop-in"}`} src={src} alt={alt} draggable={false} style={{ "--t-develop": t.toFixed(3) } as CSSProperties} />
    </>
  );
}

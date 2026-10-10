// A result filling the frame, the way a finished image deserves to be seen: the picture as large
// as it fits over a blurred, dimmed copy of itself, a slow push-in, and the change shown by a
// soft wipe (`reveal`), a before/after handle (`split`) or, for an expanded image, the frame
// opening outward from the original (`expandFrom`).
import { Img, staticFile } from "remotion";

const src = (id: string) => staticFile(`img/${id}.jpg`);
const SIZE: Record<string, [number, number]> = {
  street: [2752, 1536],
  "street-removed": [2752, 1536],
  "street-cat": [2752, 1536],
  cabin: [2200, 1228],
  "cabin-winter": [2200, 1228],
  bakery: [2200, 1635],
  "bakery-text": [2200, 1635],
  potter: [1635, 2200],
  "potter-watercolor": [1635, 2200],
  room: [2200, 1635],
  "room-sofa": [2200, 1635],
  lighthouse: [1536, 1536],
  "lighthouse-wide": [2764, 1536],
  sneaker: [1536, 1536],
  "sneaker-ad": [1536, 1536],
};

/**
 * Where an image of aspect `ar` sits in the 1920×1080 frame: wide images fill it, others fit its
 * height; portraits sit right of centre, leaving the left side to the captions.
 */
export function fullBox(ar: number) {
  if (ar >= 1.7) return { x: 0, y: 0, w: 1920, h: 1080 };
  const h = 1080 - 120;
  const w = h * ar;
  const cx = ar < 1 ? 1200 : 960;
  return { x: cx - w / 2, y: 60, w, h };
}

export function FullImage({
  before,
  after,
  reveal = 0,
  split,
  push = 1,
  expandFrom,
  dim = 0,
}: {
  before: string;
  after?: string;
  /** 0–1: how far the after image has wiped in from the left. */
  reveal?: number;
  /** 0–1: before/after handle position (before on the left); replaces `reveal` when set. */
  split?: number;
  push?: number;
  /** For an outpaint: the before image's width as a fraction of the after image; reveal opens it. */
  expandFrom?: number;
  /** 0–1: darkens the picture under captions. */
  dim?: number;
}) {
  const main = after ?? before;
  const [w, h] = SIZE[main] ?? [16, 9];
  const box = fullBox(w / h);
  const pct = (v: number) => `${v * 100}%`;
  const img = (id: string, extra = {}) => <Img src={src(id)} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", ...extra }} />;
  const edge = split ?? reveal;
  const showEdge = after && !expandFrom && (split !== undefined || (reveal > 0 && reveal < 1));
  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden", background: "#050505" }}>
      {box.w < 1920 && (
        <div style={{ position: "absolute", inset: -80, filter: "blur(60px) brightness(0.32) saturate(1.2)" }}>
          {/* Fades with the wipe: switching at once makes the glow around the picture flicker. */}
          {img(before)}
          {after && img(after, { opacity: split !== undefined ? 1 - split : reveal })}
        </div>
      )}
      <div
        style={{
          position: "absolute",
          left: box.x,
          top: box.y,
          width: box.w,
          height: box.h,
          transform: `scale(${push})`,
          borderRadius: box.w < 1920 ? 18 : 0,
          overflow: "hidden",
          boxShadow: box.w < 1920 ? "0 40px 120px rgba(0,0,0,0.6)" : undefined,
        }}
      >
        {expandFrom ? (
          <>
            {/* The original sits in the middle; the expanded frame opens out to both sides. */}
            <div style={{ position: "absolute", inset: 0, background: "#050505" }} />
            {img(main, { clipPath: `inset(0 ${pct(((1 - expandFrom) / 2) * (1 - reveal))})` })}
            <div style={{ position: "absolute", top: 0, bottom: 0, left: pct((1 - expandFrom) / 2), width: pct(expandFrom), opacity: 1 - reveal }}>{img(before)}</div>
          </>
        ) : (
          <>
            {img(before)}
            {after && img(after, { clipPath: split !== undefined ? `inset(0 0 0 ${pct(split)})` : `inset(0 ${pct(1 - reveal)} 0 0)` })}
          </>
        )}
        {showEdge && (
          <div style={{ position: "absolute", top: 0, bottom: 0, left: pct(edge), width: 3, marginLeft: -1.5, background: "rgba(255,255,255,0.95)", boxShadow: "0 0 24px rgba(255,255,255,0.7)" }}>
            {split !== undefined && (
              <div style={{ position: "absolute", top: "50%", left: 1.5, width: 44, height: 44, margin: "-22px 0 0 -22px", borderRadius: 22, background: "rgba(255,255,255,0.95)", boxShadow: "0 6px 20px rgba(0,0,0,0.4)" }} />
            )}
          </div>
        )}
        {dim > 0 && <div style={{ position: "absolute", inset: 0, background: `rgba(0,0,0,${dim})` }} />}
      </div>
    </div>
  );
}

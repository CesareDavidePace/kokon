// Full-screen viewing: a shell that takes the window full screen and shows any content
// (the live before/after comparison, or a zoomable image).

import { Maximize2, X } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { native } from "../lib/native";

const isTyping = (t: EventTarget | null) => t instanceof HTMLElement && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));

/** Button for the top-right corner of an image; F also opens it. `content` is what to show
 * full screen; `hint` is shown next to the close button. */
export function FullscreenButton({ content, hint }: { content: ReactNode; hint?: string }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (open) return;
    const down = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "f" && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey && !isTyping(e.target)) setOpen(true);
    };
    window.addEventListener("keydown", down);
    return () => window.removeEventListener("keydown", down);
  }, [open]);
  return (
    <>
      <button
        className="fullscreen-btn"
        title="Full screen (F)"
        aria-label="View full screen"
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          setOpen(true);
        }}
        onDoubleClick={(e) => e.stopPropagation()}
      >
        <Maximize2 size={15} />
      </button>
      {open &&
        createPortal(
          <FullscreenShell hint={hint} onClose={() => setOpen(false)}>
            {content}
          </FullscreenShell>,
          document.body,
        )}
    </>
  );
}

function FullscreenShell({ children, hint, onClose }: { children: ReactNode; hint?: string; onClose: () => void }) {
  // Real full screen for the window while open; restored on close unless it already was.
  useEffect(() => {
    let wasFull = false;
    let alive = true;
    native
      .isFullscreen()
      .then((f) => {
        wasFull = f;
        if (alive && !f) return native.setFullscreen(true);
      })
      .catch(() => {});
    return () => {
      alive = false;
      if (!wasFull) native.setFullscreen(false).catch(() => {});
    };
  }, []);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (isTyping(e.target)) return;
      if (e.key === "Escape" || (e.key.toLowerCase() === "f" && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey)) {
        e.preventDefault();
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", down, true);
    return () => window.removeEventListener("keydown", down, true);
  }, [onClose]);

  // React events from a portal still bubble to its React parents: without this, a press
  // anywhere here reaches the stage image underneath (which captures the pointer for the
  // slider / press-to-compare) and the buttons in full screen never get their click.
  const stop = (e: React.SyntheticEvent) => e.stopPropagation();
  return (
    <div
      className="viewer"
      role="dialog"
      aria-label="Full screen"
      onPointerDown={stop}
      onPointerMove={stop}
      onPointerUp={stop}
      onPointerCancel={stop}
      onClick={stop}
      onDoubleClick={stop}
      onContextMenu={stop}
    >
      {children}
      <div className="viewer-bar">
        <span className="viewer-hint">{hint ? `${hint} · ` : ""}Esc to exit</span>
        <button className="viewer-close" title="Exit full screen (Esc)" aria-label="Exit full screen" onClick={onClose}>
          <X size={18} />
        </button>
      </div>
    </div>
  );
}

/** Fit to screen; click for 100 % at that point, drag to pan, click again to fit. */
export function ZoomImage({ src, alt }: { src: string; alt: string }) {
  const scroller = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState<{ fx: number; fy: number } | null>(null);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const drag = useRef<{ x: number; y: number; sl: number; st: number; moved: boolean } | null>(null);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (!el || !zoom || !natural) return;
    el.scrollLeft = zoom.fx * natural.w - el.clientWidth / 2;
    el.scrollTop = zoom.fy * natural.h - el.clientHeight / 2;
  }, [zoom, natural]);

  const el = scroller.current;
  const canZoom = !natural || !el || natural.w > el.clientWidth || natural.h > el.clientHeight;

  return (
    <div
      ref={scroller}
      className={`viewer-scroll ${zoom ? "zoomed" : ""}`}
      onPointerDown={(e) => {
        if (!zoom || e.button !== 0) return;
        drag.current = { x: e.clientX, y: e.clientY, sl: e.currentTarget.scrollLeft, st: e.currentTarget.scrollTop, moved: false };
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (!d) return;
        if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 4) d.moved = true;
        e.currentTarget.scrollLeft = d.sl - (e.clientX - d.x);
        e.currentTarget.scrollTop = d.st - (e.clientY - d.y);
      }}
      onPointerUp={() => {
        const d = drag.current;
        drag.current = null;
        if (zoom && d && !d.moved) setZoom(null);
      }}
    >
      <img
        className="viewer-img"
        src={src}
        alt={alt}
        draggable={false}
        style={zoom && natural ? { width: natural.w, height: natural.h, maxWidth: "none", maxHeight: "none", cursor: "inherit" } : { cursor: canZoom ? "zoom-in" : "default" }}
        onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        onClick={(e) => {
          if (zoom || !canZoom) return;
          const r = e.currentTarget.getBoundingClientRect();
          setZoom({ fx: (e.clientX - r.left) / r.width, fy: (e.clientY - r.top) / r.height });
        }}
      />
    </div>
  );
}

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useZoomPan, type UseZoomPanOptions, type UseZoomPanReturn } from "./useZoomPan";

// Configure React act environment
// @ts-expect-error React testing environment flag
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

describe("useZoomPan hook", () => {
  let container: HTMLDivElement;
  let root: ReturnType<typeof createRoot>;
  let hookResult: UseZoomPanReturn;

  function TestComponent(props: { options?: UseZoomPanOptions }) {
    hookResult = useZoomPan(props.options);
    return (
      <div
        {...hookResult.containerProps}
        style={{ ...hookResult.containerProps.style, width: 800, height: 600, position: "relative" }}
      >
        <div ref={hookResult.contentRef} />
      </div>
    );
  }

  beforeEach(() => {
    // Mock ResizeObserver
    globalThis.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    };

    // Mock getBoundingClientRect
    Element.prototype.getBoundingClientRect = function () {
      return {
        width: 800,
        height: 600,
        top: 0,
        left: 0,
        bottom: 600,
        right: 800,
        x: 0,
        y: 0,
        toJSON: () => {},
      };
    };

    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("initializes with default fit transform for given contentSize", () => {
    act(() => {
      root.render(<TestComponent options={{ contentSize: { width: 1600, height: 1200 } }} />);
    });

    expect(hookResult).toBeDefined();
    expect(hookResult.containerSize).toEqual({ width: 800, height: 600 });
    // Fit of 1600x1200 in 800x600 is scale 0.5, centered at (0, 0)
    expect(hookResult.transform.scale).toBe(0.5);
    expect(hookResult.transform.x).toBe(0);
    expect(hookResult.transform.y).toBe(0);
    expect(hookResult.isZoomed).toBe(false);
    expect(hookResult.isPanning).toBe(false);
  });

  it("canZoom is true for larger images", () => {
    act(() => {
      root.render(<TestComponent options={{ contentSize: { width: 2000, height: 1500 } }} />);
    });
    expect(hookResult.canZoom).toBe(true);
  });

  it("zoomIn, zoomOut, and zoomFit update transform correctly", () => {
    act(() => {
      root.render(<TestComponent options={{ contentSize: { width: 1600, height: 1200 } }} />);
    });

    const initialScale = hookResult.transform.scale;
    expect(initialScale).toBe(0.5);

    act(() => {
      hookResult.zoomIn();
    });
    expect(hookResult.transform.scale).toBeGreaterThan(initialScale);
    expect(hookResult.isZoomed).toBe(true);

    act(() => {
      hookResult.zoomOut();
    });
    expect(hookResult.transform.scale).toBeLessThanOrEqual(initialScale * 1.5);

    act(() => {
      hookResult.zoomFit();
    });
    expect(hookResult.transform.scale).toBe(hookResult.fitTransform.scale);
    expect(hookResult.isZoomed).toBe(false);
  });

  it("zoom100 sets scale to 1.0", () => {
    act(() => {
      root.render(<TestComponent options={{ contentSize: { width: 1200, height: 900 } }} />);
    });

    act(() => {
      hookResult.zoom100();
    });
    expect(hookResult.transform.scale).toBe(1.0);
    expect(hookResult.isZoomed).toBe(true);
  });

  it("toggleZoom switches between fit and 100%", () => {
    act(() => {
      root.render(<TestComponent options={{ contentSize: { width: 2000, height: 1500 } }} />);
    });

    expect(hookResult.isZoomed).toBe(false);

    // Toggle -> should zoom to 100%
    act(() => {
      hookResult.toggleZoom();
    });
    expect(hookResult.transform.scale).toBe(1.0);
    expect(hookResult.isZoomed).toBe(true);

    // Toggle again -> should return to fit
    act(() => {
      hookResult.toggleZoom();
    });
    expect(hookResult.isZoomed).toBe(false);
  });

  it("sets cursorStyle to default when enableToggleZoom is false and not zoomed", () => {
    act(() => {
      root.render(<TestComponent options={{ contentSize: { width: 2000, height: 1500 }, enableToggleZoom: false }} />);
    });
    expect(hookResult.containerProps.style.cursor).toBe("default");

    act(() => {
      root.render(<TestComponent options={{ contentSize: { width: 2000, height: 1500 }, enableToggleZoom: true }} />);
    });
    expect(hookResult.containerProps.style.cursor).toBe("zoom-in");
  });

  it("setTransform clamps offsets to bounds", () => {
    act(() => {
      root.render(<TestComponent options={{ contentSize: { width: 1000, height: 1000 } }} />);
    });

    act(() => {
      hookResult.setTransform({ x: 99999, y: 99999, scale: 2.0 });
    });

    // Offset must be clamped, not arbitrarily out of bounds
    expect(hookResult.transform.x).toBeLessThan(99999);
    expect(hookResult.transform.y).toBeLessThan(99999);
  });
});

import { ZoomPan } from "../components/ZoomPan";
import { setState, updateComposer } from "./store";

describe("ZoomPan component & Stage integration", () => {
  let container: HTMLDivElement;
  let root: ReturnType<typeof createRoot>;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    setState({ overlay: null });
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    setState({ overlay: null });
  });

  it("renders the zoom toolbar", () => {
    act(() => {
      root.render(<ZoomPan contentSize={{ width: 1600, height: 1200 }} />);
    });

    expect(container.querySelector(".zoom-controls")).not.toBeNull();
  });

  it("remounting with a different key resets zoom/pan transform", () => {
    let zoomHandle: any;

    act(() => {
      root.render(
        <ZoomPan
          key="item-1"
          zoomRef={(ref) => {
            zoomHandle = ref;
          }}
          contentSize={{ width: 1600, height: 1200 }}
        />,
      );
    });

    // Zoom in on version 1
    act(() => {
      zoomHandle?.zoomIn();
      zoomHandle?.zoomIn();
    });
    expect(zoomHandle.transform.scale).toBeGreaterThan(0.5);

    // Switch to version 2 (simulating timeline version switch via key change)
    act(() => {
      root.render(
        <ZoomPan
          key="item-2"
          zoomRef={(ref) => {
            zoomHandle = ref;
          }}
          contentSize={{ width: 1600, height: 1200 }}
        />,
      );
    });

    // Version 2 starts cleanly at initial fit scale (0.5 for 1600x1200 in 800x600)
    expect(zoomHandle.transform.scale).toBe(0.5);
    expect(zoomHandle.transform.x).toBe(0);
    expect(zoomHandle.transform.y).toBe(0);
  });

  it("guards keyboard zoom shortcuts when overlay or typing is active", () => {
    let zoomHandle: any;

    act(() => {
      root.render(
        <ZoomPan
          zoomRef={(ref) => {
            zoomHandle = ref;
          }}
          contentSize={{ width: 1600, height: 1200 }}
        />,
      );
    });

    const initialScale = zoomHandle.transform.scale;

    // Simulate typing in prompt/input
    const input = document.createElement("input");
    document.body.appendChild(input);
    input.focus();
    act(() => {
      input.dispatchEvent(new KeyboardEvent("keydown", { key: "+", bubbles: true }));
    });
    expect(zoomHandle.transform.scale).toBe(initialScale); // Not changed

    input.remove();

    // Simulate modal overlay active (e.g. settings)
    setState({ overlay: "settings" });
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "+", bubbles: true }));
    });
    expect(zoomHandle.transform.scale).toBe(initialScale); // Not changed

    setState({ overlay: null });

    // With no overlay and not typing: shortcut works
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "+", bubbles: true }));
    });
    expect(zoomHandle.transform.scale).toBeGreaterThan(initialScale);

    // Numpad Subtract zooms out
    const zoomedScale = zoomHandle.transform.scale;
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Subtract", bubbles: true }));
    });
    expect(zoomHandle.transform.scale).toBeLessThan(zoomedScale);

    // Numpad Add zooms in
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Add", bubbles: true }));
    });
    expect(zoomHandle.transform.scale).toBeGreaterThan(initialScale);
  });

  it("resets transform to fit when contentSize changes without remounting", () => {
    let zoomHandle: any;

    act(() => {
      root.render(
        <ZoomPan
          zoomRef={(ref) => {
            zoomHandle = ref;
          }}
          contentSize={{ width: 1600, height: 1200 }}
        />,
      );
    });

    expect(zoomHandle.transform.scale).toBe(0.5);

    // User zooms in on image 1
    act(() => {
      zoomHandle.zoomIn();
      zoomHandle.zoomIn();
    });
    expect(zoomHandle.transform.scale).toBeGreaterThan(0.5);

    // Content size changes (new image loaded into same component instance)
    act(() => {
      root.render(
        <ZoomPan
          zoomRef={(ref) => {
            zoomHandle = ref;
          }}
          contentSize={{ width: 800, height: 600 }}
        />,
      );
    });

    // Content change should automatically reset to fit (fit of 800x600 in 800x600 is 1.0)
    expect(zoomHandle.transform.scale).toBe(1.0);
    expect(zoomHandle.transform.x).toBe(0);
    expect(zoomHandle.transform.y).toBe(0);
  });

  it("resets transform to fit when resetKey changes", () => {
    let zoomHandle: any;

    act(() => {
      root.render(
        <ZoomPan
          zoomRef={(ref) => {
            zoomHandle = ref;
          }}
          src="gen-1.png"
          contentSize={{ width: 1600, height: 1200 }}
        />,
      );
    });

    expect(zoomHandle.transform.scale).toBe(0.5);

    // User zooms in
    act(() => {
      zoomHandle.zoomIn();
      zoomHandle.zoomIn();
    });
    expect(zoomHandle.transform.scale).toBeGreaterThan(0.5);

    // Image src changes with identical dimensions
    act(() => {
      root.render(
        <ZoomPan
          zoomRef={(ref) => {
            zoomHandle = ref;
          }}
          src="gen-2.png"
          contentSize={{ width: 1600, height: 1200 }}
        />,
      );
    });

    // Should reset to fit
    expect(zoomHandle.transform.scale).toBe(0.5);
    expect(zoomHandle.transform.x).toBe(0);
    expect(zoomHandle.transform.y).toBe(0);
  });
});

import { CompareFrame, CompareFullscreen } from "../components/Compare";
import type { CompareBase } from "./logic";
import type { GenerationRecord } from "./types";
import { vi } from "vitest";

describe("CompareFrame & ZoomPan Integration", () => {
  let container: HTMLDivElement;
  let root: ReturnType<typeof createRoot>;

  const mockGen = {
    id: "gen1",
    mode: "edit",
    prompt: "make it sunset",
    width: 1600,
    height: 1200,
    references: [{ index: 0, input_id: "in1", width: 1600, height: 1200 }],
    outpaint: null,
  } as unknown as GenerationRecord;

  const mockBase: CompareBase = {
    inputId: "in1",
    rect: { x: 0, y: 0, w: 1, h: 1 },
  };

  beforeEach(() => {
    if (!Element.prototype.setPointerCapture) Element.prototype.setPointerCapture = () => {};
    if (!Element.prototype.releasePointerCapture) Element.prototype.releasePointerCapture = () => {};
    if (!Element.prototype.hasPointerCapture) Element.prototype.hasPointerCapture = () => false;

    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    setState({ overlay: null, compare: "slider" });
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    setState({ overlay: null, compare: "slider" });
  });

  it("renders the full-screen CompareFrame inside ZoomPan with content matching generation dimensions", () => {
    act(() => {
      root.render(<CompareFrame g={mockGen} base={mockBase} fullscreen />);
    });

    expect(container.querySelector(".zoom-pan-viewport")).not.toBeNull();

    const content = container.querySelector(".zoom-pan-content") as HTMLElement;
    expect(content).not.toBeNull();
    // A browser lays the content out flat, at its drawn size (ZoomPan `flat`): the image's
    // shape, with the comparison box filling it and no scale transform.
    const w = parseFloat(content.style.width);
    const h = parseFloat(content.style.height);
    expect(w).toBeGreaterThan(0);
    expect(w / h).toBeCloseTo(1600 / 1200, 5);
    expect(content.style.transform).not.toContain("scale");
    expect((container.querySelector(".compare-box") as HTMLElement).style.width).toBe("100%");

    // Both before and after images must be mounted
    const afterImg = container.querySelector(".compare-img") as HTMLImageElement;
    expect(afterImg).not.toBeNull();
    expect(afterImg.src).toContain("gen1");

    const beforeContainer = container.querySelector(".compare-before");
    expect(beforeContainer).not.toBeNull();
    const beforeImg = beforeContainer?.querySelector("img") as HTMLImageElement;
    expect(beforeImg).not.toBeNull();
    expect(beforeImg.src).toContain("in1");
  });

  it("renders interactive split slider in content space at 50% split", () => {
    act(() => {
      root.render(<CompareFrame g={mockGen} base={mockBase} />);
    });

    const handle = container.querySelector(".compare-handle") as HTMLElement;
    expect(handle).not.toBeNull();
    expect(handle.style.left).toBe("50%");

    const before = container.querySelector(".compare-before") as HTMLElement;
    expect(before.style.clipPath).toBe("inset(0 50% 0 0)");

    // Labels
    expect(container.textContent).toContain("Before");
    expect(container.textContent).toContain("After");
  });

  it("tracks slider dragging via pointer events across content space", () => {
    act(() => {
      root.render(<CompareFrame g={mockGen} base={mockBase} />);
    });

    const handle = container.querySelector(".compare-handle") as HTMLElement;
    expect(handle).not.toBeNull();

    // Pointer down at 50% (400px of 800px width)
    act(() => {
      handle.dispatchEvent(new PointerEvent("pointerdown", { clientX: 400, button: 0, bubbles: true }));
    });

    // Move pointer to 75% (600px of 800px width)
    act(() => {
      handle.dispatchEvent(new PointerEvent("pointermove", { clientX: 600, bubbles: true }));
    });

    expect(handle.style.left).toBe("75%");
    const before = container.querySelector(".compare-before") as HTMLElement;
    expect(before.style.clipPath).toBe("inset(0 25% 0 0)");

    // Pointer up to complete drag
    act(() => {
      handle.dispatchEvent(new PointerEvent("pointerup", { clientX: 600, bubbles: true }));
    });

    expect(handle.style.left).toBe("75%");
  });

  it("hold 'B' key reveals the full original image and restores on keyup", () => {
    act(() => {
      root.render(<CompareFrame g={mockGen} base={mockBase} />);
    });

    const before = container.querySelector(".compare-before") as HTMLElement;
    expect(before.style.clipPath).toBe("inset(0 50% 0 0)");

    // Press and hold 'B'
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "b" }));
    });

    // In holding mode, clipPath is cleared to show full original
    expect(before.style.clipPath).toBe("");
    expect(before.style.visibility).toBe("visible");

    // Release 'B'
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keyup", { key: "b" }));
    });

    // Restores split slider clipPath
    expect(before.style.clipPath).toBe("inset(0 50% 0 0)");
  });

  it("flicker mode flips phase automatically using timer", () => {
    vi.useFakeTimers();
    setState({ compare: "flicker", flickerMs: 150 });

    act(() => {
      root.render(<CompareFrame g={mockGen} base={mockBase} />);
    });

    const before = container.querySelector(".compare-before") as HTMLElement;
    // Initial phase in flicker is "before"
    expect(before.style.visibility).toBe("visible");

    // Advance 150ms -> flips to "after" (before hidden)
    act(() => {
      vi.advanceTimersByTime(150);
    });
    expect(before.style.visibility).toBe("hidden");

    // Advance 150ms -> flips back to "before"
    act(() => {
      vi.advanceTimersByTime(150);
    });
    expect(before.style.visibility).toBe("visible");

    vi.useRealTimers();
  });

  it("does not zoom on the stage: no viewport or zoom controls, CompareBar and full-screen button kept", () => {
    act(() => {
      root.render(<CompareFrame g={mockGen} base={mockBase} />);
    });

    expect(container.querySelector(".zoom-pan-viewport")).toBeNull();
    expect(container.querySelector(".zoom-controls")).toBeNull();
    expect(container.querySelector(".compare-overlay .compare-bar")).not.toBeNull();
    expect(container.querySelector(".fullscreen-btn")).not.toBeNull();
  });

  it("mouse press-and-hold in mode 'off' shows original and restores on pointerup/cancel", () => {
    setState({ compare: "off" });
    act(() => {
      root.render(<CompareFrame g={mockGen} base={mockBase} />);
    });

    const box = container.querySelector(".compare-box") as HTMLElement;
    const before = container.querySelector(".compare-before") as HTMLElement;
    expect(box).not.toBeNull();
    expect(before.style.visibility).toBe("hidden");

    // Pointer down on box shows original with Before label
    act(() => {
      box.dispatchEvent(new PointerEvent("pointerdown", { button: 0, bubbles: true }));
    });
    expect(before.style.visibility).toBe("visible");
    expect(container.textContent).toContain("Before");

    // Pointer up restores result
    act(() => {
      box.dispatchEvent(new PointerEvent("pointerup", { button: 0, bubbles: true }));
    });
    expect(before.style.visibility).toBe("hidden");

    // Pointer down followed by pointer cancel also restores result
    act(() => {
      box.dispatchEvent(new PointerEvent("pointerdown", { button: 0, bubbles: true }));
    });
    expect(before.style.visibility).toBe("visible");
    act(() => {
      box.dispatchEvent(new PointerEvent("pointercancel", { bubbles: true }));
    });
    expect(before.style.visibility).toBe("hidden");
  });

  it("clicking on compare-box in slider mode updates split position when unzoomed", () => {
    setState({ compare: "slider" });
    act(() => {
      root.render(<CompareFrame g={mockGen} base={mockBase} />);
    });

    const box = container.querySelector(".compare-box") as HTMLElement;
    const handle = container.querySelector(".compare-handle") as HTMLElement;
    expect(handle.style.left).toBe("50%");

    // Click at 25% (200px of 800px width)
    act(() => {
      box.dispatchEvent(new PointerEvent("pointerdown", { clientX: 200, button: 0, bubbles: true }));
    });
    expect(handle.style.left).toBe("25%");

    // Drag to 60% (480px)
    act(() => {
      box.dispatchEvent(new PointerEvent("pointermove", { clientX: 480, bubbles: true }));
    });
    expect(handle.style.left).toBe("60%");

    act(() => {
      box.dispatchEvent(new PointerEvent("pointerup", { clientX: 480, bubbles: true }));
    });
    expect(handle.style.left).toBe("60%");
  });

  it("hides slider handle and Before/After tags when hold 'B' is active in slider mode", () => {
    setState({ compare: "slider" });
    act(() => {
      root.render(<CompareFrame g={mockGen} base={mockBase} />);
    });

    expect(container.querySelector(".compare-handle")).not.toBeNull();
    expect(container.textContent).toContain("Before");
    expect(container.textContent).toContain("After");

    // Press 'B'
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "b" }));
    });

    // Handle is hidden, tags are replaced by single "Before" indicator
    expect(container.querySelector(".compare-handle")).toBeNull();
    expect(container.textContent).toContain("Before");
    expect(container.textContent).not.toContain("After");

    // Release 'B'
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keyup", { key: "b" }));
    });

    expect(container.querySelector(".compare-handle")).not.toBeNull();
    expect(container.textContent).toContain("Before");
    expect(container.textContent).toContain("After");
  });

  it("renders DiffCanvas inside compare-box with correct layout in diff mode", () => {
    setState({ compare: "diff" });
    act(() => {
      root.render(<CompareFrame g={mockGen} base={mockBase} />);
    });

    const box = container.querySelector(".compare-box") as HTMLElement;
    expect(box).not.toBeNull();
    const diffCanvas = box.querySelector(".compare-diff");
    expect(diffCanvas).not.toBeNull();
  });

  it("renders CompareFullscreen with viewer-frame and external compare bar", () => {
    act(() => {
      root.render(<CompareFullscreen g={mockGen} base={mockBase} />);
    });

    // Fullscreen viewer-frame
    expect(container.querySelector(".viewer-frame")).not.toBeNull();
    expect(container.querySelector(".zoom-pan-viewport")).not.toBeNull();
    expect(container.querySelector(".zoom-controls")).not.toBeNull();

    // Fullscreen has compare bar outside the viewer frame
    expect(container.querySelector(".viewer-compare-bar .compare-bar")).not.toBeNull();
  });

  it("switches across all compare modes in full screen without resetting the zoom transform", () => {
    setState({ compare: "slider" });

    act(() => {
      root.render(<CompareFrame g={mockGen} base={mockBase} fullscreen />);
    });

    const content = container.querySelector(".zoom-pan-content") as HTMLElement;
    const initial = content.style.transform;
    expect(initial).not.toBe("");
    expect(container.querySelector(".compare-handle")).not.toBeNull();

    for (const mode of ["flicker", "diff", "off", "slider"] as const) {
      act(() => {
        setState({ compare: mode });
      });
      expect(content.style.transform).toBe(initial);
    }
    expect(container.querySelector(".compare-handle")).not.toBeNull();
  });
});

import { InpaintEditor } from "../components/InpaintEditor";

describe("InpaintEditor & ZoomPan Integration", () => {
  let container: HTMLDivElement;
  let root: ReturnType<typeof createRoot>;

  beforeEach(() => {
    if (!Element.prototype.setPointerCapture) Element.prototype.setPointerCapture = () => {};
    if (!Element.prototype.releasePointerCapture) Element.prototype.releasePointerCapture = () => {};
    if (!Element.prototype.hasPointerCapture) Element.prototype.hasPointerCapture = () => false;

    HTMLCanvasElement.prototype.getContext = (() => ({
      globalCompositeOperation: "source-over",
      clearRect: () => {},
      drawImage: () => {},
      getImageData: () => ({ data: new Uint8ClampedArray(100) }),
      putImageData: () => {},
      beginPath: () => {},
      arc: () => {},
      fill: () => {},
      moveTo: () => {},
      lineTo: () => {},
      stroke: () => {},
      fillRect: () => {},
    })) as any;

    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    setState({ overlay: null });
    updateComposer({ refs: [{ key: "r1", source: "input:in1", width: 1600, height: 900, name: "photo.png" }], mask: null });
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("renders InpaintEditor wrapped inside ZoomPan with inpaint-zoom-viewport", () => {
    act(() => {
      root.render(
        <InpaintEditor />,
      );
    });

    const viewport = container.querySelector(".inpaint-zoom-viewport");
    expect(viewport).not.toBeNull();
    const maskWrap = container.querySelector(".mask-wrap") as HTMLElement;
    expect(maskWrap).not.toBeNull();
    expect(maskWrap.style.width).toBe("1600px");
    expect(maskWrap.style.height).toBe("900px");
  });

  it("toggles pan mode via Spacebar hold and Hand tool button", () => {
    act(() => {
      root.render(
        <InpaintEditor />,
      );
    });

    const canvas = container.querySelector("canvas") as HTMLCanvasElement;
    expect(canvas).not.toBeNull();

    // Default: brush active, cursor is none (custom circle cursor)
    expect(canvas.style.cursor).toBe("none");

    // Press and hold Spacebar -> switches to pan mode with grab cursor
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { code: "Space" }));
    });
    expect(canvas.style.cursor).toBe("grab");

    // Release Spacebar -> returns to brush mode
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keyup", { code: "Space" }));
    });
    expect(canvas.style.cursor).toBe("none");

    // Press 'H' -> activates Pan/Hand tool
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "h" }));
    });
    expect(canvas.style.cursor).toBe("grab");

    // Press 'B' -> switches back to Brush
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "b" }));
    });
    expect(canvas.style.cursor).toBe("none");
  });

  it("adjusts brush size via [ and ] shortcuts", () => {
    act(() => {
      root.render(
        <InpaintEditor />,
      );
    });

    const sizeInput = container.querySelector('input[type="range"][min="4"]') as HTMLInputElement;
    expect(sizeInput).not.toBeNull();
    const initialSize = Number(sizeInput.value);

    // Increase size with ']'
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "]" }));
    });
    expect(Number(sizeInput.value)).toBe(initialSize + 8);

    // Decrease size with '['
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "[" }));
    });
    expect(Number(sizeInput.value)).toBe(initialSize);
  });

  it("always shows the zoom controls and has no Feather slider (fixed soft edge)", () => {
    act(() => {
      root.render(
        <InpaintEditor />,
      );
    });

    expect(container.querySelector(".zoom-controls")).not.toBeNull();
    expect(container.querySelector('button[title="Hide zoom controls"]')).toBeNull();
    expect(container.textContent).not.toContain("Feather");
  });
});

describe("Gestures, Inertia & MiniMap", () => {
  let container: HTMLDivElement;
  let root: ReturnType<typeof createRoot>;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    setState({ overlay: null });
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("handles multi-touch pinch to zoom in and out", () => {
    let hookResult!: UseZoomPanReturn;
    function Test() {
      hookResult = useZoomPan({ contentSize: { width: 1600, height: 1200 } });
      return <div {...hookResult.containerProps} />;
    }

    act(() => {
      root.render(<Test />);
    });

    const initialScale = hookResult.transform.scale;

    // Finger 1 down
    act(() => {
      hookResult.containerProps.onPointerDownCapture({
        pointerId: 1,
        pointerType: "touch",
        clientX: 300,
        clientY: 300,
        stopPropagation: () => {},
      } as unknown as React.PointerEvent<HTMLDivElement>);
    });
    expect(hookResult.isPinching).toBe(false);

    // Finger 2 down
    act(() => {
      hookResult.containerProps.onPointerDownCapture({
        pointerId: 2,
        pointerType: "touch",
        clientX: 500,
        clientY: 300,
        stopPropagation: () => {},
      } as unknown as React.PointerEvent<HTMLDivElement>);
    });
    expect(hookResult.isPinching).toBe(true);

    // Spread fingers apart (move finger 2 from 500 to 700)
    act(() => {
      hookResult.containerProps.onPointerMoveCapture({
        pointerId: 2,
        clientX: 700,
        clientY: 300,
        stopPropagation: () => {},
      } as unknown as React.PointerEvent<HTMLDivElement>);
    });
    expect(hookResult.transform.scale).toBeGreaterThan(initialScale);

    // Finger up
    act(() => {
      hookResult.containerProps.onPointerUpCapture({
        pointerId: 2,
      } as unknown as React.PointerEvent<HTMLDivElement>);
    });
    expect(hookResult.isPinching).toBe(false);
  });

  it("supports middle-mouse click and drag panning", () => {
    let hookResult!: UseZoomPanReturn;
    function Test() {
      hookResult = useZoomPan({ contentSize: { width: 1600, height: 1200 }, enablePan: false });
      return <div {...hookResult.containerProps} />;
    }

    act(() => {
      root.render(<Test />);
    });

    act(() => {
      hookResult.zoom100(); // Zoom in to enable panning room
    });

    const startX = hookResult.transform.x;
    const startY = hookResult.transform.y;

    // Middle-click down (button === 1)
    act(() => {
      hookResult.containerProps.onPointerDown({
        button: 1,
        clientX: 400,
        clientY: 300,
        preventDefault: () => {},
        currentTarget: { setPointerCapture: () => {} },
      } as unknown as React.PointerEvent<HTMLDivElement>);
    });

    // Move
    act(() => {
      hookResult.containerProps.onPointerMove({
        clientX: 350,
        clientY: 250,
      } as unknown as React.PointerEvent<HTMLDivElement>);
    });

    expect(hookResult.isPanning).toBe(true);
    expect(hookResult.transform.x).toBe(startX - 50);
    expect(hookResult.transform.y).toBe(startY - 50);

    // Up
    act(() => {
      hookResult.containerProps.onPointerUp({
        button: 1,
        clientX: 350,
        clientY: 250,
        currentTarget: { releasePointerCapture: () => {}, hasPointerCapture: () => false },
      } as unknown as React.PointerEvent<HTMLDivElement>);
    });
    expect(hookResult.isPanning).toBe(false);
  });

  it("shows the MiniMap navigator by itself while zoomed in", () => {
    act(() => {
      root.render(<ZoomPan src="/photo.png" contentSize={{ width: 1600, height: 1200 }} />);
    });

    // Initially at fit scale: no MiniMap, Fit disabled
    const fit = () => container.querySelector('button[title="Fit to screen (0)"]') as HTMLButtonElement;
    expect(container.querySelector(".zoom-minimap")).toBeNull();
    expect(fit().disabled).toBe(true);

    act(() => {
      (container.querySelector('button[aria-label="Zoom in"]') as HTMLButtonElement).click();
    });
    expect(fit().disabled).toBe(false);

    // MiniMap navigator and viewport rect should be rendered
    const minimap = container.querySelector(".zoom-minimap");
    expect(minimap).not.toBeNull();
    expect(container.querySelector(".zoom-minimap-viewport")).not.toBeNull();

    // Pointer on the MiniMap
    const minimapEl = minimap as HTMLElement;
    minimapEl.getBoundingClientRect = () => ({
      width: 140,
      height: 105,
      left: 600,
      top: 400,
      right: 740,
      bottom: 505,
      x: 600,
      y: 400,
      toJSON: () => {},
    });

    act(() => {
      const pEvent = new MouseEvent("pointerdown", {
        bubbles: true,
        clientX: 670,
        clientY: 452.5,
      });
      minimapEl.dispatchEvent(pEvent);
    });

    expect(container.querySelector(".zoom-minimap")).not.toBeNull();
  });

  it("keeps the same toolbar items at every zoom level", () => {
    act(() => {
      root.render(<ZoomPan src="/photo.png" contentSize={{ width: 1600, height: 1200 }} />);
    });

    const items = () => [...container.querySelectorAll(".zoom-controls > *")].map((el) => el.getAttribute("title") ?? el.className);
    const atFit = items();
    expect(atFit).toHaveLength(4);
    act(() => {
      (container.querySelector('button[aria-label="Zoom in"]') as HTMLButtonElement).click();
    });
    expect(items()).toEqual(atFit);
  });
});



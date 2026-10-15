// Keyboard and mouse shortcuts, opened with "?".

import { Keyboard, X } from "lucide-react";
import { Fragment, useEffect } from "react";
import { isMac } from "../lib/platform";
import { setState } from "../lib/store";

const MOD = isMac ? "⌘" : "Ctrl";

/** Each row: what it does, then the alternatives (each a list of keys pressed together). */
const SECTIONS: { title: string; rows: [string, string[][]][] }[] = [
  {
    title: "Everywhere",
    rows: [
      ["Generate (Enter in the prompt box)", [["Enter"], [MOD, "Enter"]]],
      ["New line in the prompt", [["Shift", "Enter"]]],
      ["Bring back the last prompt (empty box)", [["↑"]]],
      ["Paste an image", [[MOD, "V"]]],
      ["Previous / next version", [["←"], ["→"]]],
      ["Undo / redo a version", [[MOD, "Z"], [MOD, "Shift", "Z"]]],
      ["Rename a project (in the list)", [["F2"]]],
      ["Show these shortcuts", [["?"]]],
    ],
  },
  {
    title: "Viewing",
    rows: [
      ["Full screen", [["F"]]],
      ["Browse in full screen", [["←"], ["→"]]],
      ["See the original (hold)", [["B"]]],
      ["Close", [["Esc"]]],
    ],
  },
  {
    title: "Zoom (full screen and painting)",
    rows: [
      ["Zoom", [["Scroll"], ["+"], ["−"]]],
      ["Fit / 100 %", [["0"], ["1"]]],
      ["Move around", [["Drag"]]],
    ],
  },
  {
    title: "Painting an area",
    rows: [
      ["Brush / eraser / move", [["B"], ["E"], ["H"]]],
      ["Smaller / larger brush", [["["], ["]"]]],
      ["Use the other tool", [["Right-click"]]],
      ["Move around", [["Space", "Drag"], ["Middle-drag"]]],
      ["Undo / redo a stroke", [[MOD, "Z"], [MOD, "Shift", "Z"]]],
      ["Cancel", [["Esc"]]],
    ],
  },
];

export function Shortcuts() {
  const close = () => setState({ overlay: null });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => (e.key === "Escape" || e.key === "?") && close();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="modal-scrim" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="tips shortcuts" role="dialog" aria-label="Keyboard shortcuts">
        <div className="tips-head">
          <Keyboard size={20} />
          <h1>Shortcuts</h1>
          <span className="spacer" />
          <button className="icon-btn" onClick={close} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="tips-body shortcuts-body">
          {SECTIONS.map((s) => (
            <section className="tips-rule" key={s.title}>
              <h2>{s.title}</h2>
              {s.rows.map(([what, alts]) => (
                <div className="shortcut-row" key={what}>
                  <span>{what}</span>
                  <span className="keys">
                    {alts.map((keys, i) => (
                      <Fragment key={i}>
                        {i > 0 && <span className="muted">or</span>}
                        {keys.map((k) => (
                          <kbd key={k}>{k}</kbd>
                        ))}
                      </Fragment>
                    ))}
                  </span>
                </div>
              ))}
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

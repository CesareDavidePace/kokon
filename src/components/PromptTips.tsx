import { Check, ExternalLink, Lightbulb, X } from "lucide-react";
import { useEffect, useState } from "react";
import { native } from "../lib/native";
import { focusPrompt, setState, toastError, updateComposer, useStore } from "../lib/store";
import bikePlain from "../assets/prompt-tips/bike-plain.webp";
import bikeRich from "../assets/prompt-tips/bike-rich.webp";
import deskBad from "../assets/prompt-tips/desk-bad.webp";
import deskGood from "../assets/prompt-tips/desk-good.webp";
import monkeyBad from "../assets/prompt-tips/monkey-bad.webp";
import monkeyGood from "../assets/prompt-tips/monkey-good.webp";
import posterBad from "../assets/prompt-tips/poster-bad.webp";
import posterGood from "../assets/prompt-tips/poster-good.webp";
import roomBetter from "../assets/prompt-tips/room-better.webp";
import roomBlueChair from "../assets/prompt-tips/room-blue-chair.webp";
import roomBlueWall from "../assets/prompt-tips/room-blue-wall.webp";
import roomCozy from "../assets/prompt-tips/room-cozy.webp";
import room from "../assets/prompt-tips/room.webp";
import streetMulti from "../assets/prompt-tips/street-multi.webp";
import street from "../assets/prompt-tips/street.webp";

export const PROMPT_GUIDE_URL = "https://kokon.app/prompt-guide/";

type Tab = "create" | "edit";
interface Side { kind: "bad" | "ok" | "good" | "src"; label: string; prompt: string; img: string; note?: string }
/** `before`: the photo an edit starts from, shown first. */
interface Rule { title: string; why: string; before?: string; sides: Side[]; lesson: string }

// Every image is a real Kokon result (Qwen-Image 2.1 Turbo, Fast, same seed on both sides).
const RULES: Record<Tab, Rule[]> = {
  create: [
    {
      title: "Say what's happening, and where",
      why: "With a short prompt the model draws the subject and fills the rest with the most ordinary choice.",
      sides: [
        { kind: "bad", label: "Vague", prompt: "a monkey playing basketball", img: monkeyBad, note: "Sitting on a post, with a football." },
        { kind: "good", label: "Clear", prompt: "Photograph of a monkey dribbling a basketball on an outdoor court, full body, sunny afternoon", img: monkeyGood, note: "Now it plays." },
      ],
      lesson: "Who, doing what, where, in what light.",
    },
    {
      title: "Never write “no”",
      why: "The model reads the words, not the grammar: “no laptop” puts a laptop in its head.",
      sides: [
        { kind: "bad", label: "Negative", prompt: "Photograph of a minimalist wooden desk by a window, no laptop, no computer", img: deskBad, note: "There's the laptop." },
        { kind: "good", label: "Positive", prompt: "Photograph of a minimalist wooden desk by a window, the desk is clean and empty except for a small potted plant and a ceramic cup", img: deskGood },
      ],
      lesson: "Say what is there. “An empty beach”, not “no people”.",
    },
    {
      title: "Text goes in quotes, and short",
      why: "Big titles come out right almost every time. Long lines of small text are where it slips.",
      sides: [
        { kind: "bad", label: "Too much small text", prompt: 'Minimal concert poster, a white crescent moon, title "NIGHT JAZZ", below it smaller text "Friday 24 October · Teatro Verdi · 21:00"', img: posterBad, note: "The small line appears twice, with a typo." },
        { kind: "good", label: "Short and placed", prompt: 'Minimal concert poster, dark blue background, a large white crescent moon in the center. Only two lines of text: the title "NIGHT JAZZ" in big bold letters at the top, and "24 OCTOBER" in small letters at the bottom. Flat graphic style.', img: posterGood },
      ],
      lesson: "Lots of small text? Switch Quality to Quality (native 2K).",
    },
    {
      title: "Details make it your picture",
      why: "A short prompt isn't wrong, just generic. Every detail is a choice you make instead of the model.",
      sides: [
        { kind: "ok", label: "Fine", prompt: "a girl riding a bicycle", img: bikePlain },
        { kind: "good", label: "Yours", prompt: "Photograph of a young woman riding a vintage blue bicycle along a seaside promenade, her hair blowing in the wind, a basket of flowers on the handlebar, late afternoon golden light, shot from the side, full body", img: bikeRich },
      ],
      lesson: "Long prompts are fine. Plain sentences, most important things first.",
    },
  ],
  edit: [
    {
      title: "“Better” means nothing",
      why: "Ask to “improve” a photo and you get it back almost unchanged.",
      before: room,
      sides: [
        { kind: "bad", label: "Vague", prompt: "make it better", img: roomBetter, note: "Same room." },
        { kind: "good", label: "Clear", prompt: "Make the room cozy for a winter evening: dusk outside the window, the floor lamp turned on with warm light, a knitted cream blanket on the armchair, a stack of books and a mug of tea on the rug", img: roomCozy },
      ],
      lesson: "Turn “better” into things you can see: warmer light, tidier, a sunset.",
    },
    {
      title: "Name the thing you want changed",
      why: "Without a name, the model picks the most eye-catching thing in the photo.",
      before: room,
      sides: [
        { kind: "bad", label: "Ambiguous", prompt: "make it blue", img: roomBlueChair, note: "You meant the wall." },
        { kind: "good", label: "Named", prompt: "Paint the wall behind the armchair a dusty blue. Keep the yellow armchair and everything else unchanged.", img: roomBlueWall },
      ],
      lesson: "Many similar things? Use Paint area and write only the change.",
    },
    {
      title: "Ask for several changes at once",
      why: "One clear sentence can carry many edits. If one doesn't happen, edit the result again.",
      before: street,
      sides: [
        { kind: "good", label: "Five changes", prompt: "Remove the red scooter, make it a rainy evening with wet reflective cobblestones, turn on the café lights, add a ginger cat sitting on a chair, and add an old man with an umbrella walking on the right", img: streetMulti },
      ],
      lesson: "Add “keep everything else unchanged” when an edit changes too much.",
    },
  ],
};

const CHECKLIST: Record<Tab, string[]> = {
  create: ["Subject, action, place, light, framing", "Words in quotes for text", "Describe what is there, never “no…”", "Not right? Regenerate before rewriting"],
  edit: ["Verb + the exact thing + the new look", "Left, right, corner: positions work", "“Image 1, image 2…” to combine photos", "Paint an area for small or repeated things"],
};

/** The prompt guide inside the app: the same examples as kokon.app/prompt-guide, smaller,
 * bundled so it works offline. "Try it" copies an example into the prompt box. */
export function PromptTips() {
  const hasImages = useStore((s) => s.composer.refs.length > 0);
  const [tab, setTab] = useState<Tab>(hasImages ? "edit" : "create");
  const close = () => setState({ overlay: null });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const tryIt = (prompt: string) => {
    updateComposer({ prompt });
    close();
    focusPrompt(true);
  };

  return (
    <div className="modal-scrim" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="tips" role="dialog" aria-label="Prompt tips">
        <div className="tips-head">
          <Lightbulb size={20} />
          <h1>Write better prompts</h1>
          <span className="spacer" />
          <button className="icon-btn" onClick={close} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <p className="tips-lead">The model does what you write, quite literally. Real results, same seed on both sides.</p>
        <div className="seg tips-tabs" role="tablist">
          <button className={tab === "create" ? "on" : ""} role="tab" aria-selected={tab === "create"} onClick={() => setTab("create")}>
            New image
          </button>
          <button className={tab === "edit" ? "on" : ""} role="tab" aria-selected={tab === "edit"} onClick={() => setTab("edit")}>
            Edit a photo
          </button>
        </div>

        <div className="tips-body">
          {RULES[tab].map((r, i) => (
            <section className="tips-rule" key={r.title}>
              <h2>
                <span className="tips-num">{String(i + 1).padStart(2, "0")}</span>
                {r.title}
              </h2>
              <p className="tips-why">{r.why}</p>
              <div className="tips-pair" style={{ gridTemplateColumns: `repeat(${r.sides.length + (r.before ? 1 : 0)}, minmax(0, 1fr))` }}>
                {[...(r.before ? [{ kind: "src", label: "Original", prompt: "", img: r.before } as Side] : []), ...r.sides].map((s) => (
                  <div className="tips-side" key={s.label}>
                    <span className={`tips-tag ${s.kind}`}>
                      {s.kind === "bad" ? <X size={12} /> : s.kind === "good" ? <Check size={12} /> : null}
                      {s.label}
                    </span>
                    <img src={s.img} alt="" loading="lazy" />
                    {s.prompt && <div className="tips-prompt">{s.prompt}</div>}
                    {s.note && <span className="tips-note">{s.note}</span>}
                    {s.kind === "good" && s.prompt && tab === "create" && (
                      <button className="btn tips-try" onClick={() => tryIt(s.prompt)}>
                        Try it
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <p className="tips-lesson">{r.lesson}</p>
            </section>
          ))}

          <section className="tips-rule">
            <h2>Checklist</h2>
            <ul className="tips-check">
              {CHECKLIST[tab].map((c) => (
                <li key={c}>
                  <Check size={14} /> {c}
                </li>
              ))}
            </ul>
          </section>
        </div>

        <button className="btn tips-more" onClick={() => native.openLink(PROMPT_GUIDE_URL).catch(toastError)}>
          More examples in the full guide <ExternalLink size={13} />
        </button>
      </div>
    </div>
  );
}

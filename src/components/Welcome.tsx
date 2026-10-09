import { ChevronLeft, ChevronRight, CloudOff, Cpu, EyeOff, Infinity as Unlimited } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import mark from "../assets/kokon-mark.svg?raw";
import demo from "../assets/showcase/kokon-demo.mp4";
import expand from "../assets/showcase/expand.jpg";
import poster from "../assets/showcase/fatto-in-casa.jpg";
import relight from "../assets/showcase/relight.jpg";
import remove from "../assets/showcase/remove.jpg";
import replace from "../assets/showcase/replace.jpg";
import restyle from "../assets/showcase/restyle.jpg";
import text from "../assets/showcase/text.jpg";
import { api } from "../lib/api";
import { isMac } from "../lib/platform";
import { setState, toastError, useStore } from "../lib/store";

const machine = isMac ? "Mac" : "PC";

/** The Kokon mark in the current text color (light and dark themes). */
export function KokonMark({ size = 40 }: { size?: number }) {
  const svg = mark.replace(/fill="#[0-9A-Fa-f]{6}"/g, 'fill="currentColor"').replace(/width="\d+" height="\d+"/, `width="${size}" height="${size}"`);
  return <span className="kokon-mark" aria-hidden dangerouslySetInnerHTML={{ __html: svg }} />;
}

/** Remembers that the welcome screen was seen (in the app config, so it survives updates). */
async function markSeen() {
  try {
    const cfg = await api.config();
    const ui = (cfg.ui ?? {}) as Record<string, unknown>;
    if (!ui.welcomeSeen) await api.saveConfig({ ...cfg, ui: { ...ui, welcomeSeen: true } });
  } catch (e) {
    toastError(e);
  }
}

const PROMISES = [
  { icon: Cpu, title: `Made on your ${machine}`, text: "Your graphics chip creates every image. Photos and prompts never leave this computer." },
  { icon: CloudOff, title: "Online only once", text: "Setup downloads the model. After that Kokon works without an internet connection." },
  { icon: Unlimited, title: "No limits", text: "No account, no credits, no subscription, no queue. Create as much as you like." },
  { icon: EyeOff, title: "Private mode", text: "Click the eye icon and nothing is kept: no gallery, no files, no history." },
];

/** First launch (and Settings → General → Show welcome screen): what Kokon is and why it is private. */
export function Welcome() {
  const needsSetup = useStore((s) => s.needsSetup);
  const close = () => {
    setState({ overlay: null });
    markSeen();
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return (
    <div className="modal-scrim">
      <div className="welcome" role="dialog" aria-label="Welcome to Kokon">
        <KokonMark size={44} />
        <h1>Welcome to Kokon</h1>
        <p className="welcome-lead">
          A private image studio that lives on your {machine}. Describe a new image, or a change to a photo, in plain words. Like a cocoon, Kokon is a closed space where something new
          takes shape, and nothing gets out unless you take it out.
        </p>
        <div className="welcome-grid">
          {PROMISES.map(({ icon: Icon, title, text }) => (
            <div className="welcome-item" key={title}>
              <Icon size={20} />
              <strong>{title}</strong>
              <span>{text}</span>
            </div>
          ))}
        </div>
        <button className="btn primary big" onClick={close}>
          {needsSetup ? "Set up Kokon" : "Start creating"}
        </button>
        <span className="muted welcome-foot">You can open this again from Settings → General.</span>
      </div>
    </div>
  );
}

const SLIDES: { src: string; video?: boolean; title: string; prompt: string }[] = [
  { src: demo, video: true, title: "One project, start to finish", prompt: "Create, edit, combine and expand: the whole flow in under a minute." },
  { src: remove, title: "Remove", prompt: "“Remove all the people from the beach”" },
  { src: replace, title: "Replace", prompt: "“Replace the orange car with a blue vintage Vespa scooter”" },
  { src: relight, title: "Relight", prompt: "“Make it golden hour, warm low sunlight”" },
  { src: restyle, title: "Restyle", prompt: "“Turn it into a watercolor illustration”" },
  { src: text, title: "Change text", prompt: "“Change the neon sign to say ‘OPEN LATE’”" },
  { src: expand, title: "Expand", prompt: "Expand to 16:9 · “continue the beach”" },
];

/** What Kokon can do, shown while the model downloads (the wait is the best moment for it). */
export function Showcase() {
  const [i, setI] = useState(0);
  const video = useRef<HTMLVideoElement>(null);
  const slide = SLIDES[i];
  const go = (d: number) => setI((n) => (n + d + SLIDES.length) % SLIDES.length);
  // Photos advance every 6 s; the video advances when it ends.
  useEffect(() => {
    if (slide.video) return;
    const t = window.setTimeout(() => go(1), 6000);
    return () => window.clearTimeout(t);
  }, [i]);
  return (
    <div className="showcase">
      <div className="showcase-head">
        <strong>While you wait: what you can do</strong>
        <span className="muted">
          {i + 1} / {SLIDES.length}
        </span>
      </div>
      <div className="showcase-media">
        {slide.video ? (
          <video ref={video} key={slide.src} src={slide.src} poster={poster} autoPlay muted playsInline onEnded={() => go(1)} />
        ) : (
          <img key={slide.src} src={slide.src} alt={`${slide.title}: before and after`} />
        )}
        <button className="showcase-nav prev" onClick={() => go(-1)} aria-label="Previous">
          <ChevronLeft size={18} />
        </button>
        <button className="showcase-nav next" onClick={() => go(1)} aria-label="Next">
          <ChevronRight size={18} />
        </button>
      </div>
      <div className="showcase-caption">
        <strong>{slide.title}</strong>
        <span>{slide.prompt}</span>
      </div>
      <div className="showcase-dots">
        {SLIDES.map((s, n) => (
          <button key={s.src} className={n === i ? "on" : ""} onClick={() => setI(n)} aria-label={s.title} />
        ))}
      </div>
    </div>
  );
}

/** Shown once everything is installed, before the first Start. */
export function ReadyTips() {
  return (
    <div className="ready-tips">
      <strong>Three ways to begin</strong>
      <ol>
        <li>Drop or paste a photo, then describe the change: “make it golden hour”.</li>
        <li>
          Switch to <b>New image</b> and describe something that does not exist yet.
        </li>
        <li>
          Use <b>Paint area</b> to brush over part of a photo, and only that part changes.
        </li>
      </ol>
      <span className="muted">Want nothing saved? Turn on Private mode with the eye icon at the top.</span>
    </div>
  );
}

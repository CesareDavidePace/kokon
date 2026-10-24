// The clock of the "Kokon 0.11" video (composition Update011): the music's bar grid, the shots
// (cut on bars or beats), the voice-over cues (voiceover-0.11.md) and the frames of the actions
// the voice names. Frames at 30 fps from the start of the video.
import voice from "./voice-0.11.json";
import { FPS, type Cue, type Lines, sentenceAt, type VoiceSet } from "./timeline";

// Music: one track played whole under the video (no cuts, one steady background), on
// the video's beat grid (118 BPM): each plays from its first bar line. `stop`: where it ends, its own
// dead stop or the end of a fade; `end`: the video's last frame; `fade`: frames of fade-out before `end`.
const BEAT = 60 / 118;
/** Frame of beat `k`, and of bar `k` (4 beats). */
export const beat = (k: number) => Math.round(k * BEAT * FPS);
export const bar = (k: number) => beat(4 * k);
/** The launch film's house-4 (92 s), chosen over two new ACE-Step takes: an 8-s intro,
 * the groove, a softer part from bar 24 (safe by default, the server), the groove back on bar 32
 * ("Kokon 0.11"); faded out over the end card. */
export const MUSIC = { src: "music-0.11-house4.m4a", from: 0.288, stop: bar(38), end: bar(38), fade: 75 };
export const STOP = MUSIC.stop;

/** Shot starts (frames). */
export const SHOT = {
  hook: 0,
  idea: bar(2),
  /** Setup, step 1: Settings → Other devices on the computer. */
  password: bar(7),
  /** Step 2: the phone's camera on the code, then the sign-in. */
  scan: bar(10),
  login: bar(12),
  /** Step 3: Add to Home Screen, the icon. */
  home: bar(14),
  use: bar(16),
  safe: bar(22),
  away: beat(4 * 23 + 3),
  server: bar(26),
  scene: bar(32),
  end: bar(34),
};
export const UPDATE011_FRAMES = MUSIC.end;

const json = voice as unknown as Lines;
export const VOICE_011: VoiceSet = { dir: "voice-0.11", lines: json };
const span = (line: number, i: number) => VOICE_011.lines[String(line)].sentences[i];
const cue = (line: number, at: number, from = 0, to = from): Cue => ({ line, from, to, at, set: VOICE_011 });
const all = (line: number, at: number): Cue => cue(line, at, 0, VOICE_011.lines[String(line)].sentences.length - 1);

/** Line 3's second sentence comes in three parts (gen_voice.py V011_CUTS): sentences 1, 2, 3; line
 * 4's first in two ("Pick a photo," / "describe the change."), so "Your PC…" is its sentence 2. */
export const CUES_011: Cue[] = [
  all(1, 12),
  all(2, SHOT.idea + 12),
  cue(3, SHOT.password + 12, 0),
  cue(3, SHOT.scan + 16, 1),
  cue(3, SHOT.login + 30, 2),
  cue(3, SHOT.home + 10, 3),
  cue(4, SHOT.use + 14, 0, 1),
  // "Your PC does the work…" as the photo leaves the phone (set below, from the composer's clock).
  cue(4, 0, 2),
  cue(5, SHOT.safe + 12, 0),
  cue(5, SHOT.away + 8, 1),
  all(6, SHOT.server + 10),
  cue(7, SHOT.scene + 10, 0, 1),
];
const cueOf = (n: number, i: number) => CUES_011.find((c) => c.line === n && c.from <= i && i <= c.to)!;
/** Frame where sentence `i` of line `n` starts and ends. */
export const said = (n: number, i: number) => sentenceAt(cueOf(n, i), i);
export const saidEnd = (n: number, i: number) => said(n, i) + Math.round((span(n, i)[1] - span(n, i)[0]) * FPS);

/** The phone in use (shot `use`): the library, Beach day tapped (it opens on its sunset version);
 * on "Pick a photo" a swipe back to the beach photo; the prompt tapped (the writing screen, the
 * keyboard) before "describe", typed from it, Send; the photo travels to the computer, the job
 * runs there, the result travels back and develops into view; the actions row slid to Save, Download. */
export const PROMPT_011 = "Put him on a pink flamingo, surfing a wave";
const U = SHOT.use;
export const USE = {
  open: U + 10,
  /** The finger on the picture, dragging it right, let go: the beach slides in from the left. */
  swipe: U + 28,
  drag: 10,
  swiped: U + 44,
  prompt: 0,
  type: said(4, 1),
  cps: 26,
  send: 0,
  /** The photo on its way to the computer, the job, the result on its way back. */
  out: 0,
  job: 0,
  back: 0,
  result: 0,
  /** The actions row slid left by a finger, so Save is in reach. */
  row: 0,
  download: 0,
};
USE.prompt = USE.type - 6;
USE.send = USE.type + Math.ceil((PROMPT_011.length * FPS) / USE.cps) + 10;
USE.out = USE.send + 2;
USE.job = USE.out + 22;
USE.back = USE.job + 110;
USE.result = USE.back + 22;
USE.download = Math.min(USE.result + 40, SHOT.safe - 16);
USE.row = USE.download - 18;
CUES_011[7].at = USE.out + 4;

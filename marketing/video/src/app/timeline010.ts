// The clock of the "Kokon 0.10" update video (composition Update010): the music's beat grid,
// the shots (cut on beats), the voice-over cues (voiceover-0.10.md) and the frames of the
// actions the voice names. Frames at 30 fps from the start of the video.
import voice from "./voice-0.10.json";
import { FPS, type Cue, type Lines, sentenceAt, type VoiceSet } from "./timeline";

// Music: `src` from `from` seconds; its beats fall every BEAT seconds from BEAT0 (seconds into
// the video). Placeholder until the new take: the launch track's tail (beat 76 of its grid at
// 47.09 s), so its drop (60.3 s) lands on beat 26, the hatch.
export const MUSIC = { src: "music.m4a", from: 8.45 + 76 * (60 / 118) };
const BEAT = 60 / 118;
const BEAT0 = 0;
/** Frame of beat `k` of the video. */
export const beat = (k: number) => Math.round((BEAT0 + k * BEAT) * FPS);

/** Shot starts (frames), each on a beat. */
export const SHOT = {
  title: 0,
  composer: beat(6),
  card: beat(10),
  making: beat(18),
  logo: beat(24),
  hatch: beat(26), // the drop
  model: beat(30),
  settings: beat(42),
  /** On the beat before "Turn it on with one click." (set below, from the voice). */
  pill: 0,
  look: beat(56),
  bar: beat(59),
  silk: beat(62),
  end: beat(66),
};
export const UPDATE_FRAMES = beat(80);

// Voice-over: public/voice-0.10/NN.wav and voice-0.10.json from `gen_voice.py build`. Until the
// takes exist the json holds estimated spans ("_provisional"): the cues still time the
// captions and the music's ducking, but no voice plays.
const json = voice as unknown as Lines & { _provisional?: boolean };
export const VOICE_READY = !json._provisional;
export const UPDATE_VOICE: VoiceSet = { dir: "voice-0.10", lines: json };

const cue = (line: number, at: number, from = 0, to = from): Cue => ({ line, from, to, at, set: UPDATE_VOICE });
const all = (line: number, at: number): Cue => cue(line, at, 0, UPDATE_VOICE.lines[String(line)].sentences.length - 1);
const span = (line: number, i: number) => UPDATE_VOICE.lines[String(line)].sentences[i];

/** Prompt+ in the composer: typing on "Type a few words.", the prompt written while the voice
 * says "Prompt Plus writes the full prompt." (its sentence starts at `write`). */
const WRITE = SHOT.card + 7;
export const CUES_010: Cue[] = [
  all(1, 10),
  all(2, WRITE - Math.round(span(2, 1)[0] * FPS)),
  all(3, SHOT.hatch + 10),
  all(4, SHOT.settings + 6),
  all(5, SHOT.look + 6),
  all(6, SHOT.end + 12),
];
export const LINE = (n: number) => CUES_010[n - 1];
export const said = (n: number, i: number) => sentenceAt(LINE(n), i);
/** Frame where sentence `i` of line `n` ends. */
export const saidEnd = (n: number, i: number) => LINE(n).at + Math.round((span(n, i)[1] - span(n, 0)[0]) * FPS);
/** The last beat at or before frame `f`. */
const beatBefore = (f: number) => beat(Math.floor((f / FPS - BEAT0) / BEAT));
SHOT.pill = Math.min(beatBefore(said(4, 2) + 2), SHOT.look - 45);

/** LoRA: "Load a LoRA…" clicked as the voice says "Load one in Settings.", the pill on "one
 * click" (but before the cut to the new look). */
export const LORA_CLICK = { load: said(4, 1) + 6, pill: Math.min(saidEnd(4, 2) - 8, SHOT.look - 14) };

/** The Prompt+ shots (frames of the video), in the shape Create uses (scenes/AppScenes.tsx). */
export const PROMPT = {
  ask: SHOT.composer + 14,
  typeCps: 48,
  pill: SHOT.card - 14,
  send: SHOT.card - 1,
  job: SHOT.card,
  write: WRITE,
  written: SHOT.making - 14,
  leave: SHOT.making - 6,
  form: SHOT.logo - 1,
  done: SHOT.hatch - 6,
  hatch: SHOT.hatch,
};

// The clock of the "use cases" video (composition UseCases): the beat grid of its music, each beat's
// length (in musical beats, so every cut lands on one) and mood, where its voice-over line starts,
// and the word the wipe lands on. Frames at 30 fps from the start of the video. Retime here:
// pictures, voice, effects and music follow.
import voice from "./voice-usecases.json";
import words from "./voice-usecases-words.json";
import { FPS, type Cue, type Lines, type VoiceSet } from "./timeline";

// Music: public/music-usecases.m4a, cut by audio/make_update_music.py usecases from a close version
// of the launch film's house-4 (118 BPM), starting on a bar line. Its lighter part covers the
// serious block (restore to grandma, 14 bars), its drop lands on the flamingo cut, and it falls
// silent before the last words.
export const MUSIC = { src: "music-usecases.m4a", from: 0 };
export const BPM = 118;
const BEAT = 60 / BPM;
/** Frame of beat `k` (fractions allowed) of the video. */
export const beat = (k: number) => Math.round(k * BEAT * FPS);

export type Mood = "fun" | "serious" | "message" | "end";

/** Voice-over: public/voice-usecases/NN.wav (audio/gen_voice.py build-usecases); voice-usecases.json
 * holds where each sentence starts and ends, voice-usecases-words.json each word's start (Whisper). */
export const USECASES_VOICE: VoiceSet = { dir: "voice-usecases", lines: voice as unknown as Lines };
const WORDS = words as unknown as Record<string, [string, number][]>;
/** Seconds into line `line` where the `nth` word starting with `w` (lower case) starts. */
export function wordAt(line: number, w: string, nth = 0): number {
  const hits = WORDS[String(line)].filter(([t]) => t.toLowerCase().replace(/[^a-z']/g, "").startsWith(w));
  if (!hits[nth]) throw new Error(`no "${w}" in line ${line}`);
  return hits[nth][1];
}

/** One beat of the video: its length in musical beats, its mood, its voice-over line, how long
 * after the cut the line starts (s), and for the photo beats the word the change lands on. */
export interface Seg {
  id: string;
  len: number;
  mood: Mood;
  line: number;
  lead: number;
  word?: string;
}

export const SEGS: Seg[] = [
  { id: "why", len: 15, mood: "fun", line: 1, lead: 0.25 },
  { id: "tourists", len: 11, mood: "fun", line: 2, lead: 0.3, word: "gone" },
  { id: "blink", len: 10, mood: "fun", line: 3, lead: 0.3, word: "fixed" },
  // The music's lighter part, 14 bars, played lower (MUSIC_GAIN).
  { id: "restore", len: 13, mood: "serious", line: 4, lead: 0.4, word: "restore" },
  { id: "colour", len: 9, mood: "serious", line: 5, lead: 1.2, word: "back" },
  { id: "message", len: 12, mood: "message", line: 6, lead: 0.5 },
  { id: "child", len: 12, mood: "serious", line: 7, lead: 0.4, word: "they" },
  { id: "grandma", len: 10, mood: "serious", line: 8, lead: 0.4, word: "bring" },
  // The music's drop is on this cut.
  { id: "flamingo", len: 10, mood: "fun", line: 9, lead: 0.3, word: "upgraded" },
  { id: "hair", len: 8, mood: "fun", line: 10, lead: 0.5, word: "before" },
  { id: "jacket", len: 8, mood: "fun", line: 11, lead: 0.9, word: "before" },
  // The montage: three edits from beats 0, 3 and 6 of it (MONTAGE), each cut on the beat.
  { id: "home", len: 11, mood: "fun", line: 12, lead: 0.3, word: "before" },
  { id: "cv", len: 10, mood: "fun", line: 13, lead: 0.9, word: "into" },
  { id: "invite", len: 9, mood: "fun", line: 14, lead: 1.1, word: "writes" },
  { id: "end", len: 20, mood: "end", line: 15, lead: 0.6 },
];
/** Beats of each montage shot, from the start of the montage. */
export const MONTAGE = [0, 3, 6];

/** Start frame of each beat, by id, and the video's length. */
export const START: Record<string, number> = {};
let k = 0;
for (const s of SEGS) {
  START[s.id] = beat(k);
  k += s.len;
}
export const TOTAL_BEATS = k;
export const USECASES_FRAMES = beat(k);
const beatOf = (id: string) => {
  let b = 0;
  for (const s of SEGS) {
    if (s.id === id) return b;
    b += s.len;
  }
  throw new Error(id);
};
export const seg = (id: string) => SEGS.find((s) => s.id === id)!;
/** Frame (from the segment's start) of beat `b` of the segment. */
export const at = (id: string, b: number) => beat(beatOf(id) + b) - START[id];
export const segFrames = (id: string) => beat(beatOf(id) + seg(id).len) - START[id];

/** Frame (from the segment's start) where its line starts, and where a word of it starts. */
export const voiceAt = (id: string) => Math.round(seg(id).lead * FPS);
export const wordFrame = (id: string, w: string, nth = 0) => voiceAt(id) + Math.round(wordAt(seg(id).line, w, nth) * FPS);
/** Frame (from the segment's start) where sentence `i` of its line starts. */
export const sentenceFrame = (id: string, i: number) => voiceAt(id) + Math.round(USECASES_VOICE.lines[String(seg(id).line)].sentences[i][0] * FPS);

/**
 * A photo beat, in frames from its start: the prompt box comes in, the prompt is typed, Send, the
 * wipe, and the answer, which lands with the spoken word (`word`). Fun wipes are quick, serious
 * ones slow.
 */
export function photoTimes(id: string) {
  const s = seg(id);
  const punch = wordFrame(id, s.word!) + 2;
  const wipeLen = s.mood === "fun" ? 12 : 24;
  const wipe = punch - wipeLen;
  const send = wipe - 4;
  return { chip: 4, from: 8, to: send - 6, send, wipe, wipeEnd: punch, punch };
}

/** The whole voice-over: each line from its segment's lead. */
export const CUES_USECASES: Cue[] = SEGS.map((s) => ({ line: s.line, from: 0, to: USECASES_VOICE.lines[String(s.line)].sentences.length - 1, at: START[s.id] + voiceAt(s.id), set: USECASES_VOICE }));

/** The music's level by mood (before the voice ducks it): a little lower through the serious block,
 * lower still under the privacy line. Each change ramps over the beat before its cut, so the cut
 * (and the drop on the flamingo) lands at the new level. */
export function musicGain(f: number): number {
  const ramp = beat(1);
  const level = (m: Mood) => (m === "serious" ? 0.62 : m === "message" ? 0.45 : 1);
  let g = level(SEGS[0].mood);
  for (const s of SEGS.slice(1)) {
    const a = START[s.id];
    const next = level(s.mood);
    if (f >= a) g = next;
    else {
      if (f > a - ramp) g += ((next - g) * (f - (a - ramp))) / ramp;
      break;
    }
  }
  return g;
}

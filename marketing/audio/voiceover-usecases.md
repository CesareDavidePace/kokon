# Voiceover script: "use cases" video (2026-10-23)

About 85 s, English, for the site and YouTube. The voice tells the story; the on-screen captions stay
as they are. Honest claims only (no "keeps your face", no "best", "free for personal use", "no daily
limit" means usage limits); every image on screen is an example made with Kokon.

One narrator, the launch voice, in a different tone per section (`Tone`, the settings are in
gen_voice.py `USECASES_TONES`):
- **punchy** (the opening): base model, more exaggeration, firm and quick.
- **playful** (fun beats): Chatterbox Turbo, cloned from the launch voice, with its paralinguistic tags
  (`[chuckle]`, `[happy]`...) where a smile helps.
- **tender** (grandparents, family, kids): base model, low exaggeration, slower and softer.
- **intimate** (the privacy line, alone on dark): base model, the lowest exaggeration, slowest.
- **warm** (the close): base model, the launch take's settings.

Order: beat 10 (kids) comes right after beat 5 (privacy), so the serious part is one block (3, 4, 5,
10, 6) and the music has one quiet section in the middle.

| # | Beat | Tone | Line |
|---|---|---|---|
| 1 | Why Kokon? (opening) | punchy | Why Kokon? Private: your photos never leave your computer. Unlimited: no subscription, no daily limit. |
| 2 | 01 tourists | playful | Photobombers in your holiday photo? [chuckle] Gone. |
| 3 | 02 blink | playful | Someone always blinks. Add the other shot, and it's fixed. |
| 4 | 03 restore | tender | Your grandparents' wedding photo, creased and scratched. Restore it… |
| 5 | 03 colour | tender | and bring it back in colour. |
| 6 | privacy, text on dark | intimate | Your family's memories are yours. They never leave your computer. |
| 7 | 05 child | tender | The same goes for photos of your kids. They don't belong on someone else's server. |
| 8 | 04 grandma | tender | Someone missing from the photo? Bring them in. |
| 9 | 17 flamingo | playful | [happy] And when it's time to have fun: your holiday, upgraded. |
| 10 | 13 hair | playful | Try a new haircut before the hairdresser. |
| 11 | 14 jacket | playful | Try on a jacket before you buy it. |
| 12 | montage 10, 11, 12 | playful | Redo your home before you spend a cent. |
| 13 | 16 CV | playful | Turn a kitchen selfie into a headshot. |
| 14 | 15 invite | playful | It even writes the text. |
| 15 | end card | warm | Kokon. Edit photos by describing the change. On your own computer, free for personal use. Get it at kokon dot app. |

Takes: `gen_voice.py audition-usecases` (every sentence, a few seeds each, in voice/usecases/audition-<stamp>),
then `gen_voice.py build-usecases --from <that folder> --pick LINE.SENTENCE:SEED,...`, which writes
voice/usecases/final-<stamp>/ and copies the lines to video/public/voice-usecases with their sentence spans in
video/src/app/voice-usecases.json. The launch and 0.10 takes are never touched.

# Voiceover script: "Kokon 0.11" video (2026-10-23)

About 70 s, for Reddit (it autoplays muted, so the on-screen text carries every message on its
own), then YouTube. Same voice as the 0.10 video (Chatterbox base, built-in voice, one sentence
at a time). "0.11" is spelled out for the voice. Honest claims only: the other devices use the
computer's GPU over the home network (plain HTTP, so a network you trust); off until a password
is set; 5 wrong tries make that device wait; from outside, Tailscale or another VPN, never port
forwarding; Docker image for NVIDIA only; requirements 12 GB GPU or Apple Silicon.

| # | Shot | Window | Line |
|---|---|---|---|
| 1 | Hook: iPhone, the flamingo photo | 0:00–0:04 | Kokon now works on your phone. |
| 2 | The idea: PC, MacBook, iPhone, Home Wi-Fi | 0:04–0:14 | Your computer with the graphics card makes the images. Your phone, tablet or laptop just opens Kokon in a browser. No cloud, no account. |
| 3 | Setup in 3 steps | 0:14–0:32 | Open Settings, Other devices, and choose a password. Scan the code, sign in once, and add Kokon to your home screen. |
| 4 | Using it from the phone | 0:32–0:44 | Pick a photo, describe the change. Your PC does the work and sends it back. |
| 5 | Safe by default | 0:44–0:52 | It stays off until you set a password. Away from home, use Tailscale or another VPN, never an open port. |
| 6 | Server and Docker | 0:52–1:04 | Have a home server? Run Kokon on Linux, or in Docker with an Nvidia card. One file, one command. |
| 7 | Close | 1:04–1:12 | Kokon zero point eleven. Your GPU, on every screen. |

Takes: `gen_voice.py audition-011` (every sentence in two tones, calmer and expressive, three
seeds each, in voice/update-0.11/audition-<stamp>, with pick.html to choose by ear), then
`gen_voice.py build-011 --from <that folder> --pick LINE.SENTENCE:SEED,...`, which writes
voice/update-0.11/final-<stamp>/ and copies the lines to video/public/voice-0.11 with their
sentence spans in video/src/app/voice-0.11.json.

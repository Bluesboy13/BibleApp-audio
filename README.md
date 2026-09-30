# BibleApp audio

Recorded King James Bible for the [KJV Bible app](https://github.com/Bluesboy13/BibleApp), read by
"Daniel" — the Kokoro TTS voice `bm_daniel` (Apache-2.0; a synthetic voice, not a recording of a real person).

- `<book 01-66>/<chapter 001>.m4a` — one chapter, AAC mono.
- `<book>/<chapter>.json` — verse start (`s`) and end (`e`) times in seconds, and total duration (`d`).

Served with GitHub Pages so the app can stream it. The **Record audio** workflow
(`.github/workflows/record.yml`) records everything in 20 parallel slices on GitHub Actions and
commits the results here; re-running it only records chapters that are missing.

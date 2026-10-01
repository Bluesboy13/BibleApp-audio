"""Transcribe each verse of a recorded Greek chapter and compare with the text (catches clipped or garbled verses)."""
import difflib, json, sys, unicodedata, re, subprocess
from faster_whisper import WhisperModel
sys.path.insert(0, "app/tools/greek")
from generate import monotonic, load_bible

def norm(t):
    t = unicodedata.normalize("NFD", t.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]|\d|_", "", "".join(c for c in t if not unicodedata.combining(c)))).replace("ς", "σ").strip()

b, c, base = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t = json.load(open(base + ".json"))
verses = load_bible()[b][c - 1]
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", base + ".m4a", "-ar", "16000", "-ac", "1", "full.wav"], check=True)
m = WhisperModel("small", device="cpu", compute_type="int8")
out = []
for i, (s, e) in enumerate(zip(t["s"], t["e"])):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "full.wav", "-ss", str(max(0, s - 0.1)), "-to", str(e + 0.1), "v.wav"], check=True)
    segs, _ = m.transcribe("v.wav", language="el", beam_size=5)
    heard = " ".join(x.text for x in segs)
    want = monotonic(verses[i])
    r = difflib.SequenceMatcher(None, norm(heard), norm(want)).ratio()
    out.append((r, i + 1, want, heard))
    print(f"{i + 1:3d} {r:.2f} | {want}\n          heard: {heard}", flush=True)
print("mean", sum(x[0] for x in out) / len(out), "below 0.6:", [x[1] for x in out if x[0] < 0.6])

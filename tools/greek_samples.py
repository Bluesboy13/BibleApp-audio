"""Greek voice samples with Chatterbox Multilingual (MIT). Writes samples/greek/*.mp3 and timing info."""
import os, subprocess, time, unicodedata, urllib.request
import numpy as np, soundfile as sf, torch

OUT = "samples/greek"
os.makedirs(OUT, exist_ok=True)
TEXTS = {
    "john1": "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν Θεόν, καὶ Θεὸς ἦν ὁ λόγος. Οὗτος ἦν ἐν ἀρχῇ πρὸς τὸν Θεόν. Πάντα δι᾿ αὐτοῦ ἐγένετο, καὶ χωρὶς αὐτοῦ ἐγένετο οὐδὲ ἓν ὃ γέγονεν.",
    "psalm22": "Κύριος ποιμαίνει με, καὶ οὐδέν με ὑστερήσει. Εἰς τόπον χλόης ἐκεῖ με κατεσκήνωσεν· ἐπὶ ὕδατος ἀναπαύσεως ἐξέθρεψέ με.",
}


def monotonic(t):
    """Polytonic -> modern monotonic Greek (keep one acute accent), which the model reads best."""
    t = unicodedata.normalize("NFD", t)
    t = t.replace("̀", "́").replace("͂", "́")
    t = "".join(ch for ch in t if ch not in "̓̔ͅ᾽᾿’")
    return unicodedata.normalize("NFC", t)


def daniel_reference():
    from kokoro_onnx import Kokoro
    base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    for f in ["kokoro-v1.0.onnx", "voices-v1.0.bin"]:
        if not os.path.exists(f):
            urllib.request.urlretrieve(base + f, f)
    k = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    a, sr = k.create("In the beginning God created the heaven and the earth. And the earth was without form, and void; "
                     "and darkness was upon the face of the deep.", voice="bm_daniel", speed=0.9, lang="en-gb")
    sf.write("daniel_ref.wav", a, sr)
    return "daniel_ref.wav"


def mp3(wav, sr, name):
    sf.write(f"{OUT}/{name}.wav", wav, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{OUT}/{name}.wav", "-ac", "1", "-b:a", "96k", f"{OUT}/{name}.mp3"], check=True)
    os.remove(f"{OUT}/{name}.wav")


from chatterbox.mtl_tts import ChatterboxMultilingualTTS
model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
ref = daniel_reference()
log = []
for key, text in TEXTS.items():
    for voice, prompt in (("default", None), ("daniel", ref)):
        torch.manual_seed(5)
        t0 = time.time()
        kw = {"audio_prompt_path": prompt} if prompt else {}
        wav = model.generate(monotonic(text), language_id="el", **kw)
        a = wav.squeeze().cpu().numpy()
        secs = len(a) / model.sr
        mp3(a, model.sr, f"{key}_{voice}")
        log.append(f"{key}_{voice}: {secs:.1f}s audio in {time.time() - t0:.0f}s")
        print(log[-1], flush=True)
open(f"{OUT}/timing.txt", "w").write("\n".join(log) + "\n")

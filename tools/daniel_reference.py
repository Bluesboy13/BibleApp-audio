"""Make a short clip of the Kokoro 'Daniel' voice (bm_daniel) to use as the voice to copy."""
import os, urllib.request
import soundfile as sf
from kokoro_onnx import Kokoro

base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
for f in ["kokoro-v1.0.onnx", "voices-v1.0.bin"]:
    if not os.path.exists(f):
        urllib.request.urlretrieve(base + f, f)
k = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
a, sr = k.create("In the beginning God created the heaven and the earth. And the earth was without form, and void; "
                 "and darkness was upon the face of the deep.", voice="bm_daniel", speed=0.9, lang="en-gb")
sf.write("daniel_ref.wav", a, sr)
print("daniel_ref.wav", round(len(a) / sr, 1), "s")

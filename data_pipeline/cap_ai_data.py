import os
import torchaudio
from utils.audio_utils import load_and_normalize

SRC = "ai voice"
DST = "processed/ai"

os.makedirs(DST, exist_ok=True)

idx = 0
for lang in os.listdir(SRC):
    lang_dir = os.path.join(SRC, lang)
    if not os.path.isdir(lang_dir):
        continue

    for f in os.listdir(lang_dir):
        if not f.lower().endswith((".wav", ".mp3", ".aac")):
            continue

        path = os.path.join(lang_dir, f)
        wav = load_and_normalize(path)
        torchaudio.save(
            os.path.join(DST, f"ai_{idx}.wav"),
            wav.unsqueeze(0),
            16000
        )
        idx += 1

print("AI clean files:", idx)

import os
import torch
import torchaudio
import random

INPUT_BASE = "processed/ai"
OUTPUT_BASE = "processed/ai_augmented"

os.makedirs(OUTPUT_BASE, exist_ok=True)

def degrade(wav, sr):
    # Add noise
    noise = torch.randn_like(wav) * 0.005
    wav = wav + noise

    # Band-limit (telephone-like)
    wav = torchaudio.functional.lowpass_biquad(wav, sr, 3400)

    # Random gain
    wav = wav * random.uniform(0.7, 1.1)

    return wav.clamp(-1, 1)

count = 0

for lang in os.listdir(INPUT_BASE):
    in_dir = os.path.join(INPUT_BASE, lang)
    out_dir = os.path.join(OUTPUT_BASE, lang)

    if not os.path.isdir(in_dir):
        continue

    os.makedirs(out_dir, exist_ok=True)

    for f in os.listdir(in_dir):
        if not f.lower().endswith(".wav"):
            continue

        in_path = os.path.join(in_dir, f)
        out_path = os.path.join(out_dir, f)

        wav, sr = torchaudio.load(in_path)
        wav = wav.mean(dim=0, keepdim=True)

        wav = degrade(wav, sr)

        torchaudio.save(out_path, wav, sr)
        count += 1

print("AI artifact dataset complete:", count)

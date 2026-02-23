import os
import random
import torch
import torchaudio
import torchaudio.functional as F

# ================= CONFIG =================

INPUT_DIR = "human voice"
OUTPUT_DIR = "human_degraded"

TARGET_SR = 16000
MAX_SEC = 5
MAX_SAMPLES = TARGET_SR * MAX_SEC

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= AUDIO UTILS =================

def load_audio(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)

    if sr != TARGET_SR:
        wav = F.resample(wav, sr, TARGET_SR)

    if len(wav) > MAX_SAMPLES:
        wav = wav[:MAX_SAMPLES]

    wav = wav / (wav.abs().max() + 1e-9)
    return wav

# ================= DEGRADATIONS =================

def add_noise(wav):
    noise = torch.randn_like(wav) * random.uniform(0.002, 0.01)
    return wav + noise

def band_limit(wav):
    return torchaudio.functional.lowpass_biquad(
        wav, TARGET_SR, random.choice([3000, 3400, 4000])
    )

def lowpass(wav):
    return torchaudio.functional.lowpass_biquad(
        wav, TARGET_SR, random.choice([5000, 6000])
    )

def reduce_bit_depth(wav):
    levels = random.choice([64, 128])
    return torch.round(wav * levels) / levels

def resample_degrade(wav):
    tmp_sr = random.choice([8000, 12000])
    wav = F.resample(wav, TARGET_SR, tmp_sr)
    return F.resample(wav, tmp_sr, TARGET_SR)

def volume_jitter(wav):
    return wav * random.uniform(0.6, 1.1)

DEGRADATIONS = [
    add_noise,
    band_limit,
    lowpass,
    reduce_bit_depth,
    resample_degrade,
    volume_jitter,
]

# ================= MAIN =================

def process_language(lang):
    src = os.path.join(INPUT_DIR, lang)
    dst = os.path.join(OUTPUT_DIR, lang)
    os.makedirs(dst, exist_ok=True)

    files = [
        f for f in os.listdir(src)
        if f.lower().endswith((".wav", ".mp3"))
    ]

    print(f"{lang}: {len(files)} files")

    for fname in files:
        in_path = os.path.join(src, fname)
        out_path = os.path.join(dst, fname.replace(".mp3", ".wav"))

        try:
            wav = load_audio(in_path)

            # Apply 2–3 random degradations
            for fx in random.sample(DEGRADATIONS, k=random.randint(2, 3)):
                wav = fx(wav)

            wav = wav / (wav.abs().max() + 1e-9)
            torchaudio.save(out_path, wav.unsqueeze(0), TARGET_SR)

        except Exception as e:
            print("Skipping:", fname, "|", e)

# ================= RUN =================

if __name__ == "__main__":
    for lang in os.listdir(INPUT_DIR):
        lang_path = os.path.join(INPUT_DIR, lang)
        if os.path.isdir(lang_path):
            process_language(lang)

    print("\nHuman degraded dataset created successfully")
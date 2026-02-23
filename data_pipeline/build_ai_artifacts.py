import os
import random
import torchaudio
import torch

# ================= CONFIG =================
SRC_ROOT = "processed/ai"
OUT_ROOT = "processed/ai_artifacts"

TARGET_SR = 16000
MAX_PER_LANGUAGE = 1000   # upper limit only
random.seed(42)

# =========================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_resample(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    wav = wav / (wav.abs().max() + 1e-6)
    return wav

# ---------- AI ARTIFACT TRANSFORMS ----------
def codec_artifact(wav):
    # simulate vocoder / codec damage
    wav = torchaudio.functional.lowpass_biquad(
        wav, TARGET_SR, random.choice([2800, 3200, 3600])
    )
    wav = torchaudio.functional.highpass_biquad(
        wav, TARGET_SR, random.choice([60, 90, 120])
    )
    return wav

def temporal_instability(wav):
    # slight time jitter
    shift = random.randint(-80, 80)
    return torch.roll(wav, shifts=shift, dims=-1)

def harmonic_smear(wav):
    noise = torch.randn_like(wav) * random.uniform(0.0005, 0.002)
    return wav + noise

def artifact_chain(wav):
    if random.random() < 0.8:
        wav = codec_artifact(wav)
    if random.random() < 0.6:
        wav = temporal_instability(wav)
    if random.random() < 0.6:
        wav = harmonic_smear(wav)
    return wav.clamp(-1, 1)

# =========================================
def main():
    ensure_dir(OUT_ROOT)

    total = 0
    for lang in os.listdir(SRC_ROOT):
        src_lang = os.path.join(SRC_ROOT, lang)
        if not os.path.isdir(src_lang):
            continue

        out_lang = os.path.join(OUT_ROOT, lang)
        ensure_dir(out_lang)

        files = [f for f in os.listdir(src_lang) if f.endswith(".wav")]
        random.shuffle(files)
        files = files[:MAX_PER_LANGUAGE]

        print(f"Building AI ARTIFACTS [{lang}] → {len(files)} files")

        for i, fname in enumerate(files):
            src_path = os.path.join(src_lang, fname)

            try:
                wav = load_resample(src_path)
                wav = artifact_chain(wav)
                out_path = os.path.join(out_lang, f"ai_art_{i}.wav")
                torchaudio.save(out_path, wav, TARGET_SR)
                total += 1
            except Exception as e:
                print("Skipped:", src_path, e)

    print("\nAI ARTIFACT DATASET COMPLETE")
    print("Total AI artifact files:", total)


if __name__ == "__main__":
    main()
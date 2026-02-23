# audio_utils.py
import torchaudio
import torch

# FORCE SAFE BACKEND
torchaudio.set_audio_backend("sox_io")

TARGET_SR = 16000
MAX_SEC = 6
MAX_SAMPLES = TARGET_SR * MAX_SEC

def load_and_normalize(path):
    """
    Windows-safe streaming audio loader
    NEVER loads full file into memory
    """

    try:
        wav, sr = torchaudio.load(
            path,
            frame_offset=0,
            num_frames=MAX_SAMPLES
        )
    except Exception as e:
        raise RuntimeError(f"load failed: {e}")

    # convert to mono
    if wav.dim() > 1:
        wav = wav.mean(dim=0)

    # resample
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    # pad short audio
    if wav.numel() < MAX_SAMPLES:
        wav = torch.nn.functional.pad(
            wav, (0, MAX_SAMPLES - wav.numel())
        )

    # reject silent / broken files
    if wav.abs().mean() < 1e-4:
        raise RuntimeError("silent or corrupt")

    # normalize
    wav = wav / (wav.abs().max() + 1e-9)

    return wav

import torch
import torchaudio
import torch.nn.functional as F

# =========================
# CONFIG
# =========================
TARGET_SR = 16000
CHUNK_SECONDS = 4
CHUNK_SAMPLES = TARGET_SR * CHUNK_SECONDS


# =========================
# LOAD + NORMALIZE
# =========================
def load_audio(path: str) -> torch.Tensor:
    """
    Loads audio file and returns mono float tensor [T]
    """
    wav, sr = torchaudio.load(path)

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)

    # Resample
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    # Normalize safely
    wav = wav / (wav.abs().max() + 1e-9)

    return wav


# =========================
# CHUNKING (FOR LONG AUDIO)
# =========================
def chunk_audio(wav: torch.Tensor) -> list[torch.Tensor]:
    """
    Splits waveform into fixed-size chunks
    Returns list of [T] tensors
    """
    chunks = []

    if wav.numel() < CHUNK_SAMPLES:
        # Pad short audio
        pad = CHUNK_SAMPLES - wav.numel()
        wav = F.pad(wav, (0, pad))
        return [wav]

    for i in range(0, wav.numel(), CHUNK_SAMPLES):
        chunk = wav[i : i + CHUNK_SAMPLES]
        if chunk.numel() == CHUNK_SAMPLES:
            chunks.append(chunk)

    return chunks


# =========================
# FULL PIPELINE
# =========================
def load_chunks(path: str, device: str = "cpu") -> torch.Tensor:
    """
    Returns tensor of shape [B, T] ready for Wav2Vec2
    """
    wav = load_audio(path)
    chunks = chunk_audio(wav)

    batch = torch.stack(chunks)  # [B, T]
    return batch.to(device)
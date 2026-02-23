import torch
import torchaudio

SR = 16000
CHUNK_SEC = 3
CHUNK = SR * CHUNK_SEC

def load_chunk(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)

    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)

    wav = wav / (wav.abs().max() + 1e-9)

    if len(wav) < CHUNK:
        wav = torch.nn.functional.pad(wav, (0, CHUNK - len(wav)))
    else:
        wav = wav[:CHUNK]

    return wav
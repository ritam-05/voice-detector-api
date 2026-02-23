import torch
import torchaudio
import random

def degrade(wav, sr=16000):
    if random.random() < 0.5:
        wav = torchaudio.functional.lowpass_biquad(wav, sr, 3000)
    if random.random() < 0.5:
        wav = wav + torch.randn_like(wav) * 0.005
    return wav

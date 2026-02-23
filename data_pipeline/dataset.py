import os
import torch
import torchaudio
from torch.utils.data import Dataset

TARGET_SR = 16000
CHUNK_SAMPLES = TARGET_SR * 4

def degrade(wav):
    if torch.rand(1) < 0.5:
        wav = wav + 0.01 * torch.randn_like(wav)
    if torch.rand(1) < 0.5:
        wav = torchaudio.functional.lowpass_biquad(wav, TARGET_SR, 3000)
    return wav

def load_chunk(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    if len(wav) < CHUNK_SAMPLES:
        wav = torch.nn.functional.pad(wav, (0, CHUNK_SAMPLES - len(wav)))
    else:
        wav = wav[:CHUNK_SAMPLES]

    wav = wav / (wav.abs().max() + 1e-9)
    return wav

class VoiceDataset(Dataset):
    def __init__(self, root="processed"):
        self.samples = []

        for label, name in [(0, "human"), (1, "ai")]:
            folder = os.path.join(root, name)
            if not os.path.exists(folder):
                continue
            for f in os.listdir(folder):
                if f.endswith(".wav"):
                    self.samples.append((os.path.join(folder, f), label))

        print("Total audio files:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav = load_chunk(path)

        if label == 1:
            wav = degrade(wav)

        return wav.unsqueeze(0), torch.tensor([label], dtype=torch.float32)

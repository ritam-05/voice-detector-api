import os
import torch
import torchaudio
from torch.utils.data import Dataset
import random

TARGET_SR = 16000
CHUNK_SEC = 4
CHUNK_SAMPLES = TARGET_SR * CHUNK_SEC

class Stage1Dataset(Dataset):
    def __init__(self, root):
        self.samples = []

        for label, y in [("human", 0), ("ai", 1)]:
            folder = os.path.join(root, label)
            for f in os.listdir(folder):
                if f.endswith(".wav"):
                    self.samples.append((os.path.join(folder, f), y))

        random.shuffle(self.samples)
        print(f"Total audio files: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_chunk(self, path):
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)

        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

        if wav.shape[0] < CHUNK_SAMPLES:
            wav = torch.nn.functional.pad(wav, (0, CHUNK_SAMPLES - wav.shape[0]))
            return wav

        start = random.randint(0, wav.shape[0] - CHUNK_SAMPLES)
        return wav[start:start + CHUNK_SAMPLES]

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = self._load_chunk(path)
        return x, torch.tensor(y, dtype=torch.float32)

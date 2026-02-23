import os
import torch
from torch.utils.data import Dataset
from utils.stage2_audio_utils import load_chunk

class Stage2Dataset(Dataset):
    def __init__(self, root):
        self.items = []

        for label, name in [(0, "human"), (1, "ai")]:
            folder = os.path.join(root, name)
            for f in os.listdir(folder):
                if f.endswith(".wav"):
                    self.items.append((os.path.join(folder, f), label))

        if len(self.items) == 0:
            raise RuntimeError("Stage2 dataset empty")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav = load_chunk(path)
        return wav, torch.tensor(label, dtype=torch.float32)
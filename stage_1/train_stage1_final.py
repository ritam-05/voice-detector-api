import os
import random

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Model


DATA_ROOT = os.path.join("stage_1", "stage1_data")
OUT_DIR = os.path.join("backend", "models")

TARGET_SR = 16000
CHUNK_SECONDS = 4
CHUNK_SAMPLES = TARGET_SR * CHUNK_SECONDS

BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-2

UNFREEZE_LAST_N_BLOCKS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Stage1Dataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for cls, label in (("human", 0), ("ai", 1)):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Missing directory: {cls_dir}")
            for name in os.listdir(cls_dir):
                if name.lower().endswith(".wav"):
                    self.samples.append((os.path.join(cls_dir, name), label))

        if not self.samples:
            raise RuntimeError(f"No .wav files found under {root}")

        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} wav files from {root}")

    def __len__(self):
        return len(self.samples)

    def _load_random_chunk(self, path):
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

        peak = wav.abs().max()
        wav = wav / (peak + 1e-9)

        if wav.numel() < CHUNK_SAMPLES:
            wav = torch.nn.functional.pad(wav, (0, CHUNK_SAMPLES - wav.numel()))
            return wav

        start = random.randint(0, wav.numel() - CHUNK_SAMPLES)
        return wav[start : start + CHUNK_SAMPLES]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        chunk = self._load_random_chunk(path)
        return chunk, torch.tensor(label, dtype=torch.float32)


class Stage1Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        h = self.encoder(x).last_hidden_state
        pooled = h.mean(dim=1)
        return self.classifier(pooled).squeeze(1)


def set_trainable_layers(model, unfreeze_last_n):
    for p in model.encoder.parameters():
        p.requires_grad = False

    if unfreeze_last_n > 0:
        blocks = model.encoder.encoder.layers
        for block in blocks[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True


def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = Stage1Dataset(DATA_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = Stage1Detector().to(DEVICE)
    set_trainable_layers(model, UNFREEZE_LAST_N_BLOCKS)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(OUT_DIR, f"stage1_detector_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()

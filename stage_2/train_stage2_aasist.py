import os
import sys

# ============================
# VRAM SAFE CUDA CONFIG
# ============================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Add stage_2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stage_2'))

from data_pipeline.dataset_stage2 import Stage2Dataset
from transformers import Wav2Vec2Model
from backend.aasist_backend import AASIST_Backend

# ============================
# CONFIG (4GB GPU SAFE)
# ============================
DATA_DIR = "stage2_data"

BATCH = 1                    #  MUST be 1 for 4GB GPU
ACCUMULATION_STEPS = 8       #  Simulate batch size 8
EPOCHS = 3
LR = 1e-4
USE_AMP = True               #  Mandatory for 4GB GPUs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# MODEL: Stage2 AASIST Verifier
# ============================
class Stage2AASISTVerifier(nn.Module):
    def __init__(self):
        super().__init__()

        # wav2vec2 encoder
        self.encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        #  Freeze encoder completely
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

        #  Lightweight AASIST Backend
        self.backend = AASIST_Backend(input_dim=768)

    def train(self, mode=True):
        """Always keep encoder frozen in eval mode"""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        #  No gradients stored for encoder
        with torch.no_grad():
            features = self.encoder(x).last_hidden_state  # (B,T,768)

        #  Trainable backend
        return self.backend(features)


# ============================
# MAIN TRAINING FUNCTION
# ============================
def main():
    print("===================================")
    print(" Stage2 AASIST Training (4GB Safe)")
    print("===================================")
    print("Using device:", DEVICE)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    ds = Stage2Dataset(DATA_DIR)

    dl = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0
    )

    # Clear memory before model init
    gc.collect()
    torch.cuda.empty_cache()

    # Load model
    print("\nInitializing model...")
    model = Stage2AASISTVerifier().to(DEVICE)

    #  Weighted loss (punish AI slipping as human)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([2.5]).to(DEVICE)
    )

    #  Train only backend
    optimizer = torch.optim.AdamW(
        model.backend.parameters(),
        lr=LR
    )

    #  AMP scaler
    scaler = GradScaler()

    print("\nStarting training...\n")

    # ============================
    # TRAIN LOOP
    # ============================
    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()

        for step, (x, y) in enumerate(pbar):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Forward with AMP
            with autocast(enabled=USE_AMP):
                logits = model(x)
                loss = loss_fn(logits, y)

            # Normalize loss for accumulation
            loss = loss / ACCUMULATION_STEPS

            # Backward with AMP scaler
            scaler.scale(loss).backward()

            # Step optimizer every accumulation steps
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix({"loss": loss.item() * ACCUMULATION_STEPS})

            # Cleanup
            del x, y, logits, loss

            # Periodic cache cleanup
            if (step + 1) % 50 == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(dl)
        print(f"\n Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_path = os.path.join(
            "models",
            f"stage2_aasist_epoch{epoch+1}.pt"
        )
        torch.save(model.state_dict(), save_path)

        print(f" Saved checkpoint: {save_path}\n")

    print("===================================")
    print(" Training complete.")
    print("===================================")


# ============================
# RUN
# ============================
if __name__ == "__main__":
    main()

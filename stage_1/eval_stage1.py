import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from zz_archives.model import Detector
from utils.audio_utils import load_chunks

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "stage1_detector1.pt")
DATA_DIR = os.path.join(BASE_DIR, "stage1_data")

# ======================
# LOAD MODEL
# ======================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = Detector()
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    return model

# ======================
# PREDICT SINGLE FILE
# ======================
@torch.no_grad()
def predict_file(model, path):
    chunks = load_chunks(path)
    probs = []

    for c in chunks:
        c = c.unsqueeze(0).to(DEVICE)  # [1, T]
        logit = model(c)
        prob = torch.sigmoid(logit).item()
        probs.append(prob)

    # Conservative: strongest evidence wins
    return max(probs)

# ======================
# EVALUATION
# ======================
def evaluate():
    model = load_model()

    y_true = []
    y_pred = []

    classes = {
        "human": 0,
        "ai": 1
    }

    for label, idx in classes.items():
        folder = os.path.join(DATA_DIR, label)
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".wav", ".mp3"))
        ]

        print(f"\nEvaluating {label}: {len(files)} files")

        for f in tqdm(files):
            score = predict_file(model, f)
            pred = 1 if score >= 0.5 else 0

            y_true.append(idx)
            y_pred.append(pred)

    # ======================
    # REPORT
    # ======================
    print("\n=== STAGE 1 EVALUATION ===")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Human", "AI"],
        digits=4
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# ======================
# ENTRY
# ======================
if __name__ == "__main__":
    evaluate()
import os


models_dir = os.path.join(os.path.dirname(__file__), "models")

REQUIRED_MODELS = [
    "stage1_detector_epoch3.pt",
    "stage2_aasist_epoch3.pt"
]

def ensure_models():
    """Verify that required models exist locally. Do NOT download."""
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found at {models_dir}")

    missing = []
    for filename in REQUIRED_MODELS:
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            missing.append(filename)
        else:
            print(f"[MODEL] Found {path}")

    if missing:
        raise FileNotFoundError(f"Missing model files in {models_dir}: {missing}. Please download them manually.")
    print("All models verified locally.")

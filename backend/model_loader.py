import os
import urllib.request

MODELS = {
    "models/stage1.pt": "https://huggingface.co/ritam-05/voice-detector-models/resolve/main/stage1_detector_epoch3.pt",
    "models/stage2_human.pt": "https://huggingface.co/ritam-05/voice-detector-models/resolve/main/stage2_aasist_epoch3.pt",
    "models/stage2_aasist.pt": "https://huggingface.co/ritam-05/voice-detector-models/resolve/main/stage2_verifier_epoch5.pt",
}

def ensure_models():
    os.makedirs("models", exist_ok=True)

    for path, url in MODELS.items():
        if not os.path.exists(path):
            print(f"[MODEL] Downloading {path}")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"[MODEL] Found {path}")

import os
import urllib.request

models_dir = os.path.join(os.path.dirname(__file__), "models")

REQUIRED_MODELS = [
    "stage1_detector_epoch3.pt",
    "stage2_aasist_epoch3.pt"
]

REPO_BASE_URL = "https://huggingface.co/ritam-05/voice-detector-models/resolve/main"

def ensure_models():
    """Delegated to HF hub download within `final_inference_logic._resolve_model_path()`"""
    print("Model fetching is dynamically handled via huggingface_hub during loading.")

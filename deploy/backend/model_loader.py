import os
import urllib.request

models_dir = os.path.join(os.path.dirname(__file__), "models")

REQUIRED_MODELS = [
    "stage1_detector_epoch3.pt",
    "stage2_aasist_epoch3.pt"
]

REPO_BASE_URL = "https://huggingface.co/ritam-05/voice-detector-models/resolve/main"

def ensure_models():
    """Verify models exist, and download from Hugging Face if they are missing."""
    os.makedirs(models_dir, exist_ok=True)
    
    # --- OLD STRICT LOCAL CHECK ---
    # if not os.path.exists(models_dir):
    #     raise FileNotFoundError(f"Models directory not found at {models_dir}")
    # missing = []
    # for filename in REQUIRED_MODELS:
    #     path = os.path.join(models_dir, filename)
    #     if not os.path.exists(path):
    #         missing.append(filename)
    #     else:
    #         print(f"[MODEL] Found {path}")
    # if missing:
    #     raise FileNotFoundError(f"Missing model files in {models_dir}: {missing}. Please download them manually.")
    # print("All models verified locally.")
    # ------------------------------

    # --- NEW ONLINE LOADER ---
    for filename in REQUIRED_MODELS:
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            print(f"[MODEL] Downloading {filename} from Hugging Face. This may take a minute...")
            url = f"{REPO_BASE_URL}/{filename}"
            try:
                # Set a basic User-Agent so we don't get blocked
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                    import shutil
                    shutil.copyfileobj(response, out_file)
                print(f"[MODEL] Successfully downloaded {filename}.")
            except Exception as e:
                # If it failed, delete the corrupted/partial file so we don't try to load it
                if os.path.exists(path):
                    os.unlink(path)
                raise Exception(f"Failed to download model {filename}: {str(e)}")
        else:
            print(f"[MODEL] {filename} already exists locally.")

    print("All models verified and ready.")

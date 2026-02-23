import os
import numpy as np
from tqdm import tqdm
from audio_features import extract_audio_features

BASE_DIR = "processed"

X = []
y = []

def collect_files():
    files = []

    # -------------------------
    # HUMAN = 0
    # -------------------------
    human_base = os.path.join(BASE_DIR, "human")
    for lang in os.listdir(human_base):
        lang_dir = os.path.join(human_base, lang)
        if not os.path.isdir(lang_dir):
            continue

        for f in os.listdir(lang_dir):
            if f.endswith(".wav"):
                files.append((os.path.join(lang_dir, f), 0))

    # -------------------------
    # AI (AUGMENTED) = 1
    # -------------------------
    ai_base = os.path.join(BASE_DIR, "ai_augmented")
    for lang in os.listdir(ai_base):
        lang_dir = os.path.join(ai_base, lang)
        if not os.path.isdir(lang_dir):
            continue

        for f in os.listdir(lang_dir):
            if f.endswith(".wav"):
                files.append((os.path.join(lang_dir, f), 1))

    return files


if __name__ == "__main__":
    files = collect_files()
    print(f"Total audio files found: {len(files)}")

    for path, label in tqdm(files):
        try:
            feat = extract_audio_features(path)
            X.append(feat)
            y.append(label)
        except Exception as e:
            # Skip broken files safely
            continue

    X = np.array(X)
    y = np.array(y)

    np.save("X.npy", X)
    np.save("y.npy", y)

    print("\nDATASET BUILD COMPLETE")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

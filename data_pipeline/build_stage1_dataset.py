import os
import random
import shutil

# ================= CONFIG =================
PROCESSED = "processed"
OUT = "stage_1/stage1_data"

OUT_AI = os.path.join(OUT, "ai")
OUT_HUMAN = os.path.join(OUT, "human")

TOTAL_PER_CLASS = 15000   # final size per class
random.seed(42)

# =========================================
def collect(folder):
    paths = []
    base = os.path.join(PROCESSED, folder)
    if not os.path.exists(base):
        return []
    for root, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith(".wav"):
                paths.append(os.path.join(root, f))
    return paths


def safe_sample(lst, n):
    if len(lst) <= n:
        return lst
    return random.sample(lst, n)


def main():
    os.makedirs(OUT_AI, exist_ok=True)
    os.makedirs(OUT_HUMAN, exist_ok=True)

    # ================= AI =================
    ai_clean = collect("ai")
    ai_aug = collect("ai_augmented")
    ai_art = collect("ai_artifacts")

    print("AI clean:", len(ai_clean))
    print("AI augmented:", len(ai_aug))
    print("AI artifacts:", len(ai_art))

    ai_all = ai_clean + ai_aug + ai_art
    if len(ai_all) == 0:
        raise RuntimeError("No AI files found")

    ai_final = safe_sample(ai_all, TOTAL_PER_CLASS)
    random.shuffle(ai_final)

    for i, f in enumerate(ai_final):
        shutil.copy(f, os.path.join(OUT_AI, f"ai_{i}.wav"))

    print("AI STAGE-1:", len(ai_final))

    # ================= HUMAN =================
    human_clean = collect("human")
    human_deg = collect("human_degraded")

    print("Human clean:", len(human_clean))
    print("Human degraded:", len(human_deg))

    human_all = human_clean + human_deg
    if len(human_all) < TOTAL_PER_CLASS:
        raise RuntimeError("Not enough human files")

    human_final = random.sample(human_all, TOTAL_PER_CLASS)
    random.shuffle(human_final)

    for i, f in enumerate(human_final):
        shutil.copy(f, os.path.join(OUT_HUMAN, f"human_{i}.wav"))

    print("HUMAN STAGE-1:", len(human_final))
    print(" STAGE-1 DATASET READY")


if __name__ == "__main__":
    main()

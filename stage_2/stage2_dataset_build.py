import os
import random
import shutil

# =========================
# CONFIG
# =========================
LANGS = ["english", "hindi", "tamil", "telugu", "malayalam"]

HUMAN_SOURCES = [
    "processed/human",
    "processed/human_degraded",
]

AI_SOURCES = [
    "processed/ai",
    "processed/ai_augmented",
    "processed/ai_artifacts",
]

OUT_HUMAN = "stage2_data/human"
OUT_AI = "stage2_data/ai"

TOTAL_PER_CLASS = 12000
PER_LANG = TOTAL_PER_CLASS // len(LANGS)

random.seed(42)

os.makedirs(OUT_HUMAN, exist_ok=True)
os.makedirs(OUT_AI, exist_ok=True)

# =========================
# HELPERS
# =========================
def collect_files(sources, lang):
    files = []
    for src in sources:
        d = os.path.join(src, lang)
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(".wav"):
                files.append(os.path.join(d, f))
    return files

def copy_files(files, out_dir, prefix, start_idx):
    for i, src in enumerate(files):
        dst = os.path.join(out_dir, f"{prefix}_{start_idx + i:05d}.wav")
        shutil.copyfile(src, dst)

# =========================
# BUILD DATASET
# =========================
human_all = []
ai_all = []

for lang in LANGS:
    print(f"\nProcessing language: {lang}")

    h_files = collect_files(HUMAN_SOURCES, lang)
    a_files = collect_files(AI_SOURCES, lang)

    print(f"  Human found: {len(h_files)}")
    print(f"  AI found   : {len(a_files)}")

    random.shuffle(h_files)
    random.shuffle(a_files)

    human_all.extend(h_files[:PER_LANG])
    ai_all.extend(a_files[:PER_LANG])

print("\n=========================")
print(f"Selected HUMAN files: {len(human_all)}")
print(f"Selected AI files   : {len(ai_all)}")
print("=========================\n")

# =========================
# COPY
# =========================
print("Copying HUMAN files...")
copy_files(human_all, OUT_HUMAN, "human", 0)

print("Copying AI files...")
copy_files(ai_all, OUT_AI, "ai", 0)

print("\nStage-2 HUMAN + AI dataset built successfully")
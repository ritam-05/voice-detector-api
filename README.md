# VOICE DETECTOR API

🚀 **Live Application:** [https://voiceguard12.netlify.app/](https://voiceguard12.netlify.app/)

## Dataset Update

The current human speech dataset was collected and prepared as follows:

- Mozilla Common Voice: English, Hindi, Tamil
- OpenSLR: Telugu, Malayalam
- All downloaded clips were preprocessed and converted into fixed 3-second chunks for training and inference.

## Rebuild Dataset After Local Deletion

If you delete all local datasets, recreate them with the steps below.

### 0) Install dependencies (one time per machine)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### 1) Create required folders

```powershell
New-Item -ItemType Directory -Force -Path processed\human\english,processed\human\hindi,processed\human\tamil,processed\human\telugu,processed\human\malayalam | Out-Null
New-Item -ItemType Directory -Force -Path processed\ai,processed\ai_augmented,processed\ai_artifacts,processed\human_degraded | Out-Null
```

### 2) Download source speech data again

- Mozilla Common Voice (English, Hindi, Tamil): https://commonvoice.mozilla.org/
- OpenSLR (Telugu, Malayalam): https://openslr.org/

Extract downloaded archives and collect audio files (`.wav`/`.mp3`) per language.

### 3) Preprocess all human clips into fixed 3-second WAV chunks

Use your preprocessing pipeline to:

- convert to mono
- resample to 16 kHz
- normalize amplitude
- split into fixed 3-second segments

Note: training/inference loaders in this repo use 4-second internal chunks; 3-second files are still valid and get padded automatically when needed.

Save final chunks under:

- `processed/human/english`
- `processed/human/hindi`
- `processed/human/tamil`
- `processed/human/telugu`
- `processed/human/malayalam`

### 4) Recreate AI-side processed folders (if training again)

Put clean AI WAV files in `processed/ai/<language>/`, then run:

```powershell
python data_pipeline/build_augmented_ai.py
python data_pipeline/build_ai_artifacts.py
```

If you need degraded human data, run `data_pipeline/build_human_degraded.py` and place output in `processed/human_degraded/<language>/`.

### 5) Rebuild Stage-1 training dataset

```powershell
python data_pipeline/build_stage1_dataset.py
```

This generates:

- `stage_1/stage1_data/human`
- `stage_1/stage1_data/ai`

### 6) Optional classical-feature dataset rebuild

```powershell
python data_pipeline/build_dataset_audio_only.py
```

This regenerates `X.npy` and `y.npy`. (used by baseline model)

## Data Scripts Used

### Data pipeline (`data_pipeline/`)

- `build_dataset_audio_only.py`: builds/organizes the base audio-only dataset structure.
- `build_stage1_dataset.py`: prepares Stage-1 dataset folders (`ai` / `human`).
- `augment_ai.py`: applies synthetic degradations/perturbations to AI audio.
- `build_augmented_ai.py`: generates and saves augmented AI samples at scale.
- `build_human_degraded.py`: creates degraded/noisy human variants for robustness.
- `dataset.py`: common/general dataset loader utilities.
- `dataset_stage1.py`: Stage-1 dataset loader utilities.
- `dataset_stage2.py`: Stage-2 dataset loader utilities.

### Stage-1 scripts (`stage_1/`)

- `build_stage1_dataset.py`: Stage-1 local dataset preparation.
- `dataset_stage1.py`: wav loading, resampling, chunk sampling for Stage-1.
- `stage1_feature_extraction.py`: feature extraction pipeline for feature-based Stage-1 experiments.
- `stage1_model.py`: Stage-1 model definition used by inference/training code.
- `train_stage1_final.py`: final Stage-1 wav2vec2 training script (epoch checkpoints like `stage1_detector_epochN.pt`).
- `eval_stage1.py`: evaluation script for Stage-1 model performance checks.

### Training scripts (`training/`)

- `train_stage1_final.py`: training entrypoint for Stage-1 detector.
- `train_stage2_aasist.py`: training entrypoint for Stage-2 AASIST verifier.
- `stage1_model.py`: training-side Stage-1 model module.

---

# 1️⃣ Evolution of the System

## 🔹 Phase 1 – LGBM LoLo Baseline Model

The first version of the system used a **LightGBM (LGBM) classifier** trained on low-level handcrafted acoustic features.

### 📊 Performance

- Accuracy: **~85%**
- Weakness: Failed on **high-quality studio recordings**
- Training bias: Mostly low-quality, noisy recordings
- Could not generalize to:
  - Clean AI
  - Studio human voices
  - Augmented / compressed audio

This version served as a strong baseline but exposed the need for:

- Better representation learning
- Robustness to degradation
- AI artifact detection

---

# 2️⃣ Phase 2 – Hand-Crafted Human / AI Feature Enhancements

To fix baseline weaknesses, we introduced signal-level feature engineering.

### 🧠 Human-Degraded Detection

We defined degraded audio using:

#### Signal-to-Noise Ratio (SNR)

[
SNR_{dB} = 20 \log_{10} \left( \frac{RMS}{NoiseFloor} \right)
]

Where:

- ( RMS = \sqrt{\frac{1}{N} \sum x^2} )
- NoiseFloor = 5th percentile amplitude

Degraded if:

- SNR < 18 dB

---

#### Spectral Flatness

[
Flatness = \frac{\exp(\text{mean}(\log(P)))}{\text{mean}(P)}
]

Where:

- (P) = power spectrum

Higher flatness → more noise-like
Lower flatness → tonal/studio

---

#### Clipping Ratio

[
Clipped = \frac{#(|x| > 0.99)}{N}
]

Used to detect distortion.

---

### 🤖 AI Artifact Detection

We engineered AI-specific indicators:

- **Repetition Score**
  - MFCC cosine similarity across chunks
  - High similarity → looped patterns

- **Pitch Variance**
  - Estimated F0 via autocorrelation
  - Low variance → synthetic

- **Vocoder Artifact Score**
  [
  Score = Flatness + 0.02(1 - HF_{ratio})
  ]

- **Dynamics Ratio**
  [
  dyn = \frac{\sigma(RMS)}{\mu(RMS)}
  ]

Low dynamics → monotone AI

---

This improved robustness, but classical ML still lacked deep representation power.

---

# 3️⃣ Phase 3 – Stage 1 Deep Model (Wav2Vec2)

We upgraded to a **representation learning model**:

### Architecture

- `facebook/wav2vec2-base`
- Mean pooled embeddings
- MLP head:
  - 768 → 256 → 1

### Stage 1 Role

Binary classifier:

- Outputs probability of AI

### Stage 1 Constraints & Thresholds

```python
S1_HUMAN_RECHECK_THRESHOLD = 0.40
S1_AI_CHECK_THRESHOLD = 0.75
S1_VERY_CONFIDENT = 0.90
S1_CONFIDENT = 0.82
```

### Decision Logic

| Range     | Meaning                     |
| --------- | --------------------------- |
| < 0.40    | HUMAN (trusted immediately) |
| 0.40–0.75 | Ambiguous                   |
| > 0.75    | AI candidate                |
| > 0.90    | Very confident AI           |

Stage 1 became strong — but occasionally overconfident.

So we built Stage 2.

---

# 4️⃣ Stage 2 – AASIST Advanced Verifier

## What is AASIST?

AASIST is a **graph attention-based backend** that analyzes:

- Frame-level temporal patterns
- Spectral artifacts
- Subtle vocoder cues
- Cross-frame dependencies

Architecture:

```
Wav2Vec2 Encoder → AASIST Backend → Classifier
```

---

# 🧠 Core Innovation

## Confidence-Weighted Adaptive Verification

Instead of hard thresholds:

> Different confidence levels require different strategies.

---

# 🎯 Decision Flow

---

## Case 1: Stage 1 says HUMAN (< 0.40)

✔ Trust immediately
✔ No AASIST verification
✔ Minimizes false AI labeling

---

## Case 2: Stage 1 says AI (> 0.75)

Tiered adaptive logic:

---

### TIER 1: Very Confident AI (> 0.90)

| AASIST    | Decision                 |
| --------- | ------------------------ |
| ≥ 0.40    | AI (70% S1 + 30% AASIST) |
| 0.20–0.40 | AI (trust S1)            |
| < 0.20    | HUMAN                    |

---

### TIER 2: Confident AI (0.82–0.90)

| AASIST    | Decision                |
| --------- | ----------------------- |
| ≥ 0.45    | AI (65/35 weighted)     |
| 0.25–0.45 | Weighted check          |
| < 0.25    | Feature-based tie break |

---

### TIER 3: Moderate AI (0.75–0.82)

| AASIST    | Decision     |
| --------- | ------------ |
| ≥ 0.50    | AI (50/50)   |
| 0.30–0.50 | INCONCLUSIVE |
| < 0.30    | HUMAN        |

---

## Case 3: Stage 1 Ambiguous (0.40–0.75)

Trust AASIST more:

[
FinalScore = 0.4 \cdot S1 + 0.6 \cdot AASIST
]

| AASIST    | Decision          |
| --------- | ----------------- |
| ≥ 0.55    | AI                |
| < 0.35    | HUMAN             |
| 0.35–0.55 | Weighted decision |

---

# 📊 Why This Is Powerful

### ✅ Reduces False Positives

Human voices rarely cross S1 < 0.40.

### ✅ Reduces False Negatives

Very confident S1 AI (>0.90) no longer blocked by mild AASIST uncertainty.

### ✅ Handles Studio Audio

Hand-crafted features compensate for over-clean recordings.

### ✅ Handles Augmented AI

Vocoder + repetition detection catches artifacts.

---

# 📈 Example

Example:

```
S1 = 0.85
AASIST = 0.38
```

Weighted:

[
0.6(0.85) + 0.4(0.38) = 0.662
]

→ AI

Old system → INCONCLUSIVE
New system → Correct AI

---

# 🏗 Architecture Overview

```
Audio Input
    ↓
Chunking (4s, 50% overlap)
    ↓
Stage 1 (Wav2Vec2 + MLP)
    ↓
Hand-crafted feature extraction
    ↓
AASIST (conditional verification)
    ↓
Confidence-weighted decision fusion
    ↓
Final Label + Explanation
```

---

# 🔍 Explanation Engine

The system produces structured layman explanations based on:

- Breathing detection
- Pitch variance
- Dynamics ratio
- Repetition score
- Vocoder artifacts
- Clipping
- Reverb
- Micro-noises

This makes the system explainable — critical for hackathon judging.

---

# 🚀 Final System Characteristics

| Feature           | Supported    |
| ----------------- | ------------ |
| Studio Human      | ✅           |
| Low Quality Human | ✅           |
| Augmented AI      | ✅           |
| Clean AI          | ✅           |
| Edge Cases        | INCONCLUSIVE |
| Explainability    | Yes          |

---

It is a **multi-stage adaptive AI verification system** built to handle real-world edge cases.
This system is LANGUAGE INDEPENDENT.

All models uploaded at: https://huggingface.co/ritam-05/voice-detector-models/tree/main

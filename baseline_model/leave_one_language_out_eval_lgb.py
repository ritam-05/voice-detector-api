import os
import numpy as np
import lightgbm as lgb
from data_pipeline.audio_features import extract_audio_features
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = "processed"
TEST_LANGUAGE = "malayalam"  # change this

X_train, y_train = [], []
X_test, y_test = [], []

def load_split(cls_label, base_path, is_test):
    for f in os.listdir(base_path):
        if f.endswith(".wav"):
            feat = extract_audio_features(os.path.join(base_path, f))
            if is_test:
                X_test.append(feat)
                y_test.append(cls_label)
            else:
                X_train.append(feat)
                y_train.append(cls_label)

# -------------------------
# HUMAN = 0
# -------------------------
for lang in os.listdir(f"{BASE_DIR}/human"):
    path = f"{BASE_DIR}/human/{lang}"
    load_split(0, path, lang == TEST_LANGUAGE)

# -------------------------
# AI = 1
# -------------------------
for lang in os.listdir(f"{BASE_DIR}/ai_augmented"):
    path = f"{BASE_DIR}/ai_augmented/{lang}"
    load_split(1, path, lang == TEST_LANGUAGE)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# -------------------------
# Train model
# -------------------------
model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    num_leaves=31,
    min_child_samples=50,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight={0: 1.0, 1: 3.0},  # 🔑 penalize missing AI
    objective="binary",
    random_state=42,
    n_jobs=-1
)
print("\n Training LightGBM (LOLO)...")

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
y_pred = (probs > 0.20).astype(int)

print("\n LEAVE-ONE-LANGUAGE-OUT RESULTS")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(model, "voice_detector_lgbm_lolo.pkl")
print("Model saved as voice_detector_lgbm_lolo.pkl")

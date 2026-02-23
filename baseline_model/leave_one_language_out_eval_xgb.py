import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from data_pipeline.audio_features import extract_audio_features

BASE_DIR = "processed"
TEST_LANGUAGE = "malayalam"   # change this to test others
THRESHOLD = 0.7

X_train, y_train = [], []
X_test, y_test = [], []

def load_data(base_path, label, is_test):
    for f in os.listdir(base_path):
        if f.endswith(".wav"):
            feat = extract_audio_features(os.path.join(base_path, f))
            if is_test:
                X_test.append(feat)
                y_test.append(label)
            else:
                X_train.append(feat)
                y_train.append(label)

# -------------------------
# HUMAN = 0
# -------------------------
for lang in os.listdir(f"{BASE_DIR}/human"):
    lang_path = f"{BASE_DIR}/human/{lang}"
    load_data(lang_path, 0, lang == TEST_LANGUAGE)

# -------------------------
# AI (AUGMENTED) = 1
# -------------------------
for lang in os.listdir(f"{BASE_DIR}/ai_augmented"):
    lang_path = f"{BASE_DIR}/ai_augmented/{lang}"
    load_data(lang_path, 1, lang == TEST_LANGUAGE)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------------------------
# XGBOOST MODEL (LOCKED)
# -------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42
)

print("\n Training XGBoost (LOLO)...")
model.fit(X_train, y_train)

# -------------------------
# EVALUATION
# -------------------------
probs = model.predict_proba(X_test)[:, 1]
y_pred = (probs > THRESHOLD).astype(int)

print("\n LEAVE-ONE-LANGUAGE-OUT RESULTS (XGBoost)")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

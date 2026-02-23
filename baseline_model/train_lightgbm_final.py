import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


X = np.load("X.npy")
y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    num_leaves=31,
    min_child_samples=50,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    n_jobs=-1
)

print("\n Training LightGBM...")
model.fit(X_train, y_train)

THRESHOLD = 0.7

probs = model.predict_proba(X_test)[:, 1]
y_pred = (probs > THRESHOLD).astype(int)

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

print(" Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, "voice_detector_lgbm.pkl")
print("\nModel saved as voice_detector_lgbm.pkl")

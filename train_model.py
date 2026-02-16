import os
import sys
import numpy as np
import joblib
import importlib.util

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =====================================================
# ROOT DIRECTORY
# =====================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("ROOT_DIR =", ROOT_DIR)


# =====================================================
# LOAD audio_utils.py DYNAMICALLY
# =====================================================
AUDIO_UTILS_PATH = os.path.join(ROOT_DIR, "app", "audio_utils.py")

if not os.path.exists(AUDIO_UTILS_PATH):
    print(" audio_utils.py not found at:", AUDIO_UTILS_PATH)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("audio_utils", AUDIO_UTILS_PATH)
audio_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_utils)

extract_features = audio_utils.extract_features


# =====================================================
# DATASET LOADING
# =====================================================
X, y = [], []

def load_folder(folder_relative_path, label):
    folder_path = os.path.join(ROOT_DIR, folder_relative_path)

    if not os.path.isdir(folder_path):
        print(f" Dataset folder not found: {folder_path}")
        return

    for file in os.listdir(folder_path):
        if file.lower().endswith((".wav", ".mp3")):
            file_path = os.path.join(folder_path, file)
            try:
                features = extract_features(file_path)
                features = np.nan_to_num(features)

                X.append(features)
                y.append(label)
                print(f"✔ Processed: {file}")

            except Exception as e:
                print(f"⚠ Skipped {file}: {e}")


# =====================================================
# LOAD DATA
# =====================================================
print("\nLoading HUMAN samples...")
load_folder("dataset/human", 0)

print("\nLoading AI samples...")
load_folder("dataset/ai", 1)

X = np.array(X)
y = np.array(y)

print("\nTotal samples loaded:", len(X))

if len(X) < 10:
    print(" Too few samples. Please add more audio files.")
    sys.exit(1)


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# =====================================================
# MODEL TRAINING
# =====================================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("\nTraining RandomForest model...")
model.fit(X_train, y_train)


# =====================================================
# EVALUATION
# =====================================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["HUMAN", "AI"]
))


# =====================================================
# SAVE MODEL
# =====================================================
MODEL_PATH = os.path.join(ROOT_DIR, "voice_detector.pkl")
joblib.dump(model, MODEL_PATH)

print("\n Model saved at:", MODEL_PATH)

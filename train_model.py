
import os
import cv2
import numpy as np
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

DATASET = "train"
TESTSET = "test"

LABELS = ["angry", "happy", "sad", "neutral"]
IMAGE_SIZE = (48, 48)

def load_images(folder):
    features = []
    labels = []

    for label in LABELS:
        path = os.path.join(folder, label)

        if not os.path.isdir(path):
            print(f"Missing folder: {path}")
            continue

        files = os.listdir(path)
        print(f"Loading {label}: {len(files)} images")

        for filename in files:
            image_path = os.path.join(path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            features.append(image.flatten() / 255.0)
            labels.append(label)

    return np.array(features, dtype=np.float32), np.array(labels)

print("Loading training images...")
X_train, y_train = load_images(DATASET)

print("Loading test images...")
X_test, y_test = load_images(TESTSET)

if len(X_train) == 0 or len(X_test) == 0:
    raise SystemExit("No images loaded. Check the dataset folders.")

print("\nTraining our own model...")
model = make_pipeline(
    StandardScaler(),
    SGDClassifier(
        loss="log_loss",
        max_iter=20,
        tol=None,
        random_state=42
    )
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, "emotion_model.pkl")
print("\nModel saved as emotion_model.pkl")

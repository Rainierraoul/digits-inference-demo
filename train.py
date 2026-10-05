"""Train a reproducible digits classifier and persist its preprocessing pipeline."""
import hashlib
import json
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACTS = Path(__file__).parent / "artifacts"


def train():
    data = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, stratify=data.target, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    if accuracy < 0.90:
        raise RuntimeError(f"Holdout accuracy below acceptance threshold: {accuracy}")
    ARTIFACTS.mkdir(exist_ok=True)
    model_path = ARTIFACTS / "model.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "model": "digits-standardscaler-logisticregression-v1",
        "sklearn_version": sklearn.__version__,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "holdout_accuracy": accuracy,
        "train_samples": len(y_train), "test_samples": len(y_test), "seed": 42,
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (Path(__file__).parent / "example.json").write_text(
        json.dumps({"instances": [x_test[0].tolist()]}) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    train()

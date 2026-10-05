"""Check API predictions against held-out labels and enforce the input contract."""
import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from app import ARTIFACTS, app


def holdout():
    data = load_digits()
    _, x, _, y = train_test_split(
        data.data, data.target, test_size=0.2, stratify=data.target, random_state=42
    )
    return x, y


def test_api_holdout_accuracy_and_probabilities():
    x, y = holdout()
    predictions = []
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ready"
        for start in range(0, len(x), 32):
            response = client.post("/predict", json={"instances": x[start:start+32].tolist()})
            assert response.status_code == 200
            result = response.json()
            assert result["model_sha256"] == health["sha256"]
            assert np.allclose(np.sum(result["probabilities"], axis=1), 1)
            predictions.extend(result["predictions"])
    assert np.mean(np.asarray(predictions) == y) >= 0.90


@pytest.mark.parametrize("instances", [[], [[0]*63], [[17]*64], [[-1]*64],
                                           [["0"]*64], [[True]*64], [[0]*64]*33])
def test_invalid_input(instances):
    with TestClient(app) as client:
        assert client.post("/predict", json={"instances": instances}).status_code == 422


def test_corrupted_model_fails_startup(tmp_path, monkeypatch):
    import app as service
    (tmp_path / "model.joblib").write_bytes(b"corrupt")
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text())
    assert metadata["sha256"] != hashlib.sha256(b"corrupt").hexdigest()
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(service, "ARTIFACTS", tmp_path)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        with TestClient(app):
            pass


def test_version_mismatch_fails_startup(tmp_path, monkeypatch):
    import app as service
    (tmp_path / "model.joblib").write_bytes((ARTIFACTS / "model.joblib").read_bytes())
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text())
    metadata["sklearn_version"] = "incompatible"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(service, "ARTIFACTS", tmp_path)
    with pytest.raises(RuntimeError, match="versions differ"):
        with TestClient(app):
            pass

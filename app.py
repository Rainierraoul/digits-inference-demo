"""Serve a trusted, locally trained model with a bounded batch inference API."""
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated

import joblib
import numpy as np
import sklearn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

ARTIFACTS = Path(__file__).parent / "artifacts"
logger = logging.getLogger("uvicorn.error")
Pixel = Annotated[float, Field(ge=0, le=16, allow_inf_nan=False, strict=True)]
Image = Annotated[list[Pixel], Field(min_length=64, max_length=64)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: Annotated[list[Image], Field(min_length=1, max_length=32)]


@asynccontextmanager
async def lifespan(app):
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text())
    model_path = ARTIFACTS / "model.joblib"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise RuntimeError("Model artifact checksum mismatch; rerun train.py")
    if metadata["sklearn_version"] != sklearn.__version__:
        raise RuntimeError("Training and serving sklearn versions differ")
    # joblib can execute code: load only artifacts produced by trusted train.py.
    app.state.model = joblib.load(model_path)
    app.state.metadata = metadata
    yield


app = FastAPI(title="Digits inference service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ready", **app.state.metadata}


@app.post("/predict")
def predict(request: PredictionRequest):
    started = perf_counter()
    probabilities = app.state.model.predict_proba(np.asarray(request.instances))
    labels = app.state.model.classes_[probabilities.argmax(axis=1)]
    elapsed = round((perf_counter() - started) * 1000, 3)
    logger.info("inference batch=%d latency_ms=%.3f", len(labels), elapsed)
    return {
        "model": app.state.metadata["model"],
        "model_sha256": app.state.metadata["sha256"],
        "predictions": labels.tolist(),
        "probabilities": probabilities.tolist(),
        "latency_ms": elapsed,
    }

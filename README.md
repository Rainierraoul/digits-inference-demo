# Digits inference service

A small machine-learning integration demonstration prepared for the ASD AI/ML
engineering application. It trains a scikit-learn digit classifier, saves the
complete preprocessing and classification pipeline, and serves predictions
through a FastAPI HTTP API. The model is loaded once at startup, not per request.

The example uses public data packaged with scikit-learn, runs on a CPU, and needs
no API keys or external model services. There are 174 lines of Python in total,
including training and tests. AI assistance was used to prepare the code and
documentation. This is an application demonstration, not a production deployment.

## Architecture

```text
Public digits dataset -> stratified split -> scaler + logistic regression
                                          -> model.joblib + metadata.json
HTTP request -> validate batch -> saved pipeline -> probabilities + labels
                               -> log batch size and inference latency
```

`train.py` trains and evaluates the model, writes its artifact and metadata, and
generates an example request from the held-out split. `app.py` checks the artifact
checksum and scikit-learn version before loading it. It serves `/health`,
`/predict` and generated API documentation at `/docs`. `test_service.py` checks
predictions through the API against held-out labels, rejects malformed inputs,
and verifies startup fails for a corrupt or incompatible artifact.

## Run locally

Use Python 3.12. From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python train.py
python -m pytest -q
uvicorn app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @example.json
```

Training generates `artifacts/` and `example.json`; both are excluded from Git.
The service does not download data or retrain on startup. Running from a clean
checkout without first training fails startup rather than serving an empty model.

## API contract

`POST /predict` accepts `{"instances": [[...64 pixel values...]]}`.

- A request contains between 1 and 32 images.
- Each image is an 8 by 8 grayscale grid flattened in row-major order.
- Each pixel is a finite JSON number between 0 and 16 inclusive. Strings and
  booleans are rejected. Extra top-level fields are rejected.
- Valid responses include the model identifier and artifact SHA-256, digit labels
  in input order, ten class probabilities per image (classes 0 through 9), and
  inference latency in milliseconds. Latency excludes HTTP and validation time.
- Invalid requests receive HTTP 422 with validation details.

`GET /health` returns readiness, model identifier, artifact SHA-256, library
version, train/test counts, random seed and held-out accuracy. Readiness is
available only after successful model loading. Logs record batch size and
inference time without recording pixel payloads.

## Evaluation

The fixed, stratified 80/20 split uses seed 42: 1,437 training examples and 360
held-out examples. StandardScaler is fitted only on the training split and saved
with LogisticRegression, preserving the same preprocessing at inference time.
Training refuses to export a model below 90% held-out accuracy.

Local validation on 5 October 2026 with Python 3.12 and scikit-learn 1.9.1 achieved
97.22% held-out accuracy (350/360 correct). All 10 tests passed. The accuracy test
sends every held-out image through `/predict` in bounded batches. A fixed split
is a reproducible demonstration result; it does not establish generalisation to
new handwriting sources. Probability values are not calibrated confidence.

A separate local Uvicorn process was also checked over HTTP: `/health` reported
ready and `/predict` returned a prediction for the generated example request.

## Container and continuous integration

```bash
docker build -t digits-inference .
docker run --rm -p 127.0.0.1:8000:8000 digits-inference
```

The multi-stage Dockerfile trains at build time and copies the model into the
serving image. Serving runs as a non-root user. GitHub Actions trains the model,
runs the integration tests and builds the container. The
[published GitHub Actions run](https://github.com/Rainierraoul/digits-inference-demo/actions/runs/37259164083)
passed training, all 10 tests and the Docker build on 5 October 2026. Docker is
unavailable locally; the container build was verified on the hosted runner.

## Limits and engineering choices

The small CPU model keeps the example runnable without GPUs, credentials or
external inference costs. It demonstrates the boundary between a trained model
and a service, rather than model novelty or production-scale performance.
The saved pipeline prevents training/serving preprocessing drift, and bounded
batches limit inference work per accepted request. Direct dependencies are pinned;
transitive dependencies and the Docker base image are not fully locked.

Only load model files generated by a trusted copy of `train.py`. Joblib uses
pickle-based deserialisation and can execute code. The checksum detects accidental
artifact changes; it is not a signature and does not authenticate an untrusted
model or metadata file. No endpoint accepts model uploads.

For operational use, additional work would include authentication, ingress body
limits, rate limiting, deployment-specific timeouts, load tests, monitoring and
alerting, and a controlled artifact release process. Schema checks alone do not
detect distribution shift. The model is limited to the digits dataset's pixel
format, and this project is not an enterprise MLOps platform.

## Data and references

The data is scikit-learn's bundled copy of UCI Optical Recognition of Handwritten
Digits, supplied with dataset provenance by `load_digits`. Data attribution is
separate from project-code authorship.

Dataset attribution: Alpaydin, E. and Kaynak, C. (1998), Optical Recognition of
Handwritten Digits, UCI Machine Learning Repository, DOI 10.24432/C50P49,
licensed under CC BY 4.0. This demonstration resplits the scikit-learn subset
for evaluation; it does not reproduce the original UCI training/test partition.

- [scikit-learn digits dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [UCI dataset and licence](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- [scikit-learn model persistence and security guidance](https://scikit-learn.org/stable/model_persistence.html)
- [FastAPI testing documentation](https://fastapi.tiangolo.com/tutorial/testing/)

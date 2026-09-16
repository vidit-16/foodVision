# Food Vision

[![CI](https://github.com/vidit-16/foodVision/actions/workflows/ci.yml/badge.svg)](https://github.com/vidit-16/foodVision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

A ConvNeXt-Tiny fine-tuned on Food-101, served as a FastAPI inference endpoint.

Upload a food photo, get back the ranked food categories the model thinks it is,
with confidences. 101 classes, one endpoint, one Docker image.

```bash
curl -F "file=@dinner.jpg" http://localhost:8000/predict
```

```json
{
  "prediction": "pad_thai",
  "predictions": [
    {"label": "pad_thai", "confidence": 0.874},
    {"label": "fried_rice", "confidence": 0.041},
    {"label": "spring_rolls", "confidence": 0.019}
  ]
}
```

## Status

**91.89% top-1, 98.76% top-5 on the 25,250 held-out Food-101 test images.**

The served checkpoint (release `v2.0.0`, ConvNeXt-Tiny) was trained with
`training/train.py` on the official train split only; the test split was never
seen during training. Per-class accuracy and the most common confusions are in
`evaluation/RESULTS.md`. What errors remain are mostly between genuinely similar
dishes — steak vs. filet mignon, chocolate cake vs. mousse, beef vs. tuna tartare.

| Release | Model | Top-1 | Top-5 |
| --- | --- | --- | --- |
| `v2.0.0` (served) | ConvNeXt-Tiny, improved recipe | 91.89% | 98.76% |
| `v1.1.0` | ResNet-50, original recipe | 82.19% | 96.27% |
| `v1.0.0` | ResNet-50, trained on test data | not measurable | — |

Details of each in `training/README.md`.

## Architecture

```mermaid
flowchart LR
    C[Client] -->|POST /predict image| M[app/main.py<br/>size + top_k validation]
    M --> P[app/preprocessing.py<br/>RGB, resize 256, crop 224, normalise]
    P --> I[app/predict.py<br/>softmax + top-k]
    I --> W[app/model.py<br/>ConvNeXt-Tiny via timm]
    W -->|first use| R[(GitHub release<br/>checkpoint)]
    R -->|SHA-256 verified| K[(~/.cache/foodvision)]
    K --> W
    E[evaluation/evaluate.py] --> P
```

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/): 101
categories, 1,000 images each, with a fixed split of 750 train and 250 test
images per class.

`train.py` uses only the train split, holding back 10% for validation and
leaving the 25,250 test images for evaluation.

The dataset is downloaded on demand by torchvision and is not stored here.

## Model

ConvNeXt-Tiny (28M parameters) from [timm](https://github.com/huggingface/pytorch-image-models),
initialised from ImageNet-22k weights (`convnext_tiny.fb_in22k_ft_in1k`), with a
101-way head. Fine-tuned end to end on the official train split for 12 epochs:
AdamW with warmup and cosine decay, label smoothing, stochastic depth,
TrivialAugment and random erasing, mixed precision. The released artifact is a
plain float32 `state_dict`, about 110MB.

Weights are **not committed to git**. They are published as a release asset and
downloaded on first use into `~/.cache/foodvision`, then verified against a
SHA-256 recorded in `app/config.py`. A mismatch fails loudly rather than serving
predictions from an unknown checkpoint.

To use a local file instead:

```bash
export FOODVISION_WEIGHTS_PATH=/path/to/food_vision_convnext_tiny.pt
```

## Preprocessing

Images are converted to RGB, the short side is resized to 256 (bicubic), the
centre 224×224 is cropped, and the result is normalised with ImageNet
statistics. Aspect ratio is preserved.

The transform lives in one place (`app/preprocessing.py`) and is imported by
both the API and the evaluation script. Evaluating with a different transform
from the one being served would make the reported accuracy describe a model
nobody is running.

## API

| Route | Purpose |
| --- | --- |
| `GET /` | Health: status, whether weights are loaded, class count |
| `GET /classes` | The label set, in logit order |
| `POST /predict` | Classify an uploaded image; `?top_k=` defaults to 5, max 10 |

Uploads over 10MB return 413. Undecodable files return 400. If the checkpoint
cannot be resolved or fails verification, `/predict` returns 503 rather than
guessing. Swagger UI is at `/docs`.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Docker:

```bash
docker build -t foodvision .
docker run -p 8000:8000 foodvision
```

The image installs CPU-only torch wheels, runs as a non-root user, and carries a
healthcheck against `/`.

## Train

```bash
python training/train.py --data-dir ./data
```

About 12–15 minutes per epoch on a Colab T4. `--checkpoint-dir` plus `--resume`
continues after a disconnect. `--smoke-test` runs two batches to
check the wiring first. `training/colab_train_and_eval.ipynb` does the whole
retrain-and-score pass in one Run all, resuming from Google Drive if Colab
disconnects. The original ResNet-50 notebook sits alongside
`train.py` for provenance; `training/README.md` documents the recipe and where
the two differ.

## Evaluate

```bash
python evaluation/evaluate.py --data-dir ./data
```

Runs the full test split and writes top-1 and top-5 accuracy, per-class
accuracy, and the most frequent confusions to `evaluation/RESULTS.md`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs without the checkpoint: everything verifiable without trained
weights — the output contract, preprocessing, error handling, checksum
rejection — runs against a randomly initialised network of the same shape. That
keeps CI at about a minute and keeps accuracy where it belongs, in evaluation.

Coverage of `app/` is 93% (CI fails under 85%). Weight download, caching,
checksum and network-failure paths are covered with the network mocked, and
parity tests check that the HTTP API, `predict()` and a hand-built
transform + softmax pipeline return identical results.

**Mutation testing.** `mutmut` does not run natively on Windows, so
`scripts/mutation_test.py` applies 27 hand-picked mutants (operators,
constants, guards) to `app/` and re-runs the suite for each.
Score: **27/27 killed (100%)**. The first run left 3 survivors (resize, crop and normalisation constants), which are now pinned by a preprocessing contract test.

**A/B: ResNet-50 (v1.1.0) vs ConvNeXt-Tiny (v2.0.0).** Accuracy is from the real
test split; serving cost is measured through the exact serving path by
`scripts/benchmark_models.py` (CPU, random weights, results in
`evaluation/AB_LATENCY.md`):

| Variant | Params | float32 size | CPU p50 | CPU p90 | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| A: resnet50 (v1.1.0) | 23.7M | 95MB | 989 ms | 2524 ms | 82.19% | 96.27% |
| B: convnext_tiny (v2.0.0, served) | 27.9M | 112MB | 663 ms | 938 ms | 91.89% | 98.76% |

ConvNeXt-Tiny costs 18% more parameters but gains 9.7 points of top-1 and was
not slower on CPU in this run, which is why it is served.

CI additionally enforces that no `.pt` file is ever committed and that the git
directory stays under 50MB.

## Layout

```text
.
├── app/
│   ├── config.py         # paths, weight artifact, limits
│   ├── preprocessing.py  # the one transform definition
│   ├── model.py          # timm construction, download, checksum verification
│   ├── predict.py        # top-k inference
│   ├── schemas.py        # response contract
│   └── main.py           # routes
├── training/             # train.py + the original notebook
├── evaluation/           # evaluate.py, RESULTS.md, AB_LATENCY.md
├── scripts/              # benchmark_models.py, mutation_test.py
├── tests/
├── model/classes.json
├── .env.example          # optional FOODVISION_* settings
├── Dockerfile
└── .github/workflows/ci.yml
```

## Roadmap

- Image-level test-time augmentation behind a flag, scored with `evaluate.py`
- ONNX / dynamic-quantised export to cut CPU latency and image size
- Batch endpoint for multiple images per request
- Small web UI for drag-and-drop uploads

## Stack

Python · PyTorch · timm · torchvision · FastAPI · Pydantic · Uvicorn · Pillow · Docker ·
pytest · ruff

## License

MIT.

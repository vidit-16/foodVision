# Food Vision

A ResNet-50 fine-tuned on Food-101, served as a FastAPI inference endpoint.

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

**The released checkpoint is unvalidated, and this repository claims no accuracy
for it.** It was trained on all 101,000 Food-101 images, the official test split
included, so there is no unseen data left to score it against. Details in
`training/README.md`.

`training/train.py` fixes that — it uses the official split and never opens the
test data. Retrain with it, then run `evaluation/evaluate.py` for a number that
means something.

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/): 101
categories, 1,000 images each, with a fixed split of 750 train and 250 test
images per class.

`train.py` uses only the train split, holding back 10% for validation and
leaving the 25,250 test images for evaluation. The released checkpoint predates
that discipline — see Status above.

The dataset is downloaded on demand by torchvision and is not stored here.

## Model

ResNet-50 initialised from ImageNet-V2 weights, final layer replaced with a
101-way linear head, then fine-tuned end to end for 5 epochs with Adam at 1e-4
and mixed precision. The released artifact is a plain `state_dict` (320 tensors,
float32, 95MB).

Weights are **not committed to git**. They are published as a release asset and
downloaded on first use into `~/.cache/foodvision`, then verified against a
SHA-256 recorded in `app/config.py`. A mismatch fails loudly rather than serving
predictions from an unknown checkpoint.

To use a local file instead:

```bash
export FOODVISION_WEIGHTS_PATH=/path/to/food_vision_resnet50.pt
```

## Preprocessing

Images are converted to RGB, resized directly to 224×224, and normalised with
ImageNet statistics. The direct resize distorts aspect ratio rather than
centre-cropping — that is what the checkpoint was trained with, so inference
matches training.

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
python training/train.py --data-dir ./data --epochs 5
```

About 11 minutes per epoch on a Colab T4. `--smoke-test` runs two batches to
check the wiring first. `training/colab_train_and_eval.ipynb` does the whole
retrain-and-score pass in one Run all. The original notebook sits alongside
`train.py` for provenance; `training/README.md` documents the recipe and where
the two differ.

## Evaluate

```bash
python evaluation/evaluate.py --data-dir ./data
```

Runs the full test split and writes top-1 and top-5 accuracy, per-class
accuracy, and the most frequent confusions to `evaluation/RESULTS.md`. Only
meaningful against a checkpoint trained with `train.py`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs without the checkpoint: everything verifiable without trained
weights — the output contract, preprocessing, error handling, checksum
rejection — runs against a randomly initialised network of the same shape. That
keeps CI at about a minute and keeps accuracy where it belongs, in evaluation.

CI additionally enforces that no `.pt` file is ever committed and that the git
directory stays under 50MB.

## Layout

```text
.
├── app/
│   ├── config.py         # paths, weight artifact, limits
│   ├── preprocessing.py  # the one transform definition
│   ├── model.py          # construction, download, checksum verification
│   ├── predict.py        # top-k inference
│   ├── schemas.py        # response contract
│   └── main.py           # routes
├── training/             # train.py + the original notebook
├── evaluation/           # evaluate.py + RESULTS.md
├── tests/
├── model/classes.json
├── Dockerfile
└── .github/workflows/ci.yml
```

## Stack

Python · PyTorch · torchvision · FastAPI · Pydantic · Uvicorn · Pillow · Docker ·
pytest · ruff

## License

MIT.

# Food Vision

**A ResNet-50 image classification model exposed as a small FastAPI service.**

Food Vision takes an uploaded food image, preprocesses it to the model's expected input, runs inference with a trained ResNet-50, and returns the predicted food category as JSON.

The API is intentionally small: one model, one prediction endpoint, and a Docker image that packages the service for reproducible deployment.

---

## What it actually does

The model currently predicts **101 food categories**. The class list is stored with the model rather than hard-coded into the API, so the returned label is mapped from the model output at inference time.

The prediction path is:

```text
image upload
    ↓
RGB conversion
    ↓
resize to 224 × 224
    ↓
Tensor conversion + ImageNet normalization
    ↓
ResNet-50 inference
    ↓
predicted class
```

The model is loaded once when the application starts, switched to evaluation mode, and moved to CUDA when a GPU is available. Otherwise it runs on CPU.

## API

The service exposes:

```text
GET  /
POST /predict
```

`GET /` returns a simple health response. `POST /predict` accepts an image file and returns the predicted class:

```json
{
  "prediction": "pizza"
}
```

FastAPI also exposes the automatically generated Swagger UI at `/docs`, which makes the endpoint easy to test without writing a client.

## Model

The classifier uses **ResNet-50** with a final linear layer sized to the number of food classes. The trained weights are loaded from `model/food_vision_resnet50.pt`, while the corresponding labels live in `model/classes.json`.

The repository does not currently include a dedicated evaluation report with accuracy, per-class metrics, or a held-out test breakdown, so those numbers are deliberately not claimed here.

## Deployment

The application is packaged with Docker using Python 3.10 and runs Uvicorn on port `8000`.

Build and run it with:

```bash
docker build -t foodvision .
docker run -p 8000:8000 foodvision
```

Then open:

```text
http://localhost:8000/docs
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

For Docker, the image copies the API code, model files, and dependency list into the container and starts the same FastAPI application through Uvicorn.

## Project structure

```text
.
├── app/
│   ├── main.py       # FastAPI routes
│   ├── predict.py    # model loading + inference
│   └── utils.py      # image preprocessing
├── model/
│   ├── classes.json
│   └── food_vision_resnet50.pt
├── Dockerfile
└── requirements.txt
```

## Stack

Python · PyTorch · torchvision · ResNet-50 · FastAPI · Uvicorn · Pillow · Docker

## Scope

This project is mainly about taking a trained vision model and turning it into a usable inference service. The machine learning model does the classification; FastAPI and Docker handle the interface and deployment layer.
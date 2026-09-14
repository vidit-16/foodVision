"""FastAPI service."""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.config import DEFAULT_TOP_K, MAX_TOP_K, MAX_UPLOAD_BYTES
from app.model import WeightsUnavailableError, load_classes
from app.predict import predict
from app.preprocessing import InvalidImageError
from app.schemas import HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Food Vision",
    description="ResNet-50 food image classifier over the 101 Food-101 categories.",
    version="1.0.0",
)


@app.get("/", response_model=HealthResponse)
def health() -> HealthResponse:
    from app import predict as predict_module

    return HealthResponse(
        status="ok",
        model_loaded=predict_module._model is not None,
        num_classes=len(load_classes()),
    )


@app.get("/classes", response_model=list[str])
def classes() -> list[str]:
    """The label set, in logit order. Useful for clients building a UI."""
    return load_classes()


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
) -> PredictionResponse:
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds the {MAX_UPLOAD_BYTES} byte limit",
        )
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")

    import io

    try:
        result = predict(io.BytesIO(payload), top_k=top_k)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except WeightsUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictionResponse(**result)

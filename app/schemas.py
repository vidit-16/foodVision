"""Response models. These define the API's output contract; tests assert against them."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str = Field(description="Food-101 class name")
    confidence: float = Field(ge=0.0, le=1.0, description="Softmax probability")


class PredictionResponse(BaseModel):
    prediction: str = Field(description="Highest-confidence label")
    predictions: list[Prediction] = Field(description="Ranked, highest confidence first")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int

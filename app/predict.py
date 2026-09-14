"""Inference.

The classifier is loaded lazily rather than at import time, so importing this
module (in tests, in tooling) does not trigger a checkpoint download.
"""

from __future__ import annotations

from typing import BinaryIO

import torch
from torch import nn

from app.config import DEFAULT_TOP_K
from app.model import load_classes, load_model, select_device
from app.preprocessing import prepare

_model: nn.Module | None = None
_device: torch.device | None = None


def get_model() -> tuple[nn.Module, torch.device]:
    """Return the process-wide model, loading it on first call."""
    global _model, _device
    if _model is None:
        _device = select_device()
        _model = load_model(_device)
    return _model, _device  # type: ignore[return-value]


def set_model(model: nn.Module, device: torch.device | None = None) -> None:
    """Install a model explicitly. Used by tests to avoid the real checkpoint."""
    global _model, _device
    _model = model
    _device = device or torch.device("cpu")


def classify_tensor(
    batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, float | str]]:
    """Run a prepared batch of one image and return ranked predictions."""
    classes = load_classes()
    k = max(1, min(top_k, len(classes)))

    with torch.no_grad():
        logits = model(batch.to(device))
        probabilities = torch.softmax(logits, dim=1)[0]

    scores, indices = torch.topk(probabilities, k)
    return [
        {"label": classes[int(index)], "confidence": round(float(score), 6)}
        for score, index in zip(scores, indices, strict=True)
    ]


def predict(image_file: BinaryIO | str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Classify one uploaded image.

    Returns the top prediction alongside the ranked list, so callers that only
    want a label do not have to index into an array.
    """
    model, device = get_model()
    predictions = classify_tensor(prepare(image_file), model, device, top_k)
    return {"prediction": predictions[0]["label"], "predictions": predictions}

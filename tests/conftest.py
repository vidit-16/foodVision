"""Shared fixtures.

The test suite deliberately does not download the 95MB checkpoint. Everything
that can be verified without trained weights — the output contract, the
preprocessing, the API's error handling — is verified against a randomly
initialised network of the same shape. Accuracy is not a unit-test concern; it
is measured by `evaluation/evaluate.py` and reported in evaluation/RESULTS.md.
"""

from __future__ import annotations

import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import predict as predict_module
from app.model import build_model, load_classes


@pytest.fixture(scope="session")
def classes() -> list[str]:
    return load_classes()


@pytest.fixture(scope="session")
def untrained_model(classes: list[str]) -> torch.nn.Module:
    torch.manual_seed(0)
    model = build_model(len(classes))
    model.eval()
    return model


@pytest.fixture(autouse=True)
def installed_model(untrained_model: torch.nn.Module):
    """Install the stand-in model for every test, then reset global state."""
    predict_module.set_model(untrained_model, torch.device("cpu"))
    yield
    predict_module._model = None
    predict_module._device = None


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def make_image(size: tuple[int, int] = (400, 300), color: str = "red") -> io.BytesIO:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_image().getvalue()

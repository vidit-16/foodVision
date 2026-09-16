"""Tests added to kill mutants that survived scripts/mutation_test.py."""

from __future__ import annotations

import io

import pytest
from torchvision import transforms

from app import model as model_module
from app.model import WeightsUnavailableError, resolve_weights
from app.preprocessing import InvalidImageError, inference_transform, open_image


def test_resize_uses_bicubic_interpolation():
    resize = inference_transform().transforms[0]
    assert isinstance(resize, transforms.Resize)
    assert resize.interpolation == transforms.InterpolationMode.BICUBIC


def test_truncated_image_is_rejected_at_decode_time(jpeg_bytes):
    with pytest.raises(InvalidImageError):
        open_image(io.BytesIO(jpeg_bytes[: len(jpeg_bytes) // 2]))


def test_empty_upload_reports_empty_upload(client):
    response = client.post("/predict", files={"file": ("e.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["detail"] == "empty upload"


def test_corrupt_cache_is_removed_even_if_redownload_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(model_module, "WEIGHTS_PATH_OVERRIDE", None)
    monkeypatch.setattr(model_module, "WEIGHTS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(model_module, "WEIGHTS_SHA256", "0" * 64)
    (tmp_path / model_module.WEIGHTS_FILENAME).write_bytes(b"stale")

    def offline(url, timeout=None):
        raise OSError("offline")

    monkeypatch.setattr(model_module.urllib.request, "urlopen", offline)
    with pytest.raises(WeightsUnavailableError):
        resolve_weights()
    assert not (tmp_path / model_module.WEIGHTS_FILENAME).exists()


def test_preprocessing_matches_training_contract():
    """The checkpoint was trained with ImageNet normalisation, resize 256, crop 224."""
    resize, crop, _, normalize = inference_transform().transforms
    assert resize.size == 256
    assert crop.size == (224, 224)
    assert tuple(normalize.mean) == (0.485, 0.456, 0.406)
    assert tuple(normalize.std) == (0.229, 0.224, 0.225)

"""Model and inference contract."""

from __future__ import annotations

import json

import pytest
import torch

from app.config import CLASSES_PATH
from app.model import WeightsUnavailableError, build_model, resolve_weights
from app.predict import classify_tensor, predict
from app.preprocessing import prepare
from tests.conftest import make_image


def test_class_list_is_101_unique_food101_labels(classes):
    assert len(classes) == 101
    assert len(set(classes)) == 101
    assert all(label == label.lower() and " " not in label for label in classes)


def test_class_list_on_disk_is_a_json_list():
    with open(CLASSES_PATH, encoding="utf-8") as handle:
        assert isinstance(json.load(handle), list)


def test_head_is_sized_to_the_class_list(classes):
    model = build_model(len(classes))
    assert model.fc.out_features == len(classes)
    assert model.fc.in_features == 2048


def test_forward_pass_produces_one_logit_per_class(untrained_model, classes):
    logits = untrained_model(prepare(make_image()))
    assert logits.shape == (1, len(classes))


def test_top_k_is_ranked_normalized_and_the_right_length(untrained_model, classes):
    predictions = classify_tensor(
        prepare(make_image()), untrained_model, torch.device("cpu"), top_k=5
    )
    assert len(predictions) == 5
    confidences = [p["confidence"] for p in predictions]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert all(p["label"] in classes for p in predictions)


def test_top_k_is_clamped_to_the_number_of_classes(untrained_model, classes):
    predictions = classify_tensor(
        prepare(make_image()), untrained_model, torch.device("cpu"), top_k=999
    )
    assert len(predictions) == len(classes)


def test_predict_returns_the_top_label_and_the_ranked_list():
    result = predict(make_image(), top_k=3)
    assert result["prediction"] == result["predictions"][0]["label"]
    assert len(result["predictions"]) == 3


def test_inference_is_deterministic_for_the_same_image():
    image = make_image().getvalue()
    import io

    first = predict(io.BytesIO(image))
    second = predict(io.BytesIO(image))
    assert first == second


def test_missing_weights_override_fails_loudly(monkeypatch):
    monkeypatch.setattr("app.model.WEIGHTS_PATH_OVERRIDE", "/nonexistent/weights.pt")
    with pytest.raises(WeightsUnavailableError):
        resolve_weights()


def test_corrupt_cached_checkpoint_is_rejected(monkeypatch, tmp_path):
    bad = tmp_path / "weights.pt"
    bad.write_bytes(b"not a checkpoint")
    monkeypatch.setattr("app.model.WEIGHTS_PATH_OVERRIDE", str(bad))
    with pytest.raises(WeightsUnavailableError, match="checksum mismatch"):
        resolve_weights()

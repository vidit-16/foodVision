"""Parity: the HTTP API, the core predict() function and the manual pipeline agree."""

from __future__ import annotations

import io

import torch

from app.predict import classify_tensor, predict
from app.preprocessing import inference_transform, open_image, prepare
from tests.conftest import make_image


def test_api_matches_core_predict(client, jpeg_bytes):
    api = client.post(
        "/predict", params={"top_k": 5}, files={"file": ("m.jpg", jpeg_bytes, "image/jpeg")}
    ).json()
    core = predict(io.BytesIO(jpeg_bytes), top_k=5)
    assert api == core


def test_prepare_matches_explicit_transform_pipeline():
    raw = make_image(size=(333, 517), color="green").getvalue()
    via_prepare = prepare(io.BytesIO(raw))
    manual = inference_transform()(open_image(io.BytesIO(raw))).unsqueeze(0)
    assert torch.equal(via_prepare, manual)


def test_classify_tensor_matches_manual_softmax_topk(untrained_model, classes):
    batch = prepare(make_image(color="yellow"))
    with torch.no_grad():
        probs = torch.softmax(untrained_model(batch), dim=1)[0]
    scores, indices = torch.topk(probs, 4)
    expected = [classes[int(i)] for i in indices]
    got = classify_tensor(batch, untrained_model, torch.device("cpu"), top_k=4)
    assert [p["label"] for p in got] == expected
    assert [p["confidence"] for p in got] == [round(float(s), 6) for s in scores]


def test_smaller_top_k_is_a_prefix_of_larger(untrained_model):
    batch = prepare(make_image(color="purple"))
    small = classify_tensor(batch, untrained_model, torch.device("cpu"), top_k=3)
    large = classify_tensor(batch, untrained_model, torch.device("cpu"), top_k=10)
    assert large[:3] == small

"""HTTP surface."""

from __future__ import annotations

import io

from app.config import MAX_UPLOAD_BYTES


def test_health_reports_status_and_class_count(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["num_classes"] == 101


def test_classes_endpoint_returns_the_label_set(client, classes):
    response = client.get("/classes")
    assert response.status_code == 200
    assert response.json() == classes


def test_predict_returns_the_documented_schema(client, jpeg_bytes, classes):
    response = client.post("/predict", files={"file": ("meal.jpg", jpeg_bytes, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "predictions"}
    assert body["prediction"] in classes
    assert set(body["predictions"][0]) == {"label", "confidence"}


def test_top_k_query_parameter_controls_list_length(client, jpeg_bytes):
    response = client.post(
        "/predict", params={"top_k": 3}, files={"file": ("m.jpg", jpeg_bytes, "image/jpeg")}
    )
    assert len(response.json()["predictions"]) == 3


def test_top_k_out_of_range_is_rejected(client, jpeg_bytes):
    for value in (0, 11, -1):
        response = client.post(
            "/predict",
            params={"top_k": value},
            files={"file": ("m.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 422


def test_non_image_upload_returns_400(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"hello there", "text/plain")}
    )
    assert response.status_code == 400


def test_empty_upload_returns_400(client):
    response = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_oversized_upload_returns_413(client):
    oversized = b"\xff\xd8" + b"0" * (MAX_UPLOAD_BYTES + 10)
    response = client.post(
        "/predict", files={"file": ("big.jpg", oversized, "image/jpeg")}
    )
    assert response.status_code == 413


def test_missing_file_field_returns_422(client):
    assert client.post("/predict").status_code == 422


def test_a_mislabelled_png_is_still_classified(client):
    """Content type is not trusted; decoding decides."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), "blue").save(buffer, format="PNG")
    response = client.post(
        "/predict", files={"file": ("meal.jpg", buffer.getvalue(), "image/jpeg")}
    )
    assert response.status_code == 200


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "PredictionResponse" in schema["components"]["schemas"]

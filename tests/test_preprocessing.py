"""Preprocessing behaviour.

These tests pin the transform in place. If someone later swaps the direct
resize-and-crop for a plain resize, or changes the normalization constants, the published
evaluation numbers stop describing the deployed model — so the change should
break a test and force a re-run of evaluation.
"""

from __future__ import annotations

import io

import pytest
import torch

from app.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from app.preprocessing import InvalidImageError, open_image, prepare
from tests.conftest import make_image


def test_prepare_returns_batched_tensor_of_expected_shape():
    tensor = prepare(make_image())
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32


@pytest.mark.parametrize(
    "size", [(4000, 300), (300, 4000), (224, 224), (50, 50), (1, 1)]
)
def test_any_aspect_ratio_produces_the_same_input_shape(size):
    assert prepare(make_image(size=size)).shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_normalization_constants_are_applied():
    # A pure-black image maps to exactly -mean/std in every channel.
    tensor = prepare(make_image(color="black"))[0]
    for channel, (mean, std) in enumerate(zip(NORMALIZE_MEAN, NORMALIZE_STD, strict=True)):
        expected = -mean / std
        assert torch.allclose(
            tensor[channel], torch.full_like(tensor[channel], expected), atol=1e-5
        )


def test_grayscale_and_rgba_are_converted_to_three_channels():
    from PIL import Image

    for mode in ("L", "RGBA", "P"):
        buffer = io.BytesIO()
        Image.new(mode, (100, 100)).convert(mode).save(buffer, format="PNG")
        buffer.seek(0)
        assert open_image(buffer).mode == "RGB"


def test_non_image_upload_raises_invalid_image():
    with pytest.raises(InvalidImageError):
        prepare(io.BytesIO(b"this is not an image"))


def test_truncated_image_raises_invalid_image():
    truncated = make_image().getvalue()[:60]
    with pytest.raises(InvalidImageError):
        prepare(io.BytesIO(truncated))


def test_center_crop_preserves_aspect_ratio():
    # A wide image: red on the outer thirds, blue in the middle. A direct resize
    # would keep the red; a short-side resize plus centre crop keeps only blue.
    from PIL import Image

    image = Image.new("RGB", (900, 300), "red")
    image.paste(Image.new("RGB", (300, 300), "blue"), (300, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    tensor = prepare(buffer)[0]
    red = (1.0 - NORMALIZE_MEAN[0]) / NORMALIZE_STD[0]
    assert tensor[0].max() < red - 1.0

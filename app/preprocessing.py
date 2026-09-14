"""Image preprocessing.

The API and the evaluation script both import `inference_transform` from here.
Keeping one definition is the point of the module: if evaluation preprocesses
images differently from the served endpoint, the reported accuracy does not
describe the thing that is actually deployed.

Note on the resize: the image is resized directly to 224x224 rather than
resize-short-side-then-center-crop. This distorts aspect ratio, but it is what
the shipped checkpoint was trained with, so inference matches training.
"""

from __future__ import annotations

from typing import BinaryIO

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


class InvalidImageError(ValueError):
    """Raised when an upload cannot be decoded as an image."""


def inference_transform() -> transforms.Compose:
    """The exact transform applied before every forward pass."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def open_image(source: BinaryIO | str) -> Image.Image:
    """Decode an uploaded file or path to RGB, or raise InvalidImageError."""
    try:
        image = Image.open(source)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("file could not be decoded as an image") from exc
    return image.convert("RGB")


def prepare(source: BinaryIO | str) -> torch.Tensor:
    """Decode and transform an image into a batched model input tensor."""
    tensor = inference_transform()(open_image(source))
    return tensor.unsqueeze(0)

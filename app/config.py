"""Single source of truth for paths, the weight artifact, and request limits."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CLASSES_PATH = Path(os.getenv("FOODVISION_CLASSES", REPO_ROOT / "model" / "classes.json"))

# The network. A timm model name: the architecture comes from here, the trained
# weights from the release asset below. The in22k-pretrained variant transfers
# noticeably better to fine-grained datasets like Food-101 than the in1k one.
MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"

# The checkpoint is not stored in git. It is published as a GitHub release asset
# and downloaded on first use into the cache directory below.
WEIGHTS_URL = os.getenv(
    "FOODVISION_WEIGHTS_URL",
    "https://github.com/vidit-16/foodVision/releases/download/v2.0.0/food_vision_convnext_tiny.pt",
)
WEIGHTS_SHA256 = os.getenv(
    "FOODVISION_WEIGHTS_SHA256",
    "8c68831bc884ed0b8e5aad10f639d3d81eb7cd936b07a8e63a65767bec7d122a",
)
WEIGHTS_CACHE_DIR = Path(
    os.getenv("FOODVISION_CACHE_DIR", Path.home() / ".cache" / "foodvision")
)
WEIGHTS_FILENAME = "food_vision_convnext_tiny.pt"

# A local path may be supplied to bypass the download entirely (used by Docker
# images that bake the weights in, and by anyone working offline).
WEIGHTS_PATH_OVERRIDE = os.getenv("FOODVISION_WEIGHTS_PATH")

# Preprocessing. The short side is resized to RESIZE_SIZE, then the centre
# IMAGE_SIZE square is cropped (crop fraction 0.875, the ConvNeXt default).
# Changing any of these invalidates the published evaluation numbers.
RESIZE_SIZE = 256
IMAGE_SIZE = 224
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def _int_env(name: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    """Read a bounded integer from the environment, failing with a clear message."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{name} must be >= {minimum}{upper}, got {value}")
    return value


# API limits.
MAX_TOP_K = 10
MAX_UPLOAD_BYTES = _int_env("FOODVISION_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
# Bounded by MAX_TOP_K: a larger default would make every request without an
# explicit top_k fail FastAPI's own query validation.
DEFAULT_TOP_K = _int_env("FOODVISION_TOP_K", 5, maximum=MAX_TOP_K)
# Seconds before a stalled checkpoint download is abandoned instead of hanging.
DOWNLOAD_TIMEOUT_SECONDS = _int_env("FOODVISION_DOWNLOAD_TIMEOUT", 60)

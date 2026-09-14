"""Single source of truth for paths, the weight artifact, and request limits."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CLASSES_PATH = Path(os.getenv("FOODVISION_CLASSES", REPO_ROOT / "model" / "classes.json"))

# The checkpoint is not stored in git. It is published as a GitHub release asset
# and downloaded on first use into the cache directory below.
WEIGHTS_URL = os.getenv(
    "FOODVISION_WEIGHTS_URL",
    "https://github.com/vidit-16/foodVision/releases/download/v1.0.0/food_vision_resnet50.pt",
)
WEIGHTS_SHA256 = os.getenv(
    "FOODVISION_WEIGHTS_SHA256",
    "faa9aed3658982ac22be66618b0147a93859c201535c2745a213e8ace5a144c2",
)
WEIGHTS_CACHE_DIR = Path(
    os.getenv("FOODVISION_CACHE_DIR", Path.home() / ".cache" / "foodvision")
)
WEIGHTS_FILENAME = "food_vision_resnet50.pt"

# A local path may be supplied to bypass the download entirely (used by Docker
# images that bake the weights in, and by anyone working offline).
WEIGHTS_PATH_OVERRIDE = os.getenv("FOODVISION_WEIGHTS_PATH")

# Preprocessing. These values describe the transform the shipped checkpoint was
# trained with; changing them invalidates the published evaluation numbers.
IMAGE_SIZE = 224
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# API limits.
MAX_UPLOAD_BYTES = int(os.getenv("FOODVISION_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
DEFAULT_TOP_K = int(os.getenv("FOODVISION_TOP_K", "5"))
MAX_TOP_K = 10

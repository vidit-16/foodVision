"""Model construction and weight resolution.

The 95MB checkpoint is not tracked in git. It is resolved in this order:

1. `FOODVISION_WEIGHTS_PATH`, if set and present on disk.
2. The cache directory, if a previous run already downloaded it.
3. The GitHub release asset, downloaded once and checksum-verified.

Construction is separated from weight loading so tests can exercise the model
contract against a randomly initialised network without a 95MB download.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from app.config import (
    CLASSES_PATH,
    WEIGHTS_CACHE_DIR,
    WEIGHTS_FILENAME,
    WEIGHTS_PATH_OVERRIDE,
    WEIGHTS_SHA256,
    WEIGHTS_URL,
)

logger = logging.getLogger(__name__)


class WeightsUnavailableError(RuntimeError):
    """Raised when the checkpoint cannot be resolved or fails verification."""


@lru_cache(maxsize=1)
def load_classes() -> list[str]:
    """Load the ordered class list. Index i corresponds to logit i."""
    with open(CLASSES_PATH, encoding="utf-8") as handle:
        classes = json.load(handle)
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"{CLASSES_PATH} must contain a non-empty JSON list")
    return classes


def build_model(num_classes: int) -> nn.Module:
    """A ResNet-50 with the head resized to `num_classes`. No weights loaded."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path) -> None:
    if not WEIGHTS_SHA256:
        return
    actual = _sha256(path)
    if actual != WEIGHTS_SHA256:
        raise WeightsUnavailableError(
            f"checksum mismatch for {path}: expected {WEIGHTS_SHA256}, got {actual}"
        )


def _download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    logger.info("downloading checkpoint from %s", WEIGHTS_URL)
    try:
        with urllib.request.urlopen(WEIGHTS_URL) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise WeightsUnavailableError(f"could not download weights: {exc}") from exc
    try:
        _verify(partial)
    except WeightsUnavailableError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)


def resolve_weights() -> Path:
    """Return a local path to a verified checkpoint, downloading if needed."""
    if WEIGHTS_PATH_OVERRIDE:
        override = Path(WEIGHTS_PATH_OVERRIDE)
        if not override.is_file():
            raise WeightsUnavailableError(
                f"FOODVISION_WEIGHTS_PATH points at a missing file: {override}"
            )
        _verify(override)
        return override

    cached = WEIGHTS_CACHE_DIR / WEIGHTS_FILENAME
    if cached.is_file():
        try:
            _verify(cached)
            return cached
        except WeightsUnavailableError:
            logger.warning("cached checkpoint failed verification, re-downloading")
            cached.unlink(missing_ok=True)

    _download(cached)
    return cached


def load_model(device: str | torch.device = "cpu") -> nn.Module:
    """Build the network, load verified weights, and put it in eval mode."""
    classes = load_classes()
    model = build_model(len(classes))
    state_dict = torch.load(resolve_weights(), map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise WeightsUnavailableError(
            f"checkpoint does not match the architecture "
            f"(missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]})"
        )
    model.eval()
    model.to(device)
    return model


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

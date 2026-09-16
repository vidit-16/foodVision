"""Lightweight mutation testing (mutmut does not run natively on Windows).

Each mutant changes one operator, constant or guard in a core module, runs the
test suite, and counts as killed if any test fails. Source files are always
restored, even on Ctrl+C.

    python scripts/mutation_test.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (file, original snippet, mutated snippet). Each snippet must occur exactly once.
MUTANTS = [
    ("app/config.py", "value < minimum or", "value <= minimum or"),
    ("app/config.py", "value > maximum)", "value >= maximum)"),
    ("app/config.py", 'raw.strip() == ""', 'raw.strip() != ""'),
    ("app/config.py", "RESIZE_SIZE = 256", "RESIZE_SIZE = 224"),
    ("app/config.py", "IMAGE_SIZE = 224", "IMAGE_SIZE = 256"),
    ("app/config.py", "NORMALIZE_MEAN = (0.485,", "NORMALIZE_MEAN = (0.5,"),
    ("app/config.py", "MAX_TOP_K = 10", "MAX_TOP_K = 11"),
    ("app/predict.py", "max(1, min(top_k, len(classes)))", "max(1, top_k)"),
    ("app/predict.py", "dim=1)[0]", "dim=0)[0]"),
    ("app/predict.py", "round(float(score), 6)", "round(float(score), 1)"),
    ("app/predict.py", 'predictions[0]["label"]', 'predictions[-1]["label"]'),
    ("app/preprocessing.py", "transforms.CenterCrop(IMAGE_SIZE),", ""),
    ("app/preprocessing.py", "InterpolationMode.BICUBIC", "InterpolationMode.NEAREST"),
    ("app/preprocessing.py", 'image.convert("RGB")', "image"),
    ("app/preprocessing.py", "image.load()", "pass"),
    ("app/preprocessing.py", "tensor.unsqueeze(0)", "tensor"),
    ("app/model.py", "if actual != WEIGHTS_SHA256:", "if actual == WEIGHTS_SHA256:"),
    ("app/model.py", "if not WEIGHTS_SHA256:", "if WEIGHTS_SHA256:"),
    ("app/model.py", "if not override.is_file():", "if override.is_file():"),
    ("app/model.py", "if cached.is_file():", "if False:"),
    ("app/model.py", "timeout=DOWNLOAD_TIMEOUT_SECONDS", "timeout=None"),
    ("app/model.py", "or not classes or", "or"),
    ("app/model.py", "cached.unlink(missing_ok=True)", "pass"),
    ("app/main.py", "len(payload) > MAX_UPLOAD_BYTES", "len(payload) > MAX_UPLOAD_BYTES * 2"),
    ("app/main.py", "if not payload:", "if payload is None:"),
    ("app/main.py", "status_code=400, detail=str(exc)", "status_code=500, detail=str(exc)"),
    ("app/main.py", "ge=1, le=MAX_TOP_K", "ge=0, le=MAX_TOP_K"),
]


def main() -> int:
    killed = 0
    survivors = []
    for index, (rel, old, new) in enumerate(MUTANTS, 1):
        path = ROOT / rel
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            print(f"[{index}] ERROR snippet not unique in {rel}: {old!r}")
            return 2
        path.write_text(original.replace(old, new), encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        finally:
            path.write_text(original, encoding="utf-8")
        dead = result.returncode != 0
        killed += dead
        if not dead:
            survivors.append((rel, old, new))
        status = "killed  " if dead else "SURVIVED"
        print(f"[{index}/{len(MUTANTS)}] {status} {rel}: {old!r}", flush=True)
    total = len(MUTANTS)
    print(f"\nMutation score: {killed}/{total} = {100 * killed / total:.1f}%")
    for rel, old, new in survivors:
        print(f"  survivor {rel}: {old!r} -> {new!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

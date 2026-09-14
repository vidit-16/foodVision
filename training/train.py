"""Fine-tune ConvNeXt-Tiny on Food-101.

Uses the official Food-101 split: training sees only the 75,750-image train
split, with a slice held back for validation. The 25,250-image test split is
never opened here; `evaluation/evaluate.py` scores it afterwards.

Recipe, sized to finish on a free Colab T4 in about three hours:

- ConvNeXt-Tiny, ImageNet-22k pretrained (`app.config.MODEL_NAME`)
- AdamW, lr 2e-4, weight decay 0.05, one warmup epoch then cosine decay
- Label smoothing 0.1, stochastic depth 0.1
- RandomResizedCrop, horizontal flip, TrivialAugmentWide, random erasing
- Mixed precision, batch size 64, 12 epochs

Colab sessions disconnect. With `--checkpoint-dir` pointing at Google Drive the
full training state is saved after every epoch, and `--resume` continues from
the last completed epoch.

Usage:

    python training/train.py --data-dir ./data

    # Resumable, for Colab:
    python training/train.py --data-dir ./data \\
        --checkpoint-dir /content/drive/MyDrive/food_vision_v2 --resume

    # Prove the pipeline runs end to end without training anything:
    python training/train.py --data-dir ./data --smoke-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Food101

# Running `python training/train.py` puts only this script's folder on sys.path;
# add the repo root so `app` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD  # noqa: E402
from app.model import build_model  # noqa: E402
from app.preprocessing import inference_transform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="food_vision_convnext_tiny.pt")
    parser.add_argument("--classes-output", default="model/classes.json")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Save full training state here after every epoch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from --checkpoint-dir/last.pt if it exists.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two batches per split and write nothing.",
    )
    return parser.parse_args()


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.25, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )


def build_splits(data_dir: str, val_fraction: float, seed: int):
    """Split the official train split into train and validation subsets.

    Two dataset objects wrap the same files so the subsets can carry different
    transforms: augmentation on train, the inference transform on validation.
    The official test split is not opened here.
    """
    train_base = Food101(
        root=data_dir, split="train", transform=train_transform(), download=True
    )
    val_base = Food101(root=data_dir, split="train", transform=inference_transform())

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_base), generator=generator).tolist()
    val_size = int(len(train_base) * val_fraction)

    val_indices, train_indices = order[:val_size], order[val_size:]
    return (
        Subset(train_base, train_indices),
        Subset(val_base, val_indices),
        train_base.classes,
    )


def warmup_cosine(total_steps: int, warmup_steps: int, base_lr: float, min_lr: float):
    """Per-step LR multiplier: linear warmup, then cosine decay to min_lr."""
    floor = min_lr / base_lr

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return factor


def run_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer=None,
    scheduler=None,
    scaler=None,
    limit=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = seen = 0
    use_amp = device.type == "cuda"

    with torch.set_grad_enabled(training):
        for index, batch in enumerate(loader):
            if limit is not None and index >= limit:
                break
            images = batch[0].to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            targets = batch[1].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item() * targets.size(0)
            correct += int((logits.argmax(1) == targets).sum())
            seen += targets.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, val_set, classes = build_splits(args.data_dir, args.val_fraction, args.seed)
    print(f"train={len(train_set)} val={len(val_set)} classes={len(classes)} device={device}")

    classes_path = Path(args.classes_output)
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    classes_path.write_text(json.dumps(classes, indent=2), encoding="utf-8")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    model = build_model(len(classes), pretrained=True, drop_path_rate=args.drop_path)
    model.to(device, memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine(
            total_steps=args.epochs * steps_per_epoch,
            warmup_steps=int(args.warmup_epochs * steps_per_epoch),
            base_lr=args.lr,
            min_lr=args.min_lr,
        ),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    start_epoch = 1
    best_val = 0.0
    state_path = Path(args.checkpoint_dir) / "last.pt" if args.checkpoint_dir else None

    if args.resume and state_path is not None and state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_val = state["best_val"]
        print(f"resumed from {state_path}: epoch {state['epoch']} done, best_val={best_val:.4f}")

    limit = 2 if args.smoke_test else None
    epochs = 1 if args.smoke_test else args.epochs

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, device, criterion, optimizer, scheduler, scaler, limit
        )
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion, limit=limit)
        elapsed = time.time() - started

        print(
            f"epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"({elapsed / 60:.1f} min, ~{elapsed * (epochs - epoch) / 60:.0f} min left)",
            flush=True,
        )

        if args.smoke_test:
            continue
        if val_acc >= best_val:
            best_val = val_acc
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"  saved {args.output} (val_acc={val_acc:.4f})", flush=True)
        if state_path is not None:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            partial = state_path.with_suffix(".part")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                },
                partial,
            )
            partial.replace(state_path)

    if args.smoke_test:
        print("smoke test completed; no checkpoint written")
        return

    print(
        f"\nBest validation accuracy: {best_val:.4f}\n"
        f"Now score the held-out test split:\n"
        f"  FOODVISION_WEIGHTS_PATH={args.output} FOODVISION_WEIGHTS_SHA256= \\\n"
        f"  python evaluation/evaluate.py --data-dir {args.data_dir}"
    )


if __name__ == "__main__":
    main()

"""Fine-tune ResNet-50 on Food-101.

This is the non-notebook version of the pipeline in `training/food_vision.ipynb`,
with one substantive correction: it uses the official Food-101 train/test split.

The notebook loaded every image through a single `ImageFolder` over the Kaggle
food41 mirror, which puts all 101,000 images in the training set — including the
25,250 images of the official test split. A model trained that way cannot be
evaluated honestly, because nothing is left that it has not seen.

Here the test split is never opened. Training uses the official 75,750-image
train split, with a slice held back for validation.

Everything else follows the notebook: ImageNet-V2 initialisation, full
fine-tune, Adam at 1e-4, plain cross-entropy, mixed precision, batch size 32,
and the resize-only transform shared with the served API.

Usage:

    python training/train.py --data-dir ./data --epochs 5

    # Prove the pipeline runs end to end without training anything:
    python training/train.py --data-dir ./data --smoke-test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Food101
from torchvision.models import ResNet50_Weights, resnet50

from app.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from app.preprocessing import inference_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="food_vision_resnet50.pt")
    parser.add_argument("--classes-output", default="model/classes.json")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Add random crop, flip and jitter. Off by default so the default "
        "run reproduces the notebook recipe.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two batches per split and write nothing.",
    )
    return parser.parse_args()


def train_transform(augment: bool) -> transforms.Compose:
    if not augment:
        # The notebook applied no augmentation: its training transform and its
        # inference transform were the same object.
        return inference_transform()
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def build_splits(data_dir: str, val_fraction: float, seed: int, augment: bool):
    """Split the official train split into train and validation subsets.

    Two dataset objects wrap the same files so the subsets can carry different
    transforms: augmentation on train, the inference transform on validation.
    The official test split is not opened here.
    """
    train_base = Food101(
        root=data_dir, split="train", transform=train_transform(augment), download=True
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


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, limit=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = seen = 0
    use_amp = scaler is not None and device.type == "cuda"

    with torch.set_grad_enabled(training):
        for index, batch in enumerate(loader):
            if limit is not None and index >= limit:
                break
            images = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss) * targets.size(0)
            correct += int((logits.argmax(1) == targets).sum())
            seen += targets.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, val_set, classes = build_splits(
        args.data_dir, args.val_fraction, args.seed, args.augment
    )
    print(f"train={len(train_set)} val={len(val_set)} classes={len(classes)} device={device}")

    classes_path = Path(args.classes_output)
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    classes_path.write_text(json.dumps(classes, indent=2), encoding="utf-8")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    limit = 2 if args.smoke_test else None
    epochs = 1 if args.smoke_test else args.epochs
    best_val = 0.0

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, device, criterion, optimizer, scaler, limit
        )
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion, limit=limit)

        print(
            f"epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
            f"({time.time() - started:.0f}s)"
        )

        if args.smoke_test:
            continue
        if val_acc >= best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  saved {args.output} (val_acc={val_acc:.4f})")

    if args.smoke_test:
        print("smoke test completed; no checkpoint written")
        return

    print(
        f"\nBest validation accuracy: {best_val:.4f}\n"
        f"Now score the held-out test split:\n"
        f"  python evaluation/evaluate.py --data-dir {args.data_dir}"
    )


if __name__ == "__main__":
    main()

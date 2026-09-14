# Training

## Provenance of the released checkpoints

### v1.1.0 (served)

Trained with `train.py` via `colab_train_and_eval.ipynb` on a Colab T4, using the
default recipe below on the official train split only. It scores 82.19% top-1
and 96.27% top-5 on the 25,250 test images; see `evaluation/RESULTS.md`.

### v1.0.0 (superseded)

The original checkpoint was trained in
`food_vision.ipynb`: ResNet-50 initialised from ImageNet-V2 weights, final layer
replaced with a 101-way head, fully fine-tuned for 5 epochs with Adam at 1e-4,
plain cross-entropy, mixed precision, batch size 32. About 73 minutes on a
Colab T4.

**That run has no held-out data.** The notebook loaded the Kaggle `food41`
mirror through a single `ImageFolder` over `/content/images`, which is all
101,000 Food-101 images — the official 25,250-image test split included. The
progress bar confirms it: 3,157 batches of 32 per epoch.

So no honest accuracy number can be produced for that checkpoint. Scoring it on
the Food-101 test split would report training accuracy under a test label, and
the repository claims nothing about its accuracy.

`train.py` corrects this. It uses the official split and never opens the test
data, which is how v1.1.0 was produced.

## Files

| File | What it is |
| --- | --- |
| `food_vision.ipynb` | The original Colab notebook, kept for provenance. Outputs stripped |
| `train.py` | The same recipe with the official train/test split restored |
| `colab_train_and_eval.ipynb` | Open in Colab on a T4, Run all: trains and evaluates in one pass |

## Reproducing

```bash
pip install -r requirements.txt
python training/train.py --data-dir ./data --epochs 5
```

Food-101 downloads on first run (~5GB). Training uses the official 75,750-image
train split with 10% held back for validation; the 25,250 test images are left
for `evaluation/evaluate.py`. Roughly 2,365 batches per epoch at batch size 32,
about 11 minutes per epoch on a T4.

Check the wiring first without waiting an hour:

```bash
python training/train.py --data-dir ./data --smoke-test
```

## Recipe

| Setting | Value | Source |
| --- | --- | --- |
| Backbone | ResNet-50, `IMAGENET1K_V2` | notebook |
| Head | `Linear(2048, 101)` | notebook |
| Fine-tuning | All layers | notebook |
| Optimiser | Adam, lr 1e-4 | notebook |
| Loss | Cross-entropy | notebook |
| Precision | AMP on CUDA | notebook |
| Batch size | 32 | notebook |
| Epochs | 5 | notebook |
| Augmentation | None by default; `--augment` opts in | notebook used none |
| Split | Official train/test, 10% of train for validation | corrected here |

`--augment` adds random resized crop, horizontal flip and colour jitter. It
departs from the recipe that produced the released weights, so run the default
first if you want a comparable baseline.

## Preprocessing

Validation and inference both use `inference_transform()` from
`app/preprocessing.py` — the single definition shared with the served API.
Changing the resize or the normalisation constants invalidates both the released
checkpoint and any published evaluation numbers.

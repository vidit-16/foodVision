# Training

## Provenance of the released checkpoints

### v2.0.0 (served)

ConvNeXt-Tiny trained with `train.py` via `colab_train_and_eval.ipynb` on a
free Colab T4, using the recipe below on the official train split only. It
scores 91.89% top-1 and 98.76% top-5 on the 25,250 test images; see
`evaluation/RESULTS.md`.

### v1.1.0 (superseded)

ResNet-50 on the official train split with the original notebook recipe.
82.19% top-1, 96.27% top-5.

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
data, which is how v1.1.0 and v2.0.0 were produced.

## Files

| File | What it is |
| --- | --- |
| `food_vision.ipynb` | The original ResNet-50 Colab notebook (v1.0.0), kept for provenance. Outputs stripped |
| `train.py` | The maintained pipeline: ConvNeXt-Tiny on the official train split |
| `colab_train_and_eval.ipynb` | Open in Colab on a T4, Run all: trains, resumes after disconnects, evaluates |

## Reproducing

```bash
pip install -r requirements.txt
python training/train.py --data-dir ./data
```

Food-101 downloads on first run (~5GB). Training uses the official 75,750-image
train split with 10% held back for validation; the 25,250 test images are left
for `evaluation/evaluate.py`. About 1,065 batches per epoch at batch size 64,
roughly 12–15 minutes per epoch on a T4.

Check the wiring first:

```bash
python training/train.py --data-dir ./data --smoke-test
```

## Recipe (v2.0.0)

| Setting | Value |
| --- | --- |
| Backbone | ConvNeXt-Tiny, `convnext_tiny.fb_in22k_ft_in1k` (timm) |
| Head | `Linear(768, 101)` |
| Fine-tuning | All layers, stochastic depth 0.1 |
| Optimiser | AdamW, lr 2e-4, weight decay 0.05 |
| Schedule | 1 warmup epoch, cosine decay to 1e-6 |
| Loss | Cross-entropy, label smoothing 0.1 |
| Augmentation | RandomResizedCrop (scale 0.25–1), flip, TrivialAugmentWide, RandomErasing 0.25 |
| Precision | AMP on CUDA, channels-last |
| Batch size / epochs | 64 / 12 |
| Split | Official train/test, 10% of train for validation |

v1.1.0 used ResNet-50 with the original notebook recipe (Adam 1e-4, 5 epochs, no
augmentation, direct 224×224 resize) and scored 82.19% top-1.

## Preprocessing

Validation and inference both use `inference_transform()` from
`app/preprocessing.py` — the single definition shared with the served API:
short side to 256, centre crop 224. Changing it invalidates the published
evaluation numbers.

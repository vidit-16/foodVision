# Evaluation results

**No accuracy is reported for the current checkpoint, and none can be.**

The released weights were trained on all 101,000 Food-101 images, including the
25,250 that make up the official test split (see `training/README.md`). There is
no unseen data left to score against. Running the test split through this
checkpoint would produce a high number that describes memorisation, not
generalisation, so it is not published here.

## Getting a real number

1. Retrain with the official split, which `train.py` uses:

   ```bash
   python training/train.py --data-dir ./data --epochs 5
   ```

2. Score the held-out test split:

   ```bash
   python evaluation/evaluate.py --data-dir ./data
   ```

That overwrites this file with top-1 and top-5 accuracy over the 25,250 test
images, per-class accuracy, and the most frequent confusions — all measured
through the same preprocessing the API serves.

Until step 2 has run against a checkpoint trained in step 1, treat the model as
unvalidated.

# Three-class directed transfer and loss ablation

This controlled experiment uses the unchanged P14/P16/P19 small-to-large model
pairs and six directed transitions among ImageNet goldfish (1), monarch
butterfly (2), and volcano (7). It compares the full classification plus
linear pull/push objective with the second loss alone (classification term
removed). Model weights and attack hyperparameters are held fixed.

The six directions are 1→2, 1→7, 2→1, 2→7, 7→1, and 7→2. Each direction uses
its own source images, while the source and target reference banks are fixed,
disjoint 48-image class banks. The initial run is an 8-image controlled smoke;
it does not start 50-image or 500-image work.

Prepare (requires `IMAGENET_ROOT` and local model snapshots):

```bash
PYTHONPATH=src:experiments/2026-08-pull-push-multiclass-v4/src \
  python experiments/2026-08-pull-push-multiclass-v4/src/prepare_data.py \
  --config experiments/2026-09-threeclass-transfer/config/threeclass.yaml \
  --output-dir outputs/threeclass_transfer_8

PYTHONPATH=src:experiments/2026-08-pull-push-multiclass-v4/src \
  python experiments/2026-08-pull-push-multiclass-v4/src/screen_transitions.py \
  --config experiments/2026-09-threeclass-transfer/config/threeclass.yaml \
  --output-dir outputs/threeclass_transfer_8
```

Run the 36-cell smoke (3 model pairs × 6 directions × 2 loss arms):

```bash
bash experiments/2026-09-threeclass-transfer/run_smoke.sh
```

Outputs are written under `outputs/threeclass_transfer_8`; frozen attack images
are intentionally excluded from version control. The state JSON files retain
proxy hits, TASR, ASR, loss mode, and transition metadata.
# Tracked result snapshot

The `results/` directory contains the small, reproducible result snapshot
from the completed 8-image ablations/tuning and 50-image confirmation runs.
It includes CSV summaries, JSON state records, manifests, and diagnostics.
Large generated images and model tensors are intentionally excluded.

- `results/scale8/`: loss ablations and tuning states
- `results/scale50/`: final confirmation states, summary, and prototype-distance diagnostics

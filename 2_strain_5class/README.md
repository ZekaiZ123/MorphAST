# 2 — 5-Class Strain Identification (Untreated)

Train and evaluate a 5-way EfficientNet-B0 classifier on untreated 0-min
phase-contrast micrographs. The classes are five *E. coli*-related strains:

| Class | Strain |
|---|---|
| 0 | E. coli DH5α (wild type) |
| 1 | E. coli DH5α KanR plasmid |
| 2 | E. coli DH5α AmpR plasmid |
| 3 | AIEC (adherent-invasive *E. coli*) |
| 4 | UPEC (uropathogenic *E. coli*) |

## Headline result (paper)

5-fold cross-validated, EfficientNet-B0:
- Accuracy = **0.9590 ± 0.0031**
- Macro-F1 = **0.9598 ± 0.0050**
- Macro AUROC = **0.9985**

## Scripts

| Script | Role |
|---|---|
| `build_5class_manifest.py` | Walks the raw image directory tree, parses strain from folder names, emits a manifest CSV used as Section 1's input. |
| `train_5class.py` | 5-fold cross-validation training driver. Auto-detects the number of classes from the manifest. |
| `run_5class_benchmark.py` | Aggregates per-fold metrics into a benchmark summary JSON. |
| `extract_embeddings.py` | Loads a fold checkpoint and extracts 1280-D penultimate-layer features for held-out fields — used for t-SNE / nearest-neighbour analyses. |
| `run_5class_train.sbatch` | SBATCH template for the cluster. |

## Inputs

- The QC-filtered manifest from Section 1 (`manifest_with_segmentation_qc.csv`)
  filtered to untreated 0-min fields, with `class_id ∈ [0, 5)`.

## Outputs (per fold)

```
<out_dir>/fold_{1..5}/
├── best.pt                 # best-macro-F1 checkpoint
├── metrics.json            # per-class precision/recall/F1, confusion matrix
└── training_log.jsonl      # per-epoch loss + validation metrics
<out_dir>/
├── crossval_summary.json   # mean ± SD across folds
└── crossval_predictions.csv # one row per held-out field × all class probabilities
```

## Canonical run

```bash
# SBATCH
sbatch run_5class_train.sbatch

# or interactive
python train_5class.py \
  --manifest_csv      .../manifest_with_segmentation_qc.csv \
  --out_dir           Model/Benchmark_0min_5class/efficientnet_b0 \
  --model             efficientnet_b0 \
  --mode              crossval \
  --n_folds 5 --epochs 15 --batch_size 32 \
  --lr 3e-4 --weight_decay 1e-4 \
  --use_amp --filter_auto_dropped
```

## Hyperparameters (paper)

- Backbone: **EfficientNet-B0**, ImageNet-pretrained (from `timm`)
- Head: `LayerNorm(1280) → Dropout(0.2) → Linear(1280, 5)`
- Input: 224 × 224, grayscale → 3-channel replication, ImageNet-normalised
- Augmentation: random horizontal + vertical flip only (no rotation, no colour jitter)
- Loss: **class-balanced cross-entropy** (weight ∝ 1/n_class)
- Optimizer: **AdamW**, lr = 3 × 10⁻⁴, weight decay = 1 × 10⁻⁴, no schedule
- Mixed precision (AMP)
- 15 epochs, batch size 32
- 5-fold `StratifiedGroupKFold`, stratified on `class_id`, grouped on imaging-field ID — prevents per-field leakage
- Model selection: highest validation macro-F1 epoch per fold

## Notes

- `extract_embeddings.py` is the entry-point for the t-SNE / nearest-neighbour
  analyses; it registers a forward hook on the head's LayerNorm to capture the
  pre-head 1280-D pooled features, then writes them as a `.npy` matrix
  alongside a metadata CSV.
- Alternative backbones (`resnet18`, `swin_tiny`) are supported via `--model`;
  the paper's headline backbone-comparison panel uses these.

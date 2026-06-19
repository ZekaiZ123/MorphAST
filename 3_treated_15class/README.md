# 3 — 15-Class Strain × Treatment Classification (Treated)

Train and evaluate a 15-way EfficientNet-B0 classifier on antibiotic-treated
phase-contrast micrographs. The 15 classes are the cartesian product of 5
strains × 3 antibiotics:


Each class is observed at 6 timepoints (5, 10, 15, 30, 60, 90 min). Held-out
predictions are pooled across timepoints for headline metrics; per-timepoint
splits are also reported.

## Headline result (paper)

5-fold cross-validated, EfficientNet-B0:
- Exact-class accuracy = **0.9112 ± 0.0124**
- Macro-F1 = **0.9318 ± 0.0068**
- Treatment-level accuracy (15-way softmax collapsed to 3 treatments) = **91.1%**
- Treatment-level macro AUROC = **0.994**

## Scripts

| Script | Role |
|---|---|
| `build_15class_manifest.py` | Walks raw image directories, parses strain × treatment from folder names, emits a manifest with 15 unique class labels. Supports a 3rd `--additional_root` to mix late-arriving fields into an existing manifest. |
| `train_15class.py` | 5-fold cross-validation training driver. Auto-detects 15 classes from the manifest. |
| `run_15class_benchmark.py` | Aggregates per-fold metrics into a benchmark summary JSON. |
| `extract_gradcam.py` | Loads a fold checkpoint, computes Grad-CAM heatmaps for a set of held-out fields, and writes them as per-field PNGs + a metadata CSV. |
| `extract_gradcam_multiclass.py` | Multi-target Grad-CAM variant used for the per-treatment attribution analyses. |
| `infer_clinical_isolates.py` | Inference-only entry point for clinical UPEC isolates dosed with ampicillin — used in the supplementary clinical-translation analysis. |
| `run_15class_train.sbatch` | SBATCH template for the 15-class training run. |
| `run_13class_train.sbatch` | SBATCH template kept for reproducibility of the pre-publication 13-class baseline. |
| `run_15class_tsne_export.sbatch` | SBATCH template for the held-out t-SNE export (used to generate the embedding-space figure). |

## Inputs

- The QC-filtered manifest from Section 1 (`manifest_with_segmentation_qc.csv`)
  containing antibiotic-exposed fields, with `class_id ∈ [0, 15)`.

## Outputs (per fold)

```
<out_dir>/fold_{1..5}/
├── best.pt                 # best-macro-F1 checkpoint
├── metrics.json            # per-class precision/recall/F1, confusion matrix
└── training_log.jsonl      # per-epoch loss + validation metrics
<out_dir>/
├── crossval_summary.json   # 15-class mean ± SD across folds
├── crossval_predictions.csv # one row per held-out field × all 15 class probabilities
└── class_names.json        # canonical ordering of the 15 strain × treatment labels
```

## Canonical run

```bash
# SBATCH
sbatch run_15class_train.sbatch

# or interactive
python train_15class.py \
  --manifest_csv      .../manifest_with_segmentation_qc.csv \
  --out_dir           Model/Benchmark_Treated_Condition_15class_QC/efficientnet_b0 \
  --model             efficientnet_b0 \
  --mode              crossval \
  --n_folds 5 --epochs 15 --batch_size 32 \
  --lr 3e-4 --weight_decay 1e-4 \
  --use_amp --filter_auto_dropped
```

## Hyperparameters (paper)

Same as Section 2 (intentionally — keeps the 5-class vs 15-class headline numbers directly comparable):

- Backbone: **EfficientNet-B0**, ImageNet-pretrained (`timm`)
- Head: `LayerNorm(1280) → Dropout(0.2) → Linear(1280, 15)`
- Input: 224 × 224, grayscale → 3-channel replication, ImageNet-normalised
- Augmentation: random horizontal + vertical flip only
- Loss: **class-balanced cross-entropy**, weights recomputed per fold from the training partition
- Optimizer: **AdamW**, lr = 3 × 10⁻⁴, weight decay = 1 × 10⁻⁴, no schedule
- Mixed precision (AMP)
- 15 epochs, batch 32
- 5-fold `StratifiedGroupKFold` (random_state = 42), stratified on `class_id`, grouped on imaging-field — prevents per-field leakage
- Model selection: highest validation macro-F1 epoch per fold

## Deriving treatment-level metrics from the 15-class softmax

Three-treatment probabilities are computed by summing the five per-strain
softmax outputs that share a treatment:

```python
prob_treatment_Ampicillin = sum(prob[<strain> + " Ampicillin"]) for strain in ...)
prob_treatment_MΦ-ND      = sum(...)
prob_treatment_Kanamycin  = sum(...)
pred_treatment = argmax over the three
```

This is the canonical "3-class antimicrobial-response" prediction reported in
the paper.

## Notes

- The 13-class historical run (which omitted the two AmpR + Kana and
  KanR + Amp combinations) is preserved via `run_13class_train.sbatch` for
  comparison; new analyses should use the 15-class build.
- Grad-CAM hotspot quantile defaults of 0.40 (hotspot view) / 0.20 (overlay
  view), with a tone-gamma of 0.45, are the published values producing
  Fig 4A/B.

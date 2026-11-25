# Morphology-based Antimicrobial Susceptibility Testing

Code accompanying the manuscript on rapid bacterial strain identification and
antibiotic-response classification from label-free phase-contrast microscopy.
Deep convolutional classifiers (EfficientNet-B0) read strain identity and
antibiotic-induced morphology from raw field images, with Omnipose-based
segmentation used only for per-field quality control.

## Repository structure

The repo is organised into **three task-aligned sections**:

| # | Section | What it does |
|---|---|---|
| 1 | [`1_preprocessing_omnipose/`](./1_preprocessing_omnipose/) | Tile each raw field into a 3×3 grid, run Omnipose segmentation on every tile, and aggregate per-tile cell counts and exclude flags into per-field QC summaries. Used to filter fields before training. |
| 2 | [`2_strain_5class/`](./2_strain_5class/) | Train + evaluate an EfficientNet-B0 5-way classifier for *strain identification* on untreated 0-min images (5 strains, n ≈ 804 QC-passing fields). |
| 3 | [`3_treated_15class/`](./3_treated_15class/) | Train + evaluate an EfficientNet-B0 15-way classifier for *joint strain × antibiotic-treatment* on treated images (5 strains × 3 antibiotics, n ≈ 17,123 QC-passing fields across 6 timepoints from 5 to 90 min). Includes Grad-CAM interpretability and clinical-isolate inference. |

A small [`shared/`](./shared/) directory provides the EfficientNet-B0 wrapper,
image-loading helpers, and benchmark utilities used by Sections 2 and 3.

The corrected reviewer-requested evaluation is an additional, separate workflow:
[`Code/Training/Rebuttal/`](./Code/Training/Rebuttal/README.md). Its original source
paths are retained for compatibility with hashed evaluation records; see
[`Code/README.md`](./Code/README.md). The three publication sections above are
unchanged by this code-only port.

## Installation

A single conda environment supports everything:

```bash
conda env create -f environment.yml
conda activate morphast
```

Or with pip into an existing environment:

```bash
pip install -r requirements.txt
```

Omnipose (`omnipose` standalone package, not `cellpose-omni`) may require a
specific PyTorch/CUDA build on your cluster — re-pin if necessary.

## Typical workflow

1. **Preprocessing** — point Section 1's segmentation scripts at your raw
   microscopy directory; they produce a QC manifest of per-field rows with
   `drop_from_training_seg_qc` flagged for fields that fail.
2. **Strain ID** (untreated) — feed the 5-class manifest to Section 2's
   training script. Output: 5 fold checkpoints + a held-out predictions CSV.
3. **Treatment ID** (treated) — feed the 15-class manifest to Section 3's
   training script. Output: 5 fold checkpoints + a held-out predictions CSV
   + Grad-CAM heatmaps + per-cell morphology features (via Section 1 segmentation).

Each section's `README.md` documents inputs, outputs, canonical commands,
and the hyperparameters used in the paper.

## Reproducing the paper's headline numbers

The table below is retained from the publication workflow. These are **not the
corrected inner-selected outer-test results**. Do not substitute these values
for the corrected run's `outer_fold_metrics.csv` or `corrected_results.json`.

| Benchmark | Metric | Value |
|---|---|---|
| Section 2: 5-class strain ID | Accuracy (5-fold mean ± SD) | **0.9590 ± 0.0031** |
| Section 2: 5-class strain ID | Macro-F1 | 0.9598 ± 0.0050 |
| Section 3: 15-class strain × treatment | Accuracy (5-fold mean) | **0.9112 ± 0.0124** |
| Section 3: 15-class strain × treatment | Macro-F1 | 0.9318 ± 0.0068 |
| Section 3: 3-class treatment level (collapsed from 15-way softmax) | Accuracy | 0.9112 |

## Corrected outer-fold evaluation

The [Rebuttal workflow](./Code/Training/Rebuttal/README.md) separates checkpoint
selection from testing: each outer development partition contains a grouped
inner-validation holdout, and the selected checkpoint is evaluated on the
untouched outer test fold. It includes the five-class strain and 15-class
strain-treatment tasks, per-fold macro-F1/accuracy/balanced accuracy, corrected
confusion/ROC/PR plots, six treated-cohort figures, dataset/error audits, and the
separately labeled preparation/slide sensitivity analysis.

Use the pinned dependencies above, and supply the original benchmark metadata,
QC manifests and images separately. Full input requirements, frozen-run checks,
SLURM commands and plotting commands are in the workflow README. Existing trained
runs can be verified and plotted without rerunning training. ImageNet weights
must be cached before offline training; cluster paths and resource directives
must be configured locally. No datasets, checkpoints, predictions, feature
arrays, generated figures or audit-output directories are included in this port.

Run the synthetic test suite from the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Checkpoint-selection correction does not by itself establish biological or
acquisition-session independence, or resolve conflicting-label duplicates.
Those limitations remain explicit in the generated reports.

## Data and trained checkpoints

Raw microscopy images, intermediate manifests, and trained model weights are
**not** distributed with the source code due to size (~tens of GB). They are
available from the corresponding author on reasonable request.



## License

See [LICENSE](./LICENSE).

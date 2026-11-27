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

| Benchmark | Metric | Value |
|---|---|---|
| Section 2: 5-class strain ID | Accuracy (5-fold mean ± SD) | **0.9590 ± 0.0031** |
| Section 2: 5-class strain ID | Macro-F1 | 0.9598 ± 0.0050 |
| Section 3: 15-class strain × treatment | Accuracy (5-fold mean) | **0.9112 ± 0.0124** |
| Section 3: 15-class strain × treatment | Macro-F1 | 0.9318 ± 0.0068 |
| Section 3: 3-class treatment level (collapsed from 15-way softmax) | Accuracy | 0.9112 |

## Data and trained checkpoints

Raw microscopy images, intermediate manifests, and trained model weights are
**not** distributed with the source code due to size (~tens of GB). They are
available from the corresponding author on reasonable request.



## License

See [LICENSE](./LICENSE).

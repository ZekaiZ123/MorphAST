# Reviewer 1: checkpoint selection and outer testing

The original trainers evaluated the held-out fold every epoch, chose the epoch
with the highest macro-F1 on that fold, and reported performance on the same
fold. This runner separates checkpoint/backbone selection from outer testing.

## Frozen protocol

- Preserve the original QC populations, labels, five outer `StratifiedGroupKFold`
  partitions, group identifiers, and seed 42. Verify exact outer membership
  against original prediction files wherever available. Untreated original
  training did not save these files; verify its manifest, splitter and fold sizes.
- Within each outer development partition, take the first split of a new
  five-way `StratifiedGroupKFold`, seeded with `42 + 1000 + outer_fold`.
  Nominal train/validation/test fractions are 64/16/20%. No split/seed search.
- Train ImageNet-initialized models for 15 epochs using the original image
  preprocessing, augmentation, optimizer, batch size and learning rate.
  Calculate class weights using only the inner training subset.
- Choose checkpoints by inner-validation macro-F1, retaining the earliest epoch
  on ties. Untreated candidates are ResNet-18, EfficientNet-B0 and Swin-Tiny;
  choose the backbone on the same inner validation subset, with declared
  candidate order resolving ties. The treated candidate is EfficientNet-B0.
- Freeze all checkpoint and backbone decisions before constructing any outer-test
  loader. Evaluate each candidate once; the primary untreated prediction uses
  the internally chosen candidate. Per-backbone test scores are secondary results.
- Report individual outer-fold scores, unweighted mean, sample SD (`ddof=1`),
  population SD (`ddof=0`, for compatibility), and min/max for macro-F1, accuracy
  and balanced accuracy. Report pooled predictions separately from fold means.
- Export original-versus-corrected paired differences. These differences also
  reflect reduced training size and cannot isolate checkpoint-selection bias.

The protocol uses one inner holdout, not a full inner cross-validation search.
It does not use the newer identity-recognition datasets or external test data.
Existing image/field grouping is retained, so acquisition/replicate independence
beyond those identifiers is not established by this experiment.

## Run

From the project root, using the `omnipose_env` interpreter:

```bash
python Code/Training/Rebuttal/prepare_nested_cv.py --run_dir Model/Rebuttal_CV/run_YYYYMMDD
sbatch --output=Model/Rebuttal_CV/run_YYYYMMDD/train-%A_%a.out Code/Training/Rebuttal/run_nested_cv.sbatch /absolute/path/to/Model/Rebuttal_CV/run_YYYYMMDD
sbatch --dependency=afterany:TRAIN_JOB_ID --output=Model/Rebuttal_CV/run_YYYYMMDD/report-%j.out Code/Training/Rebuttal/summarize_nested_cv.sbatch /absolute/path/to/Model/Rebuttal_CV/run_YYYYMMDD
```

The array contains five untreated jobs (each training three backbones) followed
by five treated jobs. It uses at most four GPUs concurrently. The report job
runs after the array terminates, including on failure, so missing folds are
explicit and no partial average is presented as a completed five-fold result.

Preparation freezes manifests, split assignments, source copies and hashes in
the output directory. Training executes the frozen source. ImageNet weights
must already be cached (`HF_HUB_OFFLINE=1`). Checkpoints save immediately on
validation improvement. A completed selection may resume before testing;
an interrupted test evaluation is flagged rather than silently repeated.

Final artifacts: `corrected_results.md`, `corrected_results.json`,
`outer_fold_metrics.csv`, `legacy_comparison.csv`, and per-task
`outer_test_predictions.csv`. Model weights live in
`TASK/fold_N/BACKBONE/best.pt`; selection metadata and test metrics are adjacent.

Method reference: [scikit-learn nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

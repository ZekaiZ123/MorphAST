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

### Treated 15-class only

Preparation defaults to the original untreated/treated-13 pair. Use the explicit
`treated15` task to prepare a separate 15-class run without retraining either:

```bash
RUN_DIR="$PWD/Model/Rebuttal_CV/treated15_RUN_ID"
python Code/Training/Rebuttal/prepare_nested_cv.py --run_dir "$RUN_DIR" --tasks treated15 --limitation "Checkpoint-selection correction only; unresolved conflicting-label and cross-fold byte duplicates remain."
sbatch --array=1-5%4 --output="$RUN_DIR/train-%A_%a.out" "$RUN_DIR/source/Code/Training/Rebuttal/run_nested_cv.sbatch" "$RUN_DIR" treated15
sbatch --dependency=afterany:TRAIN_JOB_ID --output="$RUN_DIR/report-%j.out" "$RUN_DIR/source/Code/Training/Rebuttal/summarize_nested_cv.sbatch" "$RUN_DIR"
```

The single-cohort array takes the task as its second argument; each array index
is the outer fold. The 15-class baseline is
`Model/Benchmark_Treated_Condition_15class_QC/efficientnet_b0` (not the archived
`efficientnet_b0_broken_classids` directory). Exact original outer-test image
membership and all 15 labels in every inner/outer partition are checked before
submission. No label corrections, duplicate removal or regrouping are performed.
Repeat `--limitation` to record unresolved issues in both `run.json` and the final
report; preserving the legacy folds does not establish content-level independence.

Method reference: [scikit-learn nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## Dataset accounting audit

Run `audit_dataset_accounting.py --out_dir outputs/rebuttal/dataset_accounting_RUN_ID`
with a new output directory. The audit inventories the original raw source trees,
checks 12/13/15-class cohort transitions, rebuilds field QC from tile metadata,
and matches retained populations to training arguments and saved prediction rows.
It exports a Markdown report, before/after tables by strain, treatment and time,
source-folder breakdowns, row-level decisions, tile coverage, and source hashes.
Replicate identifiers remain UNKNOWN unless verified; workbook filename matches
are exported separately as candidates. Input images and manifests are read only.

## Culture/slide/session CV feasibility

Before attempting the requested higher-level split, run:

```bash
python Code/Training/Rebuttal/audit_grouped_cv_feasibility.py --out_dir outputs/rebuttal/grouped_cv_feasibility_RUN_ID --verify_images
```

This preflight uses the treated 15-class retained population, verifies the
metadata audit's source evidence, and checks that image identities, class labels,
original field groups and time points agree with the original manifest. All
copies of conflicting-label exact content are excluded BEFORE filtering for
verified metadata, so an unmatched counterpart cannot conceal a conflict.
Missing culture/preparation/slide/session metadata is excluded from this matched
analysis. Same-label exact copies remain linked. Source images and labels are
not modified; these are explicit sensitivity-population exclusions.

Culture, slide, session and exact-hash constraints are combined by connected
components: sharing ANY identifier keeps images together, including indirect
links. The original-field-preserving version is reported alongside the requested
IDs alone, field/hash grouping, preparation grouping and individual higher-level
identifiers. SCID is not used. Identifier tuples are not a safe alternative,
because the same culture could then cross partitions through different slides.

Outputs include `feasibility.md`, `feasibility.json`, a row-level inclusion ledger,
class/time exclusion counts, component membership, per-class grouping support,
and original field groups linking multiple preparations. `--verify_images`
rehashes all original retained files, including excluded conflicts; without it,
the report explicitly identifies use of the existing full-cohort hash inventory.

The preflight never launches training or writes trainable folds. A class present
in only one component cannot occur in both training and testing. Counts above
the requested fold threshold are only necessary conditions, not a validated
split: inner-validation coverage and crossed class constraints still need
checking. Do not silently discard culture/field constraints or replace the task
with unseen-class evaluation to make a split possible. Preparation/slide counts
do not establish independent biological replication.

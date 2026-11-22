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

### Corrected five-class confusion plot

```bash
python Code/Training/Rebuttal/plot_corrected_fiveclass_confusion.py \
  --run_dir Model/Rebuttal_CV/reviewer1_20261118 \
  --out_dir outputs/rebuttal/corrected_fiveclass_confusion_RUN_ID
```

This reads corrected outer-test predictions, not original publication metrics.
Frozen sources, manifests, group-disjoint splits, checkpoint/selection hashes and
complete once-per-image outer coverage are verified before plotting raw counts.
The primary inner-selected procedure and fixed EfficientNet-B0 secondary result
are exported separately; `--procedures primary` requests only the former.
Each includes PNG/PDF/SVG figures, pooled/per-fold integer CSVs, predictions and
metrics. Source/artifact hashes are recorded in `verification.json`. A new output
directory is required. No inference, retraining or earlier-output changes occur.
This corrects selection/evaluation reuse, not experimental-replicate dependence.

### Corrected five-class ROC plot

```bash
python Code/Training/Rebuttal/plot_corrected_fiveclass_roc.py \
  --run_dir Model/Rebuttal_CV/reviewer1_20261118 \
  --confusion_dir outputs/rebuttal/corrected_fiveclass_confusion_20261121/primary \
  --out_dir outputs/rebuttal/corrected_fiveclass_roc_RUN_ID
```

Five one-versus-rest curves threshold each strain's softmax probability against
the other four strains. Only the primary inner-selected outer-test predictions
are used, with scores, labels, fold/backbone assignments and confusion counts
checked against the specified verified primary confusion export. The inset
enlarges FPR 0-0.05 (95-100% specificity), TPR 0.80-1.00; the legend reports
full-range pooled AUC, not partial AUC. Curves are not smoothed or jittered.
PNG/PDF/SVG figures, a caption, threshold coordinates and TP/FP/TN/FN counts,
positive/negative denominators, pooled/per-fold AUCs and source/artifact hashes
are saved in a new directory. Fold SDs are descriptive, not confidence intervals;
pooled AUC and mean fold AUC are kept separate. No inference or retraining occurs.

### Corrected five-class per-fold performance

```bash
python Code/Training/Rebuttal/plot_corrected_fiveclass_performance.py \
  --run_dir Model/Rebuttal_CV/reviewer1_20261118 \
  --confusion_dir outputs/rebuttal/corrected_fiveclass_confusion_20261121/primary \
  --out_dir outputs/rebuttal/corrected_fiveclass_performance_RUN_ID
```

Grouped bars show macro-F1, balanced accuracy, and overall accuracy separately
for each outer test fold, using only the primary inner-selected backbones and
checkpoints. Predictions, scores, model assignments, and confusion counts are
checked against the same verified primary reference used by the ROC figure.
The percentage axis starts at zero, each bar has its numerical value, and fold
labels include test-image counts. The footer gives unweighted across-fold means
and sample SDs (ddof=1), not pooled scores or confidence intervals. No error bars
are assigned to individual folds. PNG/PDF/SVG figures, a caption, exact bar
values, fold metrics and summaries, predictions, and source/artifact hashes are
exported to a new directory without inference, retraining, or overwriting earlier
outputs. The original field grouping does not establish biological independence.

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

Add `--metadata_audit outputs/rebuttal/experimental_independence_RUN_ID` to use
the verified newer provider-workbook audit. Source hashes, exact image keys and
retained coverage are rechecked. The audit adds row-level acquisition identifiers
and additive before/after tables by strain, treatment, time, culture, preparation,
slide and session, preserving UNKNOWN categories and workbook row references.
SCID is excluded. Composite associations are not provider-file checksum matches;
recorded IDs do not establish independent biological-replicate counts. Metadata
availability never changes the historical QC population. Cohort versions and
later rebuttal sensitivity exclusions are reported as different populations.

## Raw confusion counts and error images

For the **original publication EfficientNet-B0 models**, rather than corrected
reruns, export class-specific counts, recorded acquisition IDs and raw matrices:

```bash
python Code/Training/Rebuttal/export_publication_evidence.py \
  --metadata_audit outputs/rebuttal/experimental_independence_20261120 \
  --out_dir outputs/rebuttal/publication_model_evidence_RUN_ID
```

The default exports untreated 5-class, treated 13-class and treated 15-class
results separately. Use `--cohorts treated_15class` to select one. Counts before
and after QC, actual culture/preparation/slide/session ID lists, missing-ID image
denominators, and class/time/acquisition tables retain the original populations.
Independent biological/experimental replicate counts remain NOT ESTABLISHED;
SCID is excluded and documented ID counts are not equated to replicate counts.
Exact and treatment-decision matrices contain integer counts including zeros,
with pooled and per-fold exports; treated matrices also have time/acquisition
strata. Original untreated per-image predictions were not saved, so its matrices
come directly from fold metrics, not reconstructed image-level predictions.
Population, class order, metadata coverage, per-fold matrices and metrics are
cross-checked. `verification.json` records source and output hashes. A new output
directory is required; no model, data, manuscript or earlier output is changed.
These original held-out folds were also used for checkpoint selection and must
not be presented as corrected independent-test estimates.

```bash
python Code/Training/Rebuttal/audit_error_cases.py --out_dir outputs/rebuttal/error_cases_RUN_ID
```

The new output directory contains separate 13-class and 15-class legacy-CV
audits: complete integer confusion matrices (exact labels and both treatment
decision rules), fold/time/source-stratified counts, all prediction identifiers,
all error records, and a browsable `error_gallery.html` for every exact-label or
marginal-treatment error. Original images are copied byte-for-byte with SHA256
checksums. Full-field previews and 224x224 pre-normalization input views have no
extra contrast enhancement. Deterministic DH5alpha ampicillin/kanamycin comparison
pages include correctly classified controls matched on time, fold and source,
with selection rules and missing controls explicitly recorded.

These are not corrected nested-CV predictions. Untreated original per-fold and
pooled 5x5 counts are exported separately; that trainer did not save per-image
predictions, so untreated image identities are not guessed from the counts.
Source collections are not asserted to be experimental batches. Batch/session,
biological replicate and optical modality remain UNKNOWN until acquisition
metadata is provided. An optional `--acquisition_metadata path.csv` accepts an
exact-path mapping with columns `image_path`, `batch_id`,
`acquisition_session_id`, `biological_replicate_id`, `optical_modality`, and
`metadata_source`. Use literal UNKNOWN for unavailable values; duplicate paths
and missing provenance are rejected. Regenerate into a new directory after
metadata is supplied. No paper, model, original image, or manifest is modified.

For corrected treated outer-test predictions and the cleaned sensitivity arms,
use the separate exporter (new directory required):

```bash
python Code/Training/Rebuttal/audit_corrected_error_cases.py \
  --out_dir outputs/rebuttal/corrected_error_evidence_RUN_ID \
  --metadata_audit outputs/rebuttal/experimental_independence_20261120 \
  --hash_inventory outputs/rebuttal/duplicate_conflicts_20261118_verified/all_retained_image_hashes.csv \
  --analysis corrected_13class Model/Rebuttal_CV/reviewer1_20261118 treated \
  --analysis corrected_15class Model/Rebuttal_CV/reviewer1_treated15_20261119 treated15 \
  --analysis cleaned_field Model/Rebuttal_CV/preparation_sensitivity_20261120 field_baseline \
  --analysis cleaned_preparation Model/Rebuttal_CV/preparation_sensitivity_20261120 preparation_sensitivity
```

The exporter validates frozen manifests, sources, splits, checkpoint selection,
checkpoints and saved predictions, and freshly rehashes the original retained
15-class image population. It runs no inference. Every exact-label or marginal-
treatment error receives an original-file copy, a loaded full-field preview and
a 224-pixel pre-normalization input view. Known contradictory-label copies are
also archived, including correctly classified copies. HTML galleries display
condition, time, fold, culture, preparation, slide, session, metadata-match status,
workbook row, exact-content overlap and SHA256. Batch, biological replicate and
optical contrast mode remain UNKNOWN; SCID is not used.

All raw matrices include zeros, with separate exact-label, treatment-marginal and
top-class-treatment decision rules and fold/time/acquisition/conflict strata.
DH5alpha comparison controls exclude known label conflicts and do not relax
known acquisition constraints. Cross-stratum QC summaries are descriptive, not
mechanistic tests. Original and cleaned analyses are never pooled or represented
as a pure effect of removing mislabeled copies. `presentation_review.csv` flags
draft passages and `presentation_corrections.md` supplies proposed wording without
editing the manuscript. A run is complete only when `verification.json` exists
and reports success with zero required image-export failures.

The error archive also screens exact file and model-input hashes, but that screen
is limited to exported errors/controls. For a full retained-cohort byte-duplicate
audit, including correctly classified counterparts, run:

```bash
python Code/Training/Rebuttal/audit_duplicate_conflicts.py --out_dir outputs/rebuttal/duplicate_conflicts_RUN_ID
```

All retained source paths are SHA256-hashed once across both cohorts. Reports
separate conflicting condition labels from duplicate content spanning CV folds,
include condition/time/source identifiers, and keep failed hashes unknown rather
than declaring those images unique. Neither this audit nor the error exporter
resolves labels, deletes data, changes training jobs, or reports post hoc filtered
performance as corrected evaluation.

Pass `--duplicate_audit outputs/rebuttal/duplicate_conflicts_RUN_ID` to the error
exporter to annotate the galleries/ledgers with full-cohort conflicting-label
and cross-fold duplicate flags. Exact populations, identifiers and archived
image hashes are checked before accepting those annotations. Recorded labels
must not be interpreted as verified ground truth where conflicts are present.

## Experimental independence audit

```bash
python Code/Training/Rebuttal/audit_experimental_independence.py --workbook Model/Rebuttal_CV/Dataset_A_Last_update_11_20_2026.xlsx --out_dir outputs/rebuttal/experimental_independence_RUN_ID --workers 8
```

Use a new output directory. SCID is excluded at the user's request. The audit
recovers the recorded culture, preparation, slide and acquisition-session IDs,
preserves Excel row numbers and reference notes, and checks identifier/path and
parent relationships. It does not infer independently grown biological
replicates from folder names, slide counts, dates or imaging fields.

Images are associated by exact strain, treatment, explicit timepoint and filename,
unique in both inventories, plus matching first-frame image dimensions. These
are composite metadata associations, not checksum verification of provider files.
Unmatched/ambiguous records and missing identifiers remain explicit; alternative
strain/filename candidates are exported but not accepted. No fuzzy renaming,
treatment reassignment, missing-timepoint inference or ID forward-filling occurs.

Outputs include workbook and pre/post-QC counts; a per-image mapping ledger;
retained-image identifiers and outer folds for untreated 5-class and treated
13/15-class cohorts; per-fold overlap at each identifier level for original CV
and internal-validation reruns; and necessary per-class group-count checks for
future grouped splits. Frozen split hashes/populations are verified, and treated
outer membership must agree exactly with original saved predictions. Untreated
membership is the existing frozen reconstruction because original image-level
predictions were not saved. Unknown IDs are never counted as an independent
group. The report distinguishes documented identifiers from demonstrated
biological replication and specifies remaining provenance/experiment needs.

No source image, manifest, split, model or manuscript is changed. This audit does
not resolve conflicting-label duplicates or automatically submit a new training
run. A culture-level split can be infeasible if holding out a culture removes
the only documented examples of a class; splitting its fields does not fix that.

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

## Matched preparation/slide sensitivity experiment

This is an explicitly exploratory alternative, NOT a replacement for the
infeasible culture/session-independent evaluation. The preparation arm allows
original cross-time field IDs to cross partitions because their physical meaning
has not been confirmed. Never describe it as jointly field/preparation-disjoint
or biologically independent. The command requires explicit acknowledgement:

```bash
RUN_DIR="$PWD/Model/Rebuttal_CV/preparation_sensitivity_RUN_ID"
python Code/Training/Rebuttal/prepare_preparation_sensitivity.py --run_dir "$RUN_DIR" --allow_legacy_field_overlap
sbatch --output="$RUN_DIR/train-%A_%a.out" "$RUN_DIR/source/Code/Training/Rebuttal/run_preparation_sensitivity.sbatch" "$RUN_DIR"
sbatch --dependency=afterany:TRAIN_JOB_ID --output="$RUN_DIR/report-%j.out" "$RUN_DIR/source/Code/Training/Rebuttal/summarize_preparation_sensitivity.sbatch" "$RUN_DIR"
```

Preparation rechecks the metadata-audit evidence and rehashes every retained
source image. Conflicting-label copies are identified before metadata filtering
and excluded without relabeling. Both arms receive exactly the same cleaned
images, original labels and preprocessing. The field baseline groups connected
original-field IDs and exact content; the preparation arm groups connected
preparation IDs, slide IDs and exact content. Same-label duplicates stay together.
Culture/session sharing is measured, not prevented. SCID is not used.

Both arms use the same deterministic, outcome-blind splitter: largest class-pure
components first, balancing per-class outer-test image counts with seeded tie
breaking. Each outer development partition uses a grouped inner holdout greedily
targeting 20% of its images per class while retaining training coverage. No
split/seed search or outer-test-based checkpoint selection occurs. Every class
must occur in train, validation and test. Counts and time coverage are exported;
actual training sizes can differ between arms. This is not the original outer
split reconstruction and differences cannot isolate batch bias.

The ten-job array trains fresh ImageNet-initialized EfficientNet-B0 models for
15 epochs, using the existing optimizer and checkpoint-selection code. Five
field folds precede five preparation folds, with at most four GPUs concurrently.
The report runs after the array ends, including failures, and never reports an
incomplete set of folds as a completed estimate.

Artifacts include `protocol.md`, `partition_counts.csv`,
`partition_time_counts.csv`, `identifier_overlap.csv`, frozen source/evidence,
`sensitivity_results.md/json`, `outer_fold_metrics.csv`, and
`sensitivity_comparison.csv`. Each arm exports pooled outer predictions, raw
confusion counts, per-class metrics, all error-case records, and error/confusion
counts by preparation, session, time and fold. Mean +/- sample SD and pooled
metrics are separate; fold numbers are not paired units across arms. The report
retains the unresolved physical-field, culture, session and biological-replicate
limitations regardless of the resulting performance.

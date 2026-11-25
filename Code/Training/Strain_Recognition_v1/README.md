# Nine-class bacterial identity recognition — provenance and split feasibility

This milestone creates review packets, imports immutable evidence, applies approved
integrity QC, assesses a discrete partition, and freezes it only when every dataset
gate passes. It does **not** train models or generate development features/predictions.

Current data: 8,621 Dataset_B images and 34 reserved external images. The 692 incoming
legacy-training exact matches remain explicitly tagged. No images are deleted.
The original six-artifact audit is preserved under `provenance/inventory_snapshot/`.
Its old candidate spatial groups are historical observations, not verified units.

## Run the workflow

From the repository root (NumPy/Pillow for inventory/QC; SciPy for optimization):

```bash
python Code/Training/Strain_Recognition_v1/provenance_workflow.py prepare
python Code/Training/Strain_Recognition_v1/provenance_workflow.py import --response reviewed_response.json
python Code/Training/Strain_Recognition_v1/provenance_workflow.py qc
python Code/Training/Strain_Recognition_v1/provenance_workflow.py assess
python Code/Training/Strain_Recognition_v1/provenance_workflow.py freeze
python Code/utils/identity_audit_gate.py split experiments/strain_recognition_v1
```

Use `--audit-dir` and `--repo-root` for another version/root. `prepare` is idempotent;
it does not reset reviewed values. Internal imports/assessment/QC refuse to replace a READY
partition. External-only provenance imports remain permitted while sealed, but must
preserve the exact internal evidence and split hash. New data or a revised frozen
partition require a new audit version.
The old inventory builder refuses to overwrite provenance-v2 artifacts, even with
`--refresh`. Do not manually edit the published manifest or approval flags.

This workspace's `omnipose_env` has SciPy 1.15.2 / HiGHS 1.8.0. Its Python can run
the commands above and the test suite without installing training dependencies:

```bash
/new-stg/home/zekai/miniconda/envs/omnipose_env/bin/python -m unittest discover -s tests -p 'test_identity*.py'
```

Missing solver dependencies yield UNKNOWN, never INFEASIBLE. The current unanswered
packet yields BLOCKED without invoking a solver or approving training.

## Review packet and import contract

`experiments/strain_recognition_v1/provenance/review_packet_v1/` contains series and
image-assignment CSV worksheets, draft acquisition/QC policies, a response template,
the request instructions, and a hashed packet index. There are 194 internal filename
series buckets plus one external bucket. They are review conveniences only: MRSA
series capture times interleave, and the external filename date differs from its
embedded acquisition date. Dates/coordinates never establish biological replication.

Copy the response template and return reviewed claims in JSON. All evidence file
references use repository-relative `{ "path": "...", "sha256": "..." }` records.
Responses require `review_packet_version`, named `reviewer`, timezone-qualified
`review_timestamp`, and an `evidence_reference` identifying the underlying records.
The workflow retains the original response bytes, supporting evidence copies,
response hash, and an import receipt with code version/hash and import timestamp.
Original responses remain retained even when explicitly superseded.

Example claim (illustrative placeholders, **not** laboratory evidence):

```json
{
  "image_id": "image_from_the_packet",
  "verification_status": "VERIFIED",
  "values": {
    "culture_id": "lab:C017",
    "sample_id": "lab:sample-17",
    "preparation_id": "lab:preparation-17",
    "slide_id": "lab:slide-17",
    "acquisition_session_id": "instrument:session-03",
    "field_id": "lab:original-field-42",
    "source_image_id": "lab:original-image-42"
  }
}
```

- `image_assignments`: claims like the example. `series_assignments` use
  `review_series_id` instead of `image_id` and expand to the exact packet membership.
  Use image claims for mixed series. Conflicting claims remain unresolved; an image
  claim does not silently override a series claim.
- Additional values: `source_series_id`, `coordinate_frame_id`, `instrument_id`,
  `operator`, `export_pipeline`, `verified_acquisition_date`, `verified_timezone`,
  `optics`, `verified_preparation_protocol`, and `verified_imaging_settings_json`
  (a JSON-object string). Raw acquisition metadata is never overwritten.
- `unit_reviews`: exact `image_ids` membership plus separate
  `biological_independence_status` and `acquisition_independence_status`, each
  VERIFIED / UNRESOLVED / FAILED. A review must assert independence of the resulting
  whole unit from other units, not merely confirm that its filenames differ.
  Membership changes require another unit review. Evidence IDs are attached by import.
- `edge_adjudications`: `edge_id` or an explicit `edge_ids` list and `link_state`
  CONFIRMED_MUST_LINK / CANDIDATE_LINK / REJECTED. Exact duplicates cannot be rejected.
  `additional_relationships` can document a new non-hash relationship between known
  ledger/image endpoints using `image_id_a`, `image_id_b`, `relationship`, `link_state`.
- `identity_verifications`: `class_id`, `species`, `identity_status="verified"`.
  MRSA's organism must be documented as Staphylococcus aureus before freezing.
- `external_reviews`: exact `image_ids` and `independence_status`. Evidence must
  establish independence from the internal population; no detected match is not proof.
- `acquisition_schema` / `qc_policy`: completed policy objects with status APPROVED
  and a unique version. Import freezes their content and attaches response evidence.
- `supersedes`: prior response IDs being explicitly replaced. This replaces the
  **whole** response; include all still-valid claims in the replacement. Reusing a
  policy version with different rules is prohibited. New policy versions require
  `policy_change_justification` and `not_feasibility_driven=true`. This is a recorded
  scientific attestation, not a license to optimize policy choices against results.

## Acquisition and QC policies

Every observed acquisition factor needs an explicit classification and justification:

| Role | Rule |
|---|---|
| HARD_MATCH | Exact normalized values define strata. |
| BOUNDED_VARIATION | Named, nonambiguous half-open `[min,max)` intervals define approved variation groups. |
| CROSSED_BATCH | Levels must meet approved identity-support requirements. |
| INFORMATIONAL | Recorded but unconstrained, with justification. |
| UNKNOWN_BLOCKING | Unresolved factor; no feasibility approval. |
| NOT_APPLICABLE | Explicitly inapplicable, with justification. |

Sources use manifest column names, `conditions.<raw metadata key>`, or
`verified_settings.<reviewed key>`. Supported normalization is exact, string, or
numeric. A batch requirement specifies `scope=dataset/partition/both`,
`no_class_unique_level` (boolean), and `minimum_identity_support_per_level` (1–9).
`joint_batch_factors` may specify a unique `name`, crossed-factor `sources`, and the
same requirement structure. Support counts distinct identities, never image totals.

The implemented named acquisition policy is FULLY_CROSSED_WITHIN_PARTITION. Every
**predeclared analysis-eligible acquisition stratum** must contain all nine identities
in each partition where represented. Batch checks apply separately; batch levels
are not automatically exact strata. This is a strong, explicit scientific policy.
Failure is reported against its version, with no automatic weaker fallback.

Rows carry `analysis_eligibility=ELIGIBLE/QC_EXCLUDED/UNRESOLVED`, `exclusion_reason`,
`exclusion_policy_version`, and QC evidence IDs. The initial supported QC rule is:

```json
{"id":"primary_readability","kind":"integrity_decode","on_failure":"QC_EXCLUDED"}
```

The rule must be approved before QC and optimization. Unsupported microscopy rules
fail closed: they need prespecified criteria and a tested implementation before use.
No microscopy thresholds or model-driven exclusions are invented by this milestone.
Missing/permission-limited inputs are unresolved, not evidence of bad microscopy.
Corrupt image rows remain in the inventory. Metadata extraction failure alone does
not make a decodable image QC-excluded; unresolved provenance/acquisition still blocks.

QC-excluded rows remain graph nodes and preserve all relationships. Only eligible
images contribute to the solver's incidence/count matrices or data loaders. Excluded
rows follow an assigned component for traceability; entirely excluded components use
`not_assigned_qc`. QC cannot be used to break a dependency bridge, silently remove a
class, or let the optimizer choose which images/strata survive.

## Graphs, optimization, and audit states

Confirmed culture/sample/preparation/slide/session/field/source/hash relationships
merge components. Reviewed coordinate frames permit additional confirmed geometric
overlap, including across acquisition dates. Candidate relations never merge them.
A separate possible-relationship graph identifies impactful candidates, including
collective legacy bridges. Redundant candidates inside confirmed components may
remain unresolved when they affect no other hard gate. Their reasoning is retained.

`dependency_component_id` records connectivity. `independence_status` separately
records VERIFIED_INDEPENDENT / LINKED / UNRESOLVED. Biological/acquisition evidence
and `assignment_unit_eligibility` remain explicit in the manifest and group table.
Disconnectedness is not independence. The threshold is at least three
verified-independent eligible assignment units per class, and every partition must
contain all nine identities without shared dependencies.

The solver is `scipy.optimize.milp` / HiGHS with binary whole-unit assignments and
explicit component × class incidence plus image-count and stratum/batch matrices.
A unit can contain multiple identities; assigning it moves their support together.
The lexicographic objectives are scientific constraints and coverage, then per-class
unit ratio deviation from 70/15/15, then per-class image-count ratio deviation.
Solver versions, parameters, tolerances, and stable-ID sequential tie-breaking are
recorded. A separate deterministic validator recomputes constraints before accepting
any candidate. Freeze also reimports immutable evidence to reject edited conclusions.

| Assessment | Meaning |
|---|---|
| BLOCKED | Required evidence is missing/contradictory or a relevant candidate remains impactful. |
| INFEASIBLE | Complete evidence proves failure under frozen rules, analytically or through MILP infeasibility. |
| FEASIBLE | The independent validator accepts a witness, including an interrupted-search incumbent. |
| UNKNOWN | No validated witness or infeasibility conclusion is available. |

FEASIBLE does not imply optimization completion or READY. Freeze requires the complete
recorded selection protocol. Three units can mean one per partition: feasibility PASS,
replication LIMITED. Every future metric must report held-out cultures and independent
units per class; no higher threshold or automatic strong-benchmark label is invented.

The canonical audit adds provenance context, acquisition/QC policies, a predeclared
eligible-population snapshot, and split-feasibility output to the original six artifacts.
Every publication retains a content-addressed revision and publishes status last;
interrupted publication fails integrity checks rather than appearing READY.

## Experiment and external consumer boundaries

```python
from utils.identity_audit_gate import assert_split_ready, assert_training_ready

partitions = assert_split_ready(audit_dir)  # Dataset-only readiness.
partitions = assert_training_ready(
    audit_dir, experiment_record=experiment_json,
    initialization_files=actual_weight_paths,  # All learned pipeline weights; [] for random.
)
```

Consumers must use returned eligible rows without resplitting. Paths retain the
`zip://archive!member` convention; use explicit `archive_path`/`archive_member` with
ZIP readers rather than passing the URI to an ordinary image loader.

An experiment record contains `frozen_split_hash`, `initialization_kind` (random or
pretrained), `initialization_checkpoint_hash` (null for random), an ordered
`initialization_checkpoints` list of hashed file references, and
`legacy_exposure_history={status: VERIFIED, evidence_files: [...], checkpoints: [...]}`.
Each history checkpoint binds `sha256`, scope NO_DATASET_B_EXPOSURE or
DOCUMENTED_EXPOSURE, `exposed_image_ids`, and `uses_legacy_corpus`. Known legacy hashes
cannot claim no exposure. Confirmed or possible held-out legacy exposure fails clean
experiment approval, not split readiness. Include detector and other learned weights.

The guard's `experiment` CLI operation writes an experiment-level assessment with
INITIALIZATION_COMPATIBILITY, CLEAN_EVALUATION_APPROVED, and MODEL_TRAINING_APPROVED.
The `training` operation recomputes approval; old audit-level approval flags cannot
authorize training. Historical trainers still reject identity manifests before
importing torch because they generate their own folds.

External rows remain physically/logically outside internal components and assignments.
EXTERNAL_ELIGIBILITY is CLEAN / CONTAMINATED / UNRESOLVED, separate from internal READY.
While reserved_external/sealed, `assert_external_access` permits only provenance,
file_integrity, and acquisition_metadata operations. It blocks both generation and
consumption of predictions, embeddings, morphology, feature distributions, and all
other image-derived development outputs. Unknown operations are denied.

External release orchestration remains a later model-freezing step. The external
guard requires RELEASED/clean eligibility, released row states, and a FROZEN record
binding `frozen_split_hash`, checkpoint, pipeline_config, sealed_audit_status, and
sealed_manifest. File bindings include hashes; pipeline configuration contains
preprocessing, nonempty inference_code_files, and additional_model_files (explicitly
empty if unused). The actual checkpoint/configuration must match those records.
Generation permits ready rows; consumption permits ready/evaluated rows. The saved
READY/SEALED snapshot fixes the cohort. Training is blocked after external release.

These guards enforce a documented workflow; they cannot authenticate the truth of
laboratory assertions or prevent deliberate bypass by code that ignores the guards.

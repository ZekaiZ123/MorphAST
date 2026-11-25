# Corrected outer-fold evaluation compatibility sources

This directory preserves the source layout used by the corrected outer-fold
evaluation. It is separate from MorphAST's three publication-workflow sections.
Those sections and `shared/` are not replaced or modified by this port.

The corrected workflow lives in [Training/Rebuttal](Training/Rebuttal/README.md).
It selects checkpoints using grouped inner validation and reports performance on
the held-out outer fold. Use those entrypoints for corrected evaluation; the
older trainers retained here are dependencies and historical reference code,
not alternative corrected-validation entrypoints.

## Why retain these paths?

Saved runs record `Code/...` relative paths and SHA-256 hashes of the exact
training, loading and evaluation source. Renaming the modules or substituting
the reorganized publication trainers would break source verification for those
runs. The port therefore retains the committed implementation and its required
loader, model, manifest, QC and provenance-gate dependencies unchanged.

Dependency sources were imported from original implementation commit
`19a468cbd6f55af7f4bd0867a8ccf71753a66964`. The outer-fold commits are replayed on
top of the current MorphAST history with their original commit IDs recorded in
the cherry-pick messages. The original branch's generated audit artifacts and
ancestry are not imported.

No images, manifests, acquisition workbooks, checkpoints, feature arrays,
predictions, plots or generated audit reports are included. Tests use synthetic
temporary fixtures. Reproducing real runs requires the separately supplied
dataset, original benchmark manifests/metadata, and any saved evaluation runs.
Do not edit a frozen run's source or hashes to bypass verification.

## Local configuration

Run commands from the repository root using the dependencies in
`requirements.txt`. Retain the documented `Data/` and `Model/` input layout;
stored absolute image paths must resolve on the execution host. The workflow
will fail rather than silently substitute a different dataset or split.

SLURM wrappers retain the original cluster defaults. Set `REBUTTAL_PYTHON` to
your Python executable and adapt partition/GPU directives before preparing a new
run. Historical QC launchers also contain site-specific paths and are included
for source accounting, not as portable turnkey jobs. See the Rebuttal README
for each command's required inputs and run/output directory arguments.

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The checkpoint-selection correction is not evidence of culture/session-level
independence and does not silently remove known duplicate-label conflicts.
The preparation/slide analysis is an explicitly separate sensitivity analysis.

"""Compatibility import and CLI for the provenance-v2 guards.

Old audit-level model approval cannot authorize a new experiment. This module is
stdlib-only so legacy entrypoints reject identity manifests before model imports.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.identity_provenance_gate import (  # noqa: F401
    ARTIFACTS, AuditGateError, assert_external_access, assert_external_evaluation_ready,
    assert_split_ready, assert_training_ready, checked_file, digest,
    initialization_compatibility, population_digest, preflight_legacy_cli,
    reject_v1_in_legacy_trainer, rows_from, main,
)

if __name__ == "__main__":
    main()

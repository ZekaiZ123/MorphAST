#!/usr/bin/env python3
"""Prepare/import provenance, run approved integrity QC, assess, or freeze splits."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.identity_provenance import ProvenanceError
from utils.identity_provenance_workflow import ROOT, prepare, import_response, run_qc, assessment, freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "import", "qc", "assess", "freeze"))
    parser.add_argument("--audit-dir", type=Path, default=ROOT/"experiments/strain_recognition_v1")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--response", type=Path)
    args = parser.parse_args()
    try:
        if args.operation == "import":
            if not args.response:
                parser.error("import requires --response")
            result = import_response(args.audit_dir, args.response, args.repo_root)
        else:
            result = {"prepare": prepare, "qc": run_qc, "assess": assessment, "freeze": freeze}[args.operation](
                args.audit_dir, args.repo_root)
        print(json.dumps(result, indent=2))
    except (ProvenanceError, OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit("BLOCKED: " + str(exc)) from exc


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
extract_real_mimic_to_ml_dataset.py — CLI entrypoint that runs the MIMIC-IV
adapter (backend/mimic_adapter/) against REAL, locally-extracted MIMIC-IV
raw table CSVs and produces data/real/ml_dataset.csv — the real-data
counterpart to generate_synthetic_data.py.

THIS SCRIPT DOES NOT DOWNLOAD OR FABRICATE MIMIC-IV DATA. It only reads raw
table CSVs you have already extracted (on your own machine or your own
BigQuery project, per the project's data-handling rules — see
docs/MIMIC_IV_SETUP.md) and placed at config.REAL_DATA_RAW_DIR.

It will REFUSE to run (raising a clear, actionable error, never silently
producing a broken or synthetic-substituted dataset) if:
  - the raw table files are missing (backend.mimic_adapter.loader), or
  - any required item/concept mapping in backend/mimic_adapter/concept_map.py
    is still unverified against a real MIMIC-IV dictionary.

Usage:
    export CLINTWIN_DATA_MODE=real
    python extract_real_mimic_to_ml_dataset.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from backend.mimic_adapter.loader import load_raw_mimic_tables, MimicFilesMissingError
from backend.mimic_adapter.adapter import adapt
from backend.mimic_adapter.validate_mimic import run_full_validation
from backend.mimic_adapter import concept_map


def main():
    print(f"Loading raw MIMIC-IV tables from {config.REAL_DATA_RAW_DIR} ...")
    try:
        raw = load_raw_mimic_tables()
    except MimicFilesMissingError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print("Checking item/concept mapping verification status...")
    unverified = concept_map.unverified_concepts(required_only=True)
    if unverified:
        print(
            f"REFUSING TO EXTRACT: {len(unverified)} required concept mapping(s) are "
            f"unverified against a real MIMIC-IV dictionary: {unverified}\n"
            f"See backend/mimic_adapter/concept_map.py and docs/MIMIC_IV_SETUP.md for how "
            f"to populate and verify them before running this extraction.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Running pre-extraction structural validation...")
    pre_report = run_full_validation(raw)
    print(pre_report)
    if not pre_report.ok:
        print("PRE-EXTRACTION VALIDATION FAILED — see errors above. Refusing to extract.", file=sys.stderr)
        sys.exit(1)

    print("Running MIMIC-IV adapter (cohort extraction + canonical schema construction)...")
    canonical = adapt(raw)
    cohort_stats = canonical.attrs.get("cohort_stats")
    if cohort_stats:
        print("Cohort extraction summary:")
        print(json.dumps(cohort_stats, indent=2, default=str))

    os.makedirs(config.REAL_DATA_DIR, exist_ok=True)
    out_path = config.REAL_DATA_PATH
    canonical.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(canonical)} rows)")

    print("Running post-extraction canonical-schema validation...")
    post_report = run_full_validation(raw, canonical_df=canonical)
    print(post_report)
    if not post_report.ok:
        print(
            "POST-EXTRACTION VALIDATION FAILED — the extracted file was written but does "
            "not pass validation. Do not train on it until these errors are resolved.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "\nDone. This is a REAL MIMIC-IV extraction — but item/concept mappings, cohort "
        "criteria, and the unit-conversion rules that produced it should be independently "
        "reviewed against the real MIMIC-IV dictionary before any clinical claim is made "
        "from a model trained on it. See docs/MIMIC_IV_SETUP.md section 13."
    )


if __name__ == "__main__":
    main()

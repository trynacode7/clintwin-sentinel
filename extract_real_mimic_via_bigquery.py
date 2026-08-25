#!/usr/bin/env python3
"""
extract_real_mimic_via_bigquery.py — CLI entrypoint that runs the MIMIC-IV
adapter (backend/mimic_adapter/) against a FILTERED SUBSET of REAL MIMIC-IV
pulled directly from BigQuery, instead of requiring the full raw-CSV
download extract_real_mimic_to_ml_dataset.py expects.

Identical downstream behavior to extract_real_mimic_to_ml_dataset.py —
same cohort extraction, same concept-mapping verification gate, same
adapter, same pre/post validation, same output schema — the ONLY
difference is where the raw tables come from (BigQuery query vs. local
CSV). This script does not implement any new pipeline logic; it is a thin
wrapper that swaps the loading step.

THIS SCRIPT DOES NOT DOWNLOAD, FABRICATE, OR CACHE MIMIC-IV DATA TO DISK
BY DEFAULT. It queries BigQuery in memory and writes only the derived,
already-de-identified canonical ml_dataset.csv — never raw patient rows —
to disk, exactly like the CSV-based script does.

It will REFUSE to run (clear, actionable error, never a silent synthetic
substitute) if:
  - BigQuery credentials/dataset names are wrong (backend.mimic_adapter.bigquery_loader), or
  - any required item/concept mapping in backend/mimic_adapter/concept_map.py
    is still unverified against a real MIMIC-IV dictionary.

Usage (run locally, with your own `gcloud auth application-default login`
already done, and google-cloud-bigquery + db-dtypes installed):

    export CLINTWIN_DATA_MODE=real
    python extract_real_mimic_via_bigquery.py \\
        --billing-project YOUR_GCP_PROJECT_ID \\
        --hosp-dataset mimiciv_3_1_hosp \\
        --icu-dataset mimiciv_3_1_icu \\
        --n-subjects 300

VERIFY --hosp-dataset / --icu-dataset against your own BigQuery console —
see backend/mimic_adapter/bigquery_loader.py's module docstring. Do not
guess; a wrong dataset name surfaces as a clear BigQuery error, not a
silent empty result.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from backend.mimic_adapter.bigquery_loader import load_raw_mimic_tables_from_bigquery, DEFAULT_PHYSIONET_PROJECT
from backend.mimic_adapter.adapter import adapt
from backend.mimic_adapter.validate_mimic import run_full_validation
from backend.mimic_adapter import concept_map


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--billing-project", required=True)
    p.add_argument("--hosp-dataset", required=True)
    p.add_argument("--icu-dataset", required=True)
    p.add_argument("--physionet-project", default=DEFAULT_PHYSIONET_PROJECT)
    p.add_argument("--n-subjects", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Querying a {args.n_subjects}-subject subset from BigQuery "
          f"({args.physionet_project}.{args.hosp_dataset} / {args.icu_dataset}) ...")
    try:
        raw = load_raw_mimic_tables_from_bigquery(
            billing_project=args.billing_project,
            hosp_dataset=args.hosp_dataset,
            icu_dataset=args.icu_dataset,
            physionet_project=args.physionet_project,
            n_subjects=args.n_subjects,
            seed=args.seed,
        )
    except Exception as e:
        print(f"BIGQUERY LOAD FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nChecking item/concept mapping verification status...")
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

    if len(canonical) == 0:
        print(
            "WARNING: 0 rows survived cohort filtering out of this subject subset. "
            "This is expected sometimes with a small --n-subjects sample (e.g. none of "
            "the sampled subjects had a qualifying first ICU stay >= 8h). Re-run with a "
            "larger --n-subjects rather than treating this as a pipeline bug.",
            file=sys.stderr,
        )

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
        f"\nDone. This is a REAL MIMIC-IV extraction (a {args.n_subjects}-subject subset, "
        f"seed={args.seed}, deterministic and reproducible) — but item/concept mappings, "
        f"cohort criteria, and unit-conversion rules should still be independently reviewed "
        f"against the real MIMIC-IV dictionary before any clinical claim is made from a "
        f"model trained on it. See docs/MIMIC_IV_SETUP.md section 13.\n"
        f"To scale up later, re-run with a larger --n-subjects (or switch to the full local "
        f"CSV path via extract_real_mimic_to_ml_dataset.py) — nothing downstream changes."
    )


if __name__ == "__main__":
    main()

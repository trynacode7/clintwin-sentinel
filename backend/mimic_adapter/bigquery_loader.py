"""
bigquery_loader.py — Alternative MIMIC-IV loader: pulls a FILTERED SUBSET
directly from Google BigQuery instead of requiring the full ~9GB raw-CSV
download that loader.py expects on local disk.

WHY THIS EXISTS: downloading and locally storing full chartevents/labevents
CSVs is slow. BigQuery lets the filtering (which subject_ids we want)
happen server-side, so only matching rows are ever transferred — this is
the fast path to a real, small, end-to-end verification run.

============================================================================
THIS MODULE MUST BE RUN LOCALLY, ON YOUR OWN MACHINE, WITH YOUR OWN GOOGLE
CLOUD CREDENTIALS AGAINST YOUR OWN CREDENTIALED ACCESS TO physionet-data.
It has never been run inside this AI assistant session and never will be —
no patient-level row from these queries should ever be pasted into a chat.
Only aggregate row counts, schemas, and error messages (which this module
prints) are safe to share.
============================================================================

OUTPUT SHAPE: returns the exact same Dict[str, pd.DataFrame] shape that
loader.load_raw_mimic_tables() returns (same table names, same required
columns per backend/mimic_adapter/validate_mimic.py's REQUIRED_COLUMNS) —
a drop-in alternative raw-table source. Everything downstream (cohort.py,
concept_map.py, units.py, adapter.py, validate_mimic.py) is UNMODIFIED and
does not know or care which loader produced the raw tables.

SAMPLING: deterministic, seeded via BigQuery's FARM_FINGERPRINT hash on
subject_id — NOT `ORDER BY RAND()`. Re-running with the same n_subjects and
seed reproduces the identical subject cohort.

REQUIRES VERIFICATION AGAINST YOUR OWN BIGQUERY CONSOLE — DO NOT GUESS:
  The exact MIMIC-IV v3.1 dataset names under the `physionet-data` project
  (PhysioNet has used both versioned names like `mimiciv_3_1_hosp` and
  unversioned names like `mimiciv_hosp` across different dataset releases).
  Open the physionet-data project in your BigQuery console, expand it, and
  read the actual dataset names for `hosp` and `icu` — pass them via
  --hosp-dataset / --icu-dataset. This module refuses to guess and will
  surface BigQuery's own "not found" error clearly if you get a name wrong,
  rather than silently falling back to anything.

ITEM IDs ARE STILL NOT GUESSED HERE. This module only loads raw tables. It
still calls the same concept_map.require_verified() gate (via adapter.adapt)
downstream — see extract_real_mimic_via_bigquery.py.
"""

from __future__ import annotations

import argparse
from typing import Dict

import pandas as pd

from . import loader as local_loader  # reuse REQUIRED_TABLES / OPTIONAL_TABLES / _DATETIME_COLS
from . import validate_mimic  # reuse REQUIRED_COLUMNS as the one source of truth for column lists

DEFAULT_PHYSIONET_PROJECT = "physionet-data"


def _bq_client(billing_project: str):
    from google.cloud import bigquery  # deferred import — optional dependency, not needed for CSV path
    return bigquery.Client(project=billing_project)


def _sampled_subject_ids(client, physionet_project: str, hosp_dataset: str, n_subjects: int, seed: int) -> list:
    """
    Deterministic subject_id sample via hash-modulo (reproducible across
    runs with the same seed) — deliberately NOT `ORDER BY RAND()`.
    """
    query = f"""
        SELECT subject_id
        FROM `{physionet_project}.{hosp_dataset}.patients`
        ORDER BY MOD(ABS(FARM_FINGERPRINT(CONCAT(CAST(subject_id AS STRING), '-', CAST(@seed AS STRING)))), 1000000)
        LIMIT @n_subjects
    """
    from google.cloud import bigquery
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("seed", "INT64", seed),
        bigquery.ScalarQueryParameter("n_subjects", "INT64", n_subjects),
    ])
    df = client.query(query, job_config=job_config).to_dataframe()
    return df["subject_id"].tolist()


def load_raw_mimic_tables_from_bigquery(
    billing_project: str,
    hosp_dataset: str,
    icu_dataset: str,
    physionet_project: str = DEFAULT_PHYSIONET_PROJECT,
    n_subjects: int = 300,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Pull a deterministic, seeded subset of `n_subjects` patients' worth of
    every table loader.py expects, filtered server-side to those
    subject_ids, from the physionet-data MIMIC-IV BigQuery dataset.

    billing_project: YOUR OWN GCP project ID — BigQuery bills queries to
      whichever project runs them; physionet-data itself is read-only and
      is not a billing project.
    hosp_dataset / icu_dataset: exact MIMIC-IV v3.1 dataset names — VERIFY
      in your own BigQuery console (see module docstring). Do not guess.
    """
    client = _bq_client(billing_project)

    print(f"Sampling {n_subjects} subjects (seed={seed}) from "
          f"{physionet_project}.{hosp_dataset}.patients ...")
    subject_ids = _sampled_subject_ids(client, physionet_project, hosp_dataset, n_subjects, seed)
    if not subject_ids:
        raise RuntimeError(
            "BigQuery returned zero subjects. Check --hosp-dataset / --physionet-project "
            "and that your credentials actually have access to physionet-data — this "
            "loader does not substitute anything when the query comes back empty."
        )

    table_to_dataset = {
        "patients": hosp_dataset, "admissions": hosp_dataset,
        "labevents": hosp_dataset, "prescriptions": hosp_dataset,
        "microbiologyevents": hosp_dataset, "d_labitems": hosp_dataset,
        "icustays": icu_dataset, "chartevents": icu_dataset,
        "inputevents": icu_dataset, "outputevents": icu_dataset,
        "procedureevents": icu_dataset, "d_items": icu_dataset,
    }

    from google.cloud import bigquery

    raw: Dict[str, pd.DataFrame] = {}
    all_tables = local_loader.REQUIRED_TABLES + local_loader.OPTIONAL_TABLES
    for table in all_tables:
        dataset = table_to_dataset[table]
        cols = validate_mimic.REQUIRED_COLUMNS.get(table)
        col_list = ", ".join(cols) if cols else "*"

        if table in ("d_items", "d_labitems"):
            # Small dictionaries — pull in full, never subject-filtered.
            query = f"SELECT {col_list} FROM `{physionet_project}.{dataset}.{table}`"
            job_config = None
        else:
            query = f"""
                SELECT {col_list}
                FROM `{physionet_project}.{dataset}.{table}`
                WHERE subject_id IN UNNEST(@subject_ids)
            """
            job_config = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ArrayQueryParameter("subject_ids", "INT64", subject_ids)
            ])

        try:
            df = client.query(query, job_config=job_config).to_dataframe()
        except Exception as e:
            if table in local_loader.OPTIONAL_TABLES:
                print(f"  [optional] '{table}' unavailable ({e}) — skipping, degrades gracefully.")
                continue
            raise RuntimeError(
                f"BigQuery query failed for REQUIRED table '{table}' in "
                f"'{physionet_project}.{dataset}': {e}\n"
                f"Verify the dataset/table/column names against your own BigQuery console. "
                f"This loader does not guess or substitute on failure."
            ) from e

        for col in local_loader._DATETIME_COLS.get(table, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        raw[table] = df
        print(f"  {table}: {len(df)} rows, {len(df.columns)} columns")

    return raw


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Pull a filtered, deterministic MIMIC-IV subset from BigQuery — "
                     "for a fast, real, small end-to-end pipeline run without a full CSV download."
    )
    p.add_argument("--billing-project", required=True, help="Your own GCP project ID (queries are billed here)")
    p.add_argument("--hosp-dataset", required=True, help="e.g. mimiciv_3_1_hosp — VERIFY in your BigQuery console, do not guess")
    p.add_argument("--icu-dataset", required=True, help="e.g. mimiciv_3_1_icu — VERIFY in your BigQuery console, do not guess")
    p.add_argument("--physionet-project", default=DEFAULT_PHYSIONET_PROJECT)
    p.add_argument("--n-subjects", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw = load_raw_mimic_tables_from_bigquery(
        billing_project=args.billing_project,
        hosp_dataset=args.hosp_dataset,
        icu_dataset=args.icu_dataset,
        physionet_project=args.physionet_project,
        n_subjects=args.n_subjects,
        seed=args.seed,
    )
    print("\nDone. Aggregate summary only (no patient-level data printed):")
    for t, df in raw.items():
        print(f"  {t}: {len(df)} rows x {len(df.columns)} columns")

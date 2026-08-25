"""
loader.py — Loads RAW, per-table MIMIC-IV CSVs from disk (Phase 2, section 10).

FAILS CLEARLY, NEVER FALLS BACK. If `CLINTWIN_DATA_MODE=real` and the
expected raw-table files are not present at `config.REAL_DATA_RAW_DIR`, this
raises `MimicFilesMissingError` naming exactly which files are missing, the
expected directory, and how to configure the path — it never substitutes
synthetic data or an empty table.

Expected directory layout (see docs/MIMIC_IV_SETUP.md):

    <CLINTWIN_REAL_DATA_RAW_DIR>/
        patients.csv(.gz)
        admissions.csv(.gz)
        icustays.csv(.gz)
        chartevents.csv(.gz)
        labevents.csv(.gz)
        inputevents.csv(.gz)
        outputevents.csv(.gz)
        prescriptions.csv(.gz)
        microbiologyevents.csv(.gz)
        procedureevents.csv(.gz)   [optional]
        d_items.csv(.gz)           [optional]
        d_labitems.csv(.gz)        [optional]

This module never downloads, fabricates, or mock-fills any of these files —
it only reads what's already on disk.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

import config

REQUIRED_TABLES = [
    "patients", "admissions", "icustays", "chartevents", "labevents",
    "inputevents", "outputevents", "prescriptions", "microbiologyevents",
]
OPTIONAL_TABLES = ["procedureevents", "d_items", "d_labitems"]

_DATETIME_COLS = {
    "admissions": ["admittime", "dischtime"],
    "icustays": ["intime", "outtime"],
    "chartevents": ["charttime"],
    "labevents": ["charttime"],
    "inputevents": ["starttime", "endtime"],
    "outputevents": ["charttime"],
    "prescriptions": ["starttime", "stoptime"],
    "microbiologyevents": ["charttime"],
    "procedureevents": ["starttime", "endtime"],
}


class MimicFilesMissingError(FileNotFoundError):
    pass


def _find_table_file(raw_dir: str, table: str) -> Optional[str]:
    for ext in (".csv", ".csv.gz"):
        path = os.path.join(raw_dir, f"{table}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_raw_mimic_tables(raw_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load every required (and any present optional) MIMIC-IV raw table CSV
    from `raw_dir` (default config.REAL_DATA_RAW_DIR). Raises
    MimicFilesMissingError naming every missing REQUIRED file, the expected
    directory, and the env var to reconfigure it — never returns a partial
    result silently missing a required table.
    """
    raw_dir = raw_dir or config.REAL_DATA_RAW_DIR

    if not os.path.isdir(raw_dir):
        raise MimicFilesMissingError(
            f"CLINTWIN_DATA_MODE='real' extraction requires raw MIMIC-IV table CSVs, "
            f"but the expected directory does not exist: {raw_dir}\n"
            f"Configure it with the CLINTWIN_REAL_DATA_RAW_DIR environment variable, "
            f"or place your extracted MIMIC-IV tables at that path. See "
            f"docs/MIMIC_IV_SETUP.md for the exact expected files and directory layout. "
            f"Refusing to silently fall back to synthetic data."
        )

    missing: List[str] = []
    found: Dict[str, str] = {}
    for table in REQUIRED_TABLES:
        path = _find_table_file(raw_dir, table)
        if path is None:
            missing.append(f"{table}.csv (or {table}.csv.gz)")
        else:
            found[table] = path

    if missing:
        raise MimicFilesMissingError(
            f"CLINTWIN_DATA_MODE='real' extraction is missing {len(missing)} required "
            f"MIMIC-IV table file(s) in {raw_dir}:\n  - " + "\n  - ".join(missing) +
            f"\n\nExpected directory: {raw_dir} (configure via CLINTWIN_REAL_DATA_RAW_DIR). "
            f"See docs/MIMIC_IV_SETUP.md for the full required-file list and how to obtain "
            f"them from your own PhysioNet-credentialed MIMIC-IV extraction. Refusing to "
            f"silently proceed with a partial or synthetic substitute."
        )

    for table in OPTIONAL_TABLES:
        path = _find_table_file(raw_dir, table)
        if path is not None:
            found[table] = path

    raw: Dict[str, pd.DataFrame] = {}
    for table, path in found.items():
        parse_dates = _DATETIME_COLS.get(table, [])
        raw[table] = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)

    return raw

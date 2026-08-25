# data/real/ — Real MIMIC-IV Data

This directory is where a real, PhysioNet-credentialed MIMIC-IV extraction is placed. **Nothing
in this directory has been created, read, or processed by this session or any prior phase** —
only small, clearly-labeled MIMIC-STRUCTURE TEST FIXTURES have ever been used to test the
adapter (`tests/test_mimic_adapter.py`). See `docs/MIMIC_IV_SETUP.md` for the full setup guide.

## Layout (Phase 2)

```
data/real/
  raw/            <- place your extracted MIMIC-IV table CSVs here (config.REAL_DATA_RAW_DIR)
  ml_dataset.csv  <- produced BY `extract_real_mimic_to_ml_dataset.py`, not something you write yourself
```

## Why this directory exists

`config.py` defines two data modes:

- `CLINTWIN_DATA_MODE=synthetic` (default) → reads from `data/synthetic/processed/ml_dataset.csv`
- `CLINTWIN_DATA_MODE=real` → reads from `data/real/ml_dataset.csv` (path overridable via
  `CLINTWIN_REAL_DATA_PATH`)

`preprocessing.load_data()` **never silently falls back** from one mode to the other. If
`CLINTWIN_DATA_MODE=real` and no file exists at the expected path, the pipeline raises a
`FileNotFoundError` naming exactly what's missing, the expected location, and how to produce it
— instead of quietly training on synthetic data while believing it's using real data.

## What must never happen

Per the project's data-handling rules: real MIMIC-IV patient-level data must **never** be
pasted, uploaded, or shared into any chat/AI-assistant session — this assistant is not on
PhysioNet's approved AI list, and doing so would violate the PhysioNet Data Use Agreement. All
real-data extraction, storage, and training must happen entirely on your own machine or your own
BigQuery project, outside of any AI assistant environment.

## How to populate this directory (Phase 2)

1. Extract your own PhysioNet-credentialed MIMIC-IV raw tables to `data/real/raw/` — see
   `docs/MIMIC_IV_SETUP.md` sections 2-3 for the exact required files and layout.
2. Populate and verify the item/concept mappings in `backend/mimic_adapter/concept_map.py`
   against your own real MIMIC-IV dictionary — see `docs/MIMIC_IV_SETUP.md` section 6. This
   repository ships those mappings intentionally EMPTY and unverified; nobody who built this
   integration had access to a real MIMIC-IV dictionary.
3. Run `python extract_real_mimic_to_ml_dataset.py` (with `CLINTWIN_DATA_MODE=real` set) — this
   produces `data/real/ml_dataset.csv` matching the exact schema in `DATA_CONTRACT.md` section 3,
   the same columns/types/units the synthetic pipeline already produces. No other file in the
   pipeline (`preprocessing.py` onward) needs to change.

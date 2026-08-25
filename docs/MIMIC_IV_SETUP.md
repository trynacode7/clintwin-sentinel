# MIMIC-IV Setup (Phase 2)

**The synthetic dataset (`docs/SYNTHETIC_DATA.md`) is used only for engineering/development
validation. Synthetic model metrics are not clinical evidence.**

This document explains how to bring real, PhysioNet-credentialed MIMIC-IV data into ClinTwin
Sentinel once you have access to it. **No real MIMIC-IV data has been touched, downloaded, or
fabricated anywhere in this repository or this development session** — everything described
here was built and tested against small, clearly-labeled MIMIC-STRUCTURE TEST FIXTURES only
(`tests/test_mimic_adapter.py`), never real data.

---

## 1. What MIMIC-IV access is required

You need your own **PhysioNet-credentialed** access to MIMIC-IV (`hosp` and `icu` modules), via
either:

- A local extraction (downloaded, credentialed CSV/Parquet files on your own machine), or
- Your own BigQuery project with the `physionet-data` MIMIC-IV dataset shared to it.

Real MIMIC-IV patient-level data must **never** be pasted, uploaded, or shared into any chat/AI
assistant session (including the one that built this integration) — this is a PhysioNet Data Use
Agreement requirement, not just a project preference. All real-data extraction, storage, and
model training happens entirely on your own machine or your own BigQuery project.

## 2. What files/tables are required

`backend/mimic_adapter/loader.py` expects the following **raw, per-table** MIMIC-IV extracts:

| Table | Required? | Used for |
|---|---|---|
| `patients` | Required | demographics (age computation, sex) |
| `admissions` | Required | cohort definition, admission-year for age computation |
| `icustays` | Required | the modeling unit — stay_id, intime/outtime |
| `chartevents` | Required | vitals, GCS components, FiO2 |
| `labevents` | Required | lactate, WBC, creatinine, bilirubin, platelets |
| `inputevents` | Required | vasopressor + fluid bolus |
| `outputevents` | Required | urine output |
| `prescriptions` | Required | antibiotic timing (Sepsis-3 suspected-infection criterion) |
| `microbiologyevents` | Required | culture order timing (the other half of that criterion) |
| `procedureevents` | Optional | mechanical ventilation episodes (see section 6's `ventilated` caveat) |
| `d_items` | Optional | item dictionary — useful for populating concept_map.py, not consumed by the adapter itself |
| `d_labitems` | Optional | same, for lab items |

Every column each table must carry is listed in `backend/mimic_adapter/validate_mimic.py`'s
`REQUIRED_COLUMNS` — these are the tables' public, documented MIMIC-IV structure (column names
only), not itemid values, so they're already correct and don't need verification.

## 3. Expected directory structure

```
data/real/
  raw/                          <- config.REAL_DATA_RAW_DIR (override: CLINTWIN_REAL_DATA_RAW_DIR)
    patients.csv(.gz)
    admissions.csv(.gz)
    icustays.csv(.gz)
    chartevents.csv(.gz)
    labevents.csv(.gz)
    inputevents.csv(.gz)
    outputevents.csv(.gz)
    prescriptions.csv(.gz)
    microbiologyevents.csv(.gz)
    procedureevents.csv(.gz)    [optional]
    d_items.csv(.gz)            [optional]
    d_labitems.csv(.gz)         [optional]
  ml_dataset.csv                <- config.REAL_DATA_PATH (override: CLINTWIN_REAL_DATA_PATH) —
                                    produced BY extract_real_mimic_to_ml_dataset.py, not
                                    something you place there yourself
```

`.csv` or `.csv.gz` both work (`loader.py` checks both extensions).

## 4. How DATA_MODE=real is configured

```bash
export CLINTWIN_DATA_MODE=real
# optional overrides:
export CLINTWIN_REAL_DATA_RAW_DIR=/path/to/your/raw/mimic/csvs
export CLINTWIN_REAL_DATA_PATH=/path/to/your/extracted/ml_dataset.csv
```

`config.py` reads `CLINTWIN_DATA_MODE` at import time and raises immediately if it isn't
`"synthetic"` or `"real"` — there is no third, ambiguous state. **There is no automatic
fallback in either direction**: `backend.preprocessing.load_data()` will raise a
`FileNotFoundError` naming exactly what's missing, the expected path, and how to configure it,
rather than silently substituting synthetic data for a requested real-data run (or vice versa).
See section 10 in the Phase 2 final report for a live-run transcript of this failure behavior.

## 5. How the MIMIC adapter works

```
MIMIC-IV raw tables (data/real/raw/*.csv, loaded by loader.py)
        |
        v
cohort.select_cohort()        <- separate stage: adults >=18, first ICU stay per
        |                          admission, min LOS 8h — no observation values touched
        v
adapter.adapt()                <- pivots chartevents/labevents/inputevents/outputevents,
        |                          harmonizes units, interval-joins labs to the correct
        |                          stay (Fix 7 pattern reused), computes suspected_infection
        v
canonical ml_dataset.csv-shaped DataFrame  (IDENTICAL schema to the synthetic
        |                                    pipeline's extract_to_ml_dataset.py output —
        |                                    see DATA_CONTRACT.md section 3)
        v
existing preprocessing.py / feature_engineering.py / labels_sofa.py / dataset.py /
train.py / evaluate.py   <- UNCHANGED, does not know or care about the source
```

Only `backend/mimic_adapter/` (plus `extract_real_mimic_to_ml_dataset.py`, the CLI entrypoint)
contains any MIMIC-IV table name or item ID. Run it with:

```bash
export CLINTWIN_DATA_MODE=real
python extract_real_mimic_to_ml_dataset.py
```

This loads the raw tables, checks that every required concept mapping is verified (refuses to
proceed otherwise — see section 6), runs cohort extraction, runs the adapter, writes
`data/real/ml_dataset.csv`, and runs both pre- and post-extraction validation
(`backend/mimic_adapter/validate_mimic.py`). It will not silently produce a broken dataset.

## 6. How item/concept mappings work

`backend/mimic_adapter/concept_map.py` maps each canonical ClinTwin variable to a MIMIC-IV
source table/column and a list of item IDs. **Every item ID list ships empty, and every mapping
ships `verified=False`.** This is deliberate: nobody in this development session has access to a
real `mimiciv_icu.d_items`/`mimiciv_hosp.d_labitems` dictionary, and a guessed item ID presented
as authoritative would silently corrupt every downstream number with no error.

`backend/mimic_adapter/adapter.py`'s `adapt()` calls `concept_map.require_verified()` and
**refuses to run** while any concept the v1 canonical schema needs is unverified. To populate it:

1. On your own machine/BigQuery project (never in a chat session — though item dictionaries
   themselves are not patient-level data and aren't covered by the DUA's data-sharing
   restriction; only patient-level rows are), query `mimiciv_icu.d_items` /
   `mimiciv_hosp.d_labitems` for the concept labels referenced in each `ConceptMapping.notes`
   field in `concept_map.py` (e.g. `label ILIKE '%heart rate%'`).
2. Record **every** itemid that plausibly represents that concept — MIMIC-IV is documented to
   carry multiple itemids per concept (different monitor sources, CareVue vs. MetaVision,
   invasive vs. non-invasive BP — see DATA_CONTRACT.md section 4b/4c).
3. Fill in `itemids=[...]` and set `verified=True` for that `ConceptMapping`.
4. Run `python -m backend.mimic_adapter.validate_mimic --check-mapping` — it lists every
   remaining unverified concept and exits non-zero until all are resolved.

Section 13 below lists every concept still requiring this verification as of this Phase 2 pass.

## 7. Unit harmonization

`backend/mimic_adapter/units.py` implements fail-loud unit handling:

- `convert_temperature(value, unit)` — row-wise Celsius/Fahrenheit harmonization (real MIMIC-IV
  charts both under different itemids). Raises `UnitValidationError` on any unrecognized unit
  string rather than guessing.
- `validate_and_normalize_fio2(value)` — detects percent-scale (21-100) vs. fraction-scale
  (0.21-1.0) FiO2 entries and normalizes; raises if a value is implausible under either
  interpretation.
- `validate_unit_column(concept, unit_series, expected_unit)` — generic fail-loud check for
  concepts with one fixed expected unit; a mismatch usually means the itemid mapping is pulling
  in more than one clinical concept and needs fixing in `concept_map.py`, not silent conversion.

The canonical unit for every clinical variable is documented in DATA_CONTRACT.md section 3 and
in `units.CANONICAL_UNITS`.

## 8. Cohort extraction

`backend/mimic_adapter/cohort.py` is a **structurally separate stage** — it selects which ICU
stays are eligible before any observation values are touched, and is never mixed with model
preprocessing (`backend/preprocessing.py`, which is source-agnostic and runs identically
regardless of cohort composition).

**Inclusion criteria**: age >= 18 at ICU admission; ICU length of stay >= 8h (configurable via
`min_los_hours`) so a full 6h input window can exist before the first prediction point.

**Exclusion criteria**: any ICU stay that is not the first stay within its hospital admission
(avoids the "recovering from a prior stay" confound); stays with unresolvable age or missing
intime/outtime.

**Not implemented, documented as a real-data dependency**: per-`careunit` missingness auditing
(some ICU units chart more sparsely, a known
MIMIC-IV wrinkle) requires real chartevents missingness statistics this session cannot compute.

`select_cohort()` returns both the eligible `stay_id` list and a `CohortStats` object (counts at
every exclusion step) — `extract_real_mimic_to_ml_dataset.py` prints this automatically.

## 9. Canonical schema

The adapter's output is byte-for-byte the same schema as the synthetic pipeline's
`backend/synthetic_data/extract_to_ml_dataset.py` output — both are built from
`config.ID_COLS + [config.TIME_COL] + config.ALL_FEATURE_COLS + config.LABEL_ONLY_COLS`, not two
independently-maintained column lists, so they cannot silently drift apart. See DATA_CONTRACT.md
section 3 for the full column-by-column contract and section 4c for the MIMIC-IV-specific
per-variable mapping.

## 10. Validation

`backend/mimic_adapter/validate_mimic.py` runs before AND after extraction
(`extract_real_mimic_to_ml_dataset.py` calls both):

- **Pre-extraction**: required raw tables present, required columns present per table, concept
  mappings verified, ICU stay boundary sanity (no inverted intime/outtime, no duplicate
  stay_ids), cohort counts.
- **Post-extraction**: the SAME schema-generic `validate_ml_dataset()` the synthetic pipeline
  uses (missingness, impossible physiological values, duplicate `(stay_id, charttime)` rows, no
  fully-empty required column) run against the adapter's canonical output, plus (once a split
  exists) `validate_patient_split_disjoint()`.

Every check either passes or is reported as an error/warning — nothing is silently repaired.
A failing validation exits the CLI non-zero.

## 11. How to generate the model dataset

```bash
export CLINTWIN_DATA_MODE=real
export CLINTWIN_REAL_DATA_RAW_DIR=/path/to/your/raw/mimic/csvs   # if not the default
python extract_real_mimic_to_ml_dataset.py
```

Produces `data/real/ml_dataset.csv` (or wherever `CLINTWIN_REAL_DATA_PATH` points).

## 12. How to train/evaluate after real data is available

No new commands — this is the entire point of the canonical-schema architecture:

```bash
python -m backend.baselines
python -m backend.train --model lstm --epochs 100
python -m backend.train --model transformer --epochs 100
```

`backend.preprocessing.preprocess()` reads `config.DATA_PATH`, which is `data/real/ml_dataset.csv`
whenever `CLINTWIN_DATA_MODE=real` — nothing downstream of that path needs to change or knows
the difference.

## 13. Mappings that still require verification against the actual MIMIC-IV dictionary

As of this Phase 2 pass, **every** item-ID-based concept mapping is unverified (by design — see
section 6). Run `python -m backend.mimic_adapter.validate_mimic --check-mapping` for the current
authoritative list; as of writing it is:

`heart_rate, map, resp_rate, spo2, temp, gcs_eye, gcs_verbal, gcs_motor, fio2, ventilated,
lactate, wbc, creatinine, bilirubin, platelets, vasopressor_dose, fluid_bolus_ml,
urine_output_rate, antibiotic_administration, culture_order`

Two additional design decisions (not itemid lookups) also require real-data verification before
implementation:

- **`ventilated`**: concept_map.py flags the source table itself as ambiguous
  (`procedureevents` vs. `chartevents` ventilator-setting itemids, per the published MIT-LCP
  `mimic-code` repository's approach) — this needs a decision informed by which itemids the real
  dictionary actually populates, not just an itemid lookup.
- **`antibiotic_administration`**: `prescriptions.drug` is free text in real MIMIC-IV. This
  adapter does not invent an antibiotic drug-name list — `adapter.compute_suspected_infection_windows`
  accepts an explicit `antibiotic_drug_names` set and returns nothing qualifying if none is
  supplied. A real extraction needs a citable antibiotic name/class list (e.g. the kind used in
  the original Sepsis-3/Seymour et al. cohort-definition code), sourced deliberately rather than
  guessed.
- **`age`**: the anchor_age/anchor_year computation in `cohort.compute_age_at_admission` is
  structurally correct but has one documented MIMIC-IV edge case (patients coded as
  `anchor_age=91` represent "was >89 at some point," a ceiling, not an exact value) that should
  be sanity-checked against real data before being treated as exact.
- **`vasopressor_dose`**: per-drug norepinephrine-equivalent conversion (real `inputevents` has
  one itemid set per vasopressor drug) is documented as a required future-work item in
  DATA_CONTRACT.md section 7 and is **not implemented** — the adapter's `vasopressor_series()`
  will need this conversion table wired in once itemids are verified, before dosing values from
  multiple different drugs can be safely combined into one canonical column.

## 14. Which parts of the current pipeline remain synthetic

Everything under `data/synthetic/` and `backend/synthetic_data/` is unaffected by this phase and
continues to work exactly as documented in `docs/SYNTHETIC_DATA.md` — `CLINTWIN_DATA_MODE`
defaults to `synthetic`, and nothing here changes that default or any synthetic-generator
behavior. Specifically still synthetic / not yet real:

- The entire generated cohort used for all reported metrics so far (Phase 1 remediation report).
- Every item ID actually used to run the pipeline end-to-end (the synthetic generator's
  namespaced 90000+/91000+/92000+/93000+ IDs) — the MIMIC adapter built in this phase has been
  exercised only against small MIMIC-STRUCTURE TEST FIXTURES (`tests/test_mimic_adapter.py`),
  never real data, and cannot be until section 13's mappings are verified.
- The antibiotic name list, ventilation-source decision, and per-drug vasopressor conversion
  table (section 13).
- Per-careunit missingness auditing and lab-LOCF staleness caps (both explicitly deferred).

**No AUROC/precision/recall/F1 number produced by this project to date reflects real MIMIC-IV
data.** All reported metrics are synthetic-pipeline validation only.

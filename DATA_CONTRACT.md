# ClinTwin Sentinel — Data Contract (v1, synthetic-development phase)

This contract is derived from direct inspection of `preprocessing.py`, `feature_engineering.py`,
`labels.py`/`labels_sofa.py`, `dataset.py`, `train.py`, `evaluate.py`, `lstm_model.py`, and
`transformer_model.py`. It defines exactly what the pipeline
requires — nothing is added that the pipeline doesn't consume. It is the single source of
truth both the synthetic generator and, later, the real MIMIC-IV extraction must satisfy.

Any future MIMIC-IV extraction script only has to produce a CSV matching **Section 3**
below. Everything downstream of that CSV (`preprocessing.py` onward) is unchanged.

---

## 1. Identifiers (`config.ID_COLS`)

| Column | Type | Meaning | MIMIC-IV analogue |
|---|---|---|---|
| `subject_id` | int | Patient identifier | `patients.subject_id` |
| `hadm_id` | int | Hospital admission identifier | `admissions.hadm_id` |
| `stay_id` | int | ICU stay identifier — **this is the modeling unit**; windows never cross a `stay_id` boundary | `icustays.stay_id` |

`preprocessing.py`, `feature_engineering.py`, `dataset.py`, and `labels_sofa.py` all group by
`stay_id` for every per-stay operation (resampling, deltas, rolling stats, future-window
labeling, sliding windows). No cross-stay leakage is possible as long as every row carries a
correct `stay_id`.

## 2. Time (`config.TIME_COL`)

`charttime` — a real (or synthetic) timestamp, one row per `(stay_id, charttime)` after
resampling. Before resampling, raw event data may have multiple rows per timestamp per stay
(different `itemid`s) — see Section 4.

**Sampling grid**: `config.RESAMPLE_FREQ = "1h"` (hourly). This is a deliberate change from
the prototype's 5-minute grid — MIMIC-IV vitals are charted roughly hourly outside of code
events, so a 5-minute grid is mostly forward-filled rather than observed and drowns real
signal under missingness-indicator noise.

## 3. The flat modeling table (`ml_dataset.csv`) — what `preprocessing.load_data()` reads

This is the **stable interface**. The synthetic ETL (`backend/synthetic_data/extract_to_ml_dataset.py`)
produces it today; a real-MIMIC-IV extraction script produces the identical shape later. One
row per `(stay_id, charttime)` observation (irregular — resampling happens downstream in
`preprocessing.py`, not at extraction time, exactly as the current pipeline already assumes).

| Column | Type | Units | Plausible range | Fill strategy (`config.COLUMN_FILL_STRATEGY`) | Required for |
|---|---|---|---|---|---|
| `subject_id`, `hadm_id`, `stay_id` | int | — | — | — | identity |
| `charttime` | datetime | — | — | — | identity |
| `heart_rate` | float | bpm | 20–220 | interpolate (vital) | model input, shock_index, cardio-SOFA context |
| `map` | float | mmHg | 20–160 | interpolate (vital) | model input, shock_index, **cardiovascular SOFA** |
| `resp_rate` | float | breaths/min | 4–60 | interpolate (vital) | model input |
| `spo2` | float | % | 50–100 | interpolate (vital) | model input, respiratory SOFA (via SF ratio) |
| `temp` | float | °C | 30–43 | interpolate (vital) | model input |
| `lactate` | float | mmol/L | 0.2–20 | locf (lab, capped) | model input, Sepsis-3 hyperlactatemia context |
| `wbc` | float | K/µL | 0.1–50 | locf (lab, capped) | model input |
| `creatinine` | float | mg/dL | 0.1–15 | locf (lab, capped) | **renal SOFA** |
| `bilirubin` | float | mg/dL | 0.1–30 | locf (lab, capped) | **hepatic SOFA** |
| `platelets` | float | K/µL | 5–600 | locf (lab, capped) | **coagulation SOFA** |
| `gcs` | float (int-valued) | Glasgow Coma Scale total, 3–15 | 3–15 | locf | **CNS SOFA** |
| `fio2` | float | fraction, 0.21–1.0 | 0.21–1.0 | interpolate | **respiratory SOFA** (via SF ratio) |
| `ventilated` | float (0/1) | boolean status | 0/1 | locf (status holds until changed) | respiratory SOFA denominator (ventilated patients score higher at a given SF ratio) |
| `urine_output_rate` | float | mL/hr | 0–500 | locf | **renal SOFA** (secondary criterion) |
| `vasopressor_dose` | float | norepinephrine-equivalent mcg/kg/min, 0 if none | 0–2 | locf (infusion rate holds until changed) | **cardiovascular SOFA** |
| `fluid_bolus_ml` | float | mL administered in this bin | 0–2000 | **sum** (event amount — never carried forward) | treatment-state feature only, not a SOFA input |
| `age` | float | years | 18–100 | locf (constant per stay) | model input (demographic) |
| `sex_male` | float (0/1) | — | 0/1 | locf (constant per stay) | model input (demographic) |
| `suspected_infection` | int (0/1) | Sepsis-3 suspected-infection window active at this timestamp | 0/1 | **label-construction only — see Section 6, never fed to the model as a feature** | SOFA-based label eligibility |

**Optional / generated-but-not-modeled-in-v1** (present in the raw synthetic tables, carried
through to `ml_dataset.csv` for future use, excluded from `config.ALL_FEATURE_COLS`):
`sbp`, `dbp` (systolic/diastolic BP — the pipeline's shock-index and MAP-based SOFA logic are
hardcoded to the `map` column name; SBP/DBP are realistic-but-currently-unused signal, kept so
a future architecture change doesn't need new synthetic-generation work).

**Column-name constraint carried over from the existing codebase**: `feature_engineering.py`'s
`add_shock_index()` hardcodes the literal column names `"heart_rate"` and `"map"`. Those two
names must never change without also editing that function — documented here so nobody
renames them by accident during a future MIMIC-IV extraction.

## 4. Raw synthetic tables (MIMIC-IV-shaped, pre-extraction)

The generator does **not** emit `ml_dataset.csv` directly. It emits long-format,
event-shaped tables that mirror the real MIMIC-IV tables the eventual extraction will read
(see the table list below), and a separate ETL step
(`extract_to_ml_dataset.py`) flattens them — this ETL step is exactly what gets rewritten
against BigQuery/real MIMIC-IV later; nothing downstream of it changes.

| Table | Shape (columns) | Mirrors |
|---|---|---|
| `patients.csv` | `subject_id, age, sex_male` | `mimiciv_hosp.patients` |
| `admissions.csv` | `subject_id, hadm_id, admittime, dischtime` | `mimiciv_hosp.admissions` |
| `icustays.csv` | `subject_id, hadm_id, stay_id, intime, outtime, archetype` (archetype is a synthetic-only debug column, dropped before extraction) | `mimiciv_icu.icustays` |
| `d_items.csv` | `itemid, label, category` | `mimiciv_icu.d_items` |
| `chartevents.csv` | `subject_id, hadm_id, stay_id, charttime, itemid, valuenum, valueuom` | `mimiciv_icu.chartevents` (vitals + GCS + FiO2 + ventilated-status all live here as distinct `itemid`s, exactly as in real MIMIC-IV) |
| `d_labitems.csv` | `itemid, label` | `mimiciv_hosp.d_labitems` |
| `labevents.csv` | `subject_id, hadm_id, charttime, itemid, valuenum, valueuom` | `mimiciv_hosp.labevents` |
| `inputevents.csv` | `subject_id, hadm_id, stay_id, starttime, endtime, itemid, amount, rate` | `mimiciv_icu.inputevents` (fluids + vasopressors) |
| `outputevents.csv` | `subject_id, hadm_id, stay_id, charttime, itemid, value` | `mimiciv_icu.outputevents` (urine) |
| `antibiotics.csv` | `subject_id, hadm_id, starttime` | simplified `mimiciv_hosp.prescriptions` (antibiotic subset) |
| `microbiologyevents.csv` | `subject_id, hadm_id, charttime` | `mimiciv_hosp.microbiologyevents` (culture orders) |

### 4a. One-stay-per-admission assumption (FIX 8 — post Phase-1-audit remediation)

**The current synthetic generator only ever produces one ICU stay per hospital admission**
(`generate_cohort()` in `generator.py` assigns exactly one `hadm_id` and one `stay_id` per
`subject_id`, 1:1:1). Real MIMIC-IV does **not** guarantee this — a single `hadm_id` can have
multiple `mimiciv_icu.icustays` rows (e.g. a patient transferred back to the ICU during the
same hospitalization), and a single `subject_id` can have multiple `hadm_id`s.

This assumption was previously baked silently into `extract_to_ml_dataset.py`'s lab-to-stay
join (a blind `merge(..., on="hadm_id")`), which would have **duplicated every lab row once per
stay** for any admission with more than one ICU stay. This has been fixed: `_pivot_labevents()`
now attributes each lab to the stay whose `[intime, outtime]` window actually contains its
`charttime`, dropping labs that fall in neither stay's window (see the function's docstring).
`tests/test_multi_stay.py` hand-builds a multi-stay-per-admission fixture (not producible by the
current generator) and verifies: no duplicate lab rows, correct per-stay attribution, and that
`preprocessing.split_by_subject()` never splits one patient's stays across train/val/test.

**Any future real MIMIC-IV extraction must**:
1. Query `mimiciv_icu.icustays` per `subject_id`/`hadm_id` expecting **zero or more** rows, not
   exactly one.
2. Attribute `labevents` (and any other hospital-wide-not-ICU-scoped table) to a `stay_id` by
   time-window containment against `icustays.intime`/`outtime`, exactly as `_pivot_labevents()`
   now does — never by a blind `hadm_id` join.
3. Confirm `mimiciv_icu.chartevents`/`inputevents`/`outputevents` already carry `stay_id`
   natively in real MIMIC-IV (they do) — those tables don't need this fix, only tables that lack
   a native `stay_id` (like `labevents`) do.

### 4b. Unit and item-ID harmonization real MIMIC-IV will require (not needed for synthetic data)

The synthetic schema (`schema.py`) uses exactly one synthetic `itemid` per clinical concept and
one fixed unit per concept, by construction — there is nothing to harmonize. **Real MIMIC-IV is
not this clean**, and any real extraction must budget time for:

- **Temperature is charted in both °C and °F** under different `itemid`s in real
  `chartevents` (MetaVision and CareVue use different item dictionaries for the same
  concept) — a real extraction must identify every temperature `itemid`, convert °F readings to
  °C (or vice versa), and merge them into one consistent unit before this pipeline's `temp`
  column will be meaningful.
- **Multiple `itemid`s can represent the same clinical concept.** Real MIMIC-IV's `d_items`
  has separate entries for e.g. heart rate from different monitor sources, non-invasive vs.
  invasive blood pressure, and multiple ventilator/FiO2 charting conventions. A real extraction
  needs an explicit `itemid` → concept mapping (a many-to-one join), not the one-`itemid`-per-
  concept assumption this synthetic schema makes for simplicity.
- **CareVue vs. MetaVision differences.** MIMIC-IV's source ICU data comes from two different
  clinical information systems used in different eras of the source hospital's ICU (CareVue,
  older; MetaVision, newer), which used different `itemid`s, units, and sometimes different
  granularity/frequency conventions for the same concepts. A real extraction script must account
  for both source systems' `itemid` sets rather than assuming a single consistent vocabulary.
- **Weight-dependent units** (e.g. vasopressor dosing in mcg/kg/min) require a real patient
  weight, which this synthetic schema does not generate (see `docs/SYNTHETIC_DATA.md` — weight
  is listed as MISSING for BioGears purposes too). A real extraction will need `chartevents`
  weight entries (also multi-`itemid`, admission-weight vs. daily-weight) to compute
  weight-normalized doses correctly.
- **Per-drug vasopressor conversion** (already noted in Section 7 below) compounds with the
  above — real `inputevents` has separate `itemid`s per vasopressor drug and per unit
  convention, on top of everything above.

None of this is implemented in Phase 1 — it is scoped here so whoever builds the real extraction
in Phase 2+ isn't surprised by it, and doesn't inherit the synthetic schema's one-itemid-per-
concept simplification as if it were a property of the real data.

### 4c. MIMIC-IV per-variable data contract (Phase 2 — the definitive mapping)

This is the authoritative, per-variable contract that `backend/mimic_adapter/` implements and
that `backend/mimic_adapter/concept_map.py` encodes in code (the two are kept in sync — this
table is generated directly from that module, not maintained by hand separately). Every row
whose "Item/event identifiers" column reads **REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV
DICTIONARY** ships with an intentionally EMPTY item-ID list in `concept_map.py` — nobody in this
development session has access to a real MIMIC-IV `d_items`/`d_labitems` dictionary, and a
guessed item ID presented as authoritative would silently corrupt every downstream number with
no error. See `concept_map.py`'s module docstring for exactly how to populate and verify each
mapping once real MIMIC-IV access exists. `backend/mimic_adapter/adapter.py` refuses to run
against data claiming to be real MIMIC-IV while any required mapping below is unverified.

<!-- generated from backend/mimic_adapter/concept_map.py — regenerate rather than hand-edit -->

| Canonical variable | Clinical meaning | Source table | Source column | Item/event identifiers | Expected unit | Unit conversion required | Temporal aggregation | Missing-value semantics | Model feature? | Label-only? | BioGears target? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `heart_rate` | Heart rate | mimiciv_icu.chartevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | bpm | none | mean within each 1h bin (multiple monitor readings per hour collapse to one value, matching preprocessing.resample_stay's 'interpolate' strategy) | no observation in the bin -> NaN, forward-filled + interpolated downstream (config.COLUMN_FILL_STRATEGY) | Yes | No | Yes |
| `map` | Mean arterial pressure | mimiciv_icu.chartevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | mmHg | none | mean within each 1h bin | no observation -> NaN, interpolated downstream | Yes | No | Yes |
| `resp_rate` | Respiratory rate | mimiciv_icu.chartevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | insp/min | none | mean within each 1h bin | no observation -> NaN, interpolated downstream | Yes | No | Yes |
| `spo2` | Peripheral oxygen saturation | mimiciv_icu.chartevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | % | none | mean within each 1h bin | no observation -> NaN, interpolated downstream | Yes | No | Yes |
| `temp` | Body temperature | mimiciv_icu.chartevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | degC | Fahrenheit readings must be converted to Celsius BEFORE aggregation (see units.convert_temperature) — see DATA_CONTRACT.md section 4b. | mean within each 1h bin, AFTER unit harmonization (see units.py) | no observation -> NaN, interpolated downstream | Yes | No | Yes |
| `gcs_eye` | Glasgow Coma Scale — eye opening component (1-4) | mimiciv_icu.chartevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | score | none | last value within each 1h bin (GCS is an assessment, not continuously sampled) | no observation -> NaN, LOCF downstream | No | No | No |
| `gcs_verbal` | Glasgow Coma Scale — verbal response component (1-5) | mimiciv_icu.chartevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | score | none | last value within each 1h bin | no observation -> NaN, LOCF downstream | No | No | No |
| `gcs_motor` | Glasgow Coma Scale — motor response component (1-6) | mimiciv_icu.chartevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | score | none | last value within each 1h bin | no observation -> NaN, LOCF downstream | No | No | No |
| `fio2` | Fraction of inspired oxygen | mimiciv_icu.chartevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | fraction | Some real MIMIC-IV FiO2 entries are charted as a PERCENTAGE (21-100) rather than a fraction (0.21-1.0) — units.py must validate and convert (divide by 100) based on valueuom/magnitude, never assume. | mean within each 1h bin | not charted while a patient is on room air -> NaN (this is expected sparsity, not a data gap — see DATA_CONTRACT.md section 5) | Yes | No | Yes |
| `ventilated` | Currently receiving mechanical ventilation (0/1) | mimiciv_icu.procedureevents OR mimiciv_icu.chartevents (ambiguous — see notes) | derived (interval overlap), not a single valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | — | none | 1 for any 1h bin overlapping a ventilation episode's [starttime, endtime), else 0 | 0 (absence of a ventilation episode is a true negative, not missing — same convention as vasopressor_dose) | Yes | No | Yes |
| `lactate` | Serum/arterial lactate | mimiciv_hosp.labevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | mmol/L | none | LOCF (no interpolation) — labs are sparse point events, not continuously observed | no draw in the bin -> NaN, forward-filled downstream with a staleness cap to be tuned against real missingness statistics (not yet implemented, see docs/MIMIC_IV_SETUP.md limitations) | Yes | No | No |
| `wbc` | White blood cell count | mimiciv_hosp.labevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | K/uL | none | LOCF | no draw -> NaN, forward-filled downstream | Yes | No | No |
| `creatinine` | Serum creatinine | mimiciv_hosp.labevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | mg/dL | none | LOCF | no draw -> NaN, forward-filled downstream | Yes | No | No |
| `bilirubin` | Total bilirubin | mimiciv_hosp.labevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | mg/dL | none | LOCF | no draw -> NaN, forward-filled downstream | Yes | No | No |
| `platelets` | Platelet count | mimiciv_hosp.labevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | K/uL | none | LOCF | no draw -> NaN, forward-filled downstream | Yes | No | No |
| `vasopressor_dose` | Norepinephrine-equivalent vasopressor infusion rate | mimiciv_icu.inputevents | rate | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | norepinephrine-equivalent mcg/kg/min | REAL MIMIC-IV inputevents has a SEPARATE itemid PER vasopressor drug (norepinephrine, epinephrine, vasopressin, phenylephrine, dopamine, dobutamine), each with its own dosing unit/scale. A real extraction needs a documented per-drug norepinephrine-equivalent conversion table (a well-established but drug-specific clinical convention) — not implemented here (see section 7, 'known future-work item'). Do not sum raw doses of different drugs as if equivalent. | rate holds constant across [starttime, endtime) of each infusion segment (LOCF within the episode, 0 outside any episode) | 0 for stays/bins with no active vasopressor episode (true absence of treatment, not an unknown value — see extract_to_ml_dataset.py's no_vaso_mask handling, reused as-is in adapter.py) | Yes | No | Yes |
| `fluid_bolus_ml` | IV crystalloid/colloid bolus volume administered in this bin | mimiciv_icu.inputevents | amount | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | mL | none | sum of `amount` for all bolus-category inputevents rows whose starttime falls in the bin (event amounts, never carried forward) | 0 mL for bins with no bolus event (true absence, not unknown) | Yes | No | Yes |
| `urine_output_rate` | Urine output rate | mimiciv_icu.outputevents | value | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | mL/hr | none | sum of `value` within each 1h bin (a volume, converted to a rate by the bin width), LOCF between catheterized measurements | no measurement -> NaN, forward-filled downstream (sparser when not catheterized — see section 5) | Yes | No | No |
| `_antibiotic_administration_time` | Antibiotic administration timing (Sepsis-3 suspected-infection criterion, one half) | mimiciv_hosp.prescriptions | starttime | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | — | none | event timestamp, not aggregated | absence of a row -> no antibiotic event, not missing data | No | Yes | No |
| `_culture_order_time` | Microbiology culture order timing (Sepsis-3 suspected-infection criterion, other half) | mimiciv_hosp.microbiologyevents | charttime | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | — | none | event timestamp, not aggregated | absence of a row -> no culture event, not missing data | No | Yes | No |
| `sbp` | Systolic blood pressure (non-invasive or arterial) | mimiciv_icu.chartevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | mmHg | none | mean within each 1h bin | no observation -> NaN (optional/unused-in-v1 column per section 3) | No | No | No |
| `dbp` | Diastolic blood pressure | mimiciv_icu.chartevents | valuenum | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | mmHg | none | mean within each 1h bin | no observation -> NaN (optional/unused-in-v1 column) | No | No | No |
| `weight_kg` | Patient body weight (for weight-normalized dosing) | mimiciv_icu.chartevents OR mimiciv_icu.inputevents.patientweight | valuenum / patientweight | REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY | kg | none | last known weight (admission or most recent daily weight) | **not currently modeled in the canonical schema at all** — see section 4b / docs/SYNTHETIC_DATA.md | No | No | Yes |
| `age` | Patient age at ICU admission (years) | mimiciv_hosp.patients + mimiciv_hosp.admissions | derived: patients.anchor_age + (admissions.admittime.year - patients.anchor_year) | N/A — structural derivation, no itemid involved (anchor_age>89 ceiling edge case REQUIRES VERIFICATION) | years | none | constant per stay | should never be missing for an included stay (cohort.py excludes stays where age cannot be computed) | Yes | No | Yes |
| `sex_male` | Sex (1=male, 0=female) | mimiciv_hosp.patients | gender | N/A — structural derivation, no itemid involved | — | none | constant per stay | should never be missing | Yes | No | Yes |

`weight_kg` is listed for completeness (it will matter for weight-normalized vasopressor dosing
and BioGears patient initialization) but is **not** part of `config.ALL_FEATURE_COLS` today —
adding it is a canonical-schema change, explicitly out of scope for this Phase 2 pass (see the
Phase 2 final report's blockers list).

## 5. Sampling frequency and irregularity (raw, pre-resample)

Each source stream is generated at its own realistic, jittered, sometimes-missing cadence —
**not** on a clean grid — so the resampling/missingness-handling code in `preprocessing.py` is
genuinely exercised rather than trivially satisfied:

- Vitals (`heart_rate`, `map`/`sbp`/`dbp`, `resp_rate`, `spo2`, `temp`): nominally hourly,
  jittered ±15 min, per-variable independent missingness (device disconnects, etc.), occasional
  duplicate/rapid-repeat measurements, occasional outlier spikes.
- Labs (`lactate`, `wbc`, `creatinine`, `bilirubin`, `platelets`): every 6–24h, higher and more
  variable missingness than vitals, occasional delayed charting (result posted hours after the
  nominal draw time).
- GCS: every 4–8h, missing entirely for some stays (e.g., deeply sedated — documented as a
  known simplification, not modeled mechanistically).
- FiO2 / ventilated status: only present (non-null) while a synthetic ventilation episode is
  active; absent otherwise (mirrors real charting — FiO2 isn't documented for room-air patients).
- Urine output: hourly when catheterized, sparser otherwise.
- Vasopressor dose / fluid boluses: event-based (start/stop/rate), not periodic.

## 6. Label-construction-only fields

`suspected_infection` is **not** a model input. It exists purely so the SOFA/Sepsis-3 labeling
pipeline (`labels_sofa.py`) can compute label eligibility from data that a real Sepsis-3 cohort
query would also use (antibiotic administration + culture order timing). This mirrors the
existing repo's explicit design rule in `labels.py` ("the original `risk_flag` column is never
used as model input") — the label-defining signal is deliberately withheld from
`config.ALL_FEATURE_COLS` so the model cannot trivially key off it. See
`backend/synthetic_data/extract_to_ml_dataset.py` for exactly how it's derived from
`antibiotics.csv` + `microbiologyevents.csv`.

## 7. What this contract deliberately does not attempt

- No arterial blood gas / true PaO2 simulation — the SF (SpO2/FiO2) ratio is used as a
  documented, clinically established surrogate for the PF (PaO2/FiO2) ratio in respiratory
  SOFA scoring. This is a real, cited substitution used when ABGs aren't available — not an
  invented shortcut — but it is a simplification relative to full MIMIC-IV, where PaO2 is
  directly queryable from `labevents`.
- No comorbidity/diagnosis modeling (`diagnoses_icd` is in the blueprint's real-data table list
  but not synthesized here — deferred; see Known Limitations in the Phase 1 report).
- No clinical notes.
- Vasopressor dosing is modeled as a single norepinephrine-equivalent number, not per-drug —
  real MIMIC-IV `inputevents` has separate `itemid`s per pressor; the extraction step will need
  a norepinephrine-equivalent conversion table when real data arrives (documented as a known
  future-work item, not implemented now).

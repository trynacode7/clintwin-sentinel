# Synthetic ICU Development Data (Phase 1)

This document explains the synthetic data layer built in Phase 1 of ClinTwin Sentinel:
why it exists, exactly what it does and does not represent, and how it will be replaced by
real MIMIC-IV data in Phase 2+ with minimal architectural change.

**All results produced using this data are engineering / development validation only. They
are not clinical evidence, and must never be reported, cited, or interpreted as evidence of
real-world clinical performance.**

---

## 0. Post-audit remediation (read this first)

An independent audit of the first version of this data layer found that the generator drove
every observable variable from a **single shared latent severity scalar**, and that
`suspected_infection` (label eligibility) was assigned almost deterministically by archetype
identity. Consequence: ~98.5% of positive labels came from exactly one archetype ("sepsis"),
and that archetype was ~0.90 AUROC separable from a single raw observation — the prediction task
had collapsed into "detect the sepsis archetype from a snapshot," not genuine
temporal/multivariate deterioration forecasting.

This has been restructured (see Section 2 below and `generator.py`'s module docstring for full
detail): **seven partially independent physiological axes** (hemodynamic, respiratory, renal,
hepatic, coagulation, neurological, metabolic) replace the single severity scalar, each with its
own randomly-drawn trajectory shape and severity, chosen via archetype-conditioned
*probabilities* rather than archetype-determined certainties. `suspected_infection` is now a
stochastic function of the *realized* axis peaks (with substantial noise), never of archetype.
The remediation's validation results (single-timestep vs. temporal-window vs. full-pipeline
diagnostics, archetype/label cross-tabulation, feature-importance checks) are reported in the
Phase 1 remediation report delivered alongside this document — read that for the concrete
before/after numbers; this document describes the current (post-remediation) design.

---

## 1. Why synthetic data, and why now

Real MIMIC-IV is a credentialed, PhysioNet-governed dataset. Under the project's data-handling
rules (and the PhysioNet Data Use Agreement), real patient-level MIMIC-IV data can never be
pasted, uploaded, or otherwise shared into this chat/assistant session — all real-data work
must happen on the user's own machine or BigQuery project, outside this environment. Building
and iterating on the ML pipeline therefore requires a substitute dataset that:

1. Can be generated, inspected, and reasoned about entirely inside this environment.
2. Is *structurally* and *statistically* realistic enough that a pipeline that works on it has
   a reasonable chance of also working on real MIMIC-IV, rather than passing only because the
   data was trivial.
3. Is schema-compatible with real MIMIC-IV, so replacing it later touches exactly one module
   (see Section 6) rather than the rest of the codebase.

That is the entire purpose of this data layer — it is a development/testing substrate, not a
research contribution and not a clinical dataset.

## 2. What it approximates, and which MIMIC-IV concepts it mirrors

The raw synthetic tables (`data/synthetic/raw/*.csv`) mirror the *logical structure* of the
real MIMIC-IV tables they are named after — see `DATA_CONTRACT.md` Section 4 for the full
per-table mapping (e.g. `chartevents`/`d_items` for vitals and neuro/respiratory charting,
`labevents`/`d_labitems` for labs, `inputevents` for vasopressors/fluids, `outputevents` for
urine output, `antibiotics`/`microbiologyevents` for the Sepsis-3 suspected-infection signal).
Item IDs are synthetic (namespaced 90000+/91000+/92000+/93000+) and are never real MIMIC-IV
`itemid`s.

Patient trajectories are generated from one of five archetypes (stable, improving,
deteriorating, sepsis-like deterioration, recovery-after-intervention), but — post-remediation —
the archetype only sets *probabilities*, never outcomes. Each patient has **seven physiological
axes**, each independently assigned a trajectory shape (flat / transient dip / gradual decline /
sudden decline / decline-then-plateau / decline-then-relapse / gradual improve /
improve-then-relapse / improve-then-decline / recover-after-intervention) and a severity range,
drawn per-axis according to archetype-conditioned probabilities. A small, minority-weight
cross-axis blend (`CROSS_AXIS_BLEND` in `generator.py`) adds physiologically-motivated
correlation — e.g. hemodynamic collapse contributing to lactate, renal, hepatic, respiratory, and
neurological derangement — without making any axis a deterministic function of another. Renal and
hepatic axes additionally pass through a slower EWMA to represent organ-dysfunction lag. No
variable-value formula, treatment decision, or the infection-suspicion decision reads
`archetype` directly — every one of them reads only the *realized* axis state for that patient,
which varies substantially even within one archetype (see the Phase 1 remediation report for the
archetype/label cross-tabulation demonstrating this). Every variable is independently,
irregularly sampled with realistic missingness, occasional outliers, and delayed lab reporting;
lab-draw frequency additionally depends weakly and noisily on a composite acuity signal (sicker
patients get watched somewhat more closely), with an independent per-stay "monitoring intensity"
offset so missingness is informative but not a clean label proxy (see `generator.py`'s module
docstring for the full design rationale).

The clinical label is **computed, not injected**: the generator only produces underlying
observations (vitals, labs, treatments, antibiotic/culture events); `labels_sofa.py` computes
six SOFA organ-dysfunction subscores and the Sepsis-3-consistent suspected-infection window
from those observations, then derives the binary future-window deterioration label exactly the
way it would be derived from real MIMIC-IV-extracted data. See `tests/test_data_validation.py`
(`test_label_is_never_present_in_raw_or_extracted_data`) for the regression test proving this.
`suspected_infection` itself is now a stochastic function of the realized hemodynamic/
respiratory/metabolic/renal axis peaks (`infection_events()` in `generator.py`) — not of
archetype — which is what allows some "sepsis"-archetype patients to end up without a qualifying
infection window, and some "deteriorating"-archetype patients to end up with one.

## 3. What it does NOT represent accurately

Documented explicitly (see also `DATA_CONTRACT.md` Section 7):

- **No arterial blood gas / PaO2 simulation.** The respiratory SOFA subscore uses the SpO2/FiO2
  (SF) ratio, a published real-world surrogate for PaO2/FiO2 (Pandharipande et al. 2009; Rice
  et al. 2007), not a simulated PaO2. Real MIMIC-IV has actual ABGs for a subset of patients.
- **No comorbidity/diagnosis modeling.** `diagnoses_icd` is not represented at all.
- **No free-text clinical notes.**
- **Single norepinephrine-equivalent vasopressor dose**, not per-drug dosing. The
  cardiovascular SOFA subscore's "SOFA=2, low-dose dopamine" tier is not representable in this
  schema and is intentionally skipped (documented in `labels_sofa.py`, not silently dropped).
- **No real disease epidemiology, demographics, or comorbidity burden.** Archetype prevalence,
  age distribution, and correlation structure are engineering choices for coverage and
  balance, not fitted to any real population.
- **Missingness mechanism is authored, not learned.** It is designed to be *structurally*
  irregular (jittered timestamps, per-variable sampling frequency, occasional outliers, delayed
  lab reporting) so the same preprocessing/resampling code that will face real MIMIC-IV
  sparsity is genuinely exercised — but the specific missingness *rates* are chosen by hand,
  not fit to real ICU charting behavior.

## 4. Why synthetic performance is not clinical evidence

Any AUROC/precision/recall/F1 numbers produced by training on this data (see the Phase 1 final
report for the actual numbers from this run) demonstrate only that the **pipeline mechanics**
work: data flows correctly from generation through SOFA labeling, preprocessing, feature
engineering, windowing, and into both baseline and deep models, and produces learnable signal
because the generator encodes real clinical correlations by construction. This is expected and
intended — it validates the *pipeline*, not clinical utility. It says nothing about
generalization to real ICU patients, real documentation patterns, real comorbidity mixes, or
real label prevalence, and must never be presented as such.

## 5. How real MIMIC-IV replaces this (Phase 2 — built, not yet run against real data)

As of Phase 2, `backend/mimic_adapter/` is this equivalent extraction module — it runs the same
shaping logic (pivot chartevents/labevents to wide format, derive vasopressor/fluid-bolus
point-events from inputevents, interval-join labs to the correct ICU stay, compute the Sepsis-3
suspected-infection window from prescriptions × microbiologyevents, apply hard physiological
bounds and unit harmonization) against real MIMIC-IV raw tables instead of the in-memory
synthetic tables, and produces the byte-for-byte same `ml_dataset.csv` schema (`DATA_CONTRACT.md`
Section 3/4c). Everything downstream — `preprocessing.py`, `feature_engineering.py`,
`labels_sofa.py`, `dataset.py`, the models, `train.py`, `evaluate.py` — is schema-stable and
requires no changes; this was verified by running `labels_sofa.create_labels_for_stay()`
unmodified against MIMIC-adapter-produced canonical data in `tests/test_mimic_adapter.py`.

**This has only ever been exercised against small MIMIC-STRUCTURE TEST FIXTURES, never real
MIMIC-IV** — every item ID in `backend/mimic_adapter/concept_map.py` ships unverified by design
(no real dictionary was available to verify against). See `docs/MIMIC_IV_SETUP.md` for the full
setup guide, including exactly which mappings still need verification before real extraction can
run.

Switching modes is a configuration change, not a code change: set the `CLINTWIN_DATA_MODE`
environment variable to `real` (default `synthetic`) and point `CLINTWIN_REAL_DATA_PATH` at the
extracted real `ml_dataset.csv`. `config.py` and `preprocessing.load_data()` never silently fall
back between modes — a missing file raises an explicit, actionable error naming which mode was
requested (see `preprocessing.py`'s `load_data()`).

## 6. Exact commands

### Generate synthetic data

```bash
python generate_synthetic_data.py --patients 1000 --seed 42
```

Key options: `--patients`, `--seed`, `--min-stay-hours` (default 24), `--max-stay-hours`
(default 120), `--missingness-scale` (default 1.0), `--outlier-rate` (default 0.01),
`--output-dir`, `--archetype-weights` (JSON string, default
`{"stable":0.35,"improving":0.15,"deteriorating":0.20,"sepsis":0.20,"recovery":0.10}`).

This writes:
- `data/synthetic/raw/*.csv` — the raw, MIMIC-IV-shaped tables.
- `data/synthetic/processed/ml_dataset.csv` — the flat modeling table.
- `data/synthetic/processed/dataset_summary.json` — patient/stay/observation counts,
  archetype counts, per-column missingness, suspected-infection-positive stay count.

The CLI runs `validate_ml_dataset()` and exits non-zero if validation fails — it will not
silently produce a broken dataset.

### Train / evaluate

```bash
python -m backend.baselines            # LogReg + XGBoost tabular baselines
python -m backend.train --model lstm --epochs 100
python -m backend.train --model transformer --epochs 100
```

`train.py` loads data via `preprocessing.preprocess()`, which reads `config.DATA_PATH`
(`data/synthetic/processed/ml_dataset.csv` when `CLINTWIN_DATA_MODE=synthetic`, the default).
Trained checkpoints are saved to `models/lstm_model.pt` / `models/transformer_model.pt`
(paths from `config.py`); evaluate against the held-out `test` split using
`backend.evaluate.evaluate_model()`.

### Run the test suite

```bash
python -m pytest tests/ -v
```

## 7. Dataset schema

See `DATA_CONTRACT.md` for the authoritative, full schema (identifiers, timestamps, every
column's type/unit/range/fill-strategy, and the raw MIMIC-shaped table structure). Summary of
the flat `ml_dataset.csv` columns:

`subject_id, hadm_id, stay_id, charttime, heart_rate, sbp, dbp, map, resp_rate, spo2, temp,
lactate, wbc, creatinine, bilirubin, platelets, gcs, fio2, ventilated, urine_output_rate,
vasopressor_dose, fluid_bolus_ml, age, sex_male, suspected_infection`

`suspected_infection` is a **label-construction-only** field (`config.LABEL_ONLY_COLS`) — it is
excluded from `config.ALL_FEATURE_COLS` and is never visible to any model as an input feature;
it exists only so `labels_sofa.py` can determine Sepsis-3 label eligibility.

## 8. Label-generation process

1. The generator produces raw vitals/labs/treatments/antibiotic/culture events — never a
   precomputed label.
2. `extract_to_ml_dataset.py` derives `suspected_infection` (Sepsis-3 window: a qualifying
   culture within [-24h, +72h] of an antibiotic administration, active from the earlier of the
   two onward) from the raw `antibiotics`/`microbiologyevents` tables.
3. `labels_sofa.py` computes six SOFA organ subscores (respiratory via SF-ratio, coagulation,
   hepatic, cardiovascular — MAP < 70 mmHg (corrected; see Fix 1 in `labels_sofa.py`'s
   docstring, was previously an incorrect 65 mmHg), CNS, renal) per hourly row, sums them to a
   total SOFA score, and for each row `i` looks at the future window
   `[i + FUTURE_WINDOW_START, i + FUTURE_WINDOW_END)` (2-6 hours ahead): the label is `1` iff
   that window contains both (a) a suspected-infection flag and (b) a SOFA increase of
   `>= SOFA_INCREASE_THRESHOLD` (2) from the baseline (row `i`'s own current SOFA — a **rolling**
   baseline, not the official Sepsis-3 admission-anchored baseline; see the "FIX 9" note in
   `labels_sofa.py`'s docstring for exactly what would need to change to switch to an
   admission-anchored definition). Rows without enough future data get `NaN` and are dropped —
   never a manufactured negative.
4. This is exactly the same mechanism that will run against real, MIMIC-IV-extracted data —
   `labels_sofa.py` does not know or care whether its input came from the synthetic generator or
   a real extraction.

## 9. Known engineering limitations of this phase

- The generator's missingness rates, archetype prevalence, per-axis involvement probabilities,
  trajectory shape libraries, and cross-axis blend weights are hand-authored, not fit to any
  real distribution (see Section 3). The remediation makes the *structure* far more realistic
  (multiple partially-independent organ systems, diverse trajectory shapes, non-deterministic
  archetype-to-outcome mapping) but the specific numeric probabilities/weights remain engineering
  judgment calls, not statistically fitted parameters.
- Small cohorts (tens of patients) can produce very few val/test-split positive labels, which
  can make metrics noisy — this is expected statistical behavior of small samples, not a
  pipeline bug (use `--patients 1000` or more for stable metrics). This is more pronounced
  post-remediation since the label is no longer near-guaranteed for one archetype.
- `data/real/` is a placeholder directory (see its `README.md`) — Phase 1 does not extract or
  process any real MIMIC-IV data.
- The cross-axis correlation model (`CROSS_AXIS_BLEND` in `generator.py`) only encodes a small,
  fixed set of physiologically-motivated couplings (hemodynamic contributing to metabolic/renal/
  respiratory/hepatic/neuro derangement). Real multi-organ dysfunction has richer, more varied
  coupling patterns than this minority-weight blend captures.

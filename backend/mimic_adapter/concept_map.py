"""
concept_map.py — MIMIC-IV item/concept -> canonical ClinTwin variable mapping.

============================================================================
REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY
============================================================================
Every `ConceptMapping` below ships with `itemids=[]` and `verified=False`.
This is deliberate, not an oversight: nobody in this development session has
access to a real MIMIC-IV `mimiciv_icu.d_items` / `mimiciv_hosp.d_labitems`
dictionary, and presenting a guessed item ID as authoritative would silently
corrupt every downstream number with no error — precisely the failure mode
the Phase 2 spec (sections 5 and 12) requires this module to refuse to
allow. `adapter.py` will not run against data claiming to be real MIMIC-IV
while any concept it needs is unverified (see `require_verified`).

HOW TO POPULATE THIS FILE ONCE YOU HAVE MIMIC-IV ACCESS (on your own
machine/BigQuery project — never paste real dictionary rows into this chat,
though item dictionaries themselves are not patient-level data and are not
covered by the PhysioNet DUA's data-sharing restriction; code/config is
always fine to share):

  1. Query `mimiciv_icu.d_items` / `mimiciv_hosp.d_labitems` for the
     `label`s referenced in each concept's `notes` below (e.g.
     `label ILIKE '%heart rate%'`).
  2. Record EVERY itemid that plausibly represents that concept — MIMIC-IV
     is documented to have multiple itemids per concept (different monitor
     sources, CareVue vs. MetaVision, invasive vs. non-invasive BP — see
     DATA_CONTRACT.md section 4b/5b).
  3. Fill in `itemids=[...]` for that ConceptMapping and set `verified=True`.
  4. Run `python -m backend.mimic_adapter.validate_mimic --check-mapping`
     (see docs/MIMIC_IV_SETUP.md) — it refuses to proceed while any
     in-scope concept is unverified.

Only this file (plus units.py for conversions) should ever contain a MIMIC
item ID. Do not hardcode item IDs anywhere else in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ConceptMapping:
    canonical_var: str
    clinical_meaning: str
    source_table: str            # e.g. "mimiciv_icu.chartevents"
    source_value_column: str     # e.g. "valuenum"
    temporal_aggregation: str    # how repeated raw events collapse to one hourly value
    missing_value_semantics: str
    is_feature: bool
    label_only: bool
    biogears_target: bool        # whether TECHNICAL_BLUEPRINT.md section 6 marks this "Yes" for BioGears mapping
    expected_unit: Optional[str] = None
    unit_conversion: Optional[str] = None   # human-readable; actual logic lives in units.py
    itemids: List[int] = field(default_factory=list)   # REQUIRES VERIFICATION — intentionally empty
    verified: bool = False
    notes: str = ""


# ─── chartevents-sourced concepts (mimiciv_icu.chartevents + d_items) ───────
CHARTEVENTS_CONCEPTS: Dict[str, ConceptMapping] = {
    "heart_rate": ConceptMapping(
        canonical_var="heart_rate", clinical_meaning="Heart rate",
        source_table="mimiciv_icu.chartevents", source_value_column="valuenum",
        temporal_aggregation="mean within each 1h bin (multiple monitor readings per hour collapse to one value, matching preprocessing.resample_stay's 'interpolate' strategy)",
        missing_value_semantics="no observation in the bin -> NaN, forward-filled + interpolated downstream (config.COLUMN_FILL_STRATEGY)",
        is_feature=True, label_only=False, biogears_target=True,
        expected_unit="bpm",
        notes="REQUIRES VERIFICATION AGAINST ACTUAL MIMIC-IV DICTIONARY. Search "
              "d_items for label ILIKE '%heart rate%'. Real chartevents is documented "
              "to carry more than one itemid for HR across CareVue/MetaVision sources.",
    ),
    "sbp": ConceptMapping(
        canonical_var="sbp", clinical_meaning="Systolic blood pressure (non-invasive or arterial)",
        source_table="mimiciv_icu.chartevents", source_value_column="valuenum",
        temporal_aggregation="mean within each 1h bin",
        missing_value_semantics="no observation -> NaN (optional/unused-in-v1 column per DATA_CONTRACT.md section 3)",
        is_feature=False, label_only=False, biogears_target=False,
        expected_unit="mmHg",
        notes="REQUIRES VERIFICATION. Search for both invasive ('Arterial Blood Pressure "
              "systolic') and non-invasive ('Non Invasive Blood Pressure systolic') itemids — "
              "these are typically DIFFERENT itemids in real MIMIC-IV and may need a "
              "documented preference order (e.g. prefer invasive when both present).",
    ),
    "dbp": ConceptMapping(
        canonical_var="dbp", clinical_meaning="Diastolic blood pressure",
        source_table="mimiciv_icu.chartevents", source_value_column="valuenum",
        temporal_aggregation="mean within each 1h bin",
        missing_value_semantics="no observation -> NaN (optional/unused-in-v1 column)",
        is_feature=False, label_only=False, biogears_target=False,
        expected_unit="mmHg",
        notes="REQUIRES VERIFICATION. Same invasive/non-invasive multi-itemid caveat as sbp.",
    ),
    "map": ConceptMapping(
        canonical_var="map", clinical_meaning="Mean arterial pressure",
        source_table="mimiciv_icu.chartevents", source_value_column="valuenum",
        temporal_aggregation="mean within each 1h bin",
        missing_value_semantics="no observation -> NaN, interpolated downstream",
        is_feature=True, label_only=False, biogears_target=True,
        expected_unit="mmHg",
        notes="REQUIRES VERIFICATION. Search for 'Arterial Blood Pressure mean' and "
              "'Non Invasive Blood Pressure mean' — this is the single most important "
              "concept to get right, since it directly drives cardiovascular SOFA "
              "(labels_sofa.py) and shock_index (feature_engineering.py). If no direct "
              "MAP itemid is charted for a given source, DATA_CONTRACT.md's known "
              "clinical approximation MAP ~= DBP + (SBP-DBP)/3 from sbp/dbp is a "
              "documented fallback — implement it explicitly and flag which rows used "
              "the fallback, never silently.",
    ),
    "resp_rate": ConceptMapping(
        canonical_var="resp_rate", clinical_meaning="Respiratory rate",
        source_table="mimiciv_icu.chartevents", source_value_column="valuenum",
        temporal_aggregation="mean within each 1h bin",
        missing_value_semantics="no observation -> NaN, interpolated downstream",
        is_feature=True, label_only=False, biogears_target=True,
        expected_unit="insp/min",
        notes="REQUIRES VERIFICATION. Search d_items for label ILIKE '%respiratory rate%'.",
    ),
    "spo2": ConceptMapping(
        canonical_var="spo2", clinical_meaning="Peripheral oxygen saturation",
        source_table="mimiciv_icu.chartevents", source_value_column="valuenum",
        temporal_aggregation="mean within each 1h bin",
        missing_value_semantics="no observation -> NaN, interpolated downstream",
        is_feature=True, label_only=False, biogears_target=True,
        expected_unit="%",
        notes="REQUIRES VERIFICATION. Search for 'O2 saturation pulseoxymetry' or similar. "
              "Also feeds the SF-ratio respiratory SOFA surrogate in labels_sofa.py.",
    ),
    "temp": ConceptMapping(
        canonical_var="temp", clinical_meaning="Body temperature",
        source_table="mimiciv_icu.chartevents", source_value_column="valuenum",
        temporal_aggregation="mean within each 1h bin, AFTER unit harmonization (see units.py)",
        missing_value_semantics="no observation -> NaN, interpolated downstream",
        is_feature=True, label_only=False, biogears_target=True,
        expected_unit="degC",
        unit_conversion="Fahrenheit readings must be converted to Celsius BEFORE aggregation "
                         "(see units.convert_temperature) — see DATA_CONTRACT.md section 4b.",
        notes="REQUIRES VERIFICATION. Real MIMIC-IV charts temperature in BOTH Celsius and "
              "Fahrenheit under DIFFERENT itemids ('Temperature Celsius' vs. 'Temperature "
              "Fahrenheit') — both itemid sets must be found and mapped, with valueuom used "
              "to select the correct conversion at the row level (never assume one unit "
              "for a whole itemid without checking valueuom).",
    ),
    "gcs_eye": ConceptMapping(
        canonical_var="gcs_eye", clinical_meaning="Glasgow Coma Scale — eye opening component (1-4)",
        source_table="mimiciv_icu.chartevents", source_value_column="valuenum",
        temporal_aggregation="last value within each 1h bin (GCS is an assessment, not continuously sampled)",
        missing_value_semantics="no observation -> NaN, LOCF downstream",
        is_feature=False, label_only=False, biogears_target=False,
        expected_unit="score",
        notes="REQUIRES VERIFICATION. GCS is charted as THREE separate components in real "
              "MIMIC-IV chartevents (eye/verbal/motor), summed to a total (3-15) — there is "
              "no single 'GCS total' vital-sign itemid to rely on by default. Search d_items "
              "for 'GCS - Eye Opening'. Some derivations instead use a pre-summed itemid if "
              "one is confirmed to exist and is populated consistently — verify which "
              "approach the actual dictionary supports before choosing.",
    ),
    "gcs_verbal": ConceptMapping(
        canonical_var="gcs_verbal", clinical_meaning="Glasgow Coma Scale — verbal response component (1-5)",
        source_table="mimiciv_icu.chartevents", source_value_column="valuenum",
        temporal_aggregation="last value within each 1h bin",
        missing_value_semantics="no observation -> NaN, LOCF downstream",
        is_feature=False, label_only=False, biogears_target=False,
        expected_unit="score",
        notes="REQUIRES VERIFICATION. Search d_items for 'GCS - Verbal Response'. Sedated/"
              "intubated patients are frequently charted non-numerically ('Intubated'/'ETT' "
              "labels rather than a 1-5 score) in real MIMIC-IV — a real extraction must "
              "decide, and document, how these are scored (published SOFA/GCS pipelines "
              "commonly impute a conservative low value; do not decide this silently).",
    ),
    "gcs_motor": ConceptMapping(
        canonical_var="gcs_motor", clinical_meaning="Glasgow Coma Scale — motor response component (1-6)",
        source_table="mimiciv_icu.chartevents", source_value_column="valuenum",
        temporal_aggregation="last value within each 1h bin",
        missing_value_semantics="no observation -> NaN, LOCF downstream",
        is_feature=False, label_only=False, biogears_target=False,
        expected_unit="score",
        notes="REQUIRES VERIFICATION. Search d_items for 'GCS - Motor Response'. "
              "canonical gcs = gcs_eye + gcs_verbal + gcs_motor, computed in adapter.py "
              "ONLY once all three components are individually verified.",
    ),
    "fio2": ConceptMapping(
        canonical_var="fio2", clinical_meaning="Fraction of inspired oxygen",
        source_table="mimiciv_icu.chartevents", source_value_column="valuenum",
        temporal_aggregation="mean within each 1h bin",
        missing_value_semantics="not charted while a patient is on room air -> NaN "
                                 "(this is expected sparsity, not a data gap — see DATA_CONTRACT.md section 5)",
        is_feature=True, label_only=False, biogears_target=True,
        expected_unit="fraction",
        unit_conversion="Some real MIMIC-IV FiO2 entries are charted as a PERCENTAGE (21-100) "
                         "rather than a fraction (0.21-1.0) — units.py must validate and "
                         "convert (divide by 100) based on valueuom/magnitude, never assume.",
        notes="REQUIRES VERIFICATION. Search d_items for 'Inspired O2 Fraction'.",
    ),
    "ventilated": ConceptMapping(
        canonical_var="ventilated", clinical_meaning="Currently receiving mechanical ventilation (0/1)",
        source_table="mimiciv_icu.procedureevents OR mimiciv_icu.chartevents (ambiguous — see notes)",
        source_value_column="derived (interval overlap), not a single valuenum",
        temporal_aggregation="1 for any 1h bin overlapping a ventilation episode's [starttime, endtime), else 0",
        missing_value_semantics="0 (absence of a ventilation episode is a true negative, not missing — "
                                 "same convention as vasopressor_dose)",
        is_feature=True, label_only=False, biogears_target=True,
        notes="REQUIRES VERIFICATION AND A DESIGN DECISION, not just an itemid lookup. The "
              "published MIT-LCP mimic-code repository derives ventilation status from a "
              "COMBINATION of procedureevents (explicit ventilation start/stop, itemid "
              "unverified here) and chartevents oxygen-device/ventilator-settings itemids "
              "(also unverified). Do not implement a real extraction for this concept until "
              "the exact source table/itemid combination is confirmed — this is flagged as a "
              "design dependency, not guessed at.",
    ),
}

# ─── labevents-sourced concepts (mimiciv_hosp.labevents + d_labitems) ───────
LABEVENTS_CONCEPTS: Dict[str, ConceptMapping] = {
    "lactate": ConceptMapping(
        canonical_var="lactate", clinical_meaning="Serum/arterial lactate",
        source_table="mimiciv_hosp.labevents", source_value_column="valuenum",
        temporal_aggregation="LOCF (no interpolation) — labs are sparse point events, not continuously observed",
        missing_value_semantics="no draw in the bin -> NaN, forward-filled downstream with a "
                                 "staleness cap to be tuned against real missingness statistics "
                                 "(TECHNICAL_BLUEPRINT.md section 2 — not yet implemented, see "
                                 "docs/MIMIC_IV_SETUP.md limitations)",
        is_feature=True, label_only=False, biogears_target=False,
        expected_unit="mmol/L",
        notes="REQUIRES VERIFICATION. Search d_labitems for label ILIKE '%lactate%'. "
              "BioGears does NOT model lactate physiology (TECHNICAL_BLUEPRINT.md section 6) "
              "— biogears_target is correctly False here.",
    ),
    "wbc": ConceptMapping(
        canonical_var="wbc", clinical_meaning="White blood cell count",
        source_table="mimiciv_hosp.labevents", source_value_column="valuenum",
        temporal_aggregation="LOCF",
        missing_value_semantics="no draw -> NaN, forward-filled downstream",
        is_feature=True, label_only=False, biogears_target=False,
        expected_unit="K/uL",
        notes="REQUIRES VERIFICATION. Search d_labitems for 'White Blood Cells'.",
    ),
    "creatinine": ConceptMapping(
        canonical_var="creatinine", clinical_meaning="Serum creatinine",
        source_table="mimiciv_hosp.labevents", source_value_column="valuenum",
        temporal_aggregation="LOCF",
        missing_value_semantics="no draw -> NaN, forward-filled downstream",
        is_feature=True, label_only=False, biogears_target=False,
        expected_unit="mg/dL",
        notes="REQUIRES VERIFICATION. Search d_labitems for 'Creatinine'. Drives renal SOFA "
              "(labels_sofa.py) jointly with urine_output_rate.",
    ),
    "bilirubin": ConceptMapping(
        canonical_var="bilirubin", clinical_meaning="Total bilirubin",
        source_table="mimiciv_hosp.labevents", source_value_column="valuenum",
        temporal_aggregation="LOCF",
        missing_value_semantics="no draw -> NaN, forward-filled downstream",
        is_feature=True, label_only=False, biogears_target=False,
        expected_unit="mg/dL",
        notes="REQUIRES VERIFICATION. Search d_labitems for 'Bilirubin, Total'. Drives "
              "hepatic SOFA. BioGears does not model infection-specific hepatic dysfunction.",
    ),
    "platelets": ConceptMapping(
        canonical_var="platelets", clinical_meaning="Platelet count",
        source_table="mimiciv_hosp.labevents", source_value_column="valuenum",
        temporal_aggregation="LOCF",
        missing_value_semantics="no draw -> NaN, forward-filled downstream",
        is_feature=True, label_only=False, biogears_target=False,
        expected_unit="K/uL",
        notes="REQUIRES VERIFICATION. Search d_labitems for 'Platelet Count'. Drives "
              "coagulation SOFA.",
    ),
}

# ─── inputevents-sourced concepts (mimiciv_icu.inputevents) ─────────────────
INPUTEVENTS_CONCEPTS: Dict[str, ConceptMapping] = {
    "vasopressor_dose": ConceptMapping(
        canonical_var="vasopressor_dose",
        clinical_meaning="Norepinephrine-equivalent vasopressor infusion rate",
        source_table="mimiciv_icu.inputevents", source_value_column="rate",
        temporal_aggregation="rate holds constant across [starttime, endtime) of each infusion "
                              "segment (LOCF within the episode, 0 outside any episode)",
        missing_value_semantics="0 for stays/bins with no active vasopressor episode (true "
                                 "absence of treatment, not an unknown value — see "
                                 "extract_to_ml_dataset.py's no_vaso_mask handling, reused as-is "
                                 "in adapter.py)",
        is_feature=True, label_only=False, biogears_target=True,
        expected_unit="norepinephrine-equivalent mcg/kg/min",
        unit_conversion="REAL MIMIC-IV inputevents has a SEPARATE itemid PER vasopressor drug "
                         "(norepinephrine, epinephrine, vasopressin, phenylephrine, dopamine, "
                         "dobutamine), each with its own dosing unit/scale. A real extraction "
                         "needs a documented per-drug norepinephrine-equivalent conversion "
                         "table (a well-established but drug-specific clinical convention) — "
                         "not implemented here (see DATA_CONTRACT.md section 7, 'known future-"
                         "work item'). Do not sum raw doses of different drugs as if equivalent.",
        notes="REQUIRES VERIFICATION for EVERY vasopressor drug's itemid(s) individually. "
              "Weight-normalized (mcg/kg/min) doses additionally require a real patient "
              "weight (see the 'weight' concept below) — verify inputevents' rateuom per "
              "itemid, some are already weight-normalized and some are not.",
    ),
    "fluid_bolus_ml": ConceptMapping(
        canonical_var="fluid_bolus_ml", clinical_meaning="IV crystalloid/colloid bolus volume administered in this bin",
        source_table="mimiciv_icu.inputevents", source_value_column="amount",
        temporal_aggregation="sum of `amount` for all bolus-category inputevents rows whose "
                              "starttime falls in the bin (event amounts, never carried forward)",
        missing_value_semantics="0 mL for bins with no bolus event (true absence, not unknown)",
        is_feature=True, label_only=False, biogears_target=True,
        expected_unit="mL",
        notes="REQUIRES VERIFICATION. Real inputevents mixes continuous maintenance-fluid "
              "infusions with discrete resuscitation boluses under potentially many itemids "
              "(NS, LR, various colloids) — a real extraction needs an explicit, documented "
              "rule for which ordercategoryname/itemid combinations count as a 'bolus' versus "
              "background maintenance fluid, not an assumption that every inputevents fluid "
              "row is a bolus.",
    ),
    "weight": ConceptMapping(
        canonical_var="weight_kg", clinical_meaning="Patient body weight (for weight-normalized dosing)",
        source_table="mimiciv_icu.chartevents OR mimiciv_icu.inputevents.patientweight",
        source_value_column="valuenum / patientweight",
        temporal_aggregation="last known weight (admission or most recent daily weight)",
        missing_value_semantics="not currently modeled in the canonical schema at all — "
                                 "see DATA_CONTRACT.md section 4b / docs/SYNTHETIC_DATA.md",
        is_feature=False, label_only=False, biogears_target=True,
        expected_unit="kg",
        notes="REQUIRES VERIFICATION AND A SCHEMA DECISION. Not currently part of "
              "config.ALL_FEATURE_COLS or DATA_CONTRACT.md section 3 at all — needed only for "
              "weight-normalized vasopressor dosing and, later, BioGears patient "
              "initialization. Adding it is a canonical-schema change and is explicitly OUT "
              "OF SCOPE for this Phase 2 pass (see the Phase 2 final report's blockers list) "
              "— flagged here so it isn't silently forgotten.",
    ),
}

# ─── outputevents-sourced concepts (mimiciv_icu.outputevents) ───────────────
OUTPUTEVENTS_CONCEPTS: Dict[str, ConceptMapping] = {
    "urine_output_rate": ConceptMapping(
        canonical_var="urine_output_rate", clinical_meaning="Urine output rate",
        source_table="mimiciv_icu.outputevents", source_value_column="value",
        temporal_aggregation="sum of `value` within each 1h bin (a volume, converted to a rate "
                              "by the bin width), LOCF between catheterized measurements",
        missing_value_semantics="no measurement -> NaN, forward-filled downstream (sparser when "
                                 "not catheterized — see DATA_CONTRACT.md section 5)",
        is_feature=True, label_only=False, biogears_target=False,
        expected_unit="mL/hr",
        notes="REQUIRES VERIFICATION. Search d_items (outputevents also joins d_items) for "
              "'Foley'/'Urine' category labels — real outputevents has multiple itemids for "
              "different collection methods (Foley, void, condom catheter, etc.) that a real "
              "extraction must decide whether/how to combine.",
    ),
}

# ─── label-construction-only concepts (never model features) ────────────────
LABEL_CONCEPTS: Dict[str, ConceptMapping] = {
    "antibiotic_administration": ConceptMapping(
        canonical_var="_antibiotic_administration_time",
        clinical_meaning="Antibiotic administration timing (Sepsis-3 suspected-infection criterion, one half)",
        source_table="mimiciv_hosp.prescriptions", source_value_column="starttime",
        temporal_aggregation="event timestamp, not aggregated",
        missing_value_semantics="absence of a row -> no antibiotic event, not missing data",
        is_feature=False, label_only=True, biogears_target=False,
        notes="REQUIRES VERIFICATION. Real prescriptions.drug is free-text drug names, not a "
              "coded antibiotic flag — a real extraction needs an explicit antibiotic "
              "drug-name/class list (e.g. cross-referenced against a published antibiotic "
              "name list such as the one used in the original Sepsis-3/Seymour et al. "
              "cohort-definition code) to decide which prescriptions rows qualify. Do not "
              "invent this list; it must be sourced from a citable clinical reference before "
              "a real extraction runs.",
    ),
    "culture_order": ConceptMapping(
        canonical_var="_culture_order_time",
        clinical_meaning="Microbiology culture order timing (Sepsis-3 suspected-infection criterion, other half)",
        source_table="mimiciv_hosp.microbiologyevents", source_value_column="charttime",
        temporal_aggregation="event timestamp, not aggregated",
        missing_value_semantics="absence of a row -> no culture event, not missing data",
        is_feature=False, label_only=True, biogears_target=False,
        notes="Structurally simpler than antibiotics — every microbiologyevents row is a "
              "culture order by definition, no drug-name-list ambiguity. Still REQUIRES "
              "VERIFICATION of exact column names/semantics (charttime vs. chartdate) against "
              "the real table before use.",
    ),
}

# ─── demographic concepts ────────────────────────────────────────────────────
DEMOGRAPHIC_CONCEPTS: Dict[str, ConceptMapping] = {
    "age": ConceptMapping(
        canonical_var="age", clinical_meaning="Patient age at ICU admission (years)",
        source_table="mimiciv_hosp.patients + mimiciv_hosp.admissions",
        source_value_column="derived: patients.anchor_age + (admissions.admittime.year - patients.anchor_year)",
        temporal_aggregation="constant per stay",
        missing_value_semantics="should never be missing for an included stay (cohort.py excludes "
                                 "stays where age cannot be computed)",
        is_feature=True, label_only=False, biogears_target=True,
        expected_unit="years",
        notes="STRUCTURAL, not itemid-based, but still REQUIRES VERIFICATION of an edge case: "
              "MIMIC-IV's de-identification scheme (anchor_age/anchor_year/anchor_year_group) "
              "codes all patients aged >89 at any point as anchor_age=91 — a real extraction "
              "must decide how to treat this ceiling (documented in MIMIC-IV's own "
              "documentation) rather than silently treating 91 as an exact age.",
    ),
    "sex_male": ConceptMapping(
        canonical_var="sex_male", clinical_meaning="Sex (1=male, 0=female)",
        source_table="mimiciv_hosp.patients", source_value_column="gender",
        temporal_aggregation="constant per stay",
        missing_value_semantics="should never be missing",
        is_feature=True, label_only=False, biogears_target=True,
        notes="STRUCTURAL. patients.gender is coded 'M'/'F' in real MIMIC-IV — "
              "sex_male = (gender == 'M'). No itemid involved; still worth a real-data spot "
              "check that no other coding value appears.",
    ),
}

ALL_CONCEPTS: Dict[str, ConceptMapping] = {
    **CHARTEVENTS_CONCEPTS, **LABEVENTS_CONCEPTS, **INPUTEVENTS_CONCEPTS,
    **OUTPUTEVENTS_CONCEPTS, **LABEL_CONCEPTS, **DEMOGRAPHIC_CONCEPTS,
}

# Concepts an adapter run against the canonical DATA_CONTRACT.md-section-3
# schema actually needs to be itemid-verified for (i.e. excludes the two
# purely-structural demographic concepts and sbp/dbp, which are optional/
# unused-in-v1 per DATA_CONTRACT.md section 3).
REQUIRED_FOR_V1_SCHEMA = [
    "heart_rate", "map", "resp_rate", "spo2", "temp",
    "gcs_eye", "gcs_verbal", "gcs_motor", "fio2", "ventilated",
    "lactate", "wbc", "creatinine", "bilirubin", "platelets",
    "vasopressor_dose", "fluid_bolus_ml", "urine_output_rate",
    "antibiotic_administration", "culture_order",
]


def unverified_concepts(required_only: bool = True) -> List[str]:
    """Names of concepts still needing real-dictionary verification."""
    scope = REQUIRED_FOR_V1_SCHEMA if required_only else list(ALL_CONCEPTS.keys())
    return [name for name in scope if not ALL_CONCEPTS[name].verified]


def require_verified(required_only: bool = True) -> None:
    """Raise loudly if any in-scope concept mapping is unverified. Called by
    adapter.adapt() before touching data claiming to be real MIMIC-IV."""
    unverified = unverified_concepts(required_only=required_only)
    if unverified:
        raise RuntimeError(
            "Refusing to run the MIMIC-IV adapter: the following concept "
            f"mappings are unverified against a real MIMIC-IV dictionary: {unverified}. "
            "See backend/mimic_adapter/concept_map.py's module docstring and "
            "docs/MIMIC_IV_SETUP.md for how to populate and verify them before "
            "running this against real data. (Pass skip_verification_check=True "
            "to adapter.adapt() only for MIMIC-STRUCTURE TEST FIXTURES, never for real data.)"
        )

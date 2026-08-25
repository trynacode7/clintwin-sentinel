"""
units.py — Unit harmonization for the MIMIC-IV adapter (Phase 2, section 6).

FAIL LOUD, NEVER GUESS: every conversion function here validates the source
unit string against an explicit whitelist before converting. An unrecognized
or ambiguous unit string raises `UnitValidationError` — the adapter must
never silently assume a conversion. This mirrors DATA_CONTRACT.md section 3's
canonical units (the same units the synthetic generator already produces by
construction, so no conversion is needed on that side).
"""

from __future__ import annotations

import pandas as pd


class UnitValidationError(ValueError):
    """Raised when a source unit is missing, unrecognized, or ambiguous —
    the adapter must stop rather than assume a conversion."""


# Canonical unit per DATA_CONTRACT.md section 3 — what every downstream
# module (preprocessing.py, feature_engineering.py, labels_sofa.py) assumes.
CANONICAL_UNITS = {
    "heart_rate": "bpm", "sbp": "mmHg", "dbp": "mmHg", "map": "mmHg",
    "resp_rate": "insp/min", "spo2": "%", "temp": "degC",
    "lactate": "mmol/L", "wbc": "K/uL", "creatinine": "mg/dL",
    "bilirubin": "mg/dL", "platelets": "K/uL", "gcs": "score",
    "fio2": "fraction", "urine_output_rate": "mL/hr",
    "vasopressor_dose": "norepinephrine-equivalent mcg/kg/min",
    "fluid_bolus_ml": "mL", "age": "years",
}

_TEMP_C_ALIASES = {"degc", "deg. c", "c", "celsius", "°c"}
_TEMP_F_ALIASES = {"degf", "deg. f", "f", "fahrenheit", "°f"}


def convert_temperature(value: pd.Series, unit: pd.Series) -> pd.Series:
    """
    Row-wise Celsius/Fahrenheit harmonization. `unit` must be given PER ROW
    (real MIMIC-IV charts temperature in both units under different itemids —
    see concept_map.py's `temp` entry) — never assume one unit for an entire
    column. Any unit value that is not a recognized Celsius or Fahrenheit
    alias raises UnitValidationError.
    """
    unit_norm = unit.astype(str).str.strip().str.lower()
    is_c = unit_norm.isin(_TEMP_C_ALIASES)
    is_f = unit_norm.isin(_TEMP_F_ALIASES)
    unrecognized = ~(is_c | is_f) & value.notna()
    if unrecognized.any():
        bad = sorted(unit_norm[unrecognized].unique().tolist())
        raise UnitValidationError(
            f"Unrecognized/ambiguous temperature unit(s) {bad} on "
            f"{int(unrecognized.sum())} row(s) — refusing to guess a conversion. "
            f"Recognized: Celsius aliases {sorted(_TEMP_C_ALIASES)}, "
            f"Fahrenheit aliases {sorted(_TEMP_F_ALIASES)}."
        )
    out = value.copy().astype(float)
    out.loc[is_f] = (value.loc[is_f] - 32.0) * 5.0 / 9.0
    return out


def validate_and_normalize_fio2(value: pd.Series) -> pd.Series:
    """
    FiO2 is expected as a fraction (0.21-1.0). Some real-world charting
    represents it as a percentage (21-100) instead. Rather than silently
    assuming which convention a given row used, this treats any value > 1.5
    as a percentage-scale entry (a fraction can never legitimately exceed
    1.0; 1.5 leaves headroom for rounding/measurement noise right at 1.0)
    and divides by 100 — every such conversion is counted and can be audited
    via the returned mask, not applied silently in a way that's invisible.
    """
    value = value.astype(float)
    looks_like_percent = value.notna() & (value > 1.5)
    out = value.copy()
    out.loc[looks_like_percent] = value.loc[looks_like_percent] / 100.0
    still_out_of_range = out.notna() & ((out < 0.21) | (out > 1.0))
    if still_out_of_range.any():
        raise UnitValidationError(
            f"{int(still_out_of_range.sum())} FiO2 value(s) fall outside [0.21, 1.0] "
            f"even after percent-scale normalization — refusing to silently clip; "
            f"inspect these rows for a unit/charting error before proceeding."
        )
    return out, looks_like_percent


def validate_unit_column(concept: str, unit_series: pd.Series, expected_unit: str) -> None:
    """
    Generic fail-loud unit check for concepts with a single fixed expected
    unit (i.e. not the row-varying temperature/FiO2 cases above). Raises if
    any non-null unit value doesn't match `expected_unit` (case/whitespace-
    insensitive). Use for concepts where a unit mismatch means "this itemid
    is wrong" rather than "convert me" — e.g. a lab expected in mg/dL should
    never legitimately show up in a different unit; if it does, that's a
    concept-mapping error to fix in concept_map.py, not a value to convert.
    """
    present = unit_series.dropna().astype(str).str.strip().str.lower().unique()
    expected_norm = expected_unit.strip().lower()
    mismatched = [u for u in present if u != expected_norm]
    if mismatched:
        raise UnitValidationError(
            f"{concept}: found unit(s) {sorted(mismatched)} but expected only "
            f"{expected_unit!r}. This usually means the itemid mapping in "
            f"concept_map.py is pulling in more than one clinical concept, or a "
            f"unit convention this adapter doesn't yet handle — refusing to guess."
        )

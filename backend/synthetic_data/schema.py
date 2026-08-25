"""
schema.py — MIMIC-IV-inspired item dictionaries and raw table column definitions
for the synthetic data layer. See DATA_CONTRACT.md section 4 for the full
table-by-table mapping to real MIMIC-IV tables.

Item IDs here are synthetic (namespaced 90000+) and are NOT real MIMIC-IV
itemids — they exist only so chartevents/labevents can be genuinely long-format
(itemid + value), matching the real structure, rather than already-flat.
"""

from typing import Dict, List

# ─── chartevents item dictionary (vitals, GCS, resp support) ────────────────
# Mirrors mimiciv_icu.d_items in shape (itemid, label, category).
CHART_ITEMS: Dict[str, int] = {
    "heart_rate": 90001,
    "sbp": 90002,
    "dbp": 90003,
    "map": 90004,
    "resp_rate": 90005,
    "spo2": 90006,
    "temp": 90007,
    "gcs": 90008,
    "fio2": 90009,
    "ventilated": 90010,
}

CHART_ITEM_CATEGORY: Dict[str, str] = {
    "heart_rate": "Vitals",
    "sbp": "Vitals",
    "dbp": "Vitals",
    "map": "Vitals",
    "resp_rate": "Vitals",
    "spo2": "Vitals",
    "temp": "Vitals",
    "gcs": "Neurological",
    "fio2": "Respiratory",
    "ventilated": "Respiratory",
}

CHART_ITEM_UOM: Dict[str, str] = {
    "heart_rate": "bpm",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "map": "mmHg",
    "resp_rate": "insp/min",
    "spo2": "%",
    "temp": "degC",
    "gcs": "score",
    "fio2": "fraction",
    "ventilated": "boolean",
}

# ─── labevents item dictionary ───────────────────────────────────────────────
LAB_ITEMS: Dict[str, int] = {
    "lactate": 91001,
    "wbc": 91002,
    "creatinine": 91003,
    "bilirubin": 91004,
    "platelets": 91005,
}

LAB_ITEM_UOM: Dict[str, str] = {
    "lactate": "mmol/L",
    "wbc": "K/uL",
    "creatinine": "mg/dL",
    "bilirubin": "mg/dL",
    "platelets": "K/uL",
}

# ─── inputevents item dictionary (fluids + vasopressor) ──────────────────────
INPUT_ITEMS: Dict[str, int] = {
    "fluid_bolus": 92001,
    "vasopressor": 92002,
}

# ─── outputevents item dictionary ────────────────────────────────────────────
OUTPUT_ITEMS: Dict[str, int] = {
    "urine_output": 93001,
}

CHART_VARS: List[str] = list(CHART_ITEMS.keys())
LAB_VARS: List[str] = list(LAB_ITEMS.keys())


def d_items_table():
    """Build the synthetic d_items dictionary table (itemid, label, category)."""
    import pandas as pd
    rows = [
        {"itemid": itemid, "label": name, "category": CHART_ITEM_CATEGORY[name],
         "valueuom": CHART_ITEM_UOM[name]}
        for name, itemid in CHART_ITEMS.items()
    ]
    return pd.DataFrame(rows)


def d_labitems_table():
    """Build the synthetic d_labitems dictionary table (itemid, label)."""
    import pandas as pd
    rows = [
        {"itemid": itemid, "label": name, "valueuom": LAB_ITEM_UOM[name]}
        for name, itemid in LAB_ITEMS.items()
    ]
    return pd.DataFrame(rows)

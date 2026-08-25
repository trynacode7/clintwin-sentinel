"""
backend/mimic_adapter — Phase 2: the ONLY package in this codebase that is
allowed to know MIMIC-IV table names, item IDs, or MIMIC-specific quirks
(CareVue vs. MetaVision, temperature units, multi-itemid concepts, etc.).

MIMIC-IV raw tables
        |
        v
   backend.mimic_adapter   (this package)
        |
        v
canonical ClinTwin ICU schema  (identical shape to
                                 backend.synthetic_data.extract_to_ml_dataset's
                                 output — see DATA_CONTRACT.md section 3)
        |
        v
 existing preprocessing.py / feature_engineering.py / labels_sofa.py /
 dataset.py / train.py / evaluate.py  — UNCHANGED, source-agnostic

No MIMIC table name or item ID appears anywhere outside this package. See
docs/MIMIC_IV_SETUP.md for how to use it once real MIMIC-IV access exists.

WHAT THIS PACKAGE DOES NOT DO (by explicit instruction, Phase 2):
  - It does not download or fabricate MIMIC-IV data.
  - It does not guess item IDs — see concept_map.py: every mapping ships
    with an EMPTY itemids list and verified=False until a human fills it in
    against a real mimiciv_icu.d_items / mimiciv_hosp.d_labitems dictionary.
  - It has never been run against real MIMIC-IV inside this session — only
    against small, clearly-labeled MIMIC-STRUCTURE TEST FIXTURES (see
    tests/test_mimic_adapter.py).
"""

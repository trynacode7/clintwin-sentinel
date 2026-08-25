"""
conftest.py — shared fixtures for the Phase 1 synthetic-data-layer test suite.

A small, fast synthetic cohort is generated ONCE per test session (not per
test) so the full test suite runs in seconds, not minutes. Tests that need
the full generate -> extract -> validate -> preprocess -> engineer -> label
chain reuse this fixture rather than re-running generation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from backend.synthetic_data.generator import GeneratorConfig, generate_cohort
from backend.synthetic_data.extract_to_ml_dataset import extract


@pytest.fixture(scope="session")
def small_raw_cohort():
    """A small (60-patient) in-memory raw MIMIC-shaped cohort for fast tests."""
    cfg = GeneratorConfig(n_patients=60, seed=42)
    return generate_cohort(cfg)


@pytest.fixture(scope="session")
def small_ml_dataset(small_raw_cohort):
    """The flat ml_dataset.csv-equivalent DataFrame extracted from small_raw_cohort."""
    return extract(small_raw_cohort)

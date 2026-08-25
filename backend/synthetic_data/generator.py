"""
generator.py — Synthetic, MIMIC-IV-shaped ICU cohort generator.

Produces raw, long-format, event-shaped tables (patients, admissions, icustays,
chartevents, labevents, inputevents, outputevents, antibiotics,
microbiologyevents) that mirror the logical structure of the real MIMIC-IV
tables listed in DATA_CONTRACT.md section 4. A separate module,
extract_to_ml_dataset.py, flattens these into the modeling schema
(DATA_CONTRACT.md section 3) — that flattening step is the only piece that
gets rewritten when real MIMIC-IV becomes available.

============================================================================
REMEDIATION NOTE (post Phase-1-audit rewrite)
============================================================================
The original version of this generator drove every observable variable from
a SINGLE shared latent "severity" scalar per stay. An independent audit
(see the Phase 1 audit report) found this made the clinical prediction task
artificially easy: a logistic regression on a single raw observation could
recover archetype identity with ~0.90 AUROC, and ~98.5% of positive labels
came from exactly one archetype ("sepsis"), because that one latent variable
drove both the observable features (X) and, through archetype-gated
infection assignment, the label (y).

This version replaces the single latent variable with SEVEN partially
independent physiological axes (hemodynamic, respiratory, renal, hepatic,
coagulation, neurological, metabolic). Each axis gets its own randomly
drawn trajectory SHAPE (flat / gradual decline / sudden decline / transient
dip / decline-then-plateau / decline-then-relapse / gradual improve /
improve-then-decline / recover-after-intervention) and its own peak
severity, chosen via archetype-conditioned PROBABILITIES rather than
archetype-determined certainties — so within one archetype, which axes get
involved, how severely, and on what timeline, varies patient to patient.
A small amount of cross-axis blending (CROSS_AXIS_BLEND below) adds
physiologically-motivated correlation (e.g. shock contributing to lactate
and to renal hypoperfusion) without making any axis a deterministic
function of another.

Critically, `suspected_infection` (the Sepsis-3 label-eligibility signal,
see infection_events() below) is no longer a function of archetype identity
at all — it is a stochastic function of the REALIZED per-patient axis peaks,
with substantial noise. This is what allows: some "sepsis"-archetype
patients to end up with insufficient organ dysfunction or no infection flag
(no positive label), some "deteriorating"-archetype patients to end up
infected and organ-dysfunctional (a positive label), isolated single-axis
organ dysfunction, multi-axis involvement, and infection without severe
dysfunction or dysfunction without infection. Archetype is now only a PRIOR
that shapes probabilities at generation time — it is not read again by any
value formula, treatment-decision function, or the infection-decision
function. Downstream code never reads the `archetype` column as a feature
(see DATA_CONTRACT.md; it stays a synthetic-only debug column).
============================================================================

Design (see DATA_CONTRACT.md + TECHNICAL_BLUEPRINT.md section 3 for the
clinical rationale):

  - Each of the 7 axes has a hires target curve (shape function, in [0,1])
    plus its own Ornstein-Uhlenbeck mean-reverting noise deviation.
  - Renal and hepatic axes additionally pass through a slow exponentially-
    weighted moving average (organ-dysfunction lag: renal ~6h, hepatic
    ~10h) before use.
  - A small, minority-weight cross-axis blend (see CROSS_AXIS_BLEND) adds
    physiologically-motivated correlation (e.g. hemodynamic collapse
    contributing to renal hypoperfusion and to lactate) without making any
    axis a deterministic function of another.
  - Every physiological variable reads from exactly one axis's "effective"
    (blended) curve — no variable-value formula branches on `archetype`.
  - Treatments (vasopressor) and the infection-suspicion decision are
    functions of REALIZED axis state (with delay, thresholds, and
    clinician-like stochastic variation), never of archetype directly.
  - Each variable is independently, irregularly sampled (its own nominal
    interval + jitter + missingness + rare outliers), not read off a clean
    grid. Lab-draw frequency additionally depends weakly on a noisy
    composite acuity signal (sicker patients get watched a bit more
    closely) plus an independent per-stay "monitoring intensity" offset,
    so missingness is informative but not a clean proxy for the label.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import schema

ARCHETYPES = ["stable", "improving", "deteriorating", "sepsis", "recovery"]
DEFAULT_ARCHETYPE_WEIGHTS = {
    "stable": 0.35,
    "improving": 0.15,
    "deteriorating": 0.20,
    "sepsis": 0.20,
    "recovery": 0.10,
}

AXES = ["hemodynamic", "respiratory", "renal", "hepatic", "coagulation", "neuro", "metabolic"]

# Minority-weight cross-axis correlation: each axis's "effective" curve is
# (1 - sum(weights)) * its own curve + sum(weight_i * other_axis_i's curve).
# These are deliberately small relative weights — the point is plausible
# physiological correlation (shock contributing to lactate/renal/hepatic/
# respiratory derangement), not a return to one shared driver.
CROSS_AXIS_BLEND: Dict[str, List[tuple]] = {
    "metabolic": [("hemodynamic", 0.30)],
    "renal": [("hemodynamic", 0.20)],
    "respiratory": [("hemodynamic", 0.10)],
    "hepatic": [("hemodynamic", 0.10)],
    "neuro": [("hemodynamic", 0.12)],
}

TRAJECTORY_SHAPES = [
    "flat", "transient_dip", "gradual_decline", "sudden_decline",
    "decline_then_plateau", "decline_then_relapse",
    "gradual_improve", "improve_then_relapse", "improve_then_decline",
    "recover_after_intervention",
]

BASE_EPOCH = datetime(2024, 1, 1)  # fixed, arbitrary — never real wall-clock time (reproducibility)
# Per-patient timestamps are shifted well past BASE_EPOCH (subject_id * 37
# hours, subject_id starting at 100001 — see _StayGenerator.__init__ below),
# landing around year 2446. This mirrors real MIMIC-IV's own
# de-identification convention of shifting each patient's dates by a large
# random offset into the future; it is intentional, not a bug.


def _wide_range_datetime(s: pd.Series) -> pd.Series:
    """
    pandas <3.0 defaults `pd.to_datetime(..., errors="coerce")` to
    nanosecond-resolution datetime64, whose representable range ends at
    2262-04-11. Every one of this project's deliberately far-future
    (~year 2446, see BASE_EPOCH above) synthetic timestamps is OUT OF
    RANGE for that dtype — `errors="coerce"` doesn't raise on this, it
    silently turns every value into NaT instead, which then empties out
    every downstream time-indexed table (this is exactly what produced a
    0-row ml_dataset.csv when this generator was first run against
    pandas 2.3.3 outside the original pandas-3.0 development sandbox —
    see PRE_BIOGEARS_PHASE_REPORT.md's device-sync section).

    Building the column directly at microsecond resolution instead
    sidesteps the ns range limit entirely (datetime64[us] covers roughly
    year -290000 to 294000) and behaves identically on pandas 2.x and
    3.x, so the generator's output no longer depends on which pandas
    version runs it. Falls back to the original to_datetime(errors=
    "coerce") behavior for input that genuinely isn't already
    datetime-like (defensive; not expected on this codebase's own
    generator output, which always produces real datetime/None values).
    """
    try:
        return pd.Series(s.to_numpy(dtype=object), index=s.index, dtype="datetime64[us]")
    except (TypeError, ValueError):
        return pd.to_datetime(s, errors="coerce")


@dataclasses.dataclass
class GeneratorConfig:
    n_patients: int = 200
    seed: int = 42
    min_stay_hours: int = 24
    max_stay_hours: int = 120
    hires_step_minutes: int = 5
    archetype_weights: Dict[str, float] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_ARCHETYPE_WEIGHTS)
    )
    missingness_scale: float = 1.0   # multiplier on all base missingness rates
    outlier_rate: float = 0.01       # fraction of observations replaced with an outlier
    ou_theta: float = 0.35           # mean reversion speed (per hour) of severity noise
    ou_sigma: float = 0.06           # severity noise volatility


def _ou_deviation(n: int, dt_hours: float, theta: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Euler-Maruyama simulation of a mean-zero OU process (mean-reverting noise)."""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (0 - x[i - 1]) * dt_hours + sigma * np.sqrt(dt_hours) * rng.standard_normal()
    return x


def _logistic(t: np.ndarray, center: float, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(t - center) / max(scale, 1e-6)))


def _ewma_lag(x: np.ndarray, dt_hours: float, tau_hours: float) -> np.ndarray:
    """Slow exponentially-weighted moving average — models organ-dysfunction lag."""
    alpha = 1 - np.exp(-dt_hours / tau_hours)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def _sample_irregular_times(duration_h: float, nominal_interval_h: float, jitter_frac: float,
                             missing_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Generate irregular, jittered, thinned observation times (hours from stay start)."""
    n_nominal = max(int(duration_h / nominal_interval_h), 1)
    slots = np.arange(1, n_nominal + 1) * nominal_interval_h
    jitter = rng.uniform(-jitter_frac, jitter_frac, size=n_nominal) * nominal_interval_h
    times = np.clip(slots + jitter, 0.05, duration_h - 0.01)
    keep = rng.uniform(size=n_nominal) > missing_rate
    times = times[keep]
    # occasional duplicate / rapid-repeat measurement (device re-check)
    if len(times) > 0 and rng.uniform() < 0.08:
        idx = rng.integers(0, len(times))
        times = np.append(times, times[idx] + rng.uniform(0.02, 0.15))
    return np.sort(times)


def _sample_irregular_times_state_dependent(
    duration_h: float, nominal_interval_h: float, jitter_frac: float,
    base_missing_rate: float, sensitivity: float,
    severity_fn, rng: np.random.Generator,
) -> np.ndarray:
    """
    Like _sample_irregular_times, but the per-slot keep-probability shifts with a
    (noisy) composite acuity signal: sicker patients get watched somewhat more
    closely (Fix 6). Two independent noise sources keep this from being a clean
    proxy for any single label-relevant axis:
      - a per-stay "monitoring intensity" offset (unit/clinician-level variation,
        drawn once, unrelated to how sick the patient actually is)
      - per-slot noise (the same clinical state does not produce identical
        observation patterns every time)
    """
    n_nominal = max(int(duration_h / nominal_interval_h), 1)
    slots = np.arange(1, n_nominal + 1) * nominal_interval_h
    jitter = rng.uniform(-jitter_frac, jitter_frac, size=n_nominal) * nominal_interval_h
    times = np.clip(slots + jitter, 0.05, duration_h - 0.01)
    local_acuity = severity_fn(times)
    stay_monitoring_noise = rng.normal(0, 0.07)
    slot_noise = rng.normal(0, 0.05, size=n_nominal)
    missing_prob = np.clip(
        base_missing_rate - sensitivity * local_acuity + stay_monitoring_noise + slot_noise,
        0.05, 0.95,
    )
    keep = rng.uniform(size=n_nominal) > missing_prob
    times = times[keep]
    if len(times) > 0 and rng.uniform() < 0.08:
        idx = rng.integers(0, len(times))
        times = np.append(times, times[idx] + rng.uniform(0.02, 0.15))
    return np.sort(times)


def _interp(hires_t: np.ndarray, hires_v: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    return np.interp(query_t, hires_t, hires_v)


# ─── per-axis trajectory shape functions ─────────────────────────────────────
# Each takes (t_hours, duration_h, rng, lo, hi[, intervention_time_h]) and
# returns a severity curve in roughly [0, 1] (final clip applied by caller).
# `lo` = mild/baseline severity, `hi` = the more severe end for this axis on
# this stay. Which end the curve starts/ends at depends on the shape.

def _shape_flat(t, duration, rng, lo, hi):
    return np.full_like(t, lo)


def _shape_transient_dip(t, duration, rng, lo, hi):
    """Temporary abnormal reading that resolves on its own — no sustained dysfunction."""
    onset = rng.uniform(0.15, 0.70) * duration
    rise_scale = rng.uniform(1.0, 4.0)
    hold = rng.uniform(2.0, 8.0)
    fall_scale = rng.uniform(2.0, 6.0)
    up = _logistic(t, center=onset, scale=rise_scale)
    down = _logistic(t, center=onset + hold, scale=fall_scale)
    frac = np.clip(up - down, 0.0, 1.0)
    return lo + (hi - lo) * frac


def _shape_gradual_decline(t, duration, rng, lo, hi):
    center = duration * rng.uniform(0.35, 0.70)
    scale = duration * rng.uniform(0.10, 0.22)
    return lo + (hi - lo) * _logistic(t, center=center, scale=scale)


def _shape_sudden_decline(t, duration, rng, lo, hi):
    onset = rng.uniform(0.20, 0.80) * duration
    scale = rng.uniform(0.5, 2.5)
    return lo + (hi - lo) * _logistic(t, center=onset, scale=scale)


def _shape_decline_then_plateau(t, duration, rng, lo, hi):
    """Deterioration followed by a treatment-like partial response (not full recovery)."""
    onset = duration * rng.uniform(0.15, 0.40)
    rise_scale = duration * rng.uniform(0.06, 0.14)
    resp_delay = rng.uniform(4.0, 14.0)
    plateau = lo + (hi - lo) * rng.uniform(0.20, 0.55)
    fall_scale = duration * rng.uniform(0.08, 0.16)
    rise = _logistic(t, center=onset, scale=rise_scale)
    response = _logistic(t, center=onset + resp_delay, scale=fall_scale)
    curve = lo + (hi - lo) * rise - (hi - plateau) * response
    return np.maximum(curve, min(lo, plateau))


def _shape_decline_then_relapse(t, duration, rng, lo, hi):
    """Worsens, partially improves, then worsens again."""
    onset1 = duration * rng.uniform(0.12, 0.32)
    scale1 = duration * rng.uniform(0.06, 0.12)
    dip_delay = rng.uniform(6.0, 16.0)
    dip_scale = duration * rng.uniform(0.06, 0.12)
    relapse_delay = rng.uniform(8.0, 20.0)
    relapse_scale = duration * rng.uniform(0.06, 0.14)
    mid = lo + (hi - lo) * rng.uniform(0.25, 0.50)
    hi2 = mid + (hi - mid) * rng.uniform(0.6, 1.05)
    rise1 = _logistic(t, center=onset1, scale=scale1)
    fall1 = _logistic(t, center=onset1 + dip_delay, scale=dip_scale)
    rise2 = _logistic(t, center=onset1 + dip_delay + relapse_delay, scale=relapse_scale)
    curve = lo + (hi - lo) * rise1 - (hi - mid) * fall1 + (hi2 - mid) * rise2
    return np.clip(curve, min(lo, mid), max(hi, hi2))


def _shape_gradual_improve(t, duration, rng, lo, hi):
    """Starts elevated (hi), improves toward lo over the stay."""
    center = duration * rng.uniform(0.25, 0.45)
    scale = duration * rng.uniform(0.14, 0.22)
    return hi + (lo - hi) * _logistic(t, center=center, scale=scale)


def _shape_improve_then_relapse(t, duration, rng, lo, hi):
    """Starts elevated, improves, then deteriorates again later (relapse after apparent recovery)."""
    onset1 = duration * rng.uniform(0.10, 0.25)
    scale1 = duration * rng.uniform(0.08, 0.14)
    mid = lo + (hi - lo) * rng.uniform(0.10, 0.35)
    relapse_delay = rng.uniform(10.0, 24.0)
    relapse_scale = duration * rng.uniform(0.08, 0.16)
    hi2 = mid + (hi - mid) * rng.uniform(0.5, 1.0)
    fall = _logistic(t, center=onset1, scale=scale1)
    rise = _logistic(t, center=onset1 + relapse_delay, scale=relapse_scale)
    curve = hi + (mid - hi) * fall + (hi2 - mid) * rise
    return np.clip(curve, min(lo, mid), max(hi, hi2))


def _shape_improve_then_decline(t, duration, rng, lo, hi):
    """Looks stable/mild for a while, then a later-onset deterioration (delayed decompensation)."""
    onset = duration * rng.uniform(0.45, 0.78)
    scale = duration * rng.uniform(0.06, 0.14)
    return lo + (hi - lo) * _logistic(t, center=onset, scale=scale)


def _shape_recover_after_intervention(t, duration, rng, lo, hi, intervention_time_h):
    """Elevated until an explicit intervention time, then declines (possibly incompletely)."""
    onset = duration * rng.uniform(0.05, 0.20)
    rise_scale = duration * rng.uniform(0.05, 0.10)
    recovery_target = lo + (hi - lo) * rng.uniform(0.0, 0.35)  # partial recovery for some axes
    delay = rng.uniform(1.0, 5.0)
    fall_scale = duration * rng.uniform(0.08, 0.16)
    rise = lo + (hi - lo) * _logistic(t, center=onset, scale=rise_scale)
    fall = recovery_target + (hi - recovery_target) * (1 - _logistic(t, center=intervention_time_h + delay, scale=fall_scale))
    return np.minimum(rise, np.maximum(fall, recovery_target))


_SHAPE_FNS = {
    "flat": _shape_flat,
    "transient_dip": _shape_transient_dip,
    "gradual_decline": _shape_gradual_decline,
    "sudden_decline": _shape_sudden_decline,
    "decline_then_plateau": _shape_decline_then_plateau,
    "decline_then_relapse": _shape_decline_then_relapse,
    "gradual_improve": _shape_gradual_improve,
    "improve_then_relapse": _shape_improve_then_relapse,
    "improve_then_decline": _shape_improve_then_decline,
    "recover_after_intervention": _shape_recover_after_intervention,
}


# ─── archetype -> per-axis involvement PROBABILITIES (not certainties) ──────
# "priority" axes for an archetype get a higher chance of being involved and
# a wider severity range; all other axes still get a small chance of
# involvement (isolated/atypical presentations) or a "transient_dip" blip.
# This table is read ONLY when building each patient's axis profiles — no
# variable-value formula or treatment/infection decision reads archetype.
_PRIORITY_AXES = {
    "stable": [],
    "improving": ["hemodynamic", "respiratory", "metabolic"],
    "deteriorating": [],  # deliberately random per patient — see _build_axis_profiles
    "sepsis": ["hemodynamic", "respiratory", "metabolic"],
    "recovery": ["hemodynamic", "respiratory", "renal"],
}


def _build_axis_profiles(archetype: str, duration_h: float, rng: np.random.Generator) -> Dict[str, dict]:
    """
    Decide, per axis, whether this patient's axis is "involved" and with what
    shape/severity — probabilistically, conditioned on archetype but never
    determined by it. Returns {axis: {"shape":..., "lo":..., "hi":..., ...}}.
    """
    profiles: Dict[str, dict] = {}

    if archetype == "stable":
        for axis in AXES:
            lo = float(rng.uniform(0.03, 0.15))
            if rng.uniform() < 0.10:
                hi = float(rng.uniform(0.30, 0.55))
                profiles[axis] = {"shape": "transient_dip", "lo": lo, "hi": hi}
            else:
                profiles[axis] = {"shape": "flat", "lo": lo, "hi": lo}

    elif archetype == "improving":
        priority = set(_PRIORITY_AXES["improving"])
        for axis in AXES:
            p_involved = 0.55 if axis in priority else 0.28
            lo = float(rng.uniform(0.05, 0.18))
            if rng.uniform() < p_involved:
                hi = float(rng.uniform(0.42, 0.75))
                if rng.uniform() < 0.18:
                    profiles[axis] = {"shape": "improve_then_relapse", "lo": lo, "hi": hi}
                else:
                    profiles[axis] = {"shape": "gradual_improve", "lo": lo, "hi": hi}
            else:
                profiles[axis] = {"shape": "flat", "lo": lo, "hi": lo}

    elif archetype == "deteriorating":
        # Randomly select 1-3 axes to involve (uniform across axes) — this is
        # exactly what makes isolated single-organ and mixed multi-organ
        # presentations both possible without hardcoding which axis it is.
        k = int(rng.integers(1, 4))
        involved_axes = set(rng.choice(AXES, size=k, replace=False))
        shape_pool = ["gradual_decline", "sudden_decline", "decline_then_plateau", "decline_then_relapse",
                      "improve_then_decline"]
        shape_p = [0.34, 0.18, 0.20, 0.14, 0.14]
        for axis in AXES:
            lo = float(rng.uniform(0.04, 0.18))
            if axis in involved_axes:
                hi = float(rng.uniform(0.42, 0.88))
                shape = str(rng.choice(shape_pool, p=shape_p))
                profiles[axis] = {"shape": shape, "lo": lo, "hi": hi}
            elif rng.uniform() < 0.08:
                hi = float(rng.uniform(0.30, 0.50))
                profiles[axis] = {"shape": "transient_dip", "lo": lo, "hi": hi}
            else:
                profiles[axis] = {"shape": "flat", "lo": lo, "hi": lo}

    elif archetype == "sepsis":
        # tiered involvement probability — "typical" septic axes are LIKELY
        # but never certain to be involved, and severity varies widely, which
        # is what allows some sepsis-archetype patients to fall short of the
        # SOFA-increase threshold entirely.
        p_involved = {
            "hemodynamic": 0.78, "respiratory": 0.68, "metabolic": 0.80,
            "renal": 0.52, "coagulation": 0.42, "hepatic": 0.38, "neuro": 0.30,
        }
        shape_pool = ["gradual_decline", "sudden_decline", "decline_then_plateau", "decline_then_relapse"]
        shape_p = [0.32, 0.23, 0.28, 0.17]
        for axis in AXES:
            lo = float(rng.uniform(0.05, 0.20))
            if rng.uniform() < p_involved[axis]:
                hi = float(rng.uniform(0.50, 0.97))
                shape = str(rng.choice(shape_pool, p=shape_p))
                profiles[axis] = {"shape": shape, "lo": lo, "hi": hi}
            elif rng.uniform() < 0.10:
                hi = float(rng.uniform(0.30, 0.55))
                profiles[axis] = {"shape": "transient_dip", "lo": lo, "hi": hi}
            else:
                profiles[axis] = {"shape": "flat", "lo": lo, "hi": lo}

    elif archetype == "recovery":
        priority = set(_PRIORITY_AXES["recovery"])
        intervention_time_h = float(rng.uniform(0.10, 0.30) * duration_h)
        for axis in AXES:
            p_involved = 0.62 if axis in priority else 0.28
            lo = float(rng.uniform(0.05, 0.20))
            if rng.uniform() < p_involved:
                hi = float(rng.uniform(0.48, 0.85))
                profiles[axis] = {"shape": "recover_after_intervention", "lo": lo, "hi": hi,
                                   "intervention_time_h": intervention_time_h}
            else:
                profiles[axis] = {"shape": "flat", "lo": lo, "hi": lo}
        profiles["_intervention_time_h"] = intervention_time_h  # stashed for fluid-bolus reuse

    else:
        raise ValueError(f"unknown archetype {archetype}")

    return profiles


class _StayGenerator:
    """Generates all raw event rows for a single ICU stay."""

    def __init__(self, subject_id: int, hadm_id: int, stay_id: int, age: float, sex_male: int,
                 archetype: str, cfg: GeneratorConfig, rng: np.random.Generator):
        self.subject_id = subject_id
        self.hadm_id = hadm_id
        self.stay_id = stay_id
        self.age = age
        self.sex_male = sex_male
        self.archetype = archetype
        self.cfg = cfg
        self.rng = rng

        self.duration_h = float(rng.uniform(cfg.min_stay_hours, cfg.max_stay_hours))
        self.intime = BASE_EPOCH + timedelta(hours=float(subject_id * 37 + rng.integers(0, 24)))

        # Axis profiles are drawn BEFORE duration is finalized so a modest,
        # realized-severity-driven (not archetype-driven) stay-length stretch
        # can be applied below.
        profiles = _build_axis_profiles(archetype, self.duration_h, rng)
        self.intervention_time_h = profiles.pop("_intervention_time_h", None)
        self.axis_profiles = profiles

        any_involved = any(p["shape"] != "flat" for p in profiles.values())
        if any_involved:
            self.duration_h *= float(rng.uniform(1.0, 1.25))
        self.outtime = self.intime + timedelta(hours=self.duration_h)

        dt_h = cfg.hires_step_minutes / 60.0
        self.hires_t = np.arange(0, self.duration_h, dt_h)
        n = len(self.hires_t)

        # Build each axis's own hires curve (shape + independent OU noise),
        # applying organ-dysfunction lag (EWMA) for renal/hepatic, THEN the
        # small cross-axis blend, in that fixed order for reproducibility.
        self.axis_own: Dict[str, np.ndarray] = {}
        for axis in AXES:
            prof = self.axis_profiles[axis]
            shape_fn = _SHAPE_FNS[prof["shape"]]
            if prof["shape"] == "recover_after_intervention":
                curve = shape_fn(self.hires_t, self.duration_h, rng, prof["lo"], prof["hi"],
                                  prof["intervention_time_h"])
            else:
                curve = shape_fn(self.hires_t, self.duration_h, rng, prof["lo"], prof["hi"])
            deviation = _ou_deviation(n, dt_h, cfg.ou_theta, cfg.ou_sigma, rng)
            curve = np.clip(curve + deviation, 0.0, 1.0)
            if axis == "renal":
                curve = _ewma_lag(curve, dt_h, tau_hours=6.0)
            elif axis == "hepatic":
                curve = _ewma_lag(curve, dt_h, tau_hours=10.0)
            self.axis_own[axis] = curve

        self.axis_effective: Dict[str, np.ndarray] = {}
        for axis in AXES:
            own = self.axis_own[axis]
            blend = CROSS_AXIS_BLEND.get(axis, [])
            if not blend:
                self.axis_effective[axis] = own
                continue
            other_w = sum(w for _, w in blend)
            out = (1.0 - other_w) * own
            for other_axis, w in blend:
                out = out + w * self.axis_own[other_axis]
            self.axis_effective[axis] = np.clip(out, 0.0, 1.0)

        self.axis_peak: Dict[str, float] = {ax: float(np.max(self.axis_effective[ax])) for ax in AXES}

        # per-patient idiosyncratic flags — tied to REALIZED axis state, not archetype
        self.cold_shock = self.axis_peak["hemodynamic"] > 0.65 and rng.uniform() < 0.12
        self.leukopenic = self.axis_peak["metabolic"] > 0.60 and rng.uniform() < 0.15
        self.vaso_threshold = rng.uniform(0.42, 0.65)

    def _axis_at(self, axis: str, t: np.ndarray) -> np.ndarray:
        return _interp(self.hires_t, self.axis_effective[axis], t)

    def _composite_acuity(self, t: np.ndarray) -> np.ndarray:
        """Noisy aggregate across all 7 axes — used ONLY to calibrate lab-draw
        frequency (Fix 6) and infection risk (see infection_events()); never
        used as a driver of any individual variable's value."""
        vals = np.stack([self._axis_at(ax, t) for ax in AXES], axis=0)
        return vals.mean(axis=0)

    def _outlier_mask(self, n: int) -> np.ndarray:
        return self.rng.uniform(size=n) < (self.cfg.outlier_rate)

    def _charttime(self, hours: np.ndarray) -> List[datetime]:
        return [self.intime + timedelta(hours=float(h)) for h in hours]

    # ── individual variable generators ───────────────────────────────────
    def _gen_chart_var(self, name: str, axis: str, nominal_h: float, jitter: float, base_missing: float,
                        value_fn) -> pd.DataFrame:
        missing = np.clip(base_missing * self.cfg.missingness_scale, 0.0, 0.95)
        t = _sample_irregular_times(self.duration_h, nominal_h, jitter, missing, self.rng)
        if len(t) == 0:
            return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"])
        sev = self._axis_at(axis, t)
        vals = value_fn(t, sev)
        out_mask = self._outlier_mask(len(t))
        if out_mask.any():
            vals = vals.copy()
            spike = self.rng.choice([-1, 1], size=out_mask.sum()) * self.rng.uniform(2.5, 5.0, size=out_mask.sum())
            vals[out_mask] = vals[out_mask] + spike * (np.nanstd(vals) + 1e-6)
        return pd.DataFrame({
            "subject_id": self.subject_id, "hadm_id": self.hadm_id, "stay_id": self.stay_id,
            "charttime": self._charttime(t), "itemid": schema.CHART_ITEMS[name],
            "valuenum": vals, "valueuom": schema.CHART_ITEM_UOM[name],
        })

    def heart_rate(self):
        return self._gen_chart_var("heart_rate", "hemodynamic", 1.0, 0.25, 0.12,
                                    lambda t, s: 78 + 55 * s + self.rng.normal(0, 5, len(t)))

    def blood_pressure(self):
        sbp_df = self._gen_chart_var("sbp", "hemodynamic", 1.0, 0.25, 0.15,
                                      lambda t, s: 122 - 45 * s + self.rng.normal(0, 7, len(t)))
        dbp_df = self._gen_chart_var("dbp", "hemodynamic", 1.0, 0.25, 0.15,
                                      lambda t, s: 72 - 22 * s + self.rng.normal(0, 5, len(t)))
        map_df = self._gen_chart_var("map", "hemodynamic", 1.0, 0.25, 0.10,
                                      lambda t, s: 85 - 40 * s + self.rng.normal(0, 5, len(t)))
        return pd.concat([sbp_df, dbp_df, map_df], ignore_index=True)

    def resp_rate(self):
        return self._gen_chart_var("resp_rate", "respiratory", 1.0, 0.25, 0.15,
                                    lambda t, s: 15 + 18 * s + self.rng.normal(0, 2, len(t)))

    def spo2(self):
        return self._gen_chart_var("spo2", "respiratory", 1.0, 0.25, 0.15,
                                    lambda t, s: np.clip(98 - 20 * s + self.rng.normal(0, 1.5, len(t)), 55, 100))

    def temp(self):
        def fn(t, s):
            delta = 1.8 * s
            if self.cold_shock:
                severe = s > 0.75
                delta = np.where(severe, -3.0 * (s - 0.75), delta)
            return 36.7 + delta + self.rng.normal(0, 0.25, len(t))
        return self._gen_chart_var("temp", "metabolic", 2.0, 0.3, 0.20, fn)

    def gcs(self):
        def fn(t, s):
            val = 15 - 11.0 * s + self.rng.normal(0, 0.6, len(t))
            return np.clip(np.round(val), 3, 15)
        return self._gen_chart_var("gcs", "neuro", 6.0, 0.3, 0.35, fn)

    def resp_support(self):
        """FiO2 + ventilated status — only charted while relevant (mirrors real charting)."""
        t = _sample_irregular_times(self.duration_h, 1.0, 0.25, 0.10, self.rng)
        if len(t) == 0:
            empty = pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"])
            return empty, empty
        sev = self._axis_at("respiratory", t)
        spo2_true = np.clip(98 - 20 * sev, 55, 100)
        ventilated = (spo2_true < 88).astype(float)
        # once ventilated, stays ventilated for a run of subsequent low-spo2 periods (simple persistence)
        for i in range(1, len(ventilated)):
            if spo2_true[i] < 92 and ventilated[i - 1] == 1:
                ventilated[i] = 1
        needs_o2 = spo2_true < 94
        fio2 = np.where(needs_o2, np.clip(0.21 + 0.6 * sev, 0.21, 1.0), np.nan)
        vent_df = pd.DataFrame({
            "subject_id": self.subject_id, "hadm_id": self.hadm_id, "stay_id": self.stay_id,
            "charttime": self._charttime(t), "itemid": schema.CHART_ITEMS["ventilated"],
            "valuenum": ventilated, "valueuom": "boolean",
        })
        fio2_df = pd.DataFrame({
            "subject_id": self.subject_id, "hadm_id": self.hadm_id, "stay_id": self.stay_id,
            "charttime": self._charttime(t), "itemid": schema.CHART_ITEMS["fio2"],
            "valuenum": fio2, "valueuom": "fraction",
        }).dropna(subset=["valuenum"])
        return vent_df, fio2_df

    # ── labs ────────────────────────────────────────────────────────────
    def _gen_lab_var(self, name: str, axis: str, nominal_h: float, base_missing: float, value_fn,
                      report_delay_h: float = 0.0, sensitivity: float = 0.30) -> pd.DataFrame:
        missing = np.clip(base_missing * self.cfg.missingness_scale, 0.0, 0.95)
        t = _sample_irregular_times_state_dependent(
            self.duration_h, nominal_h, 0.3, missing, sensitivity, self._composite_acuity, self.rng,
        )
        if len(t) == 0:
            return pd.DataFrame(columns=["subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom"])
        sev = self._axis_at(axis, t)
        vals = value_fn(t, sev)
        # occasional delayed charting (result posted hours after draw)
        delay = self.rng.exponential(report_delay_h, size=len(t)) if report_delay_h > 0 else np.zeros(len(t))
        charted_t = np.clip(t + delay, 0, self.duration_h - 0.001)
        return pd.DataFrame({
            "subject_id": self.subject_id, "hadm_id": self.hadm_id,
            "charttime": self._charttime(charted_t), "itemid": schema.LAB_ITEMS[name],
            "valuenum": vals, "valueuom": schema.LAB_ITEM_UOM[name],
        })

    def lactate(self):
        return self._gen_lab_var("lactate", "metabolic", 6.0, 0.55,
                                  lambda t, s: np.clip(1.0 + 8.0 * (s ** 1.5) + self.rng.normal(0, 0.3, len(t)), 0.3, 20),
                                  report_delay_h=0.5)

    def wbc(self):
        def fn(t, s):
            base = 8.0 + 10.0 * s
            if self.leukopenic:
                base = np.where(s > 0.6, 8.0 - 6.0 * (s - 0.6) / 0.4, base)
            return np.clip(base + self.rng.normal(0, 1.2, len(t)), 0.2, 45)
        return self._gen_lab_var("wbc", "metabolic", 12.0, 0.5, fn, report_delay_h=1.0)

    def creatinine(self):
        def fn(t, s):
            return np.clip(0.9 + 2.4 * s + self.rng.normal(0, 0.12, len(t)), 0.1, 15)
        return self._gen_lab_var("creatinine", "renal", 12.0, 0.45, fn, report_delay_h=1.0)

    def bilirubin(self):
        def fn(t, s):
            return np.clip(0.6 + 3.2 * s + self.rng.normal(0, 0.08, len(t)), 0.1, 30)
        return self._gen_lab_var("bilirubin", "hepatic", 20.0, 0.6, fn, report_delay_h=2.0)

    def platelets(self):
        return self._gen_lab_var("platelets", "coagulation", 20.0, 0.55,
                                  lambda t, s: np.clip(235 - 190.0 * s + self.rng.normal(0, 15, len(t)), 5, 600),
                                  report_delay_h=1.5)

    # ── outputs / treatments ───────────────────────────────────────────
    def urine_output(self) -> pd.DataFrame:
        missing = np.clip(0.35 * self.cfg.missingness_scale, 0, 0.9)
        t = _sample_irregular_times_state_dependent(
            self.duration_h, 1.0, 0.2, missing, 0.20, self._composite_acuity, self.rng,
        )
        if len(t) == 0:
            return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"])
        sev = self._axis_at("renal", t)
        vals = np.clip(65 - 58.0 * sev + self.rng.normal(0, 8, len(t)), 2, 500)
        return pd.DataFrame({
            "subject_id": self.subject_id, "hadm_id": self.hadm_id, "stay_id": self.stay_id,
            "charttime": self._charttime(t), "itemid": schema.OUTPUT_ITEMS["urine_output"], "value": vals,
        })

    def vasopressor(self) -> pd.DataFrame:
        """
        Vasopressor use as a (noisy, delayed) clinical RESPONSE to the
        hemodynamic axis crossing a per-patient threshold — not an archetype
        lookup. Multiple episodes (escalation / withdrawal / re-escalation on
        relapse) are possible; some severely-affected patients are never
        treated (Fix 5) and some transient blips can trigger a brief episode.
        """
        empty = pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount", "rate"])
        peak_h = self.axis_peak["hemodynamic"]
        p_treated = float(np.clip(0.28 + 0.60 * peak_h, 0.08, 0.92))
        if peak_h <= self.vaso_threshold or self.rng.uniform() >= p_treated:
            return empty

        t = np.arange(0.25, self.duration_h, 0.5)
        hemo = self._axis_at("hemodynamic", t)
        above = hemo > self.vaso_threshold
        withdraw_thresh = max(self.vaso_threshold - 0.12, 0.05)
        below_withdraw = hemo < withdraw_thresh

        episodes = []
        in_episode = False
        start_idx = None
        n = len(t)
        i = 0
        while i < n:
            if not in_episode and above[i]:
                in_episode = True
                start_idx = i
            elif in_episode and below_withdraw[i] and (i + 1 >= n or below_withdraw[i + 1]):
                episodes.append((start_idx, i))
                in_episode = False
            i += 1
        if in_episode:
            episodes.append((start_idx, n - 1))
        episodes = [(s, e) for s, e in episodes if (e - s) >= 1]  # drop sub-1h noise blips

        if not episodes:
            return empty

        rows = []
        for start_idx, end_idx in episodes:
            response_delay = float(self.rng.uniform(0.5, 3.0))
            withdraw_delay = float(self.rng.uniform(1.0, 4.0))
            start_h = t[start_idx] + response_delay
            end_h = min(self.duration_h - 0.01, t[end_idx] + withdraw_delay)
            if start_h >= end_h:
                continue
            seg_t = t[(t >= start_h) & (t <= end_h)]
            if len(seg_t) < 2:
                continue
            seg_sev = self._axis_at("hemodynamic", seg_t)
            for j in range(len(seg_t) - 1):
                dose = float(np.clip(0.05 + 0.35 * (seg_sev[j] - self.vaso_threshold), 0.01, 1.0))
                rows.append({
                    "subject_id": self.subject_id, "hadm_id": self.hadm_id, "stay_id": self.stay_id,
                    "starttime": self.intime + timedelta(hours=float(seg_t[j])),
                    "endtime": self.intime + timedelta(hours=float(seg_t[j + 1])),
                    "itemid": schema.INPUT_ITEMS["vasopressor"], "amount": np.nan, "rate": dose,
                })
        return pd.DataFrame(rows) if rows else empty

    def fluid_boluses(self) -> pd.DataFrame:
        """Bolus frequency depends on realized hemodynamic/renal severity, not archetype."""
        rate = float(0.35 + 2.4 * max(self.axis_peak["hemodynamic"], 0.6 * self.axis_peak["renal"]))
        n_events = self.rng.poisson(rate)
        rows = []
        for _ in range(n_events):
            h = float(self.rng.uniform(0, max(self.duration_h - 0.1, 0.1)))
            amt = float(self.rng.choice([250, 500, 500, 1000]))
            rows.append({
                "subject_id": self.subject_id, "hadm_id": self.hadm_id, "stay_id": self.stay_id,
                "starttime": self.intime + timedelta(hours=h),
                "endtime": self.intime + timedelta(hours=h + 0.5),
                "itemid": schema.INPUT_ITEMS["fluid_bolus"], "amount": amt, "rate": np.nan,
            })
        # a resuscitation bolus near the intervention time, IF this stay actually
        # has a "recover_after_intervention" hemodynamic or renal axis (not a
        # blanket "archetype == recovery" rule)
        hemo_shape = self.axis_profiles["hemodynamic"]["shape"]
        renal_shape = self.axis_profiles["renal"]["shape"]
        if self.intervention_time_h is not None and "recover_after_intervention" in (hemo_shape, renal_shape):
            h = self.intervention_time_h
            rows.append({
                "subject_id": self.subject_id, "hadm_id": self.hadm_id, "stay_id": self.stay_id,
                "starttime": self.intime + timedelta(hours=h),
                "endtime": self.intime + timedelta(hours=h + 0.5),
                "itemid": schema.INPUT_ITEMS["fluid_bolus"], "amount": 1000.0, "rate": np.nan,
            })
        if not rows:
            return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount", "rate"])
        return pd.DataFrame(rows)

    # ── suspected-infection evidence (Sepsis-3 cohort construction inputs) ─
    def infection_events(self):
        """
        Returns (antibiotics_rows, microbiology_rows).

        Whether a stay gets a qualifying antibiotic+culture pair is a
        STOCHASTIC function of the REALIZED per-patient axis peaks (with
        substantial added noise) — never a function of `archetype`. This is
        the central fix for the audit's "same generation logic creates both
        X and y" finding: two patients drawn from the same archetype can end
        up on opposite sides of this decision if their randomly-realized
        axis severities differ, and patients from different archetypes can
        land on the same side.
        """
        risk = (0.32 * self.axis_peak["hemodynamic"] + 0.28 * self.axis_peak["respiratory"]
                + 0.28 * self.axis_peak["metabolic"] + 0.12 * self.axis_peak["renal"])
        risk += float(self.rng.normal(0, 0.10))
        p_infection = float(np.clip(1.0 / (1.0 + np.exp(-8.0 * (risk - 0.42))), 0.02, 0.93))

        abx_rows, micro_rows = [], []
        if self.rng.uniform() < p_infection:
            onset_h = float(np.clip(self.rng.uniform(0.10, 0.55) * self.duration_h, 0.5, max(self.duration_h - 1, 0.5)))
            abx_time = onset_h + float(self.rng.uniform(0.5, 5.0))
            culture_offset = onset_h + float(self.rng.uniform(-3.0, 4.0))
            abx_rows.append({"subject_id": self.subject_id, "hadm_id": self.hadm_id,
                              "starttime": self.intime + timedelta(hours=min(abx_time, self.duration_h - 0.01))})
            micro_rows.append({"subject_id": self.subject_id, "hadm_id": self.hadm_id,
                                "charttime": self.intime + timedelta(hours=float(np.clip(culture_offset, 0.0, self.duration_h - 0.01)))})
        elif self.rng.uniform() < 0.06:
            # incidental culture only (no antibiotics) — never qualifies as suspected infection,
            # but keeps antibiotics/cultures from perfectly complementing p_infection
            h = float(self.rng.uniform(0.1, 0.9) * self.duration_h)
            micro_rows.append({"subject_id": self.subject_id, "hadm_id": self.hadm_id,
                                "charttime": self.intime + timedelta(hours=h)})
        return pd.DataFrame(abx_rows), pd.DataFrame(micro_rows)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        vent_df, fio2_df = self.resp_support()
        chart = pd.concat([
            self.heart_rate(), self.blood_pressure(), self.resp_rate(), self.spo2(),
            self.temp(), self.gcs(), vent_df, fio2_df,
        ], ignore_index=True)
        labs = pd.concat([
            self.lactate(), self.wbc(), self.creatinine(), self.bilirubin(), self.platelets(),
        ], ignore_index=True)
        abx, micro = self.infection_events()
        return {
            "chartevents": chart,
            "labevents": labs,
            "outputevents": self.urine_output(),
            "inputevents": pd.concat([self.vasopressor(), self.fluid_boluses()], ignore_index=True),
            "antibiotics": abx,
            "microbiologyevents": micro,
        }


def generate_cohort(cfg: GeneratorConfig) -> Dict[str, pd.DataFrame]:
    """Generate a full synthetic cohort. Returns a dict of raw, MIMIC-shaped tables."""
    rng = np.random.default_rng(cfg.seed)
    archetype_names = list(cfg.archetype_weights.keys())
    archetype_p = np.array([cfg.archetype_weights[a] for a in archetype_names])
    archetype_p = archetype_p / archetype_p.sum()

    patients_rows, admissions_rows, icustays_rows = [], [], []
    chart_parts, lab_parts, out_parts, in_parts, abx_parts, micro_parts = [], [], [], [], [], []

    for i in range(1, cfg.n_patients + 1):
        subject_id = 100000 + i
        hadm_id = subject_id * 10 + 1
        stay_id = subject_id * 100 + 1
        age = float(np.clip(rng.normal(63, 17), 18, 96))
        sex_male = int(rng.uniform() < 0.55)
        archetype = str(rng.choice(archetype_names, p=archetype_p))

        stay_rng = np.random.default_rng(cfg.seed * 1_000_003 + i)
        gen = _StayGenerator(subject_id, hadm_id, stay_id, age, sex_male, archetype, cfg, stay_rng)
        tables = gen.generate_all()

        patients_rows.append({"subject_id": subject_id, "age": age, "sex_male": sex_male})
        admissions_rows.append({"subject_id": subject_id, "hadm_id": hadm_id,
                                 "admittime": gen.intime - timedelta(hours=float(stay_rng.uniform(0, 6))),
                                 "dischtime": gen.outtime + timedelta(hours=float(stay_rng.uniform(0, 24)))})
        icustays_rows.append({"subject_id": subject_id, "hadm_id": hadm_id, "stay_id": stay_id,
                               "intime": gen.intime, "outtime": gen.outtime, "archetype": archetype})

        for name, part_list in [("chartevents", chart_parts), ("labevents", lab_parts),
                                 ("outputevents", out_parts), ("inputevents", in_parts),
                                 ("antibiotics", abx_parts), ("microbiologyevents", micro_parts)]:
            df = tables[name]
            if len(df):
                part_list.append(df)

    def _cat(parts, cols):
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)

    def _coerce_dtypes(df: pd.DataFrame, numeric_cols=(), datetime_cols=(), int_cols=()) -> pd.DataFrame:
        # Per-stay tables are individually well-typed, but pd.concat()-ing many
        # per-stay frames — some of which may be the empty, default-object-dtype
        # fallback returned when a stay has zero irregular samples for a given
        # variable (see _gen_lab_var and similar) — can silently upcast a
        # genuinely numeric/datetime column to dtype "object" for the WHOLE
        # concatenated table, even though every individual value is still
        # numeric/datetime under the hood (pandas' historical empty-frame
        # concat dtype behavior). Left uncorrected, this breaks
        # pivot_table(aggfunc="mean") and downstream .mean(numeric_only=True)
        # calls in extract_to_ml_dataset.py, which silently DROP the affected
        # columns instead of raising. Coercing explicitly here, once, at the
        # single point where every raw table is finalized, is the smallest
        # fix that closes this off for every table and every future variable
        # added to any of them.
        if df.empty:
            return df
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in datetime_cols:
            if c in df.columns:
                df[c] = _wide_range_datetime(df[c])
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
        return df

    chartevents = _coerce_dtypes(
        _cat(chart_parts, ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"]),
        numeric_cols=["valuenum"], datetime_cols=["charttime"], int_cols=["subject_id", "hadm_id", "stay_id", "itemid"],
    )
    labevents = _coerce_dtypes(
        _cat(lab_parts, ["subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom"]),
        numeric_cols=["valuenum"], datetime_cols=["charttime"], int_cols=["subject_id", "hadm_id", "itemid"],
    )
    outputevents = _coerce_dtypes(
        _cat(out_parts, ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"]),
        numeric_cols=["value"], datetime_cols=["charttime"], int_cols=["subject_id", "hadm_id", "stay_id", "itemid"],
    )
    inputevents = _coerce_dtypes(
        _cat(in_parts, ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount", "rate"]),
        numeric_cols=["amount", "rate"], datetime_cols=["starttime", "endtime"],
        int_cols=["subject_id", "hadm_id", "stay_id", "itemid"],
    )
    antibiotics = _coerce_dtypes(
        _cat(abx_parts, ["subject_id", "hadm_id", "starttime"]),
        datetime_cols=["starttime"], int_cols=["subject_id", "hadm_id"],
    )
    microbiologyevents = _coerce_dtypes(
        _cat(micro_parts, ["subject_id", "hadm_id", "charttime"]),
        datetime_cols=["charttime"], int_cols=["subject_id", "hadm_id"],
    )

    admissions_df = pd.DataFrame(admissions_rows)
    icustays_df = pd.DataFrame(icustays_rows)
    for col in ("admittime", "dischtime"):
        admissions_df[col] = _wide_range_datetime(admissions_df[col])
    for col in ("intime", "outtime"):
        icustays_df[col] = _wide_range_datetime(icustays_df[col])

    return {
        "patients": pd.DataFrame(patients_rows),
        "admissions": admissions_df,
        "icustays": icustays_df,
        "d_items": schema.d_items_table(),
        "d_labitems": schema.d_labitems_table(),
        "chartevents": chartevents,
        "labevents": labevents,
        "outputevents": outputevents,
        "inputevents": inputevents,
        "antibiotics": antibiotics,
        "microbiologyevents": microbiologyevents,
    }

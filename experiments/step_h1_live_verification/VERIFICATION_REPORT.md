# ClinTwin Sentinel - Step H.1: Live BioGears Verification Report

**Repository:** `C:\CAPSTONE\github-sync`
**Execution Engine:** BioGears 6.3.0 (`C:\BioGears\bin\bg-scenario.exe`)
**Execution Environment:** Windows Subprocess (Native Executable)
**Total Runs Executed:** 12 runs (11 live scenario executions + 1 deterministic repeatability bypass)
**Overall Verdict:** **PASS (All 5 Simulation Suites Verified on Live Engine)**

---

## 1. Executive Summary & Verdict Table

| Suite | Focus | Executed Runs | Verdict | Key Finding |
| :--- | :--- | :---: | :---: | :--- |
| **1. Rate Sensitivity** | NE Infusion Rate (0.5, 1.0, 2.0 mL/min) | 3 runs (240s) | **PASS** | Monotonic dose-dependent MAP rise: Peak MAP = 97.20, 99.00, 102.60 mmHg (Delta Peak: +1.27, +3.07, +6.67 mmHg). |
| **2. Duration Sensitivity** | NE Duration (60, 120, 240 s) | 3 runs (180, 240, 360s) | **PASS** | Asymptotic elevation: Peak MAP = 98.00, 99.00, 100.00 mmHg (Delta Peak: +2.07, +3.07, +4.07 mmHg); simulated MAP returns toward baseline over 60s post-infusion. |
| **3. Start-Time Sensitivity** | NE Start Time (60, 120, 180 s) | 3 runs (180, 240, 300s) | **PASS** | Strict temporal translation: Response onset shifts by exactly +60s and +120s with matching peak latencies (~50-60s post-start). |
| **4. Matched Control** | Saline Only vs Saline + NE | 2 runs (540s) | **PASS** | Divergence isolated to intervention window: At t=480s, Saline+NE exceeds Saline alone by +2.80 mmHg simulated MAP with zero incremental volume expansion. |
| **5. Repeatability** | Deterministic Reproduction (Live bypass) | 2 identical runs | **PASS** | Bit-identical output across all channels: max absolute difference = `0.00000000`. |

---

## 2. Quantitative Results & Discrepancy Reconciliation

### Baseline Window Definition & Sampling Reconciliation
> **Sampling Architecture & Sample Count:**
> Across all scenarios, the pre-intervention baseline stabilization phase is configured for 60.0 seconds (`baseline_stabilization_s = 60.0`). At a recording frequency of 0.2 Hz (one sample every 5.0 seconds), the BioGears engine emits its initial output row at $t = 5.02\text{s}$ ($t=0.0\text{s}$ is the internal initial state and is not logged in the CSV output).
> Exactly **11 pre-intervention baseline samples** are logged at timestamps:
> `[5.02, 10.02, 15.02, 20.02, 25.02, 30.02, 35.02, 40.02, 45.02, 50.02, 55.02] s`.
> Sample index 11 occurs at $t = 60.02\text{s}$, representing the first post-onset timepoint after the scheduled intervention commences at $t=60.0\text{s}$.
> Across all 12 runs, the baseline MAP is the exact arithmetic mean of these 11 pre-intervention points: **95.927 mmHg**, rounded to **95.93 mmHg**.

---

### Experiment 1: Norepinephrine Rate Sensitivity
*Protocol:* Norepinephrine infusion at variable rate (0.5, 1.0, 2.0 mL/min) with standard stock concentration (14.0 ug/mL).
*Fixed parameters:* Start Time = 60.0s, Duration = 120.0s, Baseline = 60.0s, Post-Observation = 60.0s, Total Duration = 240.0s.

| Rate (mL/min) | Base MAP (mmHg) | Peak MAP (mmHg) | Peak Time (s) | Delta Peak MAP (mmHg) | Final MAP (mmHg) | Delta Final MAP (mmHg) | Peak HR (bpm) | Peak CO (L/min) | Delta Final BV (L) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.5 | 95.93 | 97.20 | 175.02s | +1.27 | 96.40 | +0.47 | 72.32 | 5.81 | +0.01 |
| 1.0 | 95.93 | 99.00 | 180.02s | +3.07 | 96.80 | +0.87 | 72.32 | 5.81 | +0.01 |
| 2.0 | 95.93 | 102.60 | 180.02s | +6.67 | 97.80 | +1.87 | 72.32 | 5.81 | +0.01 |

> **Reconciliation & Analysis:**
> - **Underlying arithmetic:** In prior preliminary documentation, narrative text cited "+1.5 to +3.3 mmHg", while the summary table listed Delta MAP values of 0.47, 0.87, and 1.87 mmHg. Recalculation from underlying timeseries clarifies that 0.47, 0.87, and 1.87 mmHg represent the **baseline-to-final** change (`Final - Baseline` at t=240s, 60s post-infusion), whereas the true **baseline-to-peak** elevations (`Peak - Baseline`) are **+1.27 mmHg (0.5 mL/min), +3.07 mmHg (1.0 mL/min), and +6.67 mmHg (2.0 mL/min)**.
> - **Dose response:** Increasing infusion rate produces a monotonically increasing peak MAP response. Peak occurs at or near the end of the 120s infusion window (t=175.0s to 180.0s).
> - **Volume expansion:** Blood volume change at final observation is negligible (+0.01 L, corresponding to 1-4 mL total fluid carrier infused), demonstrating pure vasopressor activity without volume expansion.

---

### Experiment 2: Norepinephrine Duration Sensitivity
*Protocol:* Norepinephrine infusion at variable duration (60s, 120s, 240s) at fixed rate (1.0 mL/min) and concentration (14.0 ug/mL).
*Fixed parameters:* Start Time = 60.0s, Baseline = 60.0s, Post-Observation = 60.0s.

| Duration (s) | Total Sim (s) | Base MAP (mmHg) | Peak MAP (mmHg) | Peak Time (s) | Delta Peak MAP (mmHg) | Final MAP (mmHg) | Delta Final MAP (mmHg) | Final HR (bpm) | Final CO (L/min) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 60 | 180.0 | 95.93 | 98.00 | 110.02s | +2.07 | 96.50 | +0.57 | 72.11 | 5.72 |
| 120 | 240.0 | 95.93 | 99.00 | 180.02s | +3.07 | 96.80 | +0.87 | 72.00 | 5.67 |
| 240 | 360.0 | 95.93 | 100.00 | 290.02s | +4.07 | 97.20 | +1.27 | 71.37 | 5.67 |

> **Reconciliation & Analysis:**
> - Extending infusion duration increases peak MAP from 98.00 mmHg (+2.07) to 99.00 mmHg (+3.07) to 100.00 mmHg (+4.07). The rate of increase slows between 120s and 240s, reflecting asymptotic saturation towards steady-state pharmacokinetics.
> - In all three runs, terminating the infusion via `_emit_norepi_stop` is followed by a progressive decline in simulated MAP toward baseline over the 60s post-infusion observation window, leaving modest residual elevations (+0.57, +0.87, +1.27 mmHg). We describe this strictly as an observed hemodynamic trajectory; pharmacokinetic drug clearance should not be inferred from MAP alone.

---

### Experiment 3: Start-Time Sensitivity
*Protocol:* Norepinephrine infusion (1.0 mL/min, 60s duration, 14.0 ug/mL) initiated at variable start times (60s, 120s, 180s).
*Fixed parameters:* Baseline = 60.0s, Post-Observation = 60.0s.

| Start Time (s) | Active Window (s) | Total Sim (s) | Base MAP (mmHg) | Peak MAP (mmHg) | Peak Time (s) | Peak Latency (s) | Delta Peak MAP (mmHg) | Final MAP (mmHg) | Delta Final MAP (mmHg) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 60 | 60..120 | 180.0 | 95.93 | 98.00 | 110.02s | +50.02s | +2.07 | 96.50 | +0.57 |
| 120 | 120..180 | 240.0 | 95.93 | 98.00 | 170.02s | +50.02s | +2.07 | 96.40 | +0.47 |
| 180 | 180..240 | 300.0 | 95.93 | 97.80 | 240.02s | +60.02s | +1.87 | 96.40 | +0.47 |

> **Reconciliation & Analysis:**
> - Initiating the intervention at t=60s, 120s, and 180s produces hemodynamic peaks at t=110.02s, 170.02s, and 240.02s respectively.
> - The response profile exhibits temporal shift invariance: for the 60s and 120s start times, peak latency is identical (+50.02s from start); for 180s, peak occurs at the end of infusion (+60.02s from start). Amplitude remains virtually identical (Peak MAP: 98.00, 98.00, 97.80 mmHg).

---

### Experiment 4: Matched Saline Control vs Saline Followed by Norepinephrine
*Protocol:* Both scenarios share an identical total simulation horizon of 540.0s and an identical 250 mL Normal Saline bolus (60.0s to 360.0s; rate = 50.0 mL/min).
- **Arm 1 (Saline Control):** Unperturbed post-infusion observation from 360.0s to 540.0s (180s unperturbed observation).
- **Arm 2 (Saline + Norepi):** Sequential Norepinephrine infusion (1.0 mL/min, 14.0 ug/mL) from 360.0s to 480.0s (120s infusion), followed by 60s observation (480.0s to 540.0s).

#### Table 4A: Summary Metrics (Baseline-to-Peak and Baseline-to-Final)
| Arm | Total Vol | Base MAP | Peak MAP (Time) | Delta Peak MAP | Final MAP | Delta Final MAP | Final BV | Delta Final BV | Final CO | Delta Final CO |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Saline Control (250 mL)** | 250 mL | 95.93 | 97.50 (@425.0s) | +1.57 | 97.50 | +1.57 | 5.78 | +0.25 | 6.02 | +0.21 |
| **Saline (250 mL) + NE (120s)** | 250 mL (+2mL NE) | 95.93 | 100.30 (@480.0s) | +4.37 | 97.70 | +1.77 | 5.79 | +0.26 | 6.03 | +0.22 |

#### Table 4B: Time-Matched Trajectory Comparison (Identical Simulation Clock)
| Simulation Time | Event in Timeline | Saline MAP (mmHg) | Mixed MAP (mmHg) | Difference (Mixed - Saline) | Saline BV (L) | Mixed BV (L) | Diff BV (L) | Saline CO (L/min) | Mixed CO (L/min) | Diff CO (L/min) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **t = 360.0s** | End of Saline Bolus (Both Arms) | 96.90 | 96.90 | **+0.00** | 5.78 | 5.78 | **+0.00** | 6.10 | 6.10 | **+0.00** |
| **t = 480.0s** | End of NE Infusion (Mixed Arm) | 97.50 | 100.30 | **+2.80** | 5.78 | 5.79 | **+0.01** | 6.03 | 5.78 | **-0.25** |
| **t = 540.0s** | End of Simulation Horizon | 97.50 | 97.70 | **+0.20** | 5.78 | 5.79 | **+0.01** | 6.02 | 6.03 | **+0.01** |

> **Detailed Window Comparability Analysis:**
> 1. **Time-matched vs. Descriptive comparisons:**
>    - **Descriptive post-protocol comparison (t = 540s):** Comparing final values at t=540s is descriptive rather than mechanically symmetric: the Saline Control is 180s post-intervention, while the Mixed Arm is only 60s post-intervention. At t=540s, the simulated MAP difference between the two arms declines to +0.20 mmHg (97.70 vs 97.50 mmHg), compared to +2.80 mmHg at t=480s. This reflects the post-infusion decline of the simulated arterial pressure toward the control arm level; direct pharmacokinetic clearance is not inferred from pressure alone.
>    - **Time-matched comparison at intervention cessation (t = 480s):** Comparing both arms at t=480s provides a time-matched observation. At this exact timestamp, both arms share identical elapsed duration and identical prior fluid loading. Mixed MAP is 100.30 mmHg vs 97.50 mmHg for Saline alone (**+2.80 mmHg simulated difference** in Mean Arterial Pressure between the two arms, coinciding with the active norepinephrine infusion period).
>    - **Pre-intervention baseline validation (t = 360s):** Prior to norepinephrine initiation, both arms received identical saline infusions and exhibit identical hemodynamics (MAP = 96.90 mmHg, BV = 5.78 L, CO = 6.10 L/min, difference = 0.00), verifying experimental control integrity.
> 2. **Physiological distinction between volume expansion and vasopression:**
>    - Saline infusion alone expanded circulating volume by **+0.25 L** (from 5.53 L to 5.78 L), causing preload augmentation and a rise in Cardiac Output (+0.21 L/min).
>    - Norepinephrine administration added no meaningful volume (BV diff = +0.01 L), but produced an elevation in MAP alongside lower simulated Cardiac Output (-0.25 L/min) and lower Heart Rate (-1.51 bpm) at t=480s relative to Saline control. In cardiovascular physiology models, such reductions are consistent with hypothesized afterload increase or baroreflex-mediated bradycardia, though systemic vascular resistance and autonomic feedback channels were not directly recorded in this data request set.

---

### Experiment 5: Deterministic Repeatability
*Protocol:* Re-execution of the 180s Norepinephrine run (identical to `exp2_dur_60s`) with live cache bypass (`use_cache=False`).

| Signal Channel | Max Absolute Difference | Bit-Level Status |
| :--- | :---: | :---: |
| HeartRate | `0.00000000` | **PASS (BIT-IDENTICAL)** |
| MeanArterialPressure | `0.00000000` | **PASS (BIT-IDENTICAL)** |
| CardiacOutput | `0.00000000` | **PASS (BIT-IDENTICAL)** |
| BloodVolume | `0.00000000` | **PASS (BIT-IDENTICAL)** |

> **Repeatability Conclusion:** Across all 36 recorded timepoints (5.0s sampling frequency), every physiological channel yielded exactly zero absolute floating-point variance between independent live runs, confirming complete deterministic reproducibility.

---

## 3. Implementation Correctness Checklist

1. **XML Emission & Proper Action Termination:**
   - Emitted XML actions verify `<Rate value="X" unit="mL/min" />` and `<Concentration value="14" unit="ug/mL" />`.
   - Termination actions correctly emit `<Rate value="0" unit="mL/min" />` and `<Concentration value="0" unit="ug/mL" />` at `start_time_s + duration_s`.
2. **Cache Key Uniqueness:**
   - Verified that distinct rates, durations, start times, and bag volumes produce distinct SHA-256 fingerprints, preventing collisions.
3. **Boundary Condition & Schema Rejection:**
   - Out-of-bounds rates (<0.1 or >5.0 mL/min) and durations (<10 or >600s) are cleanly rejected with `ValidationError`.
   - Non-standard stock concentrations (!= 14.0 ug/mL) are rejected as a deliberate application-level protocol constraint based on the validated BioGears SepticShock reference scenario. This ensures standardized dosage titration via rate and duration within validated operational bounds.
   - Overlapping actions (substance starting before previous bolus completes) are rejected by scenario validation.
4. **Engine Health:**
   - Across all 12 runs, exit code was strictly 0. No process timeouts or engine segmentation faults.
   - No `NaN`, `Inf`, or unphysical negative values in any physiological timeseries.

---

## 4. Governance & Limitations Notice

> **Important Clinical and Modeling Disclaimer:**
> All reported trajectories represent computational outputs generated by the BioGears physiology engine under an idealized standard adult male model (`StandardMale.xml`). These simulations are designed for mathematical sensitivity verification, educational modeling, and architectural integrity testing.
> **They do not constitute clinical treatment guidelines, evidence of clinical efficacy, or demonstrations of treatment superiority.** Norepinephrine concentration is constrained to standard stock dilution (14.0 ug/mL) as an application-level safety and standardization constraint matching the verified BioGears SepticShock reference scenario, rather than a demonstrated ODE solver numerical limitation. Titration is achieved by varying infusion rate (0.1-5.0 mL/min) and duration (10-600s).

# BioGears Integration Boundary

This document is the single reference for what a future BioGears
physiological-simulation integration needs to plug into this project.
It is documentation only — nothing here is implemented. See
`WHATIF_ARCHITECTURE.md` for the deeper technical version of most of
this; this document is the shorter, standalone version your teammate
building the BioGears bridge should read first.

**No BioGears package, binary, or Docker image has been installed,
built, downloaded, or invoked anywhere in this project's development —
neither in the cloud sandbox used to build the rest of the system nor
on your local machine. Nothing about BioGears' real input/output
format is assumed or guessed at anywhere in this codebase.**

## 1. What ClinTwin currently produces (real, already implemented)

- A real risk prediction (probability of clinical deterioration in the
  next 2–6 hours) from a trained BiLSTM/Transformer on a patient's
  recent vitals/labs/treatment window.
- A real gradient-based feature attribution explaining that
  prediction.
- A real per-window trajectory of vitals, risk score, and the model's
  internal latent state.
- A real, validated What-If request: a patient, a window, and one or
  more proposed interventions drawn from a small, real vocabulary
  (`backend/whatif.py::INTERVENABLE_FEATURES` — vasopressor dose,
  fluid bolus volume, FiO2, ventilation status), attached to a real
  baseline prediction.
- **A real ML counterfactual** (added in a follow-up phase, after this
  document was first written): `InferenceService.compute_counterfactual()`
  actually re-runs the modified input through the real
  feature-engineering pipeline and the same trained model checkpoint —
  never an edited number, an interpolation, or a fabricated
  trajectory. This is still not BioGears. It is the same statistical
  model evaluated on a hypothetical input, not a physiological
  simulation — see `WHATIF_ARCHITECTURE.md` section 1 for the fuller
  explanation of why these two are kept conceptually and visually
  separate. Only the physiological-simulation piece described below
  remains on the far side of the boundary.

## 2. Where BioGears plugs in

```
Real ClinTwin prediction/explanation/trajectory
              |
              v
      Validated WhatIfScenarioRequest        <- real, already built
              |
              v
      ML Counterfactual                      <- REAL (same trained model,
              |                                  modified input, re-engineered
              |                                  features) — NOT BioGears
              v
   ---- THIS BOUNDARY ----                   <- nothing crosses it yet
              |
              v
      Simulation Result                      <- NOT YET AVAILABLE;
              |                                 schema unknown until
              |                                 BioGears exists
              v
      ClinTwin visualization                 <- already renders the real
                                                  counterfactual; ready to
                                                  render whatever simulation
                                                  result comes back (see
                                                  section 5)
```

Concretely: `backend/whatif.py::run_scenario()` currently always
returns

```json
"simulation": {
  "available": false,
  "boundary": "BIOGEARS_INTEGRATION_BOUNDARY",
  "message": "Simulation requires BioGears integration."
}
```

The BioGears bridge's job is to make `available: true` with real
output, for real, once it exists.

## 3. What ClinTwin sends INTO the boundary today

```json
{
  "window_idx": 7,
  "interventions": [{"feature": "fio2", "value": 0.5}],
  "engine": "biogears"
}
```

| feature (as ClinTwin's model uses it today) | unit | range |
|---|---|---|
| `vasopressor_dose` | normalized dose (not a real mcg/kg/min value) | 0.0 – 5.0 |
| `fluid_bolus_ml` | mL | 0 – 2000 |
| `fio2` | fraction | 0.21 – 1.0 |
| `ventilated` | boolean | 0 or 1 |

**This is what ClinTwin's prediction model consumes as training
features — it is not a proposal for BioGears' input contract.**
BioGears will very likely need more granular, real-unit inputs (a
specific drug and dose rate, ventilator mode/PEEP/tidal volume, fluid
type and infusion duration, possibly a time course rather than a
single instantaneous value). Tell us what BioGears actually needs and
`INTERVENABLE_FEATURES` / the request schema will be adapted to match.

## 4. What is genuinely unresolved — open questions for the BioGears teammate

None of the following are assumed, guessed, or stubbed anywhere in
this codebase. They are open questions to be answered from the real
BioGears implementation once it exists:

1. **Input format** — does ClinTwin generate a BioGears native
   scenario file (e.g. its own `.xml`/JSON scenario format), or call a
   Python/C++ API directly?
2. **Output schema** — what physiological variables can BioGears
   output, under what names, in what units?
3. **Output shape** — a single end-state, or a time series (a
   "trajectory" the way ClinTwin already shows for real model
   predictions)?
4. **Run time / execution model** — does a run complete fast enough to
   answer synchronously within one HTTP request (what ClinTwin does
   today, only viable because nothing real happens yet), or does it
   need a job/polling model? Flagging this now specifically so it
   isn't a late surprise: if BioGears runs take more than a couple of
   seconds, the What-If endpoint will need a status/polling endpoint
   added before real integration lands.
5. **Deployment shape** — does BioGears run in-process/subprocess,
   reachable directly from ClinTwin's Python backend, or does it need
   its own service reached over HTTP/gRPC/a queue?

## 5. What exact information is needed before implementation can continue

From a real, working BioGears run (not real patient data — see this
project's data-handling rules), please send:
- The actual input you gave it (real field names/units/sample values).
- The actual output it produced (real field names/units/sample
  values, and whether it's one end-state or a series).
- How long that run took.
- How you're running it (library call vs. subprocess vs. separate
  service).

Once that exists, the four things that will change are:
1. New `backend/biogears_bridge.py` (or similar) — the real caller.
2. `backend/whatif.py::run_scenario()` — populate `simulation` with
   real output instead of `available: false`.
3. `SUPPORTED_ENGINES` and the relevant Pydantic response models in
   `backend/api.py` — reflect the real, agreed schema.
4. `frontend/src/components/WhatIfPanel.jsx` — add a real rendering
   for the real `simulation` payload (deliberately not built yet;
   building a chart for an unknown schema would mean guessing at it).

## 6. Non-goals right now (explicit)

- No BioGears install/build/Docker use in this repository.
- No invented BioGears variable names, units, or CSV/output schema
  anywhere in this codebase.
- No fabricated simulation result, ever — every What-If response is
  either a real validated baseline + honest "not available" boundary,
  or (after real integration) real BioGears output. No third option.
- `DATA_MODE=synthetic` remains the active data source throughout;
  this document does not touch that decision.

## 7. The project is fully functional without any of this

Everything above the boundary in section 1 works today, is tested
(see `tests/test_whatif.py`, `tests/test_end_to_end_pre_biogears.py`),
and is demonstrable end to end (see `docs/RUN_EVERYTHING.md`).
BioGears is explicitly not required for the project's current
definition of "done."

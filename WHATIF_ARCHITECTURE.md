# What-If / BioGears Integration Boundary

This document defines the boundary between ClinTwin Sentinel (this
repository) and the BioGears physiology-simulation bridge (being built
separately by a teammate). It exists so both sides can work
independently without either one guessing at the other's format.

**Ground rule this document exists to enforce:** ClinTwin does not
assume, invent, or fabricate anything about BioGears' scenario input
format, physiological variable names, units, CSV structure, or output
schema. Everything below the boundary is described generically, as an
open question for the teammate to answer — not as a spec ClinTwin has
already decided for them.

## 1. The boundary, end to end

**Update (pre-BioGears completion phase):** the ML COUNTERFACTUAL is
now REAL — not a placeholder. `POST /api/patients/{id}/whatif` re-runs
the actual trained model on the modified input through the real
feature-engineering pipeline (`InferenceService.compute_counterfactual`
in `backend/inference_service.py`). This is still not BioGears — it is
the same statistical model, re-evaluated on a hypothetical input, not
a physiological simulation. Only the PHYSIOLOGICAL SIMULATION section
below the boundary remains unavailable.

```
ClinTwin prediction / explanation / trajectory   (real — already implemented)
              |
              v
      WhatIfScenarioRequest                       (real — validated against
              |                                     real patient data)
              v
      ML Counterfactual                            (REAL — same trained
              |                                      model, modified input,
              |                                      re-engineered features)
              v
    ---- BIOGEARS INTEGRATION BOUNDARY ----        <- nothing crosses this
              |                                        yet. See section 3.
              v
      Physiological Simulation Result               (NOT YET AVAILABLE —
              |                                      schema unknown until
              |                                      the teammate provides it)
              v
      ClinTwin visualization                       (renders the real
                                                      counterfactual now;
                                                      ready for simulation —
                                                      see section 4)
```

Every What-If request ClinTwin receives is validated, attached to a
real baseline, given a REAL ML counterfactual, and returned with:

```json
"counterfactual": {
  "risk_score": 0.72,
  "status": "Critical",
  "risk_change": 0.29,
  "gradient_attribution": [...],
  "explanation_method": "gradient_x_input saliency (real, ...) — NOT a Shapley-value SHAP explanation.",
  "description": "ML COUNTERFACTUAL — the SAME trained model, re-run on this patient's real window with the requested intervention(s) applied ... not a physiological simulation ..."
},
"simulation": {
  "available": false,
  "boundary": "BIOGEARS_INTEGRATION_BOUNDARY",
  "message": "Simulation requires BioGears integration.",
  "description": "..."
}
```

That is the honest, current state of the whole system — see
`backend/whatif.py` and `InferenceService.compute_counterfactual()`.

## 2. What ClinTwin sends INTO the boundary (real, already defined)

`POST /api/patients/{stay_id}/whatif` (see `backend/api.py`,
`backend/whatif.py`) accepts:

```json
{
  "window_idx": 7,                 // optional — defaults to the latest window
  "interventions": [
    {"feature": "vasopressor_dose", "value": 0.8}
  ],
  "engine": "biogears"
}
```

`interventions[].feature` is currently limited to
`backend/whatif.py::INTERVENABLE_FEATURES` — four variables pulled
directly from this project's real, already-existing feature set
(`config.TREATMENT_COLS + config.RESP_SUPPORT_COLS + config.VENT_COLS`):

| feature            | unit (as used in THIS project's training data) | range      |
|---------------------|--------------------------------------------------|------------|
| `vasopressor_dose`  | normalized dose (not a real-world mcg/kg/min value — see note below) | 0.0 – 5.0 |
| `fluid_bolus_ml`    | mL                                               | 0 – 2000   |
| `fio2`              | fraction                                         | 0.21 – 1.0 |
| `ventilated`        | boolean (0/1)                                    | 0 or 1     |

**Important caveat for the BioGears teammate:** these four variables
and their units/ranges reflect what ClinTwin's *prediction model*
consumes, not necessarily what BioGears needs as simulation input.
BioGears almost certainly needs more granular, real-unit inputs (a
specific vasopressor drug and its actual mcg/kg/min rate, a specific
ventilator mode/PEEP/tidal volume, fluid type, infusion duration,
etc.), and may need scenario framing ClinTwin hasn't anticipated at
all (e.g. a time course of interventions, not just a single value at
one instant). **This table is what exists today, not a proposal for
what BioGears' input contract should be.** Please tell us what
BioGears actually needs, and we'll adapt `INTERVENABLE_FEATURES` and
the request schema to match — rather than trying to guess it and
having you conform to a spec that doesn't fit your engine.

## 3. What ClinTwin needs FROM the boundary (open questions, not assumptions)

We do not know, and have not guessed at:

- BioGears' scenario/input file format (its own native format, e.g.
  its `.xml`/JSON scenario files — do we generate one of those, or
  call a Python/C++ API directly?)
- What physiological variables BioGears can output, their names, and
  their units
- Whether output is a single end-state or a time series (a
  "trajectory" the way ClinTwin already shows for the real model's
  predictions — see `GET /api/patients/{id}/trajectory`)
- Run time — is this synchronous (fits in one HTTP request/response)
  or does it need a job/polling model? (Not built in this phase —
  ClinTwin currently answers every What-If request synchronously,
  which only works because there's no real simulation happening yet.
  If BioGears runs take more than a couple of seconds, we'll need to
  add a job-status endpoint before wiring this in for real — flagging
  this now so it isn't a surprise later.)
- Whether BioGears runs in a place ClinTwin's backend can reach
  directly (subprocess/library call) or needs its own service ClinTwin
  calls over HTTP/gRPC/a message queue

**Please answer these from the real BioGears implementation once it
exists**, rather than ClinTwin inventing placeholder answers now. When
you have real output to show us (even a single example run), send the
actual field names/units/sample values (not real patient data — see
this project's data-handling rules) and we will:

1. Add a real `backend/biogears_bridge.py` (or similar) that calls the
   real BioGears engine/service.
2. Extend `whatif.py::run_scenario()`'s `simulation` field to carry
   real output instead of `available: false`, once — and only once —
   a real run actually succeeds.
3. Update `SUPPORTED_ENGINES` and the Pydantic response models in
   `backend/api.py` to reflect the real, agreed schema.
4. Update this document to describe what's real, replacing the "open
   questions" above.

## 4. Where the visualization is already waiting

`frontend/src/components/WhatIfPanel.jsx` already renders whatever
`simulation` the backend returns. Right now that's always the honest
"not yet available" message. Once
`simulation.available` can be `true` with real data attached, the
panel needs a real rendering for that data — deliberately not built
yet, since building a chart for an unknown schema would mean guessing
at the schema.

## 5. Non-goals for this phase (explicit, per project priorities)

- No BioGears install, build, or Docker use happens in this
  repository or in the cloud sandbox used to develop it.
- No speculative BioGears variable mapping, CSV column, or output
  schema is invented anywhere in this codebase.
- No fabricated simulation result is ever returned — every What-If
  response is either a real validated baseline + honest "not
  available" boundary, or (after real integration) real BioGears
  output. There is no third option.
- The synthetic dataset (`DATA_MODE=synthetic`) remains the active
  data source throughout this work. Real MIMIC-IV/BigQuery activation
  is a separate, explicitly gated decision — see CLAUDE.md / project
  governing rules — and is untouched by this document.

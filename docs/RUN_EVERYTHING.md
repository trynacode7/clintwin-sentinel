# Running ClinTwin Sentinel from a Clean Environment

This is the single, exact, reproducible sequence to take the canonical
project at `C:\Users\diyap\Documents\projects\clinTwinSentinel\project`
from nothing to a fully working local demo: synthetic data → trained
models → backend API → frontend dashboard → real prediction →
real explanation → real trajectory/twin state → real ML What-If.

Everything through step 3 (synthetic data) has been run and verified
directly on your machine, through the device bridge, in this session.
Steps 4 onward (installing `torch`, training, and starting the
backend/frontend as long-running processes) could **not** be executed
through that bridge — it is a sandboxed, ephemeral, disk-constrained
VM built for file transfer, not for multi-minute installs or
persistent servers, and its network proxy blocks
`download.pytorch.org`. None of that is a limitation of your actual
machine. Run steps 4+ yourself in a normal terminal (PowerShell or
Command Prompt) on your machine — they are ordinary commands and
should work normally there.

## 0. One-time setup

```powershell
cd C:\Users\diyap\Documents\projects\clinTwinSentinel\project
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
```

`backend/requirements.txt` installs `torch` from plain PyPI by
default. If that's slow or you want the smaller CPU-only build
instead, use:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend\requirements.txt --no-deps
pip install numpy pandas scikit-learn joblib shap fastapi "uvicorn[standard]" pydantic python-multipart
```

Either way, this is real, standard `pip` usage on your own machine —
not something that could be verified inside the sandboxed bridge, but
not unusual either.

## 1. Generate the synthetic dataset

```powershell
python generate_synthetic_data.py --patients 300 --seed 42
```

**Already run and verified on your machine in this session** — the
resulting `data/synthetic/processed/ml_dataset.csv` was confirmed
byte-identical (187,776 lines / 20,448,104 bytes) to the same command
run in the development sandbox with the same seed, proving the
pipeline is genuinely deterministic and reproducible on your hardware.
If you run it again, you'll get the same file. This step also fixed a
real bug (see "What was fixed" below) that made this dataset silently
come out empty on your machine before this session — it now works
correctly.

## 2. (No separate preprocessing step needed)

Preprocessing — train/val/test split by subject_id, resampling,
feature engineering, SOFA labeling, and fitting+saving
`models/scaler.joblib` — happens automatically as the first stage of
the training command below (`backend/train.py` calls
`backend.preprocessing.preprocess()` internally). You do not run it
separately.

## 3. Verify the data/preprocessing pipeline without training (optional but recommended)

```powershell
pip install pytest
python -m pytest tests\test_data_validation.py tests\test_mimic_adapter.py tests\test_multi_stay.py tests\test_patient_splits.py tests\test_sofa_scoring.py tests\test_validate_mimic.py tests\test_whatif.py -q
```

**Already run for real on your machine in this session** (via the
device bridge) — 61/76 project tests passed. The other 15 require
`torch` (model loading/inference), which was not installed in that
bridge session. You should see all of these pass on your machine once
`torch` is installed, since nothing else about this layer changed.

## 4. Train both models

```powershell
python -m backend.train --model lstm
python -m backend.train --model transformer
```

This is the **canonical, only** training entrypoint — the same one
used to produce every checkpoint referenced anywhere else in this
project. It is deterministic (`utils.set_seed(config.SEED)`, `SEED =
42` in `config.py`) and uses early stopping (`EARLY_STOPPING_PATIENCE
= 10` in `train.py`, up to `NUM_EPOCHS = 100`). Each run saves its
best checkpoint to `models/lstm_model.pt` / `models/transformer_model.pt`
and (on the first run) `models/scaler.joblib`.

**Do not** substitute a random/untrained model, and do not copy a
checkpoint from any other machine or sandbox — this command is the
only source of truth for a real, reproducible local checkpoint. This
step could not be run through the device bridge (no PyTorch install
possible there — see "What was fixed / what remains" below) and must
be run by you, once, in your own terminal.

## 5. Verify the trained checkpoints actually load and match the feature contract

```powershell
python -m pytest tests\ -q
```

This should now show **102 passed** (the full suite — the same count
verified in the development sandbox, including three tests that prove
the What-If ML counterfactual is a real computation, not a
placeholder — see item 4 below). If any test fails here, do not
proceed to step 6 — that means the locally trained checkpoint doesn't
actually match what the API expects, and that must be fixed first,
not worked around.

## 6. Start the backend

```powershell
python -m uvicorn backend.api:app --reload --port 8000
```

Leave this running. In a browser or with `curl`, confirm:
```powershell
curl http://localhost:8000/api/roster
```
should return real patient stay_ids, not an error.

## 7. Start the frontend (separate terminal)

```powershell
cd C:\Users\diyap\Documents\projects\clinTwinSentinel\project\frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

> Note: if `npm run dev` fails with `Cannot find module
> @rollup/rollup-linux-x64-gnu` or a similar `@rollup/rollup-*`
> platform package — this is a known `npm` optional-dependency bug
> (https://github.com/npm/cli/issues/4828), not a project bug. On
> Windows the equivalent package name differs
> (`@rollup/rollup-win32-x64-msvc`); if you hit it, delete
> `node_modules` and `package-lock.json` and run `npm install` again.

## 8. Walk the real demo flow

1. Select a patient from the roster.
2. Confirm the vitals shown are real (`OBSERVED`, from the synthetic
   dataset — see `PRE_BIOGEARS_ARCHITECTURE.md` section 5).
3. Confirm a real risk score and gradient-based explanation render
   (never labeled "SHAP" — see `PRE_BIOGEARS_ARCHITECTURE.md`
   section 2's Explainability bullet for the naming rule).
4. Confirm the trajectory chart and the OBSERVED/PREDICTED/SIMULATED
   twin-state badges render.
5. Open the What-If panel, submit a scenario, and confirm you see:
   a real baseline risk, a real **ML Counterfactual** (a different
   risk score/status, plus a `+/-` change — the same trained model
   re-run on your modified input, clearly labeled "not a
   BioGears/physiological simulation"), and — separately — the honest
   "Simulation requires BioGears integration" message for the still-
   unavailable physiological simulation (see
   `docs/BIOGEARS_INTEGRATION_BOUNDARY.md`). The counterfactual number
   should change if you submit a different intervention value —
   if it never changes, something is wrong.

## What was verified directly on your machine this session (device bridge)

- Synthetic data generation (step 1) — real run, byte-identical output.
- 61 of 76 tests (step 3's list) — real `pytest` run, real pass,
  re-verified after the true ML counterfactual and the SOFA-labeling
  fix (items 4-5 below) were added — still 61/76, same reason for the
  other 15 (torch not installed in the bridge).
- Three real, previously-invisible bugs found and fixed by actually
  running on your machine (or, for the SOFA-labeling bug, discovered
  while building the true ML counterfactual and verified against the
  real synthetic dataset) rather than trusting assumptions (see below).

## What could not be run through the device bridge, and why

- **`pip install torch`** — the bridge's network proxy returns 403 for
  `download.pytorch.org` (the CPU-wheel index), and the plain-PyPI
  `torch` wheel (526 MB, plus multi-GB CUDA dependencies unless
  `--no-deps` is used) does not reliably fit the bridge's disk
  quota or complete within its per-command execution window. This is
  a constraint of that specific automated bridge VM, not of your
  actual machine.
- **Model training** (`python -m backend.train`) — depends on `torch`
  being installed, so it inherits the same blocker. Training also
  needs to run as one long-lived process; the bridge tears down every
  background process the moment each individual command finishes, so
  even with `torch` installed, a multi-epoch training run could not
  have stayed alive across the bridge's command boundaries.
- **Starting `uvicorn` and `vite` as live servers** — same
  process-lifetime issue, plus a separate, unrelated filesystem quirk:
  the bridge's mount would not allow deleting/rewriting files inside
  `frontend/node_modules/.vite` (`EPERM`), which blocked a live
  `vite` smoke test specifically through the bridge. This does not
  affect your normal Windows filesystem — it's specific to how the
  bridge mounts your folder.

None of this was worked around by substituting a fake result. Every
one of these steps is a normal command you can run in your own
terminal; they just could not be executed and verified from inside
this session.

## What was fixed this session (found only by running on your machine)

1. **`utils.py`** — `import torch` was unconditional at module level,
   so even torch-independent modules (like `preprocessing.py`, via
   `setup_logging()`) required torch just to import. Now lazy
   (imported inside `set_seed()`, `get_device()`,
   `compute_class_weights()` only). Verified: full test suite still
   99/99 in the sandbox; on-device, this alone unblocked 61 real
   passing tests that previously failed at collection.
2. **`backend/synthetic_data/generator.py` and
   `extract_to_ml_dataset.py`** — a real, previously-undiscovered bug:
   this project's synthetic timestamps are intentionally shifted into
   a far-future range (~year 2446, a MIMIC-IV-style de-identification
   convention). Pandas' default nanosecond-resolution datetime64[ns]
   cannot represent dates past 2262, so on your machine's pandas
   version (2.3.3), every such date was silently coerced to `NaT`,
   which emptied every time-indexed table and produced a 0-row
   dataset — invisible in the cloud sandbox because it happened to run
   a newer pandas (3.0.2) that auto-upgrades resolution. Fixed by
   explicitly constructing these columns at microsecond resolution
   (`datetime64[us]`, safe to roughly year 294000) in both the raw
   generator and the flat-dataset extractor, so this works identically
   on both pandas versions. This is exactly the kind of bug your
   original instruction ("must actually work on my machine, not just
   the sandbox") was written to catch — it would never have been found
   without running on your device.
3. **`frontend/node_modules`** — missing the
   `@rollup/rollup-linux-x64-gnu` optional dependency (a known `npm`
   bug, see step 7's note). Fixed on the bridge's copy for the smoke
   test; may or may not recur on your actual Windows `npm install`
   (Windows uses a different package name for this), noted above just
   in case.
4. **`backend/preprocessing.py` — a real SOFA-labeling correctness
   bug, fixed in a follow-up session** (found while implementing the
   true ML counterfactual, not while regenerating data). The pipeline
   used to Z-score raw vitals/labs BEFORE `labels_sofa.py` computed
   SOFA subscores and before `feature_engineering.py` computed
   `shock_index` — both need real clinical units (e.g. SOFA's "MAP <
   70 mmHg" threshold is meaningless applied to a z-score). Verified
   empirically: this produced a corrupted, near-saturated SOFA
   distribution (totals clustered at 11-23 out of a possible 24).
   Fixed by splitting `preprocess()` into a RAW step and a separate
   `scale_splits()` step that now runs LAST, after engineering and
   labeling — verified the corrected distribution is now realistic
   (mean ~3.9, median 3, right-skewed, max 15). **This means any
   checkpoint trained before this fix is invalid and must be
   retrained** — step 4 above already reflects the corrected pipeline,
   so simply running it (once torch is installed) produces a correct
   checkpoint; there is nothing extra you need to do differently.
5. **True ML counterfactual implemented** (new capability, not a bug
   fix): `InferenceService.compute_counterfactual()` — the What-If
   endpoint now actually re-runs the trained model on a modified input
   through the real feature-engineering pipeline, rather than only
   returning a baseline. See step 8's item 5 and
   `WHATIF_ARCHITECTURE.md`.

## Known limitations

- No real BioGears integration — by design, see
  `docs/BIOGEARS_INTEGRATION_BOUNDARY.md`.
- No real MIMIC-IV/BigQuery data — `DATA_MODE=synthetic` throughout,
  per your explicit hold on switching data sources.
- The Transformer model has no attention-visualization endpoint yet
  (only matters if you switch the live API off the BiLSTM).
- Steps 4, 6, and 7 above are the parts of "done" that only you can
  personally complete and verify, since they could not be run through
  the device bridge this session.

docs/BIOGEARS_IMPLEMENTATION.md

# ClinTwin Sentinel — BioGears Integration (Phase 1)

## 1. Overview

ClinTwin Sentinel is an ICU deterioration-prediction digital twin.

The existing ClinTwin ML system predicts the probability of clinical deterioration from a patient's recent physiological data.

BioGears is being integrated as the physiological simulation layer for the project's What-If functionality.

The intended flow is:

Patient data
↓
ClinTwin ML prediction
↓
What-If intervention
↓
BioGears physiological simulation
↓
Physiological trajectory
↓
ClinTwin visualization

### Phase 1

Phase 1 implements and verifies the BioGears integration for one intervention:

**Fluid bolus (`fluid_bolus_ml`)**

---

## 2. What Has Been Implemented

### `backend/biogears_bridge.py`

This is the execution and communication layer between ClinTwin and the BioGears engine.

The bridge:

1. Checks that the BioGears executable exists.
2. Accepts a BioGears scenario XML file.
3. Executes BioGears using Python `subprocess`.
4. Provides a controlled output location.
5. Waits for BioGears to finish.
6. Finds the generated CSV.
7. Parses the CSV into structured Python objects.
8. Extracts time-series data, columns, row count, and timing information.
9. Handles execution, timeout, missing-output, and parsing failures.
10. Cleans up temporary files.

The bridge does not decide the clinical meaning of interventions.

Its responsibility is:

```text
ClinTwin scenario
      ↓
BioGears executable
      ↓
BioGears output
      ↓
Structured Python result
3. BioGears Scenario Builder
backend/biogears_scenario_builder.py

This module converts a ClinTwin-supported intervention into a valid BioGears scenario XML.

Phase 1 supports:

Fluid bolus only

The ClinTwin value:

fluid_bolus_ml

is mapped to BioGears:

SubstanceCompoundInfusionData

using:

Substance compound: Saline
Bag volume: ClinTwin fluid_bolus_ml
Infusion rate: 50 mL/min

The 50 mL/min rate was taken from the verified BioGears SepticShock_Treatment.xml scenario.

This is a demonstration/integration mapping and is not intended to be a universal clinical dosing recommendation.

4. Why the Builder and Bridge Are Separate

The two components have different responsibilities.

Scenario Builder
ClinTwin intervention
        ↓
BioGears XML

The builder generates the XML but does not execute BioGears.

BioGears Bridge
BioGears XML
        ↓
bg-scenario.exe
        ↓
BioGears CSV
        ↓
Python result

The bridge executes BioGears and parses its output.

This separation keeps scenario generation independent from BioGears execution and makes both components easier to test.

5. BioGears Installation

BioGears is installed separately from the ClinTwin repository.

Verified executable:

C:\BioGears\bin\bg-scenario.exe

BioGears can be invoked using:

C:\BioGears\bin\bg-scenario.exe scenario <scenario.xml> results <output_prefix>

The results parameter controls the output location.

The BioGears installation itself is not committed to GitHub.

6. Verified BioGears Behavior

Before implementing the bridge, a minimal BioGears scenario was tested independently.

The verification confirmed:

BioGears exits successfully with code 0.
A short simulation can be executed successfully.
BioGears generates a CSV time series.
Requested physiological variables appear in the output.
Absolute scenario and output paths work.
A fixed working directory is not required.
The output naming pattern is:
<results_prefix>Results.csv

BioGears also performs its own patient stabilization before the requested simulation period.

7. End-to-End Fluid Bolus Test

The completed integration test verifies the complete chain:

BioGearsFluidBolus
        ↓
BioGearsScenario
        ↓
build_scenario_xml()
        ↓
Temporary XML
        ↓
BioGearsBridge.run_scenario_file()
        ↓
bg-scenario.exe
        ↓
BioGears
        ↓
CSV
        ↓
BioGearsResult

The real BioGears integration test passed.

Verified results
Runtime: approximately 85 seconds
BioGears exit code: 0
Output rows: 36
Requested physiological columns were present
Time values were monotonically increasing
The generated scenario was successfully accepted and executed by BioGears
The resulting CSV was successfully parsed by the bridge

The integration test uses temporary files and cleans them up after execution.

8. Physiological Difference Verification

A baseline BioGears run was compared with a run containing a 100 mL saline fluid bolus at 50 mL/min.

Baseline
Blood volume = 5.530 L
With 100 mL bolus
Final blood volume = 5.630 L
Difference
+0.100 L

The bolus delivery time is:

100 mL / 50 mL/min = 2 minutes

The BioGears output showed blood volume increasing during the infusion and reaching a plateau after the bolus was completed.

This verifies that the intervention is not merely accepted syntactically: the generated intervention produces an observable change in BioGears physiological output.

9. Tests
tests/test_biogears_bridge_unit.py

Contains 71 offline unit tests covering the BioGears bridge, including:

CSV parsing
column parsing
configuration
executable validation
output discovery
subprocess handling
timeout handling
error handling
temporary-file cleanup
scenario execution

These tests do not require BioGears.

tests/test_biogears_scenario_builder_unit.py

Contains 71 offline unit tests covering:

fluid-bolus validation
XML generation
BioGears XML structure
namespace handling
intervention values
invalid inputs
builder purity
tests/test_biogears_bridge_integration.py

Contains integration tests that can execute the actual BioGears executable.

The Phase 1 fluid-bolus integration test verifies:

Scenario Builder → BioGears Bridge → Real BioGears → CSV → Parsed Result
10. Current Test Status

Offline BioGears tests:

142/142 passed

Real BioGears fluid-bolus integration:

PASSED

The integration test executes the actual BioGears executable rather than mocking it.

11. What Is Not Implemented Yet

Phase 1 deliberately does not connect BioGears to the complete ClinTwin What-If pipeline.

The following are not yet connected:

backend/whatif.py
backend/api.py
Frontend What-If UI
BioGears simulation results in the frontend

The following intervention mappings are also unresolved:

Vasopressor

vasopressor_dose

ClinTwin currently represents this as a normalized value, while BioGears requires real concentration and infusion-rate information.

Patient weight and norepinephrine stock concentration also need to be defined.

FiO2

Ventilated and non-ventilated patients require different BioGears handling.

Ventilation

BioGears requires multiple ventilator configuration parameters that ClinTwin currently does not provide.

These mappings are intentionally left for later phases rather than being guessed or fabricated.

12. Current Architecture
                    ClinTwin Sentinel

                  Existing ML Pipeline
                         |
                         v
                 What-If Intervention
                         |
                         v
              Phase 1 BioGears Layer
                         |
                         v
          biogears_scenario_builder.py
                         |
                         v
                  BioGears XML
                         |
                         v
             biogears_bridge.py
                         |
                         v
                bg-scenario.exe
                         |
                         v
                     BioGears
                         |
                         v
                  Output CSV
                         |
                         v
                   BioGearsResult

The planned next phase is to connect the verified BioGears layer back into the existing What-If API and frontend.

13. Running the Offline Tests

From the ClinTwin repository:

pytest tests/test_biogears_bridge_unit.py -v

and:

pytest tests/test_biogears_scenario_builder_unit.py -v

Expected:

142 tests passed
14. Running the Real BioGears Integration Test

BioGears must be installed at:

C:\BioGears\bin\bg-scenario.exe

From the ClinTwin repository:

pytest tests/test_biogears_bridge_integration.py::test_fluid_bolus_builder_chains_to_bridge -v -s

The test executes the actual BioGears engine.

Runtime may take approximately 1–2 minutes depending on the machine because BioGears performs patient stabilization before the simulation.

Temporary XML and output files are cleaned up after the test.

15. Git Checkpoint

Phase 1 was committed to GitHub.

Commit:

9f09849

Commit message:

Integrate BioGears fluid bolus simulation

The commit contains:

backend/biogears_bridge.py
backend/biogears_scenario_builder.py
tests/test_biogears_bridge_integration.py
tests/test_biogears_bridge_unit.py
tests/test_biogears_scenario_builder_unit.py

The BioGears executable and generated simulation outputs are not committed to the repository.

16. Phase 1 Result

Phase 1 successfully demonstrates:

ClinTwin intervention
        ↓
BioGears scenario generation
        ↓
Real BioGears execution
        ↓
Real physiological CSV
        ↓
Parsed Python result

A 100 mL fluid-bolus intervention was also verified to produce an observable change in BioGears blood-volume output:

5.530 L  →  5.630 L

The next phase is to connect this verified BioGears layer to ClinTwin's existing What-If functionality.
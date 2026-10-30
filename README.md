# Artificial Pancreas: Feedback Control of Blood Glucose

Control Systems (BECE302L) project using the Bergman minimal model and feedback controllers: PID, constrained PI with insulin-on-board (IOB) safeguards, and LQR with an observer.

## Start the interactive application

**Windows:** Extract the project ZIP and double-click `START_HERE.bat`. It creates a project-local Python environment, installs the required packages, and starts the Streamlit app.

**Manual:** from the project root, run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

See [START_HERE.txt](START_HERE.txt) and [README_INTERACTIVE.md](README_INTERACTIVE.md) for full instructions and the new Continuous monitor tab.

## Run the eight numerical project phases

```bash
python run_all.py
```

Or run an individual phase from the `phases/` directory:

1. `phase1_parameters.py` — model parameters and basal equilibrium
2. `phase2_nonlinear.py` — nonlinear simulation
3. `phase3_linearization.py` — linearization and state-space model
4. `phase4_time_stability.py` — stability and time response
5. `phase5_frequency.py` — frequency response and robustness
6. `phase6_controllers.py` — controller comparisons
7. `phase7_state_space.py` — controllability, observability, and observer design
8. `phase8_report.py` — report generation

## Test the model

```bash
python -m unittest discover -s tests -v
```

## Project structure

- `ap/` — model, parameters, controllers, simulation, and analysis modules
- `app.py` — Streamlit dashboard and continuous monitor
- `food_database.csv` — food and nutrition estimate catalog
- `phases/` — eight-stage numerical analysis workflow
- `results/` — saved phase outputs, figures, and report
- `tests/` — model, linearization, and controller tests

## Modelling caveat

Parameters are illustrative literature-style values, not patient-specific data. The project is a coursework simulation, not a medical device. The Continuous monitor is a simulated trace; it does not connect to an actual CGM or insulin pump. Do not use its output to make treatment decisions or calculate real insulin doses.

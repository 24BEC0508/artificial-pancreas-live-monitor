# Continuous monitor integration notes

## Integrated additions
- Added a separate `Continuous monitor` tab using an independent session-state simulation.
- Reuses the existing Bergman model integrator and selected PID / constrained PI (IOB) / LQR + observer controller.
- Uses the bundled food catalog; the `Eat now` form timestamps a meal at the current simulated time.
- Adds a session-long expanding elapsed-time chart, glucose and insulin-delivery traces, plasma-insulin and IOB traces, meal-event history, and CSV export.
- Added Start / Resume, Pause, and Reset controls with a one-second Streamlit fragment refresh.
- The original Live dashboard remains a separate six-hour experiment.
- Preserved the explicit Plotly chart keys in the existing dashboard and controller comparison views.
- Added a Windows one-click setup/launch script and updated run instructions.

## Verification performed
- Python compilation passed for `app.py`, `ap/*.py`, `phases/*.py`, and `run_all.py`.
- `python -m unittest discover -s tests -v`: all 12 tests passed.

## Scope note
The Continuous monitor is a software simulation only. It does not connect to a real CGM or insulin pump and must not be used for medical decisions or dosing.

# sl-grid-piml

Physics-informed machine learning for locating electricity losses (theft and faults)
on distribution feeders with very few meters, using a Sierra Leone 11 kV feeder as the case study.

**Status:** Phase 1 — project skeleton and base feeder model.

## Why
Sierra Leone's distribution utility (EDSA) loses roughly half of the electricity it buys to
technical and commercial losses. Most utilities in this situation have meters at only a few
points on each feeder. This project asks: with only a handful of meters, how well can we
find *where* power is being lost, and how much?

## Approach
1. Model an 11 kV radial feeder (IEEE 33-bus test feeder, adapted).
2. Simulate realistic loads, theft, faults, missing and noisy meter data.
3. Compare classical state estimation, pure ML, and a physics-informed model.
4. Study how many meters are needed and where to place them.

All data is simulated for now. The code uses one standard feeder format, so real utility
data can replace the simulator later without changing the models.

## Setup (Windows PowerShell)
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest
python scripts\run_base_powerflow.py
```

## Layout
- `src/slgrid/feeder.py` — builds the base feeder and converts it to the standard format
- `scripts/` — runnable experiments
- `tests/` — checks that the physics is correct

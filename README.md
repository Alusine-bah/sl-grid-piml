# sl-grid-piml

Physics-informed machine learning for locating electricity losses (theft and faults)
on distribution feeders with very few meters, using a Sierra Leone 11 kV feeder as the case study.

**Status:** Phase 5 — physics-informed model working and evaluated. Next: meter placement study and edge deployment.

## Why
Sierra Leone's distribution utility (EDSA) loses roughly half of the electricity it buys to
technical and commercial losses. Most utilities in this situation have meters at only a few
points on each feeder. This project asks: with only a handful of meters, how well can we
find *where* power is being lost, and how much?

## Approach
1. Model an 11 kV radial feeder: the IEEE 33-bus layout rebuilt with real Freetown cable
   constants, a 33/11 kV substation transformer, and household demand from a 2023 Freetown survey.
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
python scripts\run_freetown_day.py
python scripts\generate_dataset.py --scenarios 100 --days 7
python scripts\run_experiments.py
```

## Grounded in Sierra Leone data
Every parameter is traced to a published source in [`docs/sources.md`](docs/sources.md), including:
- Household appliance ownership and hourly use in Freetown (Rogers et al., 2024)
- Real Freetown 11 kV cable and line constants and substation transformers (JICA, 2009)
- Freetown network structure and measured weak-bus voltages (Mansaray et al., 2024)
- Sierra Leone loss levels from five sources (30–53%)

## The Freetown-style feeder (default settings)
| Quantity | Value | Basis |
|---|---|---|
| Line length | 50 km, 29% underground | JICA area lengths (22–76 km) |
| Customers | 12,000 households (~54 transformers) + 1.2 MW commercial | ~220 households per transformer |
| Peak load | ~4.3 MW | one 11 kV main line carries ~5 MW |
| Lowest voltage at peak | ~0.90 pu | measured Freetown weak buses 0.87–0.90 pu |
| MV technical losses | ~3% of energy | MV network losses 4.8% |

## Scenarios: what is simulated and what the utility sees
- **Theft:** at a set of hotspot buses, 15%, 25% or 35% of all delivered energy is stolen
  (Sierra Leone commercial losses: 18–40%). The feeder load stays realistic; theft only
  changes how much is billed.
- **Cable faults:** in 30% of scenarios, a constant leakage of 40–150 kW starts at a random hour.
- **The utility's view:** a feeder-head meter, six transformer meters with 2% noise and 20%
  missing readings, and daily billing per transformer area. No customer smart meters.

See [`docs/data_format.md`](docs/data_format.md) for the exact data a utility would need to
provide to run the models on a real feeder.

## Results
On 25 test scenarios (full log in [`docs/results.md`](docs/results.md)):

| Method | Theft labels needed | Finds theft at unmetered transformers (AUC) | Error in total stolen energy |
|---|---|---|---|
| Energy audit (standard utility method) | none | 0.65 | 3.4% |
| Audit + transformer ratings | none | 0.90 | 47.7% |
| Pure ML | 3 labelled weeks | 0.92 | 23.6% |
| Pure ML | 75 labelled weeks | 0.97 | 12.2% |
| **Physics-informed (this project)** | **none** | **0.92** | **2.7%** |

With no theft labels, the physics-informed model finds theft as well as machine learning
trained on three weeks of labelled theft data, and estimates the stolen amount 4-17 times
more accurately than any other method that can say where the theft is.

**How it works:** for each feeder, it finds the theft pattern that best explains the
substation meter, a few transformer meters and the billing records, while obeying the
feeder's power balance and the voltage drop through its real cables (checked against full
AC power flow). Transformer kVA ratings from the asset register give it a starting guess.
It runs in about 5 seconds per feeder-week on a laptop.

## Layout
- `src/slgrid/feeder.py` — base IEEE 33-bus feeder and the standard feeder format
- `src/slgrid/freetown.py` — the Freetown-style feeder
- `src/slgrid/loads.py` — household and commercial demand profiles
- `src/slgrid/simulate.py` — time-series power flow
- `src/slgrid/anomalies.py` — theft and fault injection with labels
- `src/slgrid/metering.py` — what the utility observes (meters, billing, noise, gaps)
- `src/slgrid/dataset.py` — labelled dataset generator
- `src/slgrid/topology.py` — network position features from the line list
- `src/slgrid/evaluation.py` — common metrics for all methods
- `src/slgrid/baselines.py` — energy audit, audit + ratings, and pure ML baselines
- `src/slgrid/physics_model.py` — the physics-informed locator
- `data/sources/` — data extracted from published sources
- `docs/sources.md` — citations and how each number is used
- `scripts/` — runnable experiments
- `tests/` — checks that the physics and calibration are correct

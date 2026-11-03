# sl-grid-piml

Physics-informed machine learning for locating electricity losses (theft and faults)
on distribution feeders with very few meters, using a Sierra Leone 11 kV feeder as the case study.

**Status:** Phase 3 — theft and fault scenarios, realistic utility metering, and a labelled dataset generator.

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

## Layout
- `src/slgrid/feeder.py` — base IEEE 33-bus feeder and the standard feeder format
- `src/slgrid/freetown.py` — the Freetown-style feeder
- `src/slgrid/loads.py` — household and commercial demand profiles
- `src/slgrid/simulate.py` — time-series power flow
- `src/slgrid/anomalies.py` — theft and fault injection with labels
- `src/slgrid/metering.py` — what the utility observes (meters, billing, noise, gaps)
- `src/slgrid/dataset.py` — labelled dataset generator
- `data/sources/` — data extracted from published sources
- `docs/sources.md` — citations and how each number is used
- `scripts/` — runnable experiments
- `tests/` — checks that the physics and calibration are correct

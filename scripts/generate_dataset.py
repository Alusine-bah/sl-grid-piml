"""Generate a labelled dataset of theft and fault scenarios.

Example (about 10-20 minutes on a laptop):
    python scripts\\generate_dataset.py --scenarios 100 --days 7

Writes to data/generated/:
    <name>.npz               all arrays (see src/slgrid/dataset.py)
    <name>_scenarios.csv     one row per scenario: loss level, hotspots, fault, ...
    <name>_config.json       settings used
and a summary plot of the first scenario to outputs/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slgrid.dataset import DatasetConfig, generate

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", type=int, default=100)
ap.add_argument("--days", type=int, default=7)
ap.add_argument("--dropout", type=float, default=0.2, help="share of meter readings missing")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--name", default="freetown_v1")
args = ap.parse_args()

cfg = DatasetConfig(n_scenarios=args.scenarios, n_days=args.days,
                    dropout=args.dropout, seed=args.seed)
path = generate(cfg, "data/generated", name=args.name)
d = np.load(path)
sc = pd.read_csv(path.with_name(f"{args.name}_scenarios.csv"))

print(f"\nSaved {path} ({path.stat().st_size / 1e6:.1f} MB)")
print(f"Scenarios: {len(sc)}, each {args.days} days of hourly data")
print(f"Theft levels: {', '.join(f'{x:.0%}' for x in sorted(sc.target_ntl.unique()))}; "
      f"scenarios with a cable fault: {sc.has_fault.sum()}")
print(f"Hotspot buses per scenario: {sc.n_theft_buses.min()}-{sc.n_theft_buses.max()} of 32")
print(f"Unbilled energy the utility sees: {100*sc.unbilled_seen.min():.0f}-"
      f"{100*sc.unbilled_seen.max():.0f}%")
print(f"Missing transformer-meter readings: {100*np.isnan(d['meter_p']).mean():.0f}%")

# Plot scenario 0: what the utility sees vs what is really happening.
out = Path("outputs"); out.mkdir(exist_ok=True)
s = 0
head = d["head"][s, :, 0]
true = d["true_bus_p"][s].sum(axis=1)
theft = d["theft_bus_p"][s].sum(axis=1)
billed_daily = d["billing"][s].sum(axis=1)
fig, axes = plt.subplots(2, 1, figsize=(9, 7))
ax = axes[0]
ax.plot(head, label="Feeder-head meter (what leaves the substation)")
ax.plot(true - theft, label="Legitimate consumption (true, not observed hourly)")
ax.fill_between(range(len(true)), true - theft, true, alpha=0.3, color="red", label="Stolen")
ax.set_xlabel("Hour")
ax.set_ylabel("MW")
ax.set_title(f"Scenario {s}: theft level {sc.target_ntl[s]:.0%}, "
             f"{sc.n_theft_buses[s]} hotspot buses, fault: {bool(sc.has_fault[s])}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
stolen_share = d["theft_bus_p"][s].sum(axis=0) / d["true_bus_p"][s].sum(axis=0)
names = [n.replace("bus_", "") for n in d["bus_names"]]
colors = ["tab:red" if t else "tab:gray" for t in d["is_theft"][s]]
ax.bar(names, 100 * stolen_share, color=colors)
for i, f in enumerate(d["is_fault"][s]):
    if f:
        ax.annotate("fault", (i, 2), ha="center", color="tab:orange", fontsize=8)
ax.set_xlabel("Feeder bus")
ax.set_ylabel("Consumption stolen (%)")
ax.set_title("Where the theft is (the answer the models must find)")
ax.tick_params(axis="x", labelsize=7)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(out / f"{args.name}_scenario0.png", dpi=150)
print(f"Saved plot to {out / f'{args.name}_scenario0.png'}")

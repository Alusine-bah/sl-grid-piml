"""How many transformer meters does a feeder need when losses are spread as on real feeders?

    python scripts\\run_metering_study.py

Uses the realistic (Odisha) loss pattern of Phase 14 (data/generated/realism_odisha.npz,
made by run_realism_study.py, or generated here). On 30 feeder-weeks, places 0 to 24
transformer meters on the 32 transformer areas, at the far ends of branches ("ends") or on
the largest transformers ("largest"), with 20% of readings missing. Measures, over ALL areas
(metered ones included, since a utility cares about the whole feeder):
  top 30% reach   share of the true stolen energy in the 30% of areas ranked first, as a
                  share of the best possible
  MAE MWh         mean error in each area's stolen energy (per week)
  total err %     error in the feeder's total stolen energy

Takes roughly 20-30 minutes on a laptop. Writes results/metering.csv.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_realism_study import reach
from slgrid.baselines import combined_rules
from slgrid.dataset import DatasetConfig, generate
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator
from slgrid.studies import placement, remeter

COUNTS = [0, 3, 6, 10, 16, 24]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    path = Path("data/generated/realism_odisha.npz")
    if not path.exists():
        print("Generating realism_odisha (60 feeder-weeks)...")
        generate(DatasetConfig(n_scenarios=60, n_days=7, theft_pattern="odisha", seed=707),
                 "data/generated", name="realism_odisha")
    d0 = dict(np.load(path))
    idx = np.arange(0, 60, 2)                     # 30 feeder-weeks
    true = d0["theft_bus_p"][idx].sum(axis=1)
    rows = []
    for strategy in ["ends", "largest"]:
        for n in COUNTS:
            if n == 0 and strategy == "largest":
                continue
            buses = placement(d0, n, strategy, seed=1)
            d = remeter(d0, buses, dropout=0.2, seed=5)
            print(f"{strategy:8s} {n:2d} meters...", flush=True)
            _, st, _ = PhysicsInformedLocator(d, PhysicsConfig(ac_iterations=1, detect_faults=True)).predict(idx, n_jobs=args.jobs)
            _, st_r = combined_rules(d, idx)
            for m, s_ in [("physics-informed", st), ("combined rules", st_r)]:
                rows.append({"strategy": "none" if n == 0 else strategy, "meters": n, "method": m,
                             "top 30% reach %": 100 * reach(s_, true),
                             "MAE MWh": float(np.abs(s_ - true).mean()),
                             "total err %": 100 * float(np.mean(np.abs(s_.sum(1) - true.sum(1))
                                                                / true.sum(1)))})
            print("   physics %.1f%% reach, rules %.1f%%" % (rows[-2]["top 30% reach %"],
                                                            rows[-1]["top 30% reach %"]), flush=True)
    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/metering.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + out.round(2).to_string(index=False))
    print("\nSaved results/metering.csv")


if __name__ == "__main__":
    main()

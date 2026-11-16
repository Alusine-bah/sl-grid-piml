"""Does the method still work when theft is spread as on real feeders?

    python scripts\\run_realism_study.py

Real transformer accounts from TP Central Odisha (data/real/) show losses spread over
most transformer areas of a feeder, with half the lost energy in about 3 in 10 areas.
The original simulator put theft in a few hotspots and left most areas clean. This
study generates two datasets of 60 feeder-weeks with the same settings and seed, one per
pattern, and compares the methods on both.

Measures:
  AUC unmetered     telling theft areas from clean ones (with spread losses, an area
                    counts as theft above 15% of its consumption)
  total err %       error in the feeder's total stolen energy
  top 30% reach     share of the true stolen energy in the 30% of areas the method
                    ranks first, as a share of the best possible (an oracle ranking)
  MAE MWh           mean error in each area's stolen energy

Takes roughly 20-30 minutes on a laptop. Writes results/realism.csv.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from slgrid.baselines import combined_rules, energy_audit, metered_mask
from slgrid.dataset import DatasetConfig, generate
from slgrid.evaluation import evaluate
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator


def reach(st_hat, st_true, frac=0.3):
    out = []
    for s in range(len(st_true)):
        k = max(1, int(round(frac * st_true.shape[1])))
        got = st_true[s][np.argsort(-st_hat[s])[:k]].sum()
        best = np.sort(st_true[s])[::-1][:k].sum()
        out.append(got / best if best > 0 else np.nan)
    return float(np.nanmean(out))


def concentration(st_true):
    """Share of areas holding half the stolen energy (median over feeders)."""
    fr = []
    for s in st_true:
        x = np.sort(s)[::-1]
        fr.append((np.searchsorted(np.cumsum(x) / x.sum(), 0.5) + 1) / len(x))
    return float(np.median(fr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    rows = []
    for pattern in ["hotspot", "odisha"]:
        name = f"realism_{pattern}"
        path = Path("data/generated") / f"{name}.npz"
        if not path.exists():
            print(f"Generating {name} (60 feeder-weeks)...")
            generate(DatasetConfig(n_scenarios=60, n_days=7, theft_pattern=pattern, seed=707),
                     "data/generated", name=name)
        d = dict(np.load(path))
        idx = np.arange(len(d["is_theft"]))
        mm = metered_mask(d)
        true = d["theft_bus_p"][idx].sum(axis=1)
        share = 100 * true / d["true_bus_p"][idx].sum(axis=1)
        print(f"Evaluating {pattern}...")
        sc, st, _ = PhysicsInformedLocator(d, PhysicsConfig(ac_iterations=1, detect_faults=True)
                                           ).predict(idx, n_jobs=args.jobs)
        methods = {"P1 physics-informed": (sc, st), "B1c combined rules": combined_rules(d, idx),
                   "B1 energy audit": energy_audit(d, idx)}
        for m, (s_, st_) in methods.items():
            r = evaluate(m, s_, st_, d, idx, 0.3, mm)
            rows.append({
                "pattern": pattern, "method": m,
                "areas with theft %": 100 * d["is_theft"].mean(),
                "median area loss %": float(np.median(share)),
                "areas holding half the theft %": 100 * concentration(true),
                "AUC unmetered": r.loc[r.buses == "unmetered", "AUC"].iloc[0],
                "total err %": r["feeder err %"].iloc[0],
                "top 30% reach %": 100 * reach(st_, true),
                "MAE MWh": r.loc[r.buses == "all buses", "MAE_MWh"].iloc[0]})
    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/realism.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + out.round(3).to_string(index=False))
    print("\nSaved results/realism.csv")


if __name__ == "__main__":
    main()

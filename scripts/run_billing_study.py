"""Does the method still work with monthly bills or prepaid token sales?

    python scripts\\run_billing_study.py

Re-bills the 25 test scenarios of the dataset as (a) daily values, (b) one total per bus
for the whole period, with 3%, 10% and 20% error, then runs the physics-informed locator
and the combined-rules baseline. Writes results/billing.csv.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from slgrid.baselines import combined_rules, metered_mask
from slgrid.evaluation import evaluate, split_scenarios
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator
from slgrid.studies import rebill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/generated/freetown_v3.npz")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    d = dict(np.load(args.data))
    _, test = split_scenarios(len(d["is_theft"]))
    mm = metered_mask(d)
    rows = []
    for resolution, noise in [("daily", 0.03), ("period", 0.03), ("period", 0.10), ("period", 0.20)]:
        dd = rebill(d, resolution, noise, seed=5)
        s, st, _ = PhysicsInformedLocator(dd, PhysicsConfig(ac_iterations=1)).predict(test, n_jobs=args.jobs)
        r = evaluate("P1", s, st, dd, test, 0.3, mm)
        s2, st2 = combined_rules(dd, test)
        r2 = evaluate("B1c", s2, st2, dd, test, 0.3, mm)
        rows.append({"billing": "daily" if resolution == "daily" else "one total per period",
                     "billing error": noise,
                     "P1 AUC unmetered": r.loc[r.buses == "unmetered", "AUC"].iloc[0],
                     "P1 MAE MWh": r["MAE_MWh"].iloc[0], "P1 total err %": r["feeder err %"].iloc[0],
                     "B1c AUC unmetered": r2.loc[r2.buses == "unmetered", "AUC"].iloc[0],
                     "B1c MAE MWh": r2["MAE_MWh"].iloc[0], "B1c total err %": r2["feeder err %"].iloc[0]})
        print(f"  done: {rows[-1]['billing']}, error {noise:.0%}")
    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/billing.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + out.round(3).to_string(index=False))
    print("\nSaved results/billing.csv")


if __name__ == "__main__":
    main()

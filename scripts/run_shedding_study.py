"""How well does the method work under load shedding?

    python scripts\\run_shedding_study.py

Generates three datasets of 40 feeder-weeks each, half with a cable fault, under:
  4h        4 hours off per day
  8h        8 hours off per day
  rotation  supplied 7 pm to 7 am one day in three (2008 load-sharing programme, JICA 2009)
Demand surges for two hours when supply returns. Then runs the physics-informed locator
(with fault detection) and the combined-rules baseline.

Takes roughly 15-25 minutes on a laptop. Writes results/shedding.csv.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from slgrid.baselines import combined_rules, metered_mask
from slgrid.dataset import DatasetConfig, generate
from slgrid.evaluation import evaluate
from slgrid.faults import FaultDetector
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    rows = []
    for pattern, seed in [("4h", 301), ("8h", 302), ("rotation", 303)]:
        path = Path("data/generated") / f"shed_{pattern}.npz"
        if not path.exists():
            print(f"Generating shed_{pattern} (40 feeder-weeks)...")
            generate(DatasetConfig(n_scenarios=40, n_days=7, fault_probability=0.5,
                                   shedding=pattern, seed=seed), "data/generated", name=f"shed_{pattern}")
        print(f"Evaluating shed_{pattern}...")
        d = dict(np.load(path))
        S = len(d["is_theft"])
        idx = np.arange(S)
        mm = metered_mask(d)
        has = d["is_fault"].any(axis=1)
        s, st, _ = PhysicsInformedLocator(d, PhysicsConfig(ac_iterations=1, detect_faults=True)).predict(idx, n_jobs=args.jobs)
        r = evaluate("P1", s, st, d, idx, 0.3, mm)
        s2, st2 = combined_rules(d, idx)
        r2 = evaluate("B1c", s2, st2, d, idx, 0.3, mm)
        fd = FaultDetector(d)
        det, P, L = [], [], []
        for k in range(S):
            o = fd.analyse(k)
            det.append(o["detected"])
            if has[k] and o["detected"]:
                P.append(o["bus_prob"])
                L.append(d["is_fault"][k])
        det = np.array(det)
        rows.append({
            "shedding": pattern,
            "hours supplied per week": float(d["supply"].sum(axis=1).mean()),
            "P1 AUC unmetered": r.loc[r.buses == "unmetered", "AUC"].iloc[0],
            "P1 total err %": r["feeder err %"].iloc[0],
            "B1c AUC unmetered": r2.loc[r2.buses == "unmetered", "AUC"].iloc[0],
            "B1c total err %": r2["feeder err %"].iloc[0],
            "faults detected": f"{det[has].sum()}/{has.sum()}",
            "false alarms": f"{det[~has].sum()}/{(~has).sum()}",
            "fault location AUC": roc_auc_score(np.concatenate(L), np.concatenate(P)) if P else np.nan,
        })
    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/shedding.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + out.round(3).to_string(index=False))
    print("\nSaved results/shedding.csv")


if __name__ == "__main__":
    main()

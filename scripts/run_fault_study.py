"""Cable-fault detection study.

    python scripts\\run_fault_study.py

Generates two datasets of 60 feeder-weeks each, half with a cable fault (40-150 kW, random
start hour): one for development (seed 101), one held back for testing (seed 202). The
detection threshold (4.0) was chosen on the development set only. Then measures detection,
timing, size and location of faults, and the effect on theft estimates.

Takes roughly 20-30 minutes on a laptop. Writes results/faults.csv.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from slgrid.baselines import metered_mask
from slgrid.dataset import DatasetConfig, generate
from slgrid.evaluation import evaluate
from slgrid.faults import FaultDetector
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator


def study(d, jobs):
    fd = FaultDetector(d)
    S = len(d["is_fault"])
    has = d["is_fault"].any(axis=1)
    score, det, t_err, kw_err, top1, top3, inset, size, P, L = [], [], [], [], [], [], [], [], [], []
    for s in range(S):
        o = fd.analyse(s)
        score.append(o["score"])
        det.append(o["detected"])
        if has[s] and o["detected"]:
            j = int(np.flatnonzero(d["is_fault"][s])[0])
            f = d["fault_bus_p"][s, :, j]
            rank = int((o["bus_prob"] > o["bus_prob"][j]).sum()) + 1
            top1.append(rank == 1)
            top3.append(rank <= 3)
            inset.append(fd.buses[j] in o["likely_section"])
            size.append(len(o["likely_section"]))
            t_err.append(abs(o["start_hour"] - int(np.argmax(f > 0))))
            kw_err.append(abs(o["leak_kw"] - 1000 * f.max()) / (1000 * f.max()))
            P.append(o["bus_prob"])
            L.append(d["is_fault"][s])
    det = np.array(det)
    row = {"weeks": S, "faults": int(has.sum()),
           "detected": int(det[has].sum()), "false alarms": int(det[~has].sum()),
           "detection AUC": roc_auc_score(has, score),
           "start error h (median)": float(np.median(t_err)),
           "size error % (median)": 100 * float(np.median(kw_err)),
           "location AUC": roc_auc_score(np.concatenate(L), np.concatenate(P)),
           "true bus ranked 1st": float(np.mean(top1)), "true bus in top 3": float(np.mean(top3)),
           "true bus in 80% section": float(np.mean(inset)),
           "section size (median buses)": float(np.median(size))}
    idx = np.arange(S)
    mm = metered_mask(d)
    for flag, tag in [(False, "without"), (True, "with")]:
        model = PhysicsInformedLocator(d, PhysicsConfig(ac_iterations=1, detect_faults=flag))
        s_, st, _ = model.predict(idx, n_jobs=jobs)
        r = evaluate("P1", s_, st, d, idx, 0.3, mm)
        row[f"theft AUC unmetered, {tag} detector"] = r.loc[r.buses == "unmetered", "AUC"].iloc[0]
        row[f"theft total err %, {tag} detector"] = r["feeder err %"].iloc[0]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    rows = []
    for name, seed in [("faults_dev", 101), ("faults_test", 202)]:
        path = Path("data/generated") / f"{name}.npz"
        if not path.exists():
            print(f"Generating {name} (60 feeder-weeks)...")
            generate(DatasetConfig(n_scenarios=60, n_days=7, fault_probability=0.5, seed=seed),
                     "data/generated", name=name)
        print(f"Evaluating {name}...")
        row = study(dict(np.load(path)), args.jobs)
        rows.append({"dataset": name.replace("faults_", ""), **row})
    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/faults.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + out.set_index("dataset").T.round(3).to_string())
    print("\nSaved results/faults.csv")


if __name__ == "__main__":
    main()

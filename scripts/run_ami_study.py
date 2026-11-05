"""Large customers on smart meters (AMI): what they add, and finding tampered meters.

    python scripts\\run_ami_study.py

Generates 40 feeder-weeks (seed 505). In each, the three commercial transformer areas get
12 large customers each on smart meters, with 28 days of hourly history (see
src/slgrid/ami.py). Where the feeder simulation put theft at a commercial area, some large
customers tamper with their meters. Compares the feeder model with:
  no smart-meter data        large customers billed like everyone else (daily totals)
  smart meters               exact hourly sales and voltage at the commercial areas
  smart meters + check meter a transformer meter on each commercial transformer as well
and ranks the large customers for inspection.

Takes roughly 15-25 minutes on a laptop. Writes results/ami.csv.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from slgrid.ami import AMIConfig, ami_inputs, large_customer_buses, rank_large_customers, simulate_ami
from slgrid.baselines import metered_mask
from slgrid.dataset import DatasetConfig, generate
from slgrid.evaluation import evaluate
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator
from slgrid.studies import remeter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    path = Path("data/generated") / "ami.npz"
    if not path.exists():
        print("Generating ami (40 feeder-weeks)...")
        generate(DatasetConfig(n_scenarios=40, n_days=7, seed=505), "data/generated", name="ami")
    d = dict(np.load(path))
    S = len(d["is_theft"])
    idx = np.arange(S)
    T = d["head"].shape[1]

    print("Simulating smart meters of large customers...")
    custs, reads = [], []
    for s in range(S):
        d, c, r = simulate_ami(d, s, AMIConfig(seed=5000 + s))
        custs.append(c)
        reads.append(r)
    big = large_customer_buses(d)
    names = [str(b) for b in d["bus_names"]]
    mm = metered_mask(d)                         # transformer meters as originally placed

    with_ami = d
    for s in range(S):
        with_ami = ami_inputs(with_ami, custs[s], reads[s], s)
    extra = [names[j] for j in big if names[j] not in list(d["meter_names"])]
    checked = remeter(with_ami, list(d["meter_names"]) + extra, dropout=0.2, seed=77)

    true = d["theft_bus_p"][idx].sum(axis=1)     # [S, B] MWh
    rows = []
    for case, dd in [("no smart-meter data", d), ("smart meters", with_ami),
                     ("smart meters + check meter", checked)]:
        print(f"Evaluating: {case}...")
        sc, st, _ = PhysicsInformedLocator(dd, PhysicsConfig(ac_iterations=1, detect_faults=True)
                                           ).predict(idx, n_jobs=args.jobs)
        r = evaluate("P1", sc, st, dd, idx, 0.3, mm)
        tb = true[:, big].sum(axis=1)
        has = tb > 0
        row = {"case": case,
               "P1 AUC unmetered": r.loc[r.buses == "unmetered", "AUC"].iloc[0],
               "P1 total err %": r["feeder err %"].iloc[0],
               "large-customer areas: theft error % (median)":
                   100 * float(np.median(np.abs(st[has][:, big].sum(axis=1) / tb[has] - 1)))}
        # rank large customers using this case's area estimates
        ys, ps, pa, hits_k, hits_plain, cover, early, late = [], [], [], [], [], [], [], []
        for s in range(S):
            c = custs[s]
            share = pd.Series(sc[s], index=names)
            rk = rank_large_customers(c, reads[s], T, share).merge(
                c[["meter", "tamper", "onset_hour", "stolen_kwh"]], on="meter")
            ys.append(rk.tamper.values); ps.append(rk.priority.values); pa.append(rk.anomaly.values)
            k = int(rk.tamper.sum())
            if k:
                hits_k.append(rk.head(k).tamper.mean())
                hits_plain.append(rk.sort_values("anomaly", ascending=False).head(k).tamper.mean())
                cover.append(rk.head(10).stolen_kwh.sum() / rk.stolen_kwh.sum())
                top = rk.head(max(k, 1))
                t = rk[rk.tamper]
                early += list(t.meter.isin(top.meter)[t.onset_hour < -28 * 24])
                late += list(t.meter.isin(top.meter)[t.onset_hour >= -28 * 24])
        y, p, a = np.concatenate(ys), np.concatenate(ps), np.concatenate(pa)
        row.update({
            "weeks with tampering": len(hits_k),
            "tamperers among large customers %": 100 * y.mean(),
            "meter AUC": roc_auc_score(y, p),
            "meter AUC, smart-meter data alone": roc_auc_score(y, a),
            "hits in top k (k = number of tamperers) %": 100 * np.mean(hits_k),
            "same, smart-meter data alone %": 100 * np.mean(hits_plain),
            "stolen energy covered by top 10 %": 100 * np.mean(cover),
            "found: started during history %": 100 * np.mean(late) if late else np.nan,
            "found: already tampering at start %": 100 * np.mean(early) if early else np.nan,
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/ami.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + out.set_index("case").T.round(3).to_string())
    print("\nSaved results/ami.csv")


if __name__ == "__main__":
    main()

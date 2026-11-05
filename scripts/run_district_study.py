"""Ranking feeders and sharing inspection crews across a district.

    python scripts\\run_district_study.py

Generates 60 feeder-weeks (seed 606) with very different theft levels (5% to 45% of
consumption) and a cable fault on about 30% of them, and groups them into 10 districts of
6 feeders. For each district it asks two management questions:

  1. Which feeder loses the most to theft? Feeders are ranked by the physics-informed
     estimate of stolen energy, and, as utilities often do, by unbilled energy (which also
     contains technical losses and cable-fault leakage).
  2. With crews for only 10% of the district's transformer areas (19 of 192), which areas
     should they visit? Compared:
       district-wide   the 19 areas with the highest estimated theft, whatever the feeder
       equal crews     3 areas on every feeder (the top 3 by estimated theft)
       worst feeders   all crews to the feeders with the most unbilled energy, areas chosen
                       by the combined-rules method
       random          19 random areas
     scored by the share of the district's true stolen energy the crews reach.

Takes roughly 10-20 minutes on a laptop. Writes results/district.csv.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from slgrid.baselines import combined_rules
from slgrid.dataset import DatasetConfig, generate
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator

N_DISTRICTS, PER = 10, 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    path = Path("data/generated") / "district.npz"
    if not path.exists():
        print("Generating district (60 feeder-weeks)...")
        generate(DatasetConfig(n_scenarios=N_DISTRICTS * PER, n_days=7, seed=606,
                               ntl_levels=(0.05, 0.10, 0.15, 0.25, 0.35, 0.45),
                               fault_probability=0.3), "data/generated", name="district")
    d = dict(np.load(path))
    S = len(d["is_theft"])
    idx = np.arange(S)
    print("Running the physics-informed locator on every feeder...")
    _, st, _ = PhysicsInformedLocator(d, PhysicsConfig(ac_iterations=1, detect_faults=True)
                                      ).predict(idx, n_jobs=args.jobs)
    _, st_rules = combined_rules(d, idx)
    true = d["theft_bus_p"][idx].sum(axis=1)                       # [S, B] MWh
    head = d["head"][idx, :, 0].sum(axis=1)
    unbilled = head - d["billing"][idx].sum(axis=(1, 2))
    B = true.shape[1]
    budget = int(round(0.10 * PER * B))

    rng = np.random.default_rng(6)
    order = rng.permutation(S)
    rows = []
    for k in range(N_DISTRICTS):
        f = order[k * PER:(k + 1) * PER]
        T_f = true[f].sum(axis=1)
        E_f = st[f].sum(axis=1)
        U_f = unbilled[f]
        total = true[f].sum()
        r = {"district": k + 1,
             "spearman physics": spearmanr(E_f, T_f).statistic,
             "spearman unbilled": spearmanr(U_f, T_f).statistic,
             "top feeder right, physics": float(np.argmax(E_f) == np.argmax(T_f)),
             "top feeder right, unbilled": float(np.argmax(U_f) == np.argmax(T_f)),
             "feeder total err % (median)": 100 * float(np.median(np.abs(E_f / T_f - 1)))}
        # 1) district-wide by physics estimate
        flat_e, flat_t = st[f].ravel(), true[f].ravel()
        r["crews: district-wide"] = flat_t[np.argsort(-flat_e)[:budget]].sum() / total
        # 2) equal crews per feeder
        per = budget // PER
        got = sum(true[s][np.argsort(-st[s])[:per]].sum() for s in f)
        rest = budget - per * PER
        if rest:      # leftover crews go district-wide to the next best areas
            chosen = {(s, j) for s in f for j in np.argsort(-st[s])[:per]}
            cand = sorted(((st[s, j], s, j) for s in f for j in range(B) if (s, j) not in chosen),
                          reverse=True)[:rest]
            got += sum(true[s, j] for _, s, j in cand)
        r["crews: equal per feeder"] = got / total
        # 3) worst feeders by unbilled energy, areas by combined rules
        left, got = budget, 0.0
        for s in f[np.argsort(-U_f)]:
            take = min(left, B)
            got += true[s][np.argsort(-st_rules[s])[:take]].sum()
            left -= take
            if left == 0:
                break
        r["crews: worst feeders by unbilled"] = got / total
        # 4) random
        r["crews: random"] = budget / (PER * B)
        rows.append(r)

    res = pd.DataFrame(rows)
    summary = res.drop(columns="district").mean().to_frame("mean").T
    for c in summary.columns:
        if c.startswith("crews") or c.startswith("top feeder"):
            summary[c] = 100 * summary[c]
    summary.insert(0, "districts", N_DISTRICTS)
    summary.insert(1, "feeders per district", PER)
    summary.insert(2, "crew budget (areas)", budget)
    Path("results").mkdir(exist_ok=True)
    summary.to_csv("results/district.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nPer district:\n" + res.round(3).to_string(index=False))
    print("\nMean (crews and top-feeder rows in %):\n" + summary.T.round(3).to_string())
    print("\nSaved results/district.csv")


if __name__ == "__main__":
    main()

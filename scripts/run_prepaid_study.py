"""Prepaid vending records: token sales, customer-to-transformer mapping, suspicious meters.

    python scripts\\run_prepaid_study.py

Generates 40 feeder-weeks (seed 404) and, for each, 90 days of simulated prepaid vending
records for about 12,000 customers (see src/slgrid/prepaid.py). Then compares the locator
fed with:
  daily billing (reference)       the billing used in earlier phases
  token sales in the week         purchases made during the 7 analysed days
  30-day token average            average daily purchases over the last 30 days x 7
and with 5%, 10% and 20% of customers linked to the wrong transformer.

It also ranks individual meters for inspection and checks how many of the top 100 are
customers who really bypass their meter.

Takes roughly 30-50 minutes on a laptop. Writes results/prepaid.csv.
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
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator
from slgrid.prepaid import VendingConfig, rank_meters, simulate_vending, tokens_to_billing

CASES = [  # name, lookback (None = window only, 0 = keep daily billing), mapping error
    ("daily billing (reference)", 0, 0.0),
    ("token sales in the week", None, 0.0),
    ("30-day token average", 30, 0.0),
    ("30-day average, 5% mis-mapped", 30, 0.05),
    ("30-day average, 10% mis-mapped", 30, 0.10),
    ("30-day average, 20% mis-mapped", 30, 0.20),
]
TOP = 100


def meter_metrics(cust, ranked, share):
    """Customer-level ranking quality for one scenario."""
    r = ranked.merge(cust[["meter", "is_bypass", "stolen_kwh_day"]], on="meter")
    y = r.is_bypass.to_numpy()
    if y.all() or not y.any():
        return None
    no_area = (r["missing_kwh_day"]).to_numpy()
    top = r.index[:TOP]
    top_plain = np.argsort(-no_area)[:TOP]
    return {"meter AUC": roc_auc_score(y, r.priority),
            "meter AUC without area weight": roc_auc_score(y, no_area),
            "bypassers in top 100": y[top].mean(),
            "bypassers in top 100 without area weight": y[top_plain].mean(),
            "bypassers among all meters": y.mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--scenarios", type=int, default=40)
    args = ap.parse_args()
    path = Path("data/generated") / "prepaid.npz"
    if not path.exists():
        print("Generating prepaid (40 feeder-weeks)...")
        generate(DatasetConfig(n_scenarios=40, n_days=7, seed=404), "data/generated", name="prepaid")
    d = dict(np.load(path))
    S = min(args.scenarios, len(d["is_theft"]))
    idx = np.arange(S)
    n_days = d["head"].shape[1] // 24
    mm = metered_mask(d)
    names = [str(b) for b in d["bus_names"]]

    rows = []
    for name, lookback, mapping in CASES:
        print(f"Evaluating: {name}...")
        dd = dict(d)
        vend = []
        if lookback != 0:
            bills = []
            for s in range(len(d["is_theft"])):
                if s < S:
                    c, t = simulate_vending(d, s, VendingConfig(mapping_error=mapping, seed=1000 + s))
                    vend.append((c, t))
                    bills.append(tokens_to_billing(c, t, names, n_days, lookback))
                else:
                    bills.append(d["billing"][s].sum(axis=0, keepdims=True))
            dd["billing"] = np.stack(bills)
        sc, st, _ = PhysicsInformedLocator(dd, PhysicsConfig(ac_iterations=1, detect_faults=True)
                                           ).predict(idx, n_jobs=args.jobs)
        r = evaluate("P1", sc, st, dd, idx, 0.3, mm)
        s2, st2 = combined_rules(dd, idx)
        r2 = evaluate("B1c", s2, st2, dd, idx, 0.3, mm)
        legit = (d["true_bus_p"] - d["theft_bus_p"] - d["fault_bus_p"])[idx].sum(axis=1)
        sold = dd["billing"][idx].sum(axis=1)
        row = {"case": name,
               "sales error per area % (median)": 100 * float(np.median(np.abs(sold / legit - 1))),
               "P1 AUC unmetered": r.loc[r.buses == "unmetered", "AUC"].iloc[0],
               "P1 total err %": r["feeder err %"].iloc[0],
               "B1c AUC unmetered": r2.loc[r2.buses == "unmetered", "AUC"].iloc[0],
               "B1c total err %": r2["feeder err %"].iloc[0]}
        if vend:
            mets = []
            for k, (c, t) in enumerate(vend):
                share = pd.Series(sc[k], index=names)
                m = meter_metrics(c, rank_meters(c, t, n_days, share), share)
                if m:
                    mets.append(m)
            row.update(pd.DataFrame(mets).mean().to_dict())
        rows.append(row)

    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/prepaid.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + out.set_index("case").T.round(3).to_string())
    print("\nSaved results/prepaid.csv")


if __name__ == "__main__":
    main()

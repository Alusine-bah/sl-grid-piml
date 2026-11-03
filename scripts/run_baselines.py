"""Run the baseline methods and save a results table and chart.

    python scripts\\run_baselines.py --data data\\generated\\freetown_v3.npz

Writes results/baselines.csv, results/baselines_labels.csv and outputs/baselines.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slgrid.baselines import PureML, energy_audit, metered_mask
from slgrid.evaluation import best_threshold, evaluate, split_scenarios

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/generated/freetown_v3.npz")
args = ap.parse_args()

d = dict(np.load(args.data))
train, test = split_scenarios(len(d["is_theft"]))
mm = metered_mask(d)
print(f"{len(train)} training and {len(test)} test scenarios; "
      f"{mm.sum()} of {len(mm)} buses have a transformer meter")


def run(name, score_fn, train_idx):
    s_tr, _ = score_fn(train_idx)
    thr = best_threshold(s_tr.ravel(), d["is_theft"][train_idx].ravel())
    s, st = score_fn(test)
    return evaluate(name, s, st, d, test, thr, mm)


results = [run("B1 energy audit", lambda i: energy_audit(d, i), train)]
ml = PureML().fit(d, train)
results.append(run("B2 pure ML", lambda i: ml.predict(d, i), train))
res = pd.concat(results, ignore_index=True)

# How much labelled history does pure ML need?
curve = []
for n in [3, 10, 25, len(train)]:
    sub = train[:n]
    m = PureML().fit(d, sub)
    r = run("B2", lambda i: m.predict(d, i), sub)
    curve.append({"labelled_scenarios": n,
                  "AUC_unmetered": r.loc[r.buses == "unmetered", "AUC"].iloc[0],
                  "feeder_err_pct": r["feeder err %"].iloc[0]})
curve = pd.DataFrame(curve)

Path("results").mkdir(exist_ok=True)
res.to_csv("results/baselines.csv", index=False)
curve.to_csv("results/baselines_labels.csv", index=False)

pd.set_option("display.width", 200)
print("\nBaseline results on test scenarios:")
print(res.round(3).to_string(index=False))
print("\nPure ML with fewer labelled scenarios:")
print(curve.round(3).to_string(index=False))
print(f"\nShare of buses that are hotspots (random guess level for AP): "
      f"{d['is_theft'][test].mean():.2f}")

Path("outputs").mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
groups = ["metered", "unmetered"]
x = np.arange(len(groups))
for k, (name, g) in enumerate(res[res.buses != "all buses"].groupby("method", sort=False)):
    ax.bar(x + 0.35 * k, g.set_index("buses").loc[groups, "AUC"], 0.35, label=name)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random guess")
ax.set_xticks(x + 0.175)
ax.set_xticklabels(["Buses with a meter", "Buses without a meter"])
ax.set_ylabel("AUC (1.0 = perfect)")
ax.set_ylim(0.4, 1.02)
ax.set_title("Finding theft hotspots")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
ax = axes[1]
allb = res[res.buses == "all buses"]
ax.bar(allb["method"], allb["feeder err %"], color=["tab:blue", "tab:orange"])
ax.set_ylabel("Error in total stolen energy (%)")
ax.set_title("Estimating how much is stolen")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("outputs/baselines.png", dpi=150)
print("\nSaved results/baselines.csv and outputs/baselines.png")

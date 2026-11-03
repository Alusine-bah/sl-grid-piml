"""Compare all methods on the test scenarios.

    python scripts\\run_experiments.py --data data\\generated\\freetown_v3.npz

Methods:
  B1   energy audit                       no labels
  B1b  energy audit + transformer ratings no labels
  B1c  ratings + meters + feeder gap rules no labels (strongest rule-based method)
  B2   pure ML (+ ratings), 3 and all     needs labelled theft history
       labelled training scenarios
  P1   physics-informed locator (ours)    no labels

Writes results/experiments.csv and outputs/experiments.png.
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from slgrid.baselines import PureML, capacity_audit, combined_rules, energy_audit, metered_mask
from slgrid.evaluation import best_threshold, evaluate, split_scenarios
from slgrid.physics_model import PhysicsInformedLocator

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/generated/freetown_v3.npz")
args = ap.parse_args()

d = dict(np.load(args.data))
train, test = split_scenarios(len(d["is_theft"]))
mm = metered_mask(d)
print(f"{len(train)} training / {len(test)} test scenarios; "
      f"{mm.sum()} of {len(mm)} buses metered\n")

results, timing = [], {}


def add(name, labels, fn, thr_idx):
    t0 = time.time()
    s_thr, _ = fn(thr_idx)[:2]
    thr = best_threshold(s_thr.ravel(), d["is_theft"][thr_idx].ravel())
    out = fn(test)
    timing[name] = (time.time() - t0) / (len(test) + len(thr_idx))
    r = evaluate(name, out[0], out[1], d, test, thr, mm)
    r.insert(1, "theft labels", labels)
    results.append(r)
    print(f"  done: {name}")
    return out


# Rule-based methods need no labels; a few training scenarios only set the threshold.
calib = train[:5]
add("B1 energy audit", "none", lambda i: energy_audit(d, i), calib)
add("B1b audit + ratings", "none", lambda i: capacity_audit(d, i), calib)
add("B1c combined rules", "none", lambda i: combined_rules(d, i), calib)
for n in (3, len(train)):
    ml = PureML().fit(d, train[:n])
    add(f"B2 pure ML ({n} weeks)", f"{n} weeks", lambda i, m=ml: m.predict(d, i), train[:n])
phys = PhysicsInformedLocator(d)
print("  running physics-informed model (a few seconds per scenario)...")
p_out = add("P1 physics-informed", "none", phys.predict, calib)

res = pd.concat(results, ignore_index=True)
res["sec / scenario"] = res["method"].map(timing)
Path("results").mkdir(exist_ok=True)
res.to_csv("results/experiments.csv", index=False)

pd.set_option("display.width", 220)
show = res[res.buses != "metered"].copy()
show["buses"] = show["buses"].replace({"all buses": "all"})
print("\nResults on test scenarios (unmetered = buses with no transformer meter):")
print(show.round(3).to_string(index=False))

f = p_out[2]
lab = d["is_fault"][test]
if lab.any() and not lab.all():
    print(f"\nFault localisation by P1 leakage estimate: AUC {roc_auc_score(lab.ravel(), f.ravel()):.2f} "
          f"({lab.any(axis=1).sum()} test scenarios contain a fault)")

# Chart: localisation at unmetered buses vs accuracy of total stolen energy.
allb = res[res.buses == "all buses"].set_index("method")
unm = res[res.buses == "unmetered"].set_index("method")
fig, ax = plt.subplots(figsize=(8, 5))
for name in allb.index:
    lbl = allb.loc[name, "theft labels"]
    marker = "o" if lbl == "none" else "s"
    color = "tab:red" if name.startswith("P1") else ("tab:gray" if lbl == "none" else "tab:blue")
    if name.startswith("B1b"):
        continue   # off the chart (total error ~50%); listed in the table
    ax.scatter(allb.loc[name, "feeder err %"], unm.loc[name, "AUC"], s=120, marker=marker,
               color=color, zorder=3)
    ax.annotate(f"{name}\n(labels: {lbl})", (allb.loc[name, "feeder err %"], unm.loc[name, "AUC"]),
                textcoords="offset points", xytext=(8, -4), fontsize=8)
ax.set_xlabel("Error in total stolen energy (%)  ← better")
ax.set_ylabel("Finding theft at unmetered transformers (AUC)  → better")
ax.set_title("Physics-informed vs baselines (test scenarios)")
ax.grid(alpha=0.3)
ax.set_xlim(0, allb.drop(index=[i for i in allb.index if i.startswith("B1b")])["feeder err %"].max() * 1.35)
fig.tight_layout()
Path("outputs").mkdir(exist_ok=True)
fig.savefig("outputs/experiments.png", dpi=150)
print("\nSaved results/experiments.csv and outputs/experiments.png")

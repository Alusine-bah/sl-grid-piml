"""Meter placement and robustness studies for the physics-informed model.

    python scripts\\run_studies.py

Uses the first 12 test scenarios of the dataset and one AC correction pass (which gives
the same accuracy as two at half the time). Scenarios run in parallel on all but two CPU
cores. Expect roughly 10-30 minutes on a laptop.

Writes results/placement.csv, results/robustness.csv, outputs/placement.png and
outputs/robustness.png.
"""
import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slgrid.baselines import combined_rules, metered_mask
from slgrid.evaluation import evaluate, split_scenarios
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator
from slgrid.studies import perturb_assets, placement, remeter


def score(name, d, idx, out, **extra):
    r = evaluate(name, out[0], out[1], d, idx, 0.3, metered_mask(d))
    row = {"AUC all": r.loc[r.buses == "all buses", "AUC"].iloc[0],
           "AUC unmetered": r.loc[r.buses == "unmetered", "AUC"].iloc[0],
           "top-k hit": r["top-k hit"].iloc[0],
           "feeder err %": r["feeder err %"].iloc[0]}
    row.update(extra)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/generated/freetown_v3.npz")
    ap.add_argument("--scenarios", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    d = dict(np.load(args.data))
    _, test = split_scenarios(len(d["is_theft"]))
    idx = test[: args.scenarios]
    cfg = PhysicsConfig(ac_iterations=1)
    print(f"Using {len(idx)} test scenarios, {args.jobs} parallel workers\n")

    def run_p1(dd):
        t0 = time.time()
        out = PhysicsInformedLocator(dd, cfg).predict(idx, n_jobs=args.jobs)
        return out, time.time() - t0


    # ---------------- Meter placement ----------------
    rows = []
    counts = [0, 3, 6, 12]
    for strategy in ["ends", "largest", "random"]:
        for n in counts:
            if n == 0 and strategy != "ends":
                continue
            buses = placement(d, n, strategy, seed=1)
            dd = remeter(d, buses, seed=2)
            out, sec = run_p1(dd)
            row = score("P1", dd, idx, out, strategy=strategy if n else "none",
                        meters=n, buses=" ".join(b.replace("bus_", "") for b in buses))
            rc = score("B1c", dd, idx, combined_rules(dd, idx))
            row["B1c AUC unmetered"] = rc["AUC unmetered"]
            row["B1c feeder err %"] = rc["feeder err %"]
            rows.append(row)
            print(f"  placement {strategy:8s} {n:2d} meters: AUC all {row['AUC all']:.3f}, "
                  f"total err {row['feeder err %']:.1f}%  ({sec:.0f}s)")
    pl = pd.DataFrame(rows)
    num = ["AUC all", "AUC unmetered", "top-k hit", "feeder err %", "B1c AUC unmetered",
           "B1c feeder err %"]
    zero = pl[pl.meters == 0].iloc[0]
    for st in ["largest", "random"]:
        z = zero.copy(); z["strategy"] = st
        pl = pd.concat([pl, z.to_frame().T], ignore_index=True)
    z = zero.copy(); z["strategy"] = "ends"
    pl = pd.concat([pl[pl.strategy != "none"], z.to_frame().T], ignore_index=True)
    pl = pl.sort_values(["strategy", "meters"]).reset_index(drop=True)
    pl[num] = pl[num].astype(float)

    # ---------------- Robustness ----------------
    base_buses = [str(b) for b in d["meter_names"]]
    rows = []
    tests = [("missing meter readings", "dropout", [0.2, 0.5, 0.8]),
             ("cable data error", "cable_error", [0.0, 0.1, 0.25]),
             ("transformer rating error", "kva_error", [0.0, 0.2, 0.4])]
    for label, key, levels in tests:
        for lv in levels:
            dd = remeter(d, base_buses, dropout=lv if key == "dropout" else 0.2, seed=2)
            if key != "dropout":
                dd = perturb_assets(dd, **{key: lv}, seed=3)
            out, sec = run_p1(dd)
            row = score("P1", dd, idx, out, test=label, level=lv)
            r2 = score("B1c", dd, idx, combined_rules(dd, idx))
            row["B1c AUC unmetered"] = r2["AUC unmetered"]
            row["B1c feeder err %"] = r2["feeder err %"]
            rows.append(row)
            print(f"  robustness {label:26s} {lv:<5}: AUC all {row['AUC all']:.3f}, "
                  f"total err {row['feeder err %']:.1f}%  ({sec:.0f}s)")
    rb = pd.DataFrame(rows)

    Path("results").mkdir(exist_ok=True)
    pl.to_csv("results/placement.csv", index=False)
    rb.to_csv("results/robustness.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["strategy", "meters", "AUC unmetered", "B1c AUC unmetered", "feeder err %",
            "B1c feeder err %"]
    print("\nMeter placement: physics-informed (P1) vs combined rules (B1c), unmetered buses:")
    print(pl[cols].round(3).to_string(index=False))
    print("\nRobustness (6 meters at the default positions):")
    print(rb[["test", "level", "AUC unmetered", "B1c AUC unmetered", "feeder err %",
              "B1c feeder err %"]].round(3).to_string(index=False))

    # ---------------- Charts ----------------
    Path("outputs").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = {"ends": "far ends of branches", "largest": "largest transformers",
              "random": "random"}
    colors = {"ends": "tab:red", "largest": "tab:orange", "random": "tab:purple"}
    for st, g in pl.groupby("strategy"):
        g = g.sort_values("meters")
        axes[0].plot(g.meters, g["AUC unmetered"], marker="o", color=colors[st],
                     label=f"Physics-informed, {labels[st]}")
        axes[0].plot(g.meters, g["B1c AUC unmetered"], marker="x", linestyle=":",
                     color=colors[st], alpha=0.7, label=f"Combined rules, {labels[st]}")
        axes[1].plot(g.meters, g["feeder err %"], marker="o", color=colors[st],
                     label=f"Physics-informed, {labels[st]}")
        axes[1].plot(g.meters, g["B1c feeder err %"], marker="x", linestyle=":",
                     color=colors[st], alpha=0.7, label=f"Combined rules, {labels[st]}")
    axes[0].set_ylabel("Finding theft at unmetered buses (AUC)")
    axes[1].set_ylabel("Error in total stolen energy (%)")
    for ax in axes:
        ax.set_xlabel("Number of transformer meters (out of 32)")
        ax.set_xticks(counts)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Where to put meters")
    axes[1].set_title("How much is stolen")
    fig.tight_layout()
    fig.savefig("outputs/placement.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    xl = {"missing meter readings": "Share of meter readings missing",
          "cable data error": "Cable impedance error (relative sd)",
          "transformer rating error": "Transformer rating error (log sd)"}
    for ax, (label, g) in zip(axes, rb.groupby("test", sort=False)):
        ax.plot(g.level, g["AUC unmetered"], marker="o", color="tab:red", label="Physics-informed")
        ax.plot(g.level, g["B1c AUC unmetered"], marker="x", linestyle=":", color="gray",
                label="Combined rules")
        ax.set_xlabel(xl[label])
        ax.set_ylabel("AUC, unmetered buses")
        ax.set_ylim(0.6, 1.0)
        ax.set_title(label.capitalize())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/robustness.png", dpi=150)
    print("\nSaved results/placement.csv, results/robustness.csv and two charts in outputs/")


if __name__ == "__main__":
    main()

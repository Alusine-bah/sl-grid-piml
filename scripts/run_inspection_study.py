"""Inspection rounds in closed loop: does the planning policy find theft, and can its
numbers be trusted?

    python scripts\\run_inspection_study.py

Uses the 60 simulated feeder-weeks of the district study (1,920 transformer areas; theft
5-45%). The physics-informed locator gives every area an estimated stolen share and
stolen energy. Crews then make 8 monthly rounds of 60 visits:
  * a crew finds theft in a theft area with probability 0.8 x the share of premises it
    checks (60-100%), and never in a clean area;
  * 8% of visits to clean areas and 15% to theft areas end with no access (refusal is
    more likely where there is something to hide);
  * an area visited in the last two rounds is not visited again (it has just been dealt
    with).
Policies compared:
  greedy        the 60 areas with the highest estimated stolen energy (usual practice)
  top+random    the planner: 85% to the top areas, 15% random audit of the rest,
                logged probabilities
  top+random+learn  the same, re-weighting each round with what crews found
  spread        all visits drawn at random, tilted toward high-risk areas
  random        60 random areas
Two situations: accurate physics, and physics fooled on 15 feeders whose substation meter
reads 3x high. Measured: theft reached and found, and how well each policy's own numbers
estimate the true share of areas with theft (the question a regulator or a concession
audit asks). Finally, whether learning makes the displayed probabilities honest.

Takes about 10-15 minutes the first time (it runs the locator on the 60 feeders);
a few minutes afterwards. Writes results/inspections.csv and
results/inspections_calibration.csv.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from slgrid.dataset import DatasetConfig, generate
from slgrid.inspections import (InspectionConfig, area_posterior, clean_log, evaluate_log,
                                fit_calibration, plan_round)
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator

ROUNDS, BUDGET, REPS = 8, 60, 30


def load_population(jobs):
    cache = Path("data/generated/district_physics.npz")
    path = Path("data/generated/district.npz")
    if not path.exists():
        print("Generating district (60 feeder-weeks)...")
        generate(DatasetConfig(n_scenarios=60, n_days=7, seed=606,
                               ntl_levels=(0.05, 0.10, 0.15, 0.25, 0.35, 0.45),
                               fault_probability=0.3), "data/generated", name="district")
    d = dict(np.load(path))
    if cache.exists():
        c = np.load(cache)
        sc, st = c["score"], c["stolen"]
    else:
        print("Running the physics-informed locator on 60 feeders...")
        idx = np.arange(len(d["is_theft"]))
        sc, st, _ = PhysicsInformedLocator(d, PhysicsConfig(ac_iterations=1, detect_faults=True)
                                           ).predict(idx, n_jobs=jobs)
        np.savez(cache, score=sc, stolen=st)
    S, B = sc.shape
    true_st = d["theft_bus_p"].sum(axis=1)
    return pd.DataFrame({
        "feeder": np.repeat([f"F{s:02d}" for s in range(S)], B),
        "bus": np.tile([str(b) for b in d["bus_names"]], S),
        "model_score": sc.ravel(), "weight": np.maximum(st.ravel(), 1e-6),
        "theft": d["is_theft"].ravel().astype(bool), "true_stolen": true_st.ravel()})


def ct_error(pop, n_feeders=15, factor=3.0, seed=7):
    """Physics fooled on some feeders: the substation meter reads high (a wrong current-
    transformer ratio), so the model sees `factor` times the stolen energy everywhere on
    those feeders. Crews find what is really there."""
    rng = np.random.default_rng(seed)
    bad = rng.choice(sorted(pop.feeder.unique()), n_feeders, replace=False)
    p = pop.copy()
    m = p.feeder.isin(bad)
    s = p.loc[m, "model_score"]
    p.loc[m, "weight"] *= factor
    p.loc[m, "model_score"] = factor * s / (1 + (factor - 1) * s)
    return p


def visit(rows, pop, rng):
    """Simulated crew outcomes for the planned rows."""
    t = pop.set_index(["feeder", "bus"]).loc[list(zip(rows.feeder, rows.bus))]
    theft = t["theft"].values
    cov = rng.uniform(0.6, 1.0, len(rows))
    miss = rng.random(len(rows)) < np.where(theft, 0.15, 0.08)
    found = theft & (rng.random(len(rows)) < 0.8 * cov)
    out = rows.copy()
    out["premises"], out["premises_checked"] = 100, np.round(100 * cov)
    out["outcome"] = np.where(miss, "NO_ACCESS_REFUSED", np.where(found, "THEFT_BYPASS", "CLEAN"))
    out["kwh_per_day_found"] = np.nan
    out["true_theft"], out["true_stolen"] = theft, t["true_stolen"].values
    return out


def run(policy, pop, seed, cfg):
    rng = np.random.default_rng(seed)
    logs, cool = [], []
    for r in range(ROUNDS):
        frame = pop[~(pop.feeder + "|" + pop.bus).isin(set(sum(cool[-2:], [])))].reset_index(drop=True)
        areas = frame[["feeder", "bus", "model_score", "weight"]].copy()
        if policy.endswith("+learn") and logs:
            post = area_posterior(areas, clean_log(pd.concat(logs)), cfg)
            areas["weight"] = post["next_weight"].values
        if policy.startswith(("top+random", "spread")):
            c = InspectionConfig(policy="spread" if policy == "spread" else "top+random")
            plan = plan_round(areas, BUDGET, c, seed=seed * 100 + r, round_id=f"R{r + 1}")
        elif policy == "greedy":
            plan = areas.sort_values("weight", ascending=False).head(BUDGET).copy()
            plan["pi"], plan["selection_arm"] = 1.0, "model"
        else:
            plan = areas.sample(BUDGET, random_state=seed * 100 + r).copy()
            plan["pi"], plan["selection_arm"] = BUDGET / len(areas), "random"
        out = visit(plan, pop, rng)
        out["round"] = r
        out["frame_theft_share"] = frame.theft.mean()
        logs.append(out)
        cool.append(list(out.feeder + "|" + out.bus))
    log = clean_log(pd.concat(logs, ignore_index=True))
    ev = evaluate_log(log, n_areas=len(pop), n_boot=0)
    # what each policy would report as the share of areas with theft
    if policy.startswith(("top+random", "spread")):
        est = ev["ht_found_share"]
        est_random = ev["arms"].get("random", {}).get("hit_rate", np.nan)
    else:
        est = log["y"].mean()                  # its own hit rate is all it has
        est_random = est if policy == "random" else np.nan
    truth = pop.theft.mean()
    # allowing for theft that crews miss (sensitivity 0.8 x average coverage 0.8)
    scale = 1 / (0.8 * 0.8)
    return {"policy": policy, "seed": seed,
            "theft reached (MWh/week)": log.loc[log.true_theft, "true_stolen"].sum() / ROUNDS,
            "theft found (MWh/week)": log.loc[log.y == 1, "true_stolen"].sum() / ROUNDS,
            "hit rate, all valid visits": log["y"].mean(),
            "hit rate, model arm": ev["arms"].get("model", {}).get("hit_rate", np.nan),
            "hit rate, random arm": ev["arms"].get("random", {}).get("hit_rate", np.nan),
            "random-arm visits": ev["arms"].get("random", {}).get("n", 0),
            "estimated theft share of areas": min(est * scale, 1.0),
            "estimated, random arm only": min(est_random * scale, 1.0) if est_random == est_random else np.nan,
            "true theft share of areas": truth,
            "verdict model>random": ev["verdict"].startswith("model better")}


def calibration_check(pop, reps=10):
    """Do the displayed probabilities become honest? Start from the default map, or from a
    deliberately wrong one (as if copied from another utility), run 8 rounds with
    learning, and score P(theft) for every area against the truth (Brier score; lower
    is better)."""
    rows = []
    for scen, p in [("accurate physics", pop), ("substation meter reads 3x high on 15 feeders", ct_error(pop))]:
        y = p.theft.values.astype(float)
        for start, (a0, b0) in [("default map", (-2.2, 12.0)), ("wrong map", (-0.5, 3.0))]:
            cfg = InspectionConfig(default_a=a0, default_b=b0)
            before, after = [], []
            for rep in range(reps):
                rng = np.random.default_rng(rep)
                logs = []
                for r in range(ROUNDS):
                    areas = p[["feeder", "bus", "model_score", "weight"]]
                    if logs:
                        post = area_posterior(areas, clean_log(pd.concat(logs)), cfg)
                        areas = areas.assign(weight=post["next_weight"].values)
                    logs.append(visit(plan_round(areas, BUDGET, cfg, seed=rep * 100 + r), p, rng))
                log = clean_log(pd.concat(logs))
                a, b, _ = fit_calibration(log, cfg)
                p0 = 1 / (1 + np.exp(-(a0 + b0 * p.model_score.values)))
                p1 = 1 / (1 + np.exp(-(a + b * p.model_score.values)))
                before.append(((p0 - y) ** 2).mean())
                after.append(((p1 - y) ** 2).mean())
            rows.append({"scenario": scen, "start": start, "Brier before": np.mean(before),
                         "Brier after 8 rounds": np.mean(after)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    pop = load_population(args.jobs)
    print(f"{len(pop)} transformer areas, {pop.theft.mean():.0%} with theft.")
    cfg = InspectionConfig()
    rows = []
    for scen, p in [("accurate physics", pop), ("substation meter reads 3x high on 15 feeders", ct_error(pop))]:
        for policy in ["greedy", "top+random", "top+random+learn", "spread", "random"]:
            print(f"{scen}: {REPS} x {ROUNDS} rounds, {policy}...")
            for rep in range(REPS):
                rows.append({"scenario": scen, **run(policy, p, 1000 + rep, cfg)})
    df = pd.DataFrame(rows)
    df["estimate error (points)"] = 100 * (df["estimated theft share of areas"]
                                           - df["true theft share of areas"])
    summary = df.groupby(["scenario", "policy"], sort=False).agg(
        **{c: (c, "mean") for c in ["theft reached (MWh/week)", "theft found (MWh/week)",
                                    "hit rate, all valid visits", "hit rate, model arm",
                                    "hit rate, random arm", "random-arm visits",
                                    "true theft share of areas",
                                    "estimated theft share of areas", "verdict model>random"]},
        **{"estimate bias (points)": ("estimate error (points)", "mean"),
           "estimate spread (points, sd)": ("estimate error (points)", "std")})
    Path("results").mkdir(exist_ok=True)
    summary.reset_index().to_csv("results/inspections.csv", index=False)
    pd.set_option("display.width", 200)
    for scen, g in summary.groupby(level=0, sort=False):
        print(f"\n{scen}\n" + g.droplevel(0).T.round(3).to_string())
    print("\nCalibration of the displayed probabilities (Brier score, lower is better)...")
    cal = calibration_check(pop)
    cal.to_csv("results/inspections_calibration.csv", index=False)
    print(cal.round(4).to_string(index=False))
    print("\nSaved results/inspections.csv and results/inspections_calibration.csv")


if __name__ == "__main__":
    main()

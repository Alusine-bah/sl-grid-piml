"""Inspection rounds: plan where crews go, log what they find, measure honestly, learn.

Design (see docs/results.md, Phase 13, and the research report behind it):

* Planning.  Each round visits `budget` transformer areas, and every area's chance of
  being visited, pi_i, is fixed BEFORE the draw and logged. Default policy "top+random":
  85% of visits go to the areas with the highest expected stolen energy (pi = 1) and 15%
  to a simple random sample of all other areas (pi = n_random / areas left), so every
  area keeps a chance and the random arm measures the base rate. The alternative
  "spread" policy (tested and found to give up about half the theft reached in the
  closed-loop study, Phase 13) draws all visits at random with pi_i mixing
    - a random share rho (default 15%) spread evenly over all areas, and
    - a risk-tilted share proportional to each area's expected stolen energy,
  with a floor pi_min (default 2%) so that no area is ever certain to be skipped. A fixed-
  size draw with exactly these probabilities is made by systematic sampling in random
  order. Each visited area is then labelled "random" with probability (budget*rho/N)/pi_i,
  which makes the random arm a uniform random sample of the feeder population: its hit
  rate is an unbiased base rate. Pure top-k selection is not offered: it cannot measure
  anything except itself.

* Outcomes.  Ten codes (OUTCOMES). Theft, tampering, illegal connections, non-fraud meter
  faults and billing errors are positive findings of non-technical loss; CLEAN is a noisy
  negative (crews miss some theft, and may check only part of an area); NO_ACCESS,
  NOT_FOUND and ABORTED_SAFETY are missing, never negative.

* Measurement.  Hit rate per arm with Wilson 95% intervals; lift (model / random) with a
  bootstrap interval; the Horvitz-Thompson estimate of the share of areas with theft,
  sum(y_i / pi_i) / N; and an "insufficient evidence" verdict until the sample sizes
  needed to tell the model from random have been reached.

* Learning.  The physics estimate stays label-free. Inspections adjust it through
    - calibration: a two-parameter logistic map from the physics stolen share to the
      probability that an area has theft, fitted on valid outcomes weighted by 1/pi
      (used once there are >= 20 valid outcomes);
    - a Beta posterior per area seeded from that probability (prior weight 3 areas),
      +1 for each positive, + s*f for each clean visit (s = crew sensitivity, f = share
      of premises checked), nothing for missing outcomes;
    - a lower bound on the stolen energy of an area where theft was found and measured,
      passed to the physics model (`theta_min`).

The default numbers are labelled defaults: no published source gives crew sensitivity,
cost per inspection or repeat-theft rates.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

POSITIVE = ["THEFT_BYPASS", "METER_TAMPER", "ILLEGAL_CONNECTION", "METER_FAULT_NONFRAUD",
            "BILLING_ERROR"]
NEGATIVE = ["CLEAN"]
MISSING = ["NO_ACCESS_REFUSED", "NOT_FOUND_ADDRESS_ERROR", "ABORTED_SAFETY"]
OTHER = ["CABLE_OR_NETWORK_FAULT"]           # a finding, but not non-technical loss
OUTCOMES = {
    "THEFT_BYPASS": "Meter bypassed",
    "METER_TAMPER": "Meter tampered",
    "ILLEGAL_CONNECTION": "Illegal connection / line tap (no account)",
    "METER_FAULT_NONFRAUD": "Faulty meter, no fraud (still recoverable)",
    "BILLING_ERROR": "Billing or customer-mapping error",
    "CABLE_OR_NETWORK_FAULT": "Cable or network fault found",
    "CLEAN": "Nothing found",
    "NO_ACCESS_REFUSED": "No access / refused",
    "NOT_FOUND_ADDRESS_ERROR": "Not found / address error",
    "ABORTED_SAFETY": "Aborted for safety",
}
ARMS = ["random", "model", "tip-off", "routine", "follow-up"]
LOG_COLUMNS = ["inspection_id", "round_id", "feeder", "bus", "selection_arm", "pi",
               "model_score", "model_rank", "model_version", "date", "crew", "premises",
               "premises_checked", "outcome", "kwh_per_day_found", "kwh_recovered",
               "amount_billed", "amount_collected", "notes"]


@dataclass
class InspectionConfig:
    policy: str = "top+random"     # or "spread" (risk-tilted draw; see docstring)
    random_share: float = 0.15     # share of each round spent on a random audit (10-20%)
    pi_min: float = 0.02           # every area keeps at least this chance per round
    sensitivity: float = 0.8       # chance a crew finds theft that is there (assumption)
    prior_weight: float = 3.0      # Beta prior strength, in "areas worth" of evidence
    calibrate_after: int = 20      # valid outcomes needed before calibrating
    pool_strength: float = 3.0     # feeder pooling: expected positives worth of shrinkage
    cooldown_rounds: int = 2       # areas visited in the last rounds are not planned again
    bound_slack: float = 0.2       # recovered energy as a lower bound, minus this margin
    default_a: float = -2.2        # default calibration: P(theft) = sigmoid(a + b*share)
    default_b: float = 12.0        # (from the simulation; replaced by data when available)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------- planning
def inclusion_probabilities(weight: np.ndarray, budget: int, cfg: InspectionConfig) -> np.ndarray:
    """pi_i for a round of `budget` visits among N areas, summing to `budget`."""
    w = np.maximum(np.asarray(weight, float), 0.0)
    N = len(w)
    budget = int(min(budget, N))
    if budget <= 0:
        return np.zeros(N)
    tilt = w / w.sum() if w.sum() > 0 else np.full(N, 1 / N)
    pi = budget * (cfg.random_share / N + (1 - cfg.random_share) * tilt)
    floor = min(cfg.pi_min, budget / N)
    for _ in range(100):          # floor, cap at 1, and rescale the rest to keep the sum
        pi = np.clip(pi, floor, 1.0)
        fixed = (pi <= floor + 1e-12) | (pi >= 1 - 1e-12)
        gap = budget - pi.sum()
        if abs(gap) < 1e-9 or (~fixed).sum() == 0:
            break
        free = ~fixed
        pi[free] *= 1 + gap / pi[free].sum()
    return pi * budget / pi.sum()


def plan_round(areas: pd.DataFrame, budget: int, cfg: InspectionConfig | None = None,
               seed: int = 0, round_id: str = "R1", model_version: str = "") -> pd.DataFrame:
    """Draw a round of inspections.

    areas: one row per transformer area with columns feeder, bus, weight (expected stolen
           energy or value; higher = visit more likely), and optionally model_score.
    Returns the work list: the selected areas with pi, selection_arm and empty outcome
    columns to be filled in by crews. inclusion_probabilities() gives pi for every area.
    """
    cfg = cfg or InspectionConfig()
    rng = np.random.default_rng(seed)
    a = areas.reset_index(drop=True).copy()
    N = len(a)
    n = int(min(budget, N))
    rank = np.argsort(-a["weight"].values, kind="stable")
    if cfg.policy == "top+random":
        # model arm: the top areas by expected stolen energy (pi = 1); random arm: a simple
        # random sample of the rest (pi = n_r / remaining), so every area has a chance
        n_r = int(round(cfg.random_share * n)) if n >= 4 else 0
        n_m = n - n_r
        top = rank[:n_m]
        rest = rank[n_m:]
        pick = rng.choice(rest, n_r, replace=False) if n_r else np.array([], int)
        pi = np.full(N, n_r / max(len(rest), 1))
        pi[top] = 1.0
        chosen = np.concatenate([top, pick]).astype(int)
        arm = np.array(["model"] * n_m + ["random"] * n_r)
    else:                                     # "spread": risk-tilted draw, see docstring
        pi = inclusion_probabilities(a["weight"].values, n, cfg)
        # systematic sampling in random order: exactly n areas, area i with prob pi_i
        order = rng.permutation(N)
        cum = np.concatenate([[0.0], np.cumsum(pi[order])])
        points = rng.uniform(0, 1) + np.arange(n)
        pos = np.searchsorted(cum, points, side="right") - 1
        chosen = np.unique(order[np.clip(pos, 0, N - 1)])
        # random with probability (n*rho/N)/pi, so the random arm is uniform over areas
        p_rand = np.minimum(1.0, (n * cfg.random_share / N) / np.maximum(pi[chosen], 1e-12))
        arm = np.where(rng.random(len(chosen)) < p_rand, "random", "model")
    a["pi"] = pi
    a["model_rank"] = (-a["weight"]).rank(method="first").astype(int)
    out = a.iloc[chosen].copy()
    out["selection_arm"] = arm
    out["round_id"] = round_id
    out["model_version"] = model_version
    out = out.sort_values("model_rank").reset_index(drop=True)
    out.insert(0, "inspection_id", [f"{round_id}-{k + 1:03d}" for k in range(len(out))])
    for c in ["date", "crew", "premises", "premises_checked", "outcome", "kwh_per_day_found",
              "kwh_recovered", "amount_billed", "amount_collected", "notes"]:
        out[c] = np.nan if c not in ("date", "crew", "outcome", "notes") else ""
    return out


def recently_visited(log: pd.DataFrame | None, rounds: int) -> set:
    """(feeder, bus) of areas visited in the last `rounds` rounds of the log (rounds in the
    order they first appear). These were just dealt with; the next plan leaves them out."""
    if log is None or not len(log) or rounds <= 0 or "round_id" not in log:
        return set()
    order = list(dict.fromkeys(log["round_id"].astype(str)))
    last = set(order[-rounds:])
    r = log[log["round_id"].astype(str).isin(last)]
    return set(zip(r["feeder"].astype(str), r["bus"].astype(str)))


# ----------------------------------------------------------------------------- the log
def clean_log(log: pd.DataFrame) -> pd.DataFrame:
    """Check and normalise an inspection log; adds y (1 positive, 0 clean, NaN missing)."""
    df = log.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["feeder", "bus", "selection_arm", "pi", "outcome"]:
        if c not in df.columns:
            raise ValueError(f"The inspection log needs a '{c}' column.")
    df["outcome"] = df["outcome"].astype(str).str.strip().str.upper().replace({"NAN": ""})
    df = df[df["outcome"] != ""].copy()                      # planned but not yet visited
    bad = sorted(set(df["outcome"]) - set(OUTCOMES))
    if bad:
        raise ValueError(f"Unknown outcome code(s): {', '.join(bad[:5])}. "
                         f"Use one of: {', '.join(OUTCOMES)}.")
    df["selection_arm"] = df["selection_arm"].astype(str).str.strip().str.lower()
    df["pi"] = pd.to_numeric(df["pi"], errors="coerce")
    if (df["pi"].isna() | (df["pi"] <= 0) | (df["pi"] > 1)).any():
        raise ValueError("Every inspection needs its selection probability pi (0 < pi <= 1), "
                         "as written on the work list.")
    df["feeder"] = df["feeder"].astype(str)
    df["bus"] = df["bus"].astype(str)
    df["y"] = np.where(df["outcome"].isin(POSITIVE), 1.0,
                       np.where(df["outcome"].isin(NEGATIVE), 0.0, np.nan))
    for c in ["premises", "premises_checked", "kwh_per_day_found", "kwh_recovered"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    cov = df["premises_checked"] / df["premises"]
    df["coverage"] = cov.where(cov.between(0, 1), np.nan).fillna(1.0)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- measuring
def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def needed_per_arm(p0: float, p1: float) -> int:
    """Inspections per arm to tell hit rates p0 and p1 apart (80% power, one-sided 5%)."""
    za, zb = 1.645, 0.842
    pb = (p0 + p1) / 2
    num = (za * sqrt(2 * pb * (1 - pb)) + zb * sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return int(np.ceil(num / max((p1 - p0) ** 2, 1e-9)))


def _population_share(log: pd.DataFrame, n_areas: int, cooldown: int,
                      rest_rate: float | None = None) -> float:
    """Estimated share of all areas where a visit would find theft (see evaluate_log).

    With the top + random planner, each round splits the areas into ones whose outcome is
    known (visited from the top of the list, pi = 1, or visited in the previous rounds and
    left out) and the rest, of which the random arm is a simple random sample. The rest
    are counted at the random arm's hit rate (`rest_rate`, pooled over rounds unless
    given). Missing outcomes of known areas are counted at the model arm's hit rate.
    Other planners fall back to Horvitz-Thompson weights 1/pi per round."""
    order = list(dict.fromkeys(log["round_id"].astype(str)))
    key = log["feeder"].astype(str) + "|" + log["bus"].astype(str)
    rnd = log[(log["selection_arm"] == "random") & log["y"].notna()]
    mdl = log[(log["selection_arm"] != "random") & log["y"].notna()]
    top_plan = bool((log.loc[log["selection_arm"] != "random", "pi"] >= 1 - 1e-9).all())
    if rest_rate is None:
        rest_rate = float(rnd["y"].mean()) if len(rnd) else np.nan
    model_rate = float(mdl["y"].mean()) if len(mdl) else rest_rate
    shares = []
    for k, r in enumerate(order):
        cur = log[log["round_id"].astype(str) == r]
        prev = log[log["round_id"].astype(str).isin(order[max(0, k - cooldown):k])]
        latest = prev.assign(k=key[prev.index]).groupby("k")["y"].last()   # NaN if missing
        if top_plan and rest_rate == rest_rate:
            top = cur[cur["selection_arm"] != "random"]
            known = len(top) + len(latest)
            pos = (top["y"].fillna(model_rate).sum() + latest.fillna(model_rate).sum())
            shares.append((pos + rest_rate * max(n_areas - known, 0)) / n_areas)
        else:
            v = cur[cur["y"].notna()]
            if not len(v):
                continue
            w = 1 / v["pi"].values
            frame_rate = (v["y"].values * w).sum() / w.sum()
            cooled_pos = latest.fillna(frame_rate).sum()
            shares.append((frame_rate * (n_areas - len(latest)) + cooled_pos) / n_areas)
    return float(np.mean(shares)) if shares else np.nan


def evaluate_log(log: pd.DataFrame, n_areas: int | None = None, seed: int = 0,
                 sensitivity: float = 0.8, cooldown: int = 2, n_boot: int = 300) -> dict:
    """Honest summary of an inspection log (output of clean_log)."""
    rng = np.random.default_rng(seed)
    out = {"inspections": len(log), "missing": int(log["y"].isna().sum()),
           "refused": int((log["outcome"] == "NO_ACCESS_REFUSED").sum()),
           "network_faults": int((log["outcome"] == "CABLE_OR_NETWORK_FAULT").sum())}
    arms = {}
    valid = log[log["y"].notna()]
    for arm, g in valid.groupby("selection_arm"):
        k, n = g["y"].sum(), len(g)
        lo, hi = wilson(k, n)
        arms[arm] = {"n": int(n), "positives": int(k), "hit_rate": k / n, "lo": lo, "hi": hi,
                     "kwh_recovered": float(log.loc[log.selection_arm == arm, "kwh_recovered"].sum())}
    out["arms"] = arms
    r, m = arms.get("random"), arms.get("model")
    out["lift"] = None
    if r and m and r["n"] >= 5 and m["n"] >= 5:
        yr = valid.loc[valid.selection_arm == "random", "y"].values
        ym = valid.loc[valid.selection_arm == "model", "y"].values
        boots = []
        for _ in range(max(n_boot, 200)):
            a = rng.choice(yr, len(yr)).mean()
            b = rng.choice(ym, len(ym)).mean()
            if a > 0:
                boots.append(b / a)
        if r["hit_rate"] > 0 and boots:
            out["lift"] = {"value": m["hit_rate"] / r["hit_rate"],
                           "lo": float(np.percentile(boots, 2.5)),
                           "hi": float(np.percentile(boots, 97.5))}
    # verdict: enough evidence that the model beats random?
    if r and m:
        p0 = min(max(r["hit_rate"], 0.02), 0.97)
        need = needed_per_arm(p0, min(max(m["hit_rate"], p0 + 0.02), 0.99))
        out["needed_per_arm"] = need
        if m["lo"] > r["hi"]:
            out["verdict"] = "model better than random (intervals do not overlap)"
        elif min(r["n"], m["n"]) < need:
            out["verdict"] = (f"insufficient evidence: about {need} valid inspections per arm "
                              f"are needed (random has {r['n']}, model {m['n']})")
        else:
            out["verdict"] = "no clear difference between model and random yet"
    else:
        out["verdict"] = "insufficient evidence: both a random and a model arm are needed"
    # Share of ALL areas where crews would find theft. Each round: the areas it could
    # select (its frame) are estimated with Horvitz-Thompson weights 1/pi; the areas left
    # out because they were visited in the previous `cooldown` rounds count with their
    # latest outcome. Rounds are then averaged.
    if n_areas and len(valid) and "round_id" in log:
        det = sensitivity * float(valid["coverage"].mean())
        est = _population_share(log, n_areas, cooldown)
        rnd = log[(log["selection_arm"] == "random") & log["y"].notna()]
        if len(rnd) and (log.loc[log["selection_arm"] != "random", "pi"] >= 1 - 1e-9).all():
            # interval from the Wilson interval of the random arm's hit rate (the only
            # part estimated from a sample; outcomes of the visited top areas are known)
            r_lo, r_hi = wilson(rnd["y"].sum(), len(rnd))
            lo = _population_share(log, n_areas, cooldown, r_lo)
            hi = _population_share(log, n_areas, cooldown, r_hi)
        else:
            bs = []
            groups = [g.index.values for _, g in log.groupby(["round_id", "selection_arm"])]
            for _ in range(n_boot):
                idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
                v = _population_share(log.loc[idx].reset_index(drop=True), n_areas, cooldown)
                if v == v:
                    bs.append(v)
            lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
        if est == est:
            out["ht_found_share"] = est
            out["ht_found_interval"] = (float(lo), float(hi))
            out["ht_theft_share"] = min(est / max(det, 1e-6), 1.0)        # allowing for misses
            out["ht_theft_interval"] = (float(min(lo / det, 1)), float(min(hi / det, 1)))
    return out


# ----------------------------------------------------------------------------- learning
def fit_calibration(log: pd.DataFrame, cfg: InspectionConfig | None = None) -> tuple[float, float, int]:
    """P(theft in area) = sigmoid(a + b * physics share), fitted on valid outcomes weighted
    by 1/pi. A crew finds theft that is there with probability s*f, so the fit models
    P(report theft) = s*f*P(theft). Returns (a, b, n_used); the defaults until there are
    enough outcomes."""
    cfg = cfg or InspectionConfig()
    v = log[log["y"].notna() & log["model_score"].notna()] if "model_score" in log else log.iloc[:0]
    if len(v) < cfg.calibrate_after or v["y"].nunique() < 2:
        return cfg.default_a, cfg.default_b, len(v)
    from scipy.optimize import minimize
    x = v["model_score"].values.astype(float)
    y = v["y"].values.astype(float)
    w = 1 / v["pi"].values
    w = w / w.mean()
    d = np.clip(cfg.sensitivity * v["coverage"].values, 1e-3, 1.0)   # chance theft is found
    # A crew reports theft with probability d * p(x) (it can miss theft that is there), so
    # the likelihood uses that, weighted by 1/pi (selection), with a gentle Gaussian prior
    # around the defaults (sd 1.5 on a, 8 on b) so a few outcomes cannot swing the map.
    def nll(t):
        p = np.clip(d * _sigmoid(t[0] + t[1] * x), 1e-9, 1 - 1e-9)
        ll = w * (y * np.log(p) + (1 - y) * np.log(1 - p))
        return -ll.sum() + 0.5 * ((t[0] - cfg.default_a) / 1.5) ** 2 \
            + 0.5 * ((t[1] - cfg.default_b) / 8.0) ** 2
    res = minimize(nll, [cfg.default_a, cfg.default_b], method="L-BFGS-B",
                   bounds=[(-8, 4), (0, 60)])
    a, b = res.x
    return float(a), float(b), len(v)


def area_posterior(areas: pd.DataFrame, log: pd.DataFrame | None,
                   cfg: InspectionConfig | None = None) -> pd.DataFrame:
    """Per-area probability of theft after the inspections so far.

    areas: feeder, bus, model_score (physics stolen share), weight (expected stolen energy).
    Adds p_prior (calibrated physics), alpha, beta, p_post, visits, and next_weight: the
    expected stolen energy rescaled by how the evidence changed the belief.
    """
    cfg = cfg or InspectionConfig()
    a_, b_, n_used = fit_calibration(log if log is not None else pd.DataFrame(), cfg)
    out = areas.copy()
    out["p_prior"] = _sigmoid(a_ + b_ * out["model_score"].values)
    # feeder pooling: if crews find less (or more) theft on a feeder than its areas'
    # probabilities predicted, scale every area of that feeder (shrunk toward 1)
    out["feeder_factor"] = 1.0
    if log is not None and len(log) and "model_score" in log:
        v = log[log["y"].notna()].copy()
        v["expect"] = (_sigmoid(a_ + b_ * v["model_score"].astype(float))
                       * cfg.sensitivity * v["coverage"])
        g = v.groupby("feeder").agg(k=("y", "sum"), e=("expect", "sum"))
        fac = (g["k"] + cfg.pool_strength) / (g["e"] + cfg.pool_strength)
        out["feeder_factor"] = out["feeder"].astype(str).map(fac).fillna(1.0).values
        out["p_prior"] = np.clip(out["p_prior"] * out["feeder_factor"], 0.001, 0.999)
    out["alpha"] = cfg.prior_weight * out["p_prior"]
    out["beta"] = cfg.prior_weight * (1 - out["p_prior"])
    out["visits"] = 0
    out["positives"] = 0
    if log is not None and len(log):
        key = out["feeder"].astype(str) + "|" + out["bus"].astype(str)
        lk = log["feeder"].astype(str) + "|" + log["bus"].astype(str)
        pos = log.assign(k=lk)[log["y"] == 1].groupby("k").size()
        neg = (log.assign(k=lk, e=cfg.sensitivity * log["coverage"])[log["y"] == 0]
               .groupby("k")["e"].sum())
        vis = log.assign(k=lk)[log["y"].notna()].groupby("k").size()
        out["alpha"] += key.map(pos).fillna(0).values
        out["beta"] += key.map(neg).fillna(0).values
        out["visits"] = key.map(vis).fillna(0).astype(int).values
        out["positives"] = key.map(pos).fillna(0).astype(int).values
    out["p_post"] = out["alpha"] / (out["alpha"] + out["beta"])
    p_phys = _sigmoid(a_ + b_ * out["model_score"].values)
    out["next_weight"] = out["weight"] * out["p_post"] / np.maximum(p_phys, 1e-6)
    out.attrs["calibration"] = {"a": a_, "b": b_, "n": n_used,
                                "fitted": n_used >= cfg.calibrate_after}
    return out


def theft_lower_bounds(log: pd.DataFrame, buses: list[str], billed_mwh: np.ndarray,
                       n_days: float, cfg: InspectionConfig | None = None) -> np.ndarray:
    """Lower bounds on theta (stolen / billed) per bus from measured theft.

    Where a crew found theft and estimated the stolen energy per day (`kwh_per_day_found`,
    e.g. from the bypassed load), the area's stolen energy over the period is at least
    (1 - slack) times that. Returns theta_min [B] for the physics model."""
    cfg = cfg or InspectionConfig()
    theta_min = np.zeros(len(buses))
    if log is None or not len(log):
        return theta_min
    f = log[(log["y"] == 1) & log["kwh_per_day_found"].notna()]
    for bus, g in f.groupby("bus"):
        if bus in buses:
            j = buses.index(bus)
            mwh = g["kwh_per_day_found"].max() * n_days / 1000 * (1 - cfg.bound_slack)
            theta_min[j] = min(mwh / max(billed_mwh[j], 1e-9), 8.9)
    return theta_min

import numpy as np
import pandas as pd
import pytest

from slgrid.app_core import district_areas, locate, simulate_case, simulate_visits
from slgrid.inspections import (OUTCOMES, InspectionConfig, area_posterior, clean_log,
                                evaluate_log, fit_calibration, inclusion_probabilities,
                                needed_per_arm, plan_round, recently_visited,
                                theft_lower_bounds, wilson)


def _areas(n=200, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.random(n)
    return pd.DataFrame({"feeder": "F", "bus": [f"b{i}" for i in range(n)], "model_score": s,
                         "weight": s + 0.05 * rng.random(n), "theft": rng.random(n) < 0.1 + 0.7 * s})


def test_statistics_match_published_formulas():
    lo, hi = wilson(10, 30)
    assert lo == pytest.approx(0.192, abs=1e-3) and hi == pytest.approx(0.512, abs=1e-3)
    assert needed_per_arm(0.1, 0.2) == 157 and needed_per_arm(0.1, 0.3) == 49


def test_top_plus_random_plan():
    a = _areas()
    p = plan_round(a, 40, InspectionConfig(), seed=1)
    assert len(p) == 40 and (p.selection_arm == "random").sum() == 6
    top = set(a.sort_values("weight", ascending=False).bus.head(34))
    assert set(p[p.selection_arm == "model"].bus) == top
    assert (p[p.selection_arm == "model"].pi == 1).all()
    assert p[p.selection_arm == "random"].pi.iloc[0] == pytest.approx(6 / 166)
    assert set(p.columns) >= {"inspection_id", "round_id", "pi", "outcome", "kwh_recovered"}


def test_spread_plan_has_the_stated_probabilities():
    a = _areas(100)
    cfg = InspectionConfig(policy="spread")
    pi = inclusion_probabilities(a.weight.values, 20, cfg)
    assert pi.sum() == pytest.approx(20) and pi.min() >= 0.02 - 1e-9 and pi.max() <= 1 + 1e-9
    hits, rand = np.zeros(100), np.zeros(100)
    for s in range(1500):
        p = plan_round(a, 20, cfg, seed=s)
        i = p.bus.str[1:].astype(int).values
        hits[i] += 1
        rand[i[p.selection_arm.values == "random"]] += 1
    assert np.abs(hits / 1500 - pi).max() < 0.06
    assert (rand / 1500).std() < 0.01                # random arm is uniform over areas


def test_log_codes_and_missing_outcomes():
    log = pd.DataFrame({"feeder": "F", "bus": ["b1", "b2", "b3", "b4"], "selection_arm": "model",
                        "pi": 1.0, "outcome": ["theft_bypass", "CLEAN", "NO_ACCESS_REFUSED", ""]})
    c = clean_log(log)
    assert len(c) == 3 and c.y.tolist()[:2] == [1.0, 0.0] and np.isnan(c.y.iloc[2])
    assert len(OUTCOMES) == 10
    with pytest.raises(ValueError, match="Unknown outcome"):
        clean_log(log.assign(outcome="STOLEN"))
    with pytest.raises(ValueError, match="pi"):
        clean_log(log.assign(pi=0))


def test_population_estimate_is_unbiased_and_verdict_is_cautious():
    a = _areas(192, 3)
    cfg = InspectionConfig()
    est = []
    for rep in range(60):
        logs = []
        for r in range(3):
            frame = a
            if logs:
                rc = recently_visited(clean_log(pd.concat(logs)), 2)
                frame = a[[(f, b) not in rc for f, b in zip(a.feeder, a.bus)]]
            p = plan_round(frame, 30, cfg, seed=rep * 10 + r, round_id=f"R{r}")
            t = a.set_index("bus").loc[p.bus, "theft"].values
            p["outcome"] = np.where(t, "THEFT_BYPASS", "CLEAN")
            logs.append(p)
        ev = evaluate_log(clean_log(pd.concat(logs, ignore_index=True)), n_areas=len(a),
                          sensitivity=1.0, n_boot=0)
        est.append(ev["ht_found_share"])
    assert abs(np.mean(est) - a.theft.mean()) < 0.03
    small = clean_log(pd.concat(logs, ignore_index=True).head(12))
    assert evaluate_log(small, n_areas=192, n_boot=0)["verdict"].startswith("insufficient")


def test_calibration_recovers_from_a_wrong_start():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.random(n)
    y = rng.random(n) < 0.8 / (1 + np.exp(-(-3 + 8 * x)))      # found = 0.8 x P(theft)
    log = pd.DataFrame({"feeder": "F", "bus": [f"b{i}" for i in range(n)], "selection_arm": "random",
                        "pi": 0.5, "outcome": np.where(y, "THEFT_BYPASS", "CLEAN"), "model_score": x})
    a, b, used = fit_calibration(clean_log(log), InspectionConfig(default_a=0.0, default_b=2.0))
    assert used == n and a == pytest.approx(-3, abs=1.0) and b == pytest.approx(8, abs=3.0)


def test_simulated_round_and_lower_bounds_reach_the_model():
    d = simulate_case(0.25, 6, "ends", 0.2, False, 3, 7)
    res = locate(d)
    areas = district_areas({"F1": (d, res)})
    plan = plan_round(areas[["feeder", "bus", "model_score", "weight"]], 12, seed=2)
    log = clean_log(simulate_visits(plan, areas, seed=2))
    assert set(log.outcome) <= set(OUTCOMES)
    post = area_posterior(areas[["feeder", "bus", "model_score", "weight"]], log)
    assert post.p_post.between(0, 1).all() and post.visits.sum() == log.y.notna().sum()
    buses = [str(b) for b in d["bus_names"]]
    tm = theft_lower_bounds(log, buses, d["billing"][0].sum(axis=0), 3)
    if (tm > 0).any():
        d2 = dict(d)
        d2["theta_min"] = tm[None]
        r2 = locate(d2).set_index("bus").loc[buses]
        assert (r2.share_physics.values >= tm / (1 + tm) - 1e-6).all()

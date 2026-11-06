"""Back end of the interactive app: simulate a case, locate losses, lay out the feeder."""
from __future__ import annotations

import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd

from slgrid.ami import AMIConfig, ami_inputs, rank_large_customers, simulate_ami
from slgrid.baselines import combined_rules, energy_audit, metered_mask
from slgrid.dataset import DatasetConfig, generate
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator
from slgrid.prepaid import VendingConfig, rank_meters, simulate_vending, tokens_to_billing
from slgrid.studies import placement, rebill, remeter


def simulate_case(theft_level: float, n_meters: int, strategy: str, dropout: float,
                  fault: bool, n_days: int, seed: int, billing: str = "daily",
                  billing_noise: float = 0.03, shedding: str = "none",
                  mapping_error: float = 0.05, ami: bool = False,
                  vending_history: int = 90) -> dict:
    """Simulate one feeder period and return it in the standard format (with truth).

    billing: "daily" (one value per day per bus) or "period" (one total per bus, as with
    monthly bills or prepaid token sales). billing_noise: relative error of each value.
    shedding: load-shedding pattern (see loads.SHEDDING_PATTERNS).
    billing="prepaid": simulate 90 days of prepaid vending records for every customer, with
    `mapping_error` of meters listed under the wrong transformer; sales per area come from
    the last 30 days of purchases. The records are kept in d["vending"].
    ami: the large customers at commercial transformer areas have smart meters with 28 days
    of hourly history, and some tamper where there is theft. Kept in d["ami"].
    """
    cfg = DatasetConfig(n_scenarios=1, n_days=n_days, ntl_levels=(theft_level,),
                        fault_probability=1.0 if fault else 0.0, dropout=dropout, seed=seed,
                        shedding=shedding)
    with tempfile.TemporaryDirectory() as tmp:
        d = dict(np.load(generate(cfg, tmp, name="case", progress=False)))
    buses = placement(d, n_meters, strategy, seed=seed)
    d = remeter(d, buses, dropout=dropout, seed=seed + 1)
    if ami:
        d, lc, rd = simulate_ami(d, 0, AMIConfig(seed=seed + 4))
        d = ami_inputs(d, lc, rd)
        d["ami"] = {"customers": lc, "readings": rd}
    if billing == "prepaid":
        cust, tok = simulate_vending(d, 0, VendingConfig(history_days=vending_history,
                                                          mapping_error=mapping_error, seed=seed + 3))
        if ami:     # large customers are on smart meters, not on the prepaid vending system
            lc_buses = set(d["ami"]["customers"]["bus"])
            keep = ~cust["true_bus"].isin(lc_buses)
            tok = tok[tok.meter.isin(cust.meter[keep])].reset_index(drop=True)
            cust = cust[keep].reset_index(drop=True)
        sold = tokens_to_billing(cust, tok, d["bus_names"], n_days)[None]
        if ami:
            j = np.isfinite(d["ami_p"][0]).any(axis=0)
            sold[0, :, j] = np.nansum(d["ami_p"][0][:, j], axis=0)[:, None]
        d["billing"] = sold
        d["vending"] = {"customers": cust, "tokens": tok}
    elif billing != "daily" or abs(billing_noise - 0.03) > 1e-9:
        d = rebill(d, billing, billing_noise, seed=seed + 2)
    return d


def large_customer_list(d: dict, res: pd.DataFrame) -> pd.DataFrame | None:
    """Large customers on smart meters ranked for inspection, or None without AMI data."""
    if "ami" not in d:
        return None
    a = d["ami"]
    share = pd.Series(res.share_physics.values, index=res.bus.values)
    out = rank_large_customers(a["customers"], a["readings"], d["head"].shape[1], share)
    if "tamper" in a["customers"]:
        out = out.merge(a["customers"][["meter", "tamper", "tamper_kind", "onset_hour"]],
                        on="meter", how="left")
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def meter_list(d: dict, res: pd.DataFrame) -> pd.DataFrame | None:
    """Prepaid meters ranked for inspection, or None without vending records.

    Each meter's shortfall in buying is weighted by the physics-informed estimate of the
    stolen share in its transformer area."""
    if "vending" not in d:
        return None
    v = d["vending"]
    n_days = d["head"].shape[1] // 24
    share = pd.Series(res.share_physics.values, index=res.bus.values)
    out = rank_meters(v["customers"], v["tokens"], n_days, share)
    if "is_bypass" in v["customers"]:
        out = out.merge(v["customers"][["meter", "is_bypass", "is_vacant", "true_bus"]],
                        on="meter", how="left")
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def locate(d: dict, power_factor: float = 0.90, tech_loss: float = 0.03) -> pd.DataFrame:
    """Run the energy audit, combined rules and the physics-informed locator.

    Returns one row per load bus with each method's estimated stolen energy (MWh over the
    period) and stolen share of consumption, plus truth columns when available.
    """
    idx = np.array([0])
    _, st_audit = energy_audit(d, idx, tech_loss)
    _, st_rules = combined_rules(d, idx, tech_loss)
    phys = PhysicsInformedLocator(d, PhysicsConfig(ac_iterations=1, power_factor=power_factor,
                                                   detect_faults=True))
    fit = phys.fit_scenario(0)
    fault = fit["fault"]

    billed = d["billing"][0].sum(axis=0)
    mm = metered_mask(d)
    out = pd.DataFrame({
        "bus": [str(b) for b in d["bus_names"]],
        "kva": d["transformer_kva"][0],
        "metered": mm,
        "billed_mwh": billed,
        "stolen_physics_mwh": fit["stolen"],
        "share_physics": fit["score"],
        "leak_kw": fit["fault_mw"] * 1000,
        "fault_prob": fault["bus_prob"] if fault["detected"] else np.zeros(len(billed)),
        "in_fault_section": [str(b) in fault["likely_section"] and fault["detected"]
                             for b in d["bus_names"]],
        "stolen_rules_mwh": st_rules[0],
        "stolen_audit_mwh": st_audit[0],
    })
    out["share_rules"] = out.stolen_rules_mwh / (out.stolen_rules_mwh + billed)
    if "theft_bus_p" in d:
        true = d["theft_bus_p"][0].sum(axis=0)
        out["true_stolen_mwh"] = true
        out["true_share"] = true / d["true_bus_p"][0].sum(axis=0)
        out["true_hotspot"] = d["is_theft"][0].astype(bool)
        out["true_fault"] = d["is_fault"][0].astype(bool)
    out = out.sort_values("stolen_physics_mwh", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.attrs["fault"] = {k: fault[k] for k in ("detected", "score", "start_hour", "leak_kw",
                                                 "likely_section")}
    return out


def feeder_summary(d: dict, res: pd.DataFrame) -> dict:
    from slgrid.faults import energised_hours
    head_e = np.nansum(d["head"][0, :, 0])
    billed = d["billing"][0].sum()
    s = {
        "days": d["head"].shape[1] // 24,
        "head_mwh": head_e,
        "billed_mwh": billed,
        "unbilled_pct": 100 * (1 - billed / head_e),
        "stolen_physics": res.stolen_physics_mwh.sum(),
        "stolen_rules": res.stolen_rules_mwh.sum(),
        "n_meters": int(res.metered.sum()),
        "n_buses": len(res),
        "hours_on": int(energised_hours(d["head"][0, :, 0]).sum()),
        "hours": int(d["head"].shape[1]),
    }
    if "true_stolen_mwh" in res:
        s["stolen_true"] = res.true_stolen_mwh.sum()
        k = int(res.true_hotspot.sum())
        s["n_hotspots"] = k
        s["topk_physics"] = res.head(k).true_hotspot.mean() if k else np.nan
        s["topk_rules"] = (res.sort_values("stolen_rules_mwh", ascending=False)
                           .head(k).true_hotspot.mean()) if k else np.nan
    return s


def tree_layout(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positions for drawing a radial feeder: x = distance from substation in sections,
    y = branch lane, with the largest branch drawn straight."""
    children = defaultdict(list)
    for f, t in zip(d["line_from"], d["line_to"]):
        children[str(f)].append(str(t))
    size = {}

    def count(b):
        size[b] = 1 + sum(count(c) for c in children[b])
        return size[b]
    count("bus_0")

    pos, lane = {}, [0]

    def place(b, depth):
        kids = sorted(children[b], key=lambda c: -size[c])
        if not kids:
            pos[b] = (depth, lane[0])
            lane[0] += 1
            return
        for i, c in enumerate(kids):
            place(c, depth + 1)
        pos[b] = (depth, pos[kids[0]][1])
    place("bus_0", 0)

    nodes = pd.DataFrame([{"bus": b, "x": x, "y": -y} for b, (x, y) in pos.items()])
    edges = pd.DataFrame([{"x": pos[str(f)][0], "y": -pos[str(f)][1],
                           "x2": pos[str(t)][0], "y2": -pos[str(t)][1]}
                          for f, t in zip(d["line_from"], d["line_to"])])
    # draw branches as an elbow: vertical drop at the parent's x, then horizontal
    elbows = []
    for _, e in edges.iterrows():
        if e.y != e.y2:
            elbows.append({"x": e.x, "y": e.y, "x2": e.x, "y2": e.y2})
            elbows.append({"x": e.x, "y": e.y2, "x2": e.x2, "y2": e.y2})
        else:
            elbows.append(e.to_dict())
    return nodes, pd.DataFrame(elbows)


# ----------------------------------------------------------------------------- districts
def district_params(n_feeders: int, seed: int) -> list[dict]:
    """Settings for a simulated district: feeders that differ in theft level, number of
    transformer meters, load shedding and cable faults."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_feeders):
        out.append({"theft": float(rng.choice([0.05, 0.10, 0.15, 0.25, 0.35, 0.45])),
                    "n_meters": int(rng.choice([0, 2, 4, 6, 8])),
                    "shedding": str(rng.choice(["none", "none", "4h", "8h"])),
                    "fault": bool(rng.random() < 0.3),
                    "seed": int(seed * 100 + i)})
    return out


def simulate_district(n_feeders: int, n_days: int, seed: int) -> dict:
    """Simulate a district; returns {feeder name: d}."""
    return {f"Feeder {i + 1}": simulate_case(p["theft"], p["n_meters"], "ends", 0.2, p["fault"],
                                             n_days, p["seed"], shedding=p["shedding"])
            for i, p in enumerate(district_params(n_feeders, seed))}


def district_table(feeders: dict) -> pd.DataFrame:
    """One row per feeder, ranked by estimated stolen energy per month.

    `feeders` maps a feeder name to (d, res) as returned by locate()."""
    rows = []
    for name, (d, res) in feeders.items():
        s = feeder_summary(d, res)
        pm = 30.0 / s["days"]
        st = res.stolen_physics_mwh.values
        top5 = np.sort(st)[::-1][:5].sum() / max(st.sum(), 1e-12)
        k_half = int(np.searchsorted(np.cumsum(np.sort(st)[::-1]) / max(st.sum(), 1e-12), 0.5) + 1)
        f = res.attrs.get("fault", {})
        row = {"feeder": name, "days": s["days"], "hours_on": s["hours_on"], "hours": s["hours"],
               "energy_in_month": s["head_mwh"] * pm, "unbilled_pct": s["unbilled_pct"],
               "stolen_month": s["stolen_physics"] * pm,
               "stolen_pct": 100 * s["stolen_physics"] / max(s["head_mwh"], 1e-12),
               "top5_share": 100 * top5, "areas_for_half": k_half,
               "meters": s["n_meters"], "areas": s["n_buses"],
               "fault": bool(f.get("detected", False)),
               "fault_kw": float(f.get("leak_kw", 0.0)) if f.get("detected") else 0.0}
        if "stolen_true" in s:
            row["true_stolen_month"] = s["stolen_true"] * pm
            row["true_fault"] = bool(d["is_fault"][0].any())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("stolen_month", ascending=False, ignore_index=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    if "true_stolen_month" in out:
        out["true_rank"] = out.true_stolen_month.rank(ascending=False, method="first").astype(int)
    return out


def district_plan(feeders: dict, n_areas: int) -> pd.DataFrame:
    """The `n_areas` transformer areas with the highest estimated theft per month across all
    feeders: where to send a limited number of inspection crews."""
    parts = []
    for name, (d, res) in feeders.items():
        pm = 30.0 / (d["head"].shape[1] / 24)
        p = pd.DataFrame({"feeder": name, "bus": res.bus.values,
                          "stolen_month": res.stolen_physics_mwh.values * pm,
                          "share": res.share_physics.values})
        if "true_stolen_mwh" in res:
            p["true_stolen_month"] = res.true_stolen_mwh.values * pm
            p["true_hotspot"] = res.true_hotspot.values
        parts.append(p)
    allp = pd.concat(parts, ignore_index=True).sort_values("stolen_month", ascending=False,
                                                           ignore_index=True)
    allp["cum_pct"] = 100 * allp.stolen_month.cumsum() / max(allp.stolen_month.sum(), 1e-12)
    plan = allp.head(n_areas).copy()
    plan.insert(0, "order", range(1, len(plan) + 1))
    plan.attrs["total_stolen_month"] = float(allp.stolen_month.sum())
    if "true_stolen_month" in allp:
        plan.attrs["true_total_month"] = float(allp.true_stolen_month.sum())
    return plan


# ----------------------------------------------------------------------------- inspections
def district_areas(feeders: dict) -> pd.DataFrame:
    """All transformer areas of a district, for planning inspection rounds: feeder, bus,
    model_score (estimated stolen share), weight (estimated stolen MWh per month), and the
    truth columns when simulated."""
    parts = []
    for name, (d, res) in feeders.items():
        pm = 30.0 / (d["head"].shape[1] / 24)
        p = pd.DataFrame({"feeder": name, "bus": res.bus.values,
                          "model_score": res.share_physics.values,
                          "weight": np.maximum(res.stolen_physics_mwh.values * pm, 1e-6),
                          "kva": res.kva.values})
        if "true_hotspot" in res:
            p["true_theft"] = res.true_hotspot.values
            p["true_stolen_month"] = res.true_stolen_mwh.values * pm
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def simulate_visits(plan: pd.DataFrame, areas: pd.DataFrame, seed: int,
                    sensitivity: float = 0.8) -> pd.DataFrame:
    """Fill in a work list as simulated crews would: theft found with probability
    sensitivity x share of premises checked (60-100%); no access on 8% of visits to clean
    areas and 15% to theft areas; recovered energy for found theft."""
    rng = np.random.default_rng(seed)
    t = areas.set_index(["feeder", "bus"]).loc[list(zip(plan.feeder, plan.bus))]
    theft = t["true_theft"].values.astype(bool)
    out = plan.copy()
    n = len(out)
    cov = rng.uniform(0.6, 1.0, n)
    prem = rng.integers(60, 400, n)
    miss = rng.random(n) < np.where(theft, 0.15, 0.08)
    found = theft & ~miss & (rng.random(n) < sensitivity * cov)
    kinds = np.array(["THEFT_BYPASS", "METER_TAMPER", "ILLEGAL_CONNECTION"])
    out["premises"] = prem
    out["premises_checked"] = np.round(prem * cov).astype(int)
    out["outcome"] = np.where(miss, rng.choice(["NO_ACCESS_REFUSED", "NOT_FOUND_ADDRESS_ERROR"], n,
                                               p=[0.8, 0.2]),
                              np.where(found, rng.choice(kinds, n, p=[0.4, 0.2, 0.4]), "CLEAN"))
    daily = t["true_stolen_month"].values * 1000 / 30          # kWh per day really stolen
    out["kwh_per_day_found"] = np.where(found, np.round(daily * cov * rng.uniform(0.5, 1.0, n), 1),
                                        np.nan)
    out["kwh_recovered"] = np.where(found, np.round(out["kwh_per_day_found"] * 90), np.nan)
    out["crew"] = rng.choice(["Crew A", "Crew B", "Crew C"], n)
    return out

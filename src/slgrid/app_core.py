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

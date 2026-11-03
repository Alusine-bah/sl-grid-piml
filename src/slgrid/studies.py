"""Sensitivity studies: meter placement and robustness.

All studies reuse one generated dataset. Meter readings for any set of buses are
re-synthesised from the stored ground truth (true bus demand and voltages) with the same
noise and drop-out model as metering.py, so placements can be compared on identical
scenarios.
"""
from __future__ import annotations

import numpy as np

from slgrid.topology import bus_features

POWER_FACTOR = 0.90


def remeter(d: dict, meter_buses, dropout: float = 0.2, noise: float = 0.02,
            seed: int = 0) -> dict:
    """Return a copy of d with transformer meters at `meter_buses` (list of bus names)."""
    rng = np.random.default_rng(seed)
    out = dict(d)
    names = list(d["bus_names"])
    all_names = list(d["all_bus_names"])
    cols = [names.index(m) for m in meter_buses]
    vcols = [all_names.index(m) for m in meter_buses]
    P = d["true_bus_p"][:, :, cols]
    Q = (P - d["fault_bus_p"][:, :, cols]) * np.tan(np.arccos(POWER_FACTOR))
    V = d["vm_pu"][:, :, vcols]
    mask = rng.random(P.shape) < dropout
    noisy = lambda x, sd: x * rng.normal(1.0, sd, size=x.shape)   # noqa: E731
    out["meter_p"] = np.where(mask, np.nan, noisy(P, noise))
    out["meter_q"] = np.where(mask, np.nan, noisy(Q, noise))
    out["meter_v"] = np.where(mask, np.nan, V + rng.normal(0, 0.003, V.shape))
    out["meter_names"] = np.array(meter_buses)
    if len(meter_buses) == 0:
        for k in ("meter_p", "meter_q", "meter_v"):
            out[k] = np.zeros(P.shape[:2] + (0,))
    return out


def placement(d: dict, n: int, strategy: str, seed: int = 0) -> list[str]:
    """Choose n buses for transformer meters.

    ends     spread meters over the far ends of the feeder: repeatedly pick the bus with
             the largest path impedance from the substation that is not on the path to,
             or behind, an already chosen meter. Far-end voltages carry information about
             every bus upstream, which the physics model can use.
    largest  the n buses with the largest transformer ratings (where most energy flows).
    random   n buses at random.
    """
    names = [str(b) for b in d["bus_names"]]
    if n == 0:
        return []
    if strategy == "random":
        return list(np.random.default_rng(seed).choice(names, n, replace=False))
    if strategy == "largest":
        kva = d["transformer_kva"].mean(axis=0)
        return [names[i] for i in np.argsort(-kva)[:n]]
    if strategy == "ends":
        t = bus_features(d["line_from"], d["line_to"], d["line_r_ohm"], d["line_x_ohm"])
        t = t.loc[names]
        ancestors = {}
        for b in names:
            a, p = set(), t.loc[b, "parent"]
            while p is not None and p in t.index:
                a.add(p)
                p = t.loc[p, "parent"]
            ancestors[b] = a
        chosen = []
        z = (t["path_r"] + t["path_x"]).sort_values(ascending=False)
        # first pass: far ends of different branches; then fill by impedance
        for b in z.index:
            if len(chosen) == n:
                break
            if all(b not in ancestors[c] and c not in ancestors[b] for c in chosen):
                chosen.append(b)
        for b in z.index:
            if len(chosen) == n:
                break
            if b not in chosen:
                chosen.append(b)
        return chosen
    raise ValueError(strategy)


def perturb_assets(d: dict, cable_error: float = 0.0, kva_error: float = 0.0,
                   seed: int = 0) -> dict:
    """Give the model wrong asset data (the truth in the dataset is unchanged).

    cable_error  each line's resistance and reactance multiplied by a random factor
                 with this relative standard deviation (wrong cable type or length)
    kva_error    each transformer rating multiplied by a random log-normal factor with
                 this standard deviation (out-of-date asset register)
    """
    rng = np.random.default_rng(seed)
    out = dict(d)
    if cable_error:
        f = rng.normal(1.0, cable_error, size=d["line_r_ohm"].shape).clip(0.3, 2.0)
        out["line_r_ohm"] = d["line_r_ohm"] * f
        out["line_x_ohm"] = d["line_x_ohm"] * f
    if kva_error:
        out["transformer_kva"] = d["transformer_kva"] * rng.lognormal(
            0.0, kva_error, size=d["transformer_kva"].shape)
    return out

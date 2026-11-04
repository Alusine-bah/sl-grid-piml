"""Baseline methods a utility could use today, without physics-informed learning.

B1. Energy audit (rule-based, no training)
    The standard utility method. Feeder level: unexplained energy = head energy - billed
    energy - an assumed technical-loss allowance. At a metered transformer: stolen =
    metered energy - billed energy. The unexplained energy left over is spread over the
    unmetered buses in proportion to their billing, since there is no information to
    say where it is.

B1b. Energy audit + transformer ratings (rule-based, no training)
    Uses the asset register: physical demand behind a transformer should be roughly
    proportional to its kVA rating. Expected energy per bus = its share of installed kVA
    times the feeder-head energy. Buses billing far less than expected are suspicious.
    Gives good locations but no physics, so its total is not tied to the energy balance.

B1c. Combined rules: ratings + meters + feeder energy gap (no training, no physics)
    The strongest simple method we could design. Metered buses: stolen = metered energy -
    billed. The rest of the feeder gap (head energy x 97% - billed) is spread over the
    unmetered buses in proportion to their B1b excess (expected from kVA - billed).
    Locates like B1b and keeps the total tied to the energy balance like B1.

B2. Pure machine learning (supervised, no physics)
    A gradient-boosted tree model trained on labelled scenarios. Per bus it sees billing
    statistics, the bus's position in the network (depth, path impedance, downstream size)
    as plain numbers, the feeder-level gap, the meter readings, and the voltages at the
    meters. It must learn any relationship between these on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from slgrid.topology import bus_features


def metered_mask(d) -> np.ndarray:
    return np.isin(d["bus_names"], d["meter_names"])


def _metered_energy(d, s_idx):
    """Energy at each metered bus over the scenario, scaling up for missing hours.

    Only hours when the feeder was energised count (load shedding leaves the meters dead
    and the feeder empty), so the average over readings is scaled by energised hours.
    """
    from slgrid.faults import energised_hours
    p = d["meter_p"][s_idx]                               # [S, T, M]
    out = np.zeros(p.shape[::2])
    for i, s in enumerate(s_idx):
        on = energised_hours(d["head"][s, :, 0])
        vals = p[i][on]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(vals, axis=0) if len(vals) else np.full(p.shape[2], np.nan)
        out[i] = np.nan_to_num(mean) * on.sum()
    return out                                            # [S, M] MWh


def energy_audit(d, s_idx, tech_loss_share: float = 0.03):
    """B1. Returns (score, stolen_MWh), each [S, B]."""
    billed = d["billing"][s_idx].sum(axis=1)             # [S, B] MWh
    head = d["head"][s_idx, :, 0].sum(axis=1)            # [S] MWh
    unexplained = head * (1 - tech_loss_share) - billed.sum(axis=1)

    stolen = np.zeros_like(billed)
    mm = metered_mask(d)
    m_cols = [int(np.flatnonzero(d["bus_names"] == m)[0]) for m in d["meter_names"]]
    stolen[:, m_cols] = np.maximum(_metered_energy(d, s_idx) - billed[:, m_cols], 0)

    rest = np.maximum(unexplained - stolen[:, m_cols].sum(axis=1), 0)
    w = billed[:, ~mm] / billed[:, ~mm].sum(axis=1, keepdims=True)
    stolen[:, ~mm] = rest[:, None] * w

    score = stolen / (stolen + billed + 1e-9)
    return score, stolen


def capacity_audit(d, s_idx, tech_loss_share: float = 0.03, max_share: float = 0.9):
    """B1b. Returns (score, stolen_MWh), each [S, B]."""
    billed = d["billing"][s_idx].sum(axis=1)
    head = d["head"][s_idx, :, 0].sum(axis=1)
    kva = d["transformer_kva"][s_idx]
    expected = kva / kva.sum(axis=1, keepdims=True) * head[:, None] * (1 - tech_loss_share)
    theta = np.clip(expected / billed - 1, 0, max_share / (1 - max_share))
    stolen = billed * theta
    return theta / (1 + theta), stolen


def combined_rules(d, s_idx, tech_loss_share: float = 0.03):
    """B1c. Returns (score, stolen_MWh), each [S, B]."""
    _, excess = capacity_audit(d, s_idx, tech_loss_share)
    billed = d["billing"][s_idx].sum(axis=1)
    head = d["head"][s_idx, :, 0].sum(axis=1)
    gap = np.maximum(head * (1 - tech_loss_share) - billed.sum(axis=1), 0)
    mm = metered_mask(d)
    stolen = np.zeros_like(billed)
    if mm.any():
        m_cols = [int(np.flatnonzero(d["bus_names"] == m)[0]) for m in d["meter_names"]]
        stolen[:, m_cols] = np.maximum(_metered_energy(d, s_idx) - billed[:, m_cols], 0)
    rest = np.maximum(gap - stolen.sum(axis=1), 0)
    w = excess[:, ~mm] / np.maximum(excess[:, ~mm].sum(axis=1, keepdims=True), 1e-12)
    stolen[:, ~mm] = rest[:, None] * w
    return stolen / (stolen + billed + 1e-9), stolen


def ml_features(d, s_idx) -> pd.DataFrame:
    """One row per (scenario, bus). Uses only observed data plus the network layout."""
    topo = bus_features(d["line_from"], d["line_to"], d["line_r_ohm"], d["line_x_ohm"])
    topo = topo.loc[d["bus_names"]]
    mm = metered_mask(d)
    m_cols = [int(np.flatnonzero(d["bus_names"] == m)[0]) for m in d["meter_names"]]

    billing = d["billing"][s_idx]                         # [S, D, B]
    billed = billing.sum(axis=1)                          # [S, B]
    head_e = d["head"][s_idx, :, 0].sum(axis=1)
    feeder_gap = 1 - billed.sum(axis=1) / head_e
    metered_e = _metered_energy(d, s_idx)
    vm = np.nanmean(d["meter_v"][s_idx], axis=1)          # [S, M] mean voltage at meters
    vm_min = np.nanmin(d["meter_v"][s_idx], axis=1)
    head_v = d["head"][s_idx, :, 2].mean(axis=1)

    rows = []
    for si in range(len(s_idx)):
        share = billed[si] / billed[si].sum()
        for b in range(billed.shape[1]):
            r = {
                "billed_share": share[b],
                "billed_cv": billing[si, :, b].std() / (billing[si, :, b].mean() + 1e-9),
                "depth": topo["depth"].iloc[b],
                "path_r": topo["path_r"].iloc[b],
                "path_x": topo["path_x"].iloc[b],
                "n_downstream": topo["n_downstream"].iloc[b],
                "feeder_gap": feeder_gap[si],
                "is_metered": float(mm[b]),
                "meter_gap": np.nan,
                "head_v": head_v[si],
            }
            if "transformer_kva" in d:
                kva = d["transformer_kva"][s_idx[si]]
                expected = kva[b] / kva.sum() * head_e[si]
                r["billed_per_expected"] = billed[si, b] / expected
                r["kva_share"] = kva[b] / kva.sum()
            if mm[b]:
                k = m_cols.index(b)
                r["meter_gap"] = 1 - billed[si, b] / max(metered_e[si, k], 1e-9)
            for k, name in enumerate(d["meter_names"]):
                r[f"v_mean_{name}"] = vm[si, k]
                r[f"v_drop_{name}"] = head_v[si] - vm_min[si, k]
            rows.append(r)
    return pd.DataFrame(rows)


class PureML:
    """B2. Supervised gradient-boosted trees, no physics."""

    def __init__(self, seed: int = 0):
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                  random_state=seed)
        self.reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                 random_state=seed)

    def fit(self, d, train_idx):
        X = ml_features(d, train_idx)
        y = d["is_theft"][train_idx].ravel().astype(int)
        stolen = d["theft_bus_p"][train_idx].sum(axis=1).ravel()
        self.clf.fit(X, y)
        self.reg.fit(X, stolen)
        return self

    def predict(self, d, s_idx):
        X = ml_features(d, s_idx)
        n_b = len(d["bus_names"])
        score = self.clf.predict_proba(X)[:, 1].reshape(len(s_idx), n_b)
        stolen = np.maximum(self.reg.predict(X), 0).reshape(len(s_idx), n_b)
        return score, stolen

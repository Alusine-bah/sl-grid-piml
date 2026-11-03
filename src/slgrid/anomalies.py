"""Theft and fault scenarios injected into the Freetown-style feeder.

Two anomaly types, both invisible to billing:

- Theft: illegal connections or meter bypass at "hotspot" buses. The physical demand
  of the feeder stays at its calibrated level; theft means that a share of the
  consumption at a hotspot bus is not billed. Theft clusters in dense neighbourhoods
  (S2, S4), so a scenario has a set of hotspot buses rather than theft spread evenly.
  Stolen power follows the bus's own usage pattern, since it is used by the same kind
  of households.

  Two ways to size it:
    * `target_ntl`: stolen energy is this share of all energy delivered to customers.
      Hotspots are added until the target can be met with no bus stealing more than
      `max_share` of its consumption. Sierra Leone commercial losses: 18% (S4) up to
      about 40% (S5 total 46-53% minus technical).
    * otherwise: each of `n_theft_buses` hotspots steals a random share in
      `theft_share`.

- Fault: a high-impedance earth fault or failing cable joint (S6: 67% of 11 kV faults are
  underground cable joints). Modelled as a constant resistive leakage that starts at a
  random hour and stays on day and night. The flat shape is what separates it from theft.

Every scenario records exact labels: which buses are anomalous, the type, and the
true stolen or leaked power at every hour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ScenarioConfig:
    n_theft_buses: int = 3
    theft_share: tuple[float, float] = (0.2, 0.6)   # share of a hotspot's consumption stolen
    target_ntl: float | None = None
    max_share: float = 0.9
    n_fault_buses: int = 0
    fault_kw: tuple[float, float] = (40.0, 150.0)
    power_factor: float = 0.90
    exclude_buses: tuple[int, ...] = field(default_factory=tuple)


def _pick_hotspots(mean_load, allowed, cfg, rng):
    """Choose hotspot buses and the share of consumption stolen at each."""
    order = list(rng.permutation(allowed))
    if cfg.target_ntl is None:
        idx = order[: cfg.n_theft_buses]
        return idx, rng.uniform(*cfg.theft_share, size=len(idx))

    need = cfg.target_ntl * mean_load.sum()
    # Aim for hotspots stealing ~60% of their consumption on average, leaving room
    # for variation below max_share.
    idx = []
    for j in order:
        idx.append(j)
        if len(idx) >= cfg.n_theft_buses and mean_load[idx].sum() * 0.6 >= need:
            break
    shares = rng.uniform(0.6, 1.4, size=len(idx))
    for _ in range(50):     # rescale to hit the target exactly, respecting the cap
        stolen = (shares * mean_load[idx]).sum()
        shares = np.minimum(shares * need / stolen, cfg.max_share)
        if abs((shares * mean_load[idx]).sum() - need) < 1e-9 * need:
            break
    return idx, shares


def inject(customers: pd.DataFrame, demand_p: pd.DataFrame, cfg: ScenarioConfig,
           rng: np.random.Generator) -> dict:
    """Split physical demand into billed and stolen parts, and add faults.

    `demand_p` is the true physical consumption at each load (from loads.generate_profiles).

    Returns a dict with:
      p_mw, q_mvar     total physical demand per load incl. faults (power-flow input)
      billed_mw        legitimate, billed consumption per load and hour
      theft_mw         stolen power per load and hour
      fault_mw         fault leakage per load and hour
      labels           one row per load: feeder_bus, kind, is_theft, is_fault,
                       theft_share, fault_kw, fault_start
    """
    n_t, n_l = demand_p.shape
    buses = customers["feeder_bus"].values
    allowed = [j for j in range(n_l) if buses[j] not in cfg.exclude_buses]
    D = demand_p.values

    labels = pd.DataFrame({
        "feeder_bus": buses, "kind": customers["kind"].values,
        "is_theft": False, "is_fault": False,
        "theft_share": 0.0, "fault_kw": 0.0, "fault_start": -1,
    }, index=customers.index)

    theft = np.zeros((n_t, n_l))
    idx, shares = _pick_hotspots(D.mean(axis=0), allowed, cfg, rng)
    for j, sh in zip(idx, shares):
        hourly = (sh * rng.normal(1.0, 0.05, size=n_t)).clip(0, 0.98)
        theft[:, j] = D[:, j] * hourly
        labels.iloc[j, labels.columns.get_loc("is_theft")] = True
        labels.iloc[j, labels.columns.get_loc("theft_share")] = sh
    billed = D - theft

    fault = np.zeros((n_t, n_l))
    remaining = [j for j in allowed if j not in set(idx)]
    fault_idx = rng.choice(remaining, size=min(cfg.n_fault_buses, len(remaining)), replace=False)
    for j in fault_idx:
        kw = rng.uniform(*cfg.fault_kw)
        start = int(rng.integers(0, n_t))
        fault[start:, j] = kw / 1000.0
        labels.iloc[j, labels.columns.get_loc("is_fault")] = True
        labels.iloc[j, labels.columns.get_loc("fault_kw")] = kw
        labels.iloc[j, labels.columns.get_loc("fault_start")] = start

    tan = np.tan(np.arccos(cfg.power_factor))
    idx_t, cols = demand_p.index, demand_p.columns
    frame = lambda a: pd.DataFrame(a, index=idx_t, columns=cols)   # noqa: E731
    return {
        "p_mw": frame(D + fault),
        "q_mvar": frame(D * tan),            # faults are resistive leakage: no Q
        "billed_mw": frame(billed),
        "theft_mw": frame(theft),
        "fault_mw": frame(fault),
        "labels": labels,
    }

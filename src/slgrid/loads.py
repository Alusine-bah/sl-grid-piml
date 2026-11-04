"""Load profiles for the Freetown-style feeder.

Residential demand comes from the Freetown household survey of Rogers et al. (2024)
(docs/sources.md S1): per-household diversified demand at hour h is

    d(h) = sum_i  MDD_i * LVF_i(h) / 100

where MDD_i is the maximum diversified demand per customer of appliance i (Table 2) and
LVF_i(h) is its hourly load variation factor (Fig. 5). This gives a morning peak around
08:00 and an evening peak around 19:00 of about 0.28 kW per household.

Commercial demand has no published Freetown profile. We assume a daytime plateau from
08:00 to 18:00 with a 30% night base, matching the daytime supply hours that the NPA
load-sharing programme gave to the commercial district (S6, Table 3.4-17). This is an
assumption and is marked as such.

Random variation: each day gets a feeder-wide factor (sd 5%), and each bus-hour gets its
own factor (sd 8%), so no two days or buses are identical.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "sources"
HOURS = np.arange(24)

HOUSEHOLDS_PER_TRANSFORMER = 220   # S1 households / S5 transformer count
POWER_FACTOR = 0.90                # S6 planning value (measured 0.79 on one feeder)


def household_profile_kw() -> pd.Series:
    """Diversified demand of one Freetown household, kW, for hours 0-23."""
    app = pd.read_csv(DATA / "rogers2024_appliances.csv").set_index("appliance")
    lvf = pd.read_csv(DATA / "rogers2024_hourly_lvf.csv").set_index("hour")
    d = (lvf[app.index] / 100.0 * app["mdd_per_customer_kw"]).sum(axis=1)
    d.name = "kw_per_household"
    return d


def commercial_profile_pu() -> pd.Series:
    """Assumed commercial shape, per unit of its peak, for hours 0-23."""
    p = np.full(24, 0.30)
    p[7] = 0.6
    p[8:18] = 1.0
    p[18] = 0.7
    p[19] = 0.5
    return pd.Series(p, index=HOURS, name="commercial_pu")


def assign_customers(net, total_households: int = 12_000,
                     commercial_peak_mw: float = 1.2) -> pd.DataFrame:
    """Spread households and commercial demand over the feeder's load buses.

    Residential buses get households in proportion to their IEEE load share;
    commercial buses share `commercial_peak_mw` the same way.
    """
    L = net.load.copy()
    res = L["kind"] == "residential"
    com = ~res
    L["households"] = 0
    L.loc[res, "households"] = np.round(
        total_households * L.loc[res, "p_mw"] / L.loc[res, "p_mw"].sum()).astype(int)
    L["commercial_peak_mw"] = 0.0
    L.loc[com, "commercial_peak_mw"] = (
        commercial_peak_mw * L.loc[com, "p_mw"] / L.loc[com, "p_mw"].sum())
    L["transformers"] = np.maximum(1, np.round(L["households"] / HOUSEHOLDS_PER_TRANSFORMER)).astype(int)
    return L[["feeder_bus", "kind", "households", "commercial_peak_mw", "transformers"]]


def perturb_customers(customers: pd.DataFrame, rng: np.random.Generator,
                      sd: float = 0.3) -> pd.DataFrame:
    """Randomise how many customers sit behind each bus.

    A utility does not know the true number of households (legal or illegal) behind each
    transformer, and it differs between neighbourhoods. Each scenario therefore scales
    every bus by a random log-normal factor (sd 0.3, i.e. typically +/-30%), then rescales
    so the feeder total stays the same. This stops models from memorising a fixed
    "normal" level for each bus.
    """
    c = customers.copy()
    f = rng.lognormal(0.0, sd, size=len(c))
    hh = c["households"].values * f
    c["households"] = np.round(hh * c["households"].sum() / hh.sum()).astype(int)
    com = c["commercial_peak_mw"].values * f
    if com.sum() > 0:
        c["commercial_peak_mw"] = com * c["commercial_peak_mw"].sum() / com.sum()
    c["transformers"] = np.maximum(1, np.round(c["households"] / HOUSEHOLDS_PER_TRANSFORMER)).astype(int)
    return c


def generate_profiles(customers: pd.DataFrame, n_days: int = 1, seed: int = 0,
                      power_factor: float = POWER_FACTOR) -> dict[str, pd.DataFrame]:
    """Hourly true demand at every load bus.

    Returns {"p_mw": DataFrame, "q_mvar": DataFrame}, indexed by timestep
    (n_days * 24 rows), one column per load (same order as net.load).
    """
    rng = np.random.default_rng(seed)
    hh = household_profile_kw().values / 1000.0          # MW per household
    com = commercial_profile_pu().values

    n_t = n_days * 24
    n_l = len(customers)
    base = np.zeros((n_t, n_l))
    for d in range(n_days):
        day_factor = rng.normal(1.0, 0.05)
        for j, (_, c) in enumerate(customers.iterrows()):
            shape = c["households"] * hh + c["commercial_peak_mw"] * com
            base[d * 24:(d + 1) * 24, j] = shape * day_factor

    noise = rng.normal(1.0, 0.08, size=base.shape).clip(0.6, 1.4)
    p = base * noise
    q = p * np.tan(np.arccos(power_factor))

    idx = pd.RangeIndex(n_t, name="t")
    cols = customers.index
    return {"p_mw": pd.DataFrame(p, index=idx, columns=cols),
            "q_mvar": pd.DataFrame(q, index=idx, columns=cols)}


# ----------------------------------------------------------------------------- load shedding
SHEDDING_PATTERNS = {
    "none": "Supply all day",
    "4h": "4 hours off per day",
    "8h": "8 hours off per day",
    "rotation": "Rotation: on 7 pm to 7 am, one day in three",
}


def supply_schedule(n_days: int, pattern: str = "none", rng: np.random.Generator | None = None) -> np.ndarray:
    """Hourly supply status of the feeder: 1 = energised, 0 = switched off.

    Patterns:
      none       supply all day
      4h, 8h     one block of that many hours off per day, starting at a random hour
                 (load shedding / rationing in blocks)
      rotation   the 2008 NPA load-sharing programme for domestic feeders in Western Area:
                 supplied 7 pm to 7 am, one day in three (JICA 2009, Table 3.4-17; docs/sources.md S6)
    """
    rng = rng or np.random.default_rng(0)
    T = n_days * 24
    on = np.ones(T)
    if pattern in ("4h", "8h"):
        k = int(pattern[:-1])
        for d in range(n_days):
            start = d * 24 + int(rng.integers(0, 24))
            on[start:start + k] = 0
    elif pattern == "rotation":
        on[:] = 0
        phase = int(rng.integers(0, 3))
        for d in range(n_days):
            if (d + phase) % 3 == 0:
                on[d * 24 + 19: d * 24 + 31] = 1      # 19:00 to 07:00 next day
        on = on[:T]
    elif pattern != "none":
        raise ValueError(f"unknown shedding pattern: {pattern}")
    return on


def apply_supply(p_mw: pd.DataFrame, q_mvar: pd.DataFrame, on: np.ndarray,
                 cold_load: float = 0.3) -> dict[str, pd.DataFrame]:
    """Zero demand while the feeder is off and add the surge when it comes back.

    When supply returns after an outage, fridges, freezers, fans and water heaters all
    restart at once ("cold load pickup"), so demand in the first hours is higher than
    usual. Modelled as +cold_load in the first hour back and +cold_load/2 in the second.
    The size is an assumption (typical values are 20-50%), not a Freetown measurement.
    """
    on = np.asarray(on, dtype=float)
    mult = on.copy()
    for t in range(1, len(on)):
        if on[t] and not on[t - 1]:
            mult[t] *= 1 + cold_load
            if t + 1 < len(on) and on[t + 1]:
                mult[t + 1] *= 1 + cold_load / 2
    return {"p_mw": p_mw.mul(mult, axis=0), "q_mvar": q_mvar.mul(mult, axis=0)}

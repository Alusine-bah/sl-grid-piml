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

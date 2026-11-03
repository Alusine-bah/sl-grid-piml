"""What a utility like EDSA can actually observe.

Three data streams, each imperfect:

1. Feeder-head meter at the substation 11 kV breaker: P, Q and V every hour.
   Assumed always available, 1% noise (class 1 metering).
2. A few transformer meters at selected buses: P, Q and V every hour, 2% noise, with
   readings missing at random (communication drop-outs, outages, broken meters; S6 notes
   most switchgear meters in 2008 were not working).
3. Billing: energy sold to legitimate customers at each bus, summed per day (prepaid
   vending or monthly bills). Theft and faults never appear here. 3% noise.

Truth (actual loads, voltages, theft) is kept separately and only used for scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MeterConfig:
    meter_buses: tuple[int, ...] = (6, 12, 17, 24, 29, 32)
    head_noise: float = 0.01
    meter_noise: float = 0.02
    billing_noise: float = 0.03
    dropout: float = 0.2           # fraction of transformer-meter readings missing


def observe(res: dict, billed_mw: pd.DataFrame, customers: pd.DataFrame,
            cfg: MeterConfig, rng: np.random.Generator) -> dict:
    """Build the utility's view from a simulation result.

    `res` is the output of simulate.run_timeseries on the true (anomalous) demand.
    """
    s = res["summary"]
    n_t = len(s)

    def noisy(x, sd):
        return x * rng.normal(1.0, sd, size=np.shape(x))

    head = pd.DataFrame({
        "p_mw": noisy(s["p_head_mw"].values, cfg.head_noise),
        "q_mvar": noisy(s["q_head_mvar"].values, cfg.head_noise),
        "vm_pu": res["vm_pu"]["bus_0"].values + rng.normal(0, 0.002, n_t),
    }, index=s.index)

    cols = [f"bus_{b}" for b in cfg.meter_buses]
    load_by_bus = res["bus_p_mw"][cols]
    q_by_bus = res["bus_q_mvar"][cols]
    vm = res["vm_pu"][cols]
    mask = rng.random((n_t, len(cols))) < cfg.dropout
    meters = {
        "p_mw": noisy(load_by_bus, cfg.meter_noise).mask(mask),
        "q_mvar": noisy(q_by_bus, cfg.meter_noise).mask(mask),
        "vm_pu": (vm + rng.normal(0, 0.003, vm.shape)).mask(mask),
    }

    n_days = n_t // 24
    daily = billed_mw.values[: n_days * 24].reshape(n_days, 24, -1).sum(axis=1)
    by_bus = pd.DataFrame(daily, columns=[f"bus_{b}" for b in customers["feeder_bus"]])
    by_bus = by_bus.T.groupby(level=0).sum().T          # combine loads on the same bus
    billing = noisy(by_bus, cfg.billing_noise)
    billing.index.name = "day"

    return {"head": head, "meters": meters, "billing_mwh": billing}

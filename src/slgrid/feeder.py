"""Base feeder model and the project's standard feeder format.

The base feeder is the IEEE 33-bus radial test feeder (Baran & Wu, 1989),
adapted from 12.66 kV to 11 kV, which is EDSA's main distribution voltage.

Adaptation choices:
- Line impedances (ohm) are kept unchanged, since they describe the physical conductors.
- Loads are scaled by (11 / 12.66)^2 so the feeder keeps a similar voltage-drop
  profile at the lower voltage.
- The five normally-open tie lines are removed, so the feeder is purely radial.

Every data source (this simulator now, utility data later) must produce the
standard format returned by `to_standard()`. Models only ever see that format.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

BASE_KV = 11.0
ORIGINAL_KV = 12.66


def build_base_feeder() -> pp.pandapowerNet:
    """Return the 11 kV radial version of the IEEE 33-bus feeder."""
    net = pn.case33bw()
    net.bus["vn_kv"] = BASE_KV

    scale = (BASE_KV / ORIGINAL_KV) ** 2
    net.load["p_mw"] *= scale
    net.load["q_mvar"] *= scale

    net.line = net.line[net.line["in_service"]].copy()
    net.line.reset_index(drop=True, inplace=True)
    return net


def to_standard(net: pp.pandapowerNet) -> dict[str, pd.DataFrame]:
    """Convert a pandapower network to the project's standard feeder format.

    Returns a dict with:
      buses: bus_id, vn_kv, p_load_mw, q_load_mvar, is_source
      lines: line_id, from_bus, to_bus, r_ohm, x_ohm
    """
    loads = net.load.groupby("bus")[["p_mw", "q_mvar"]].sum()
    buses = pd.DataFrame({"bus_id": net.bus.index, "vn_kv": net.bus["vn_kv"].values})
    buses["p_load_mw"] = buses["bus_id"].map(loads["p_mw"]).fillna(0.0)
    buses["q_load_mvar"] = buses["bus_id"].map(loads["q_mvar"]).fillna(0.0)
    buses["is_source"] = buses["bus_id"].isin(net.ext_grid["bus"])

    L = net.line
    lines = pd.DataFrame({
        "line_id": L.index,
        "from_bus": L["from_bus"].values,
        "to_bus": L["to_bus"].values,
        "r_ohm": (L["r_ohm_per_km"] * L["length_km"]).values,
        "x_ohm": (L["x_ohm_per_km"] * L["length_km"]).values,
    })
    return {"buses": buses, "lines": lines}


def run_powerflow(net: pp.pandapowerNet) -> dict[str, pd.DataFrame]:
    """Run AC power flow and return bus voltages and line flows."""
    pp.runpp(net)
    buses = pd.DataFrame({
        "bus_id": net.bus.index,
        "vm_pu": net.res_bus["vm_pu"].values,
        "va_deg": net.res_bus["va_degree"].values,
    })
    lines = pd.DataFrame({
        "line_id": net.line.index,
        "p_from_mw": net.res_line["p_from_mw"].values,
        "q_from_mvar": net.res_line["q_from_mvar"].values,
        "loss_mw": net.res_line["pl_mw"].values,
        "i_ka": net.res_line["i_ka"].values,
    })
    return {"buses": buses, "lines": lines}


def is_radial(std: dict[str, pd.DataFrame]) -> bool:
    """A connected feeder is radial when it has exactly (buses - 1) lines."""
    return len(std["lines"]) == len(std["buses"]) - 1

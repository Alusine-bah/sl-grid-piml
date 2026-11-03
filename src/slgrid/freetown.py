"""Freetown-style 11 kV feeder.

Keeps the radial layout of the IEEE 33-bus feeder, but replaces its lines with real
Freetown cable and overhead-line types (JICA 2009, Table 6.4-1, see docs/sources.md S7)
and adds a 33/11 kV substation transformer at the head (Table 6.4-2).

Layout choices (documented assumptions):
- The trunk near the substation and the first sections of each lateral are underground
  Cu 185 mm2 XLPE cable (1.2 km per section); the outer parts are overhead AAAC 120 mm2
  (1.8 km per section). Total about 50 km of line, 29% underground, similar to the outer
  Freetown supply areas in S6 (Wilberforce 62 km, Wellington 76 km), which have more
  overhead line than the city centre.
- Lengths were calibrated so that, with the default load (12,000 households + 1.2 MW
  commercial, peak about 4.3 MW), the lowest voltage at peak is about 0.91 pu, between
  the measured weak Freetown buses Roportee (0.90) and Wellington (0.87) in S5, and MV
  technical losses are about 3% of energy (S5 reports 4.8% for the whole MV network).

Each load bus represents a cluster of distribution transformers. Buses 23, 24 and 29,
which carry the largest loads in the IEEE feeder, are treated as commercial; the rest
are residential.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp

from slgrid.feeder import build_base_feeder

DATA = Path(__file__).resolve().parents[2] / "data" / "sources"

COMMERCIAL_BUSES = (23, 24, 29)
# Line indices of the base feeder laid as underground cable: the trunk near the
# substation and the first sections of the three laterals.
UNDERGROUND_LINES = {0, 1, 2, 3, 4, 5, 17, 21, 24, 25, 26, 27}
UG_TYPE = "Cu185_XLPE"
OH_TYPE = "AAAC120"
UG_LENGTH_KM = 1.2
OH_LENGTH_KM = 1.8


def load_conductors() -> pd.DataFrame:
    return pd.read_csv(DATA / "jica2009_conductors.csv").set_index("conductor")


def build_freetown_feeder(source_vm_pu: float = 1.0) -> pp.pandapowerNet:
    """Return the Freetown-style feeder with nominal loads (scaled later by profiles).

    Bus 0 is the 33 kV side of the substation; bus 'sub11' is the 11 kV busbar.
    Feeder buses keep their IEEE numbers 1-32 so results stay comparable.
    """
    base = build_base_feeder()
    cond = load_conductors()
    net = pp.create_empty_network(name="freetown_style_11kv")

    bus33 = pp.create_bus(net, vn_kv=33.0, name="sub_33kV")
    pp.create_ext_grid(net, bus33, vm_pu=source_vm_pu, name="grid")

    bus_map = {}
    for b in base.bus.index:
        bus_map[b] = pp.create_bus(net, vn_kv=11.0, name=f"bus_{b}")

    # 33/11 kV substation transformer: 10 MVA, 9.5% impedance (JICA Table 6.4-2)
    pp.create_transformer_from_parameters(
        net, hv_bus=bus33, lv_bus=bus_map[0], sn_mva=10.0, vn_hv_kv=33.0, vn_lv_kv=11.0,
        vk_percent=9.5, vkr_percent=0.8, pfe_kw=8.0, i0_percent=0.2, name="sub_tx",
    )

    for i, row in base.line.iterrows():
        ctype = UG_TYPE if i in UNDERGROUND_LINES else OH_TYPE
        c = cond.loc[ctype]
        length = UG_LENGTH_KM if i in UNDERGROUND_LINES else OH_LENGTH_KM
        pp.create_line_from_parameters(
            net, from_bus=bus_map[row.from_bus], to_bus=bus_map[row.to_bus],
            length_km=length, r_ohm_per_km=c.r_ohm_per_km, x_ohm_per_km=c.x_ohm_per_km,
            c_nf_per_km=c.c_nf_per_km, max_i_ka=c.max_current_a / 1000.0,
            name=f"line_{i}", type="cs" if i in UNDERGROUND_LINES else "ol",
        )
        net.line.loc[net.line.index[-1], "std_type"] = ctype

    for _, ld in base.load.iterrows():
        kind = "commercial" if ld.bus in COMMERCIAL_BUSES else "residential"
        pp.create_load(net, bus=bus_map[ld.bus], p_mw=ld.p_mw, q_mvar=0.0,
                       name=f"{kind}_{ld.bus}")
    net.load["kind"] = np.where(net.load["name"].str.startswith("commercial"),
                                "commercial", "residential")
    net.load["feeder_bus"] = [int(n.split("_")[1]) for n in net.load["name"]]
    net.load["share"] = net.load["p_mw"] / net.load["p_mw"].sum()
    return net


def feeder_summary(net: pp.pandapowerNet) -> dict:
    L = net.line
    ug = L[L["type"] == "cs"]["length_km"].sum()
    total = L["length_km"].sum()
    return {"line_km": total, "underground_share": ug / total,
            "load_buses": len(net.load),
            "commercial_buses": int((net.load["kind"] == "commercial").sum())}

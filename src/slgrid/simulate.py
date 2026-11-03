"""Quasi-static time-series power flow: one AC power flow per timestep."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandapower as pp


def run_timeseries(net: pp.pandapowerNet, p_mw: pd.DataFrame,
                   q_mvar: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run a power flow for every row of p_mw / q_mvar.

    Returns per-timestep results:
      vm_pu      bus voltage magnitude (columns = feeder bus names)
      line_p_mw  active power entering each line at its sending end
      line_loss  active power loss of each line, MW
      summary    source power, total load, total losses, min voltage
    """
    names = net.bus["name"].values
    vm, lp, ll, summ = [], [], [], []
    for t in p_mw.index:
        net.load["p_mw"] = p_mw.loc[t].values
        net.load["q_mvar"] = q_mvar.loc[t].values
        pp.runpp(net, numba=True)
        vm.append(net.res_bus["vm_pu"].values.copy())
        lp.append(net.res_line["p_from_mw"].values.copy())
        ll.append(net.res_line["pl_mw"].values.copy())
        p_src = net.res_ext_grid["p_mw"].sum()
        p_load = net.load["p_mw"].sum()
        summ.append({
            "p_source_mw": p_src,
            "p_load_mw": p_load,
            "p_loss_line_mw": net.res_line["pl_mw"].sum(),
            "p_loss_trafo_mw": net.res_trafo["pl_mw"].sum(),
            "vm_min_pu": net.res_bus["vm_pu"].iloc[1:].min(),
            "bus_vm_min": names[1:][net.res_bus["vm_pu"].iloc[1:].values.argmin()],
        })
    idx = p_mw.index
    return {
        "vm_pu": pd.DataFrame(vm, index=idx, columns=names),
        "line_p_mw": pd.DataFrame(lp, index=idx, columns=net.line["name"].values),
        "line_loss": pd.DataFrame(ll, index=idx, columns=net.line["name"].values),
        "summary": pd.DataFrame(summ, index=idx),
    }

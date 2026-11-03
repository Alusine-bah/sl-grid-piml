"""Generate labelled scenario datasets.

Each scenario is a run of `n_days` on the Freetown-style feeder with its own random
loads, theft hotspots, possible cable fault, and metering noise and drop-outs.

Saved per dataset (one compressed .npz file plus a scenario table):

Truth (for training targets and scoring only):
  true_bus_p      [S, T, B]  physical demand at each load bus, MW
  theft_bus_p     [S, T, B]  stolen power at each load bus, MW
  fault_bus_p     [S, T, B]  fault leakage at each load bus, MW
  vm_pu           [S, T, N]  voltage at every bus (N = all feeder buses incl. head)
  line_loss       [S, T, L]  technical loss of every line, MW
  is_theft        [S, B]     bus label
  is_fault        [S, B]     bus label
  households      [S, B]     true number of households behind each bus (varies by scenario)

What the utility sees:
  head            [S, T, 3]  feeder-head P (MW), Q (MVAr), V (pu)
  meter_p, meter_q, meter_v  [S, T, M]  transformer meters, NaN where missing
  billing         [S, D, B]  billed energy per day and bus, MWh

Metadata: bus names, meter bus names, line from/to buses, line r and x in ohm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from slgrid.anomalies import ScenarioConfig, inject
from slgrid.freetown import build_freetown_feeder
from slgrid.loads import assign_customers, generate_profiles, perturb_customers
from slgrid.metering import MeterConfig, observe
from slgrid.simulate import run_timeseries


@dataclass
class DatasetConfig:
    n_scenarios: int = 50
    n_days: int = 7
    ntl_levels: tuple[float, ...] = (0.15, 0.25, 0.35)
    fault_probability: float = 0.3
    meter_buses: tuple[int, ...] = (6, 12, 17, 24, 29, 32)
    dropout: float = 0.2
    customer_spread: float = 0.3   # random variation of customers per bus between scenarios
    seed: int = 0


def generate(cfg: DatasetConfig, out_dir: str | Path, name: str = "dataset",
             progress: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    net = build_freetown_feeder()
    base_cust = assign_customers(net)
    cust = base_cust
    bus_cols = [f"bus_{b}" for b in sorted(cust["feeder_bus"].unique())]
    meter_cols = [f"bus_{b}" for b in cfg.meter_buses]
    mcfg = MeterConfig(meter_buses=cfg.meter_buses, dropout=cfg.dropout)

    def per_bus(df):
        g = df.copy()
        g.columns = [f"bus_{b}" for b in cust["feeder_bus"]]
        return g.T.groupby(level=0).sum().T[bus_cols].values

    keys = ["true_bus_p", "theft_bus_p", "fault_bus_p", "vm_pu", "line_loss",
            "is_theft", "is_fault", "households", "head", "meter_p", "meter_q", "meter_v", "billing"]
    store = {k: [] for k in keys}
    rows = []

    it = range(cfg.n_scenarios)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc="scenarios")
    for s in it:
        ntl = float(rng.choice(cfg.ntl_levels))
        n_fault = int(rng.random() < cfg.fault_probability)
        scfg = ScenarioConfig(target_ntl=ntl, n_fault_buses=n_fault)
        cust = perturb_customers(base_cust, rng, cfg.customer_spread) if cfg.customer_spread else base_cust
        demand = generate_profiles(cust, n_days=cfg.n_days, seed=int(rng.integers(1 << 31)))
        sc = inject(cust, demand["p_mw"], scfg, rng)
        res = run_timeseries(net, sc["p_mw"], sc["q_mvar"])
        obs = observe(res, sc["billed_mw"], cust, mcfg, rng)

        lab = sc["labels"].copy()
        lab["bus"] = [f"bus_{b}" for b in lab["feeder_bus"]]
        lab = lab.groupby("bus")[["is_theft", "is_fault"]].any().loc[bus_cols]

        store["true_bus_p"].append(res["bus_p_mw"][bus_cols].values)
        store["theft_bus_p"].append(per_bus(sc["theft_mw"]))
        store["fault_bus_p"].append(per_bus(sc["fault_mw"]))
        store["vm_pu"].append(res["vm_pu"].values)
        store["line_loss"].append(res["line_loss"].values)
        store["is_theft"].append(lab["is_theft"].values)
        store["is_fault"].append(lab["is_fault"].values)
        hh = cust.assign(bus=[f"bus_{b}" for b in cust["feeder_bus"]]).groupby("bus")["households"].sum()
        store["households"].append(hh.loc[bus_cols].values)
        store["head"].append(obs["head"][["p_mw", "q_mvar", "vm_pu"]].values)
        store["meter_p"].append(obs["meters"]["p_mw"][meter_cols].values)
        store["meter_q"].append(obs["meters"]["q_mvar"][meter_cols].values)
        store["meter_v"].append(obs["meters"]["vm_pu"][meter_cols].values)
        store["billing"].append(obs["billing_mwh"][bus_cols].values)

        s_ = res["summary"]
        rows.append({
            "scenario": s, "target_ntl": ntl, "n_theft_buses": int(lab["is_theft"].sum()),
            "has_fault": bool(n_fault),
            "true_ntl": sc["theft_mw"].values.sum() / s_["p_load_mw"].sum(),
            "tech_loss": (s_["p_loss_line_mw"].sum()) / s_["p_head_mw"].sum(),
            "unbilled_seen": 1 - obs["billing_mwh"].values.sum() / obs["head"]["p_mw"].sum(),
            "vm_min": s_["vm_min_pu"].min(),
        })

    arrays = {k: np.stack(v) for k, v in store.items()}
    arrays["bus_names"] = np.array(bus_cols)
    arrays["all_bus_names"] = net.bus["name"].values.astype(str)
    arrays["meter_names"] = np.array(meter_cols)
    names = net.bus["name"].values.astype(str)
    arrays["line_from"] = names[net.line["from_bus"].values]
    arrays["line_to"] = names[net.line["to_bus"].values]
    arrays["line_r_ohm"] = (net.line["r_ohm_per_km"] * net.line["length_km"]).values
    arrays["line_x_ohm"] = (net.line["x_ohm_per_km"] * net.line["length_km"]).values

    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    pd.DataFrame(rows).to_csv(out_dir / f"{name}_scenarios.csv", index=False)
    pd.Series(asdict(cfg)).to_json(out_dir / f"{name}_config.json")
    return path

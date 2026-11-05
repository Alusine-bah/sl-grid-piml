import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from slgrid.ami import AMIConfig, ami_inputs, large_customer_buses, rank_large_customers, simulate_ami
from slgrid.app_core import large_customer_list, locate, simulate_case
from slgrid.io import frames_to_standard, standard_to_frames
from slgrid.physics_model import PhysicsConfig, PhysicsInformedLocator

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "example_prepaid"


@pytest.fixture(scope="module")
def case():
    d = simulate_case(0.25, 6, "ends", 0.2, False, 7, 7)
    d2, c, r = simulate_ami(d, 0, AMIConfig(seed=7))
    return d, d2, c, r


def test_large_customers_consistent_with_feeder(case):
    d, d2, c, r = case
    big = large_customer_buses(d)
    assert len(big) == 3 and len(c) == 36
    T = d["head"].shape[1]
    w = r["hours"] >= 0
    for j in big:
        cols = np.flatnonzero(c.bus == str(d["bus_names"][j]))
        recorded = r["p_kw"][w][:, cols].sum(axis=1) / 1000
        use = r["true_p_kw"][w][:, cols].sum(axis=1) / 1000
        assert np.allclose(use, d["true_bus_p"][0][:, j] - d["fault_bus_p"][0][:, j])
        assert np.isclose(d2["billing"][0][:, j].sum(), recorded.sum())
        assert np.allclose(d2["theft_bus_p"][0][:, j], np.maximum(use - recorded, 0))
    assert np.array_equal(d2["true_bus_p"], d["true_bus_p"])          # physics unchanged
    # tampering only in areas the feeder simulation made theft hotspots
    theft_buses = {str(d["bus_names"][j]) for j in big if d["theft_bus_p"][0][:, j].sum() > 0}
    assert set(c[c.tamper].bus) <= theft_buses
    assert r["p_kw"].shape == (28 * 24 + T, 36)


def test_model_uses_smart_meters(case):
    d, d2, c, r = case
    dd = ami_inputs(d2, c, r)
    m = PhysicsInformedLocator(dd, PhysicsConfig(ac_iterations=1))
    big = large_customer_buses(d)
    assert set(big) <= set(m.v_idx)
    b = m._hourly_billed(0)
    assert np.allclose(b[:, big], dd["ami_p"][0][:, big])
    f = m.fit_scenario(0)
    assert np.isfinite(f["stolen"]).all()


def test_ranking_finds_tampered_meters(case):
    d, d2, c, r = case
    share = pd.Series(d2["theft_bus_p"][0].sum(0) / d2["true_bus_p"][0].sum(0),
                      index=[str(b) for b in d["bus_names"]])
    rk = rank_large_customers(c, r, d["head"].shape[1], share).merge(c[["meter", "tamper"]], on="meter")
    k = int(rk.tamper.sum())
    assert k > 0 and rk.head(k).tamper.mean() >= 0.6


def test_round_trip_and_example_with_smart_meters():
    d = simulate_case(0.25, 6, "ends", 0.2, False, 3, 9, ami=True, shedding="8h")
    fr = standard_to_frames(d)
    assert {"ami_customers", "ami_readings"} <= set(fr)
    d2, _ = frames_to_standard(fr)
    assert np.nanmax(np.abs(d2["ami_p"] - d["ami_p"])) < 1e-3
    res = locate(d2)
    assert len(large_customer_list(d2, res)) == 36
    files = ["lines", "transformers", "head", "meters", "customers", "tokens",
             "ami_customers", "ami_readings"]
    d3, _ = frames_to_standard({k: pd.read_csv(EXAMPLE / f"{k}.csv") for k in files})
    assert "ami" in d3 and "vending" in d3

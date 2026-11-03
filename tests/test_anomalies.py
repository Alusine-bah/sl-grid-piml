"""Checks on theft and fault injection, metering, and the dataset generator."""
import numpy as np
import pytest

from slgrid.anomalies import ScenarioConfig, inject
from slgrid.dataset import DatasetConfig, generate
from slgrid.freetown import build_freetown_feeder
from slgrid.loads import assign_customers, generate_profiles
from slgrid.metering import MeterConfig, observe
from slgrid.simulate import run_timeseries


@pytest.fixture(scope="module")
def setup():
    net = build_freetown_feeder()
    cust = assign_customers(net)
    demand = generate_profiles(cust, n_days=1, seed=2)["p_mw"]
    return net, cust, demand


@pytest.mark.parametrize("ntl", [0.15, 0.35])
def test_theft_hits_target_and_keeps_physical_load(setup, ntl):
    _, cust, demand = setup
    sc = inject(cust, demand, ScenarioConfig(target_ntl=ntl), np.random.default_rng(0))
    stolen = sc["theft_mw"].values.sum() / demand.values.sum()
    assert stolen == pytest.approx(ntl, abs=0.02)
    # billed + stolen = physical demand (no faults here)
    assert np.allclose(sc["billed_mw"].values + sc["theft_mw"].values, demand.values)
    assert (sc["billed_mw"].values >= 0).all()
    assert sc["labels"]["theft_share"].max() <= 0.9 + 1e-9


def test_labels_match_injected_power(setup):
    _, cust, demand = setup
    sc = inject(cust, demand, ScenarioConfig(target_ntl=0.25, n_fault_buses=1),
                np.random.default_rng(1))
    lab = sc["labels"]
    has_theft = sc["theft_mw"].values.sum(axis=0) > 0
    has_fault = sc["fault_mw"].values.sum(axis=0) > 0
    assert (has_theft == lab["is_theft"].values).all()
    assert (has_fault == lab["is_fault"].values).all()
    assert not (lab["is_theft"] & lab["is_fault"]).any()


def test_fault_is_flat_once_started(setup):
    _, cust, demand = setup
    sc = inject(cust, demand, ScenarioConfig(n_theft_buses=0, n_fault_buses=1),
                np.random.default_rng(3))
    lab = sc["labels"]
    j = int(np.flatnonzero(lab["is_fault"].values)[0])
    start = int(lab["fault_start"].iloc[j])
    f = sc["fault_mw"].iloc[:, j].values
    assert (f[:start] == 0).all()
    assert np.allclose(f[start:], lab["fault_kw"].iloc[j] / 1000)


def test_utility_view(setup):
    net, cust, demand = setup
    rng = np.random.default_rng(4)
    sc = inject(cust, demand, ScenarioConfig(target_ntl=0.25), rng)
    res = run_timeseries(net, sc["p_mw"], sc["q_mvar"])
    obs = observe(res, sc["billed_mw"], cust, MeterConfig(dropout=0.3), rng)
    # Billing never includes stolen energy.
    billed = obs["billing_mwh"].values.sum()
    assert billed == pytest.approx(sc["billed_mw"].values.sum(), rel=0.03)
    # Feeder head sees everything: load + losses.
    assert obs["head"]["p_mw"].sum() > sc["p_mw"].values.sum()
    # Drop-outs at roughly the requested rate.
    assert obs["meters"]["p_mw"].isna().values.mean() == pytest.approx(0.3, abs=0.1)


def test_small_dataset(tmp_path):
    path = generate(DatasetConfig(n_scenarios=2, n_days=1), tmp_path, progress=False)
    d = np.load(path)
    assert d["true_bus_p"].shape == (2, 24, 32)
    assert d["head"].shape == (2, 24, 3)
    assert d["billing"].shape == (2, 1, 32)
    assert d["is_theft"].any(axis=1).all()

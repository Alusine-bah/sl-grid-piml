"""Checks on the Freetown-style feeder and its load profiles."""
import numpy as np
import pytest

from slgrid.freetown import build_freetown_feeder, feeder_summary, load_conductors
from slgrid.loads import assign_customers, generate_profiles, household_profile_kw
from slgrid.simulate import run_timeseries


@pytest.fixture(scope="module")
def day():
    net = build_freetown_feeder()
    cust = assign_customers(net)
    prof = generate_profiles(cust, n_days=1, seed=1)
    res = run_timeseries(net, prof["p_mw"], prof["q_mvar"])
    return net, cust, prof, res


def test_feeder_uses_real_freetown_conductors():
    net = build_freetown_feeder()
    cond = load_conductors()
    for _, ln in net.line.iterrows():
        c = cond.loc[ln["std_type"]]
        assert np.isclose(ln["r_ohm_per_km"], c["r_ohm_per_km"])
        assert np.isclose(ln["x_ohm_per_km"], c["x_ohm_per_km"])
    s = feeder_summary(net)
    assert 40 <= s["line_km"] <= 80
    assert 0.2 <= s["underground_share"] <= 0.7


def test_household_profile_matches_survey_shape():
    d = household_profile_kw()
    assert 0.25 <= d.max() <= 0.31            # S1: ~0.30 kW diversified peak
    assert 17 <= d.idxmax() <= 21             # evening peak
    morning = d.loc[6:9].max()
    midday = d.loc[12:15].min()
    assert morning > midday                   # a morning peak exists


def test_customer_counts(day):
    _, cust, _, _ = day
    assert cust["households"].sum() == pytest.approx(12_000, abs=20)
    assert cust["commercial_peak_mw"].sum() == pytest.approx(1.2)


def test_profiles_are_reproducible():
    net = build_freetown_feeder()
    cust = assign_customers(net)
    a = generate_profiles(cust, seed=7)["p_mw"]
    b = generate_profiles(cust, seed=7)["p_mw"]
    assert np.allclose(a.values, b.values)


def test_energy_balance_every_hour(day):
    """Source power = load + line losses + transformer losses, at every timestep."""
    _, _, _, res = day
    s = res["summary"]
    assert np.allclose(s.p_source_mw, s.p_load_mw + s.p_loss_line_mw + s.p_loss_trafo_mw,
                       atol=1e-6)


def test_calibrated_to_freetown(day):
    _, _, _, res = day
    s = res["summary"]
    assert 3.5 <= s.p_load_mw.max() <= 5.5              # one 11 kV main line ~5 MW (S6)
    assert 0.85 <= s.vm_min_pu.min() <= 0.94            # weak but realistic (S5)
    loss = (s.p_loss_line_mw + s.p_loss_trafo_mw).sum() / s.p_source_mw.sum()
    assert 0.015 <= loss <= 0.06                        # MV technical losses only

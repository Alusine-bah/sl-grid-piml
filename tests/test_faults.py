"""Checks on cable-fault detection."""
import numpy as np
import pytest

from slgrid.app_core import locate, simulate_case


@pytest.fixture(scope="module")
def fault_case():
    return simulate_case(0.25, 6, "ends", 0.2, True, 3, 8)     # 109 kW fault at bus 14, hour 20


def test_fault_detected_timed_and_located(fault_case):
    d = fault_case
    j = int(np.flatnonzero(d["is_fault"][0])[0])
    f = d["fault_bus_p"][0, :, j]
    res = locate(d)
    fa = res.attrs["fault"]
    assert fa["detected"]
    assert abs(fa["start_hour"] - int(np.argmax(f > 0))) <= 2
    assert abs(fa["leak_kw"] - 1000 * f.max()) / (1000 * f.max()) < 0.3
    assert str(d["bus_names"][j]) in fa["likely_section"]


def test_no_false_alarm_without_fault():
    d = simulate_case(0.25, 6, "ends", 0.2, False, 3, 8)
    res = locate(d)
    assert not res.attrs["fault"]["detected"]
    assert (res.fault_prob == 0).all()


def test_fault_aware_model_keeps_total(fault_case):
    res = locate(fault_case)
    true = fault_case["theft_bus_p"][0].sum()
    assert abs(res.stolen_physics_mwh.sum() - true) / true < 0.1

"""Checks on load shedding: schedules, simulation, uploads and the model."""
import numpy as np
import pandas as pd
import pytest

from slgrid.app_core import locate, simulate_case
from slgrid.faults import energised_hours
from slgrid.io import frames_to_standard, standard_to_frames
from slgrid.loads import apply_supply, supply_schedule


def test_schedules():
    rng = np.random.default_rng(0)
    assert supply_schedule(3, "none", rng).sum() == 72
    on8 = supply_schedule(7, "8h", rng)
    assert 168 - 7 * 8 <= on8.sum() < 168
    rot = supply_schedule(6, "rotation", np.random.default_rng(1))
    assert set(np.unique(rot)) == {0, 1} and rot.sum() <= 6 * 24 / 3 + 12


def test_cold_load_pickup():
    on = np.array([1, 1, 0, 0, 1, 1, 1], dtype=float)
    p = pd.DataFrame({"a": np.ones(7)})
    out = apply_supply(p, p, on, cold_load=0.3)["p_mw"]["a"].values
    assert np.allclose(out, [1, 1, 0, 0, 1.3, 1.15, 1])


@pytest.fixture(scope="module")
def rotation_case():
    return simulate_case(0.25, 6, "ends", 0.2, False, 3, 7, shedding="rotation")


def test_outage_hours_are_dead(rotation_case):
    d = rotation_case
    off = d["supply"][0] == 0
    assert off.any()
    assert np.allclose(d["head"][0, off, 0], 0)
    assert np.isnan(d["meter_p"][0][off]).all()
    assert (energised_hours(d["head"][0, :, 0]) == ~off).all()


def test_model_handles_rotation(rotation_case):
    res = locate(rotation_case)
    true = rotation_case["theft_bus_p"][0].sum()
    assert abs(res.stolen_physics_mwh.sum() - true) / true < 0.10
    assert abs(res.stolen_rules_mwh.sum() - true) / true < 0.25
    assert not res.attrs["fault"]["detected"]


def test_upload_with_missing_outage_hours(rotation_case):
    fr = standard_to_frames(rotation_case)
    fr["head"] = fr["head"][fr["head"].p_mw > 0]          # outage hours simply absent
    d, _ = frames_to_standard(fr)
    assert d["head"].shape[1] == 72
    assert np.allclose(d["head"][0, rotation_case["supply"][0] == 0, 0], 0)
    assert len(locate(d)) == 32

"""Checks on the physics-informed locator."""
import numpy as np
import pytest

from slgrid.baselines import capacity_audit, metered_mask
from slgrid.dataset import DatasetConfig, generate
from slgrid.physics_model import PhysicsInformedLocator


@pytest.fixture(scope="module")
def d(tmp_path_factory):
    p = generate(DatasetConfig(n_scenarios=3, n_days=2, seed=5),
                 tmp_path_factory.mktemp("ds"), progress=False)
    return dict(np.load(p))


def test_dataset_has_transformer_ratings(d):
    kva = d["transformer_kva"]
    assert kva.shape == d["is_theft"].shape
    assert (kva >= 50).all() and (kva % 50 == 0).all()


def test_linear_voltage_model_close_to_ac(d):
    """LinDistFlow drop with the true loads is within 15% of the simulated AC drop."""
    m = PhysicsInformedLocator(d)
    s = 0
    P = d["true_bus_p"][s]
    Q = (P - d["fault_bus_p"][s]) * m.tan
    lin = (m.ZR[m.m_idx] @ P.T + m.ZX[m.m_idx] @ Q.T).T.mean(axis=0)
    obs = np.nanmean(d["head"][s, :, 2][:, None] - d["meter_v"][s], axis=0)
    assert np.allclose(lin, obs, rtol=0.15)


def test_locator_outputs_and_totals(d):
    m = PhysicsInformedLocator(d)
    idx = np.arange(len(d["is_theft"]))
    score, stolen, fault = m.predict(idx)
    assert score.shape == stolen.shape == fault.shape == d["is_theft"].shape
    assert (score >= 0).all() and (score <= 0.9 + 1e-6).all()
    assert (stolen >= 0).all() and (fault >= -1e-9).all()
    true_total = d["theft_bus_p"].sum(axis=(1, 2))
    assert np.allclose(stolen.sum(axis=1), true_total, rtol=0.15)


def test_physics_fixes_total_of_ratings_audit(d):
    """Physics should get the feeder total closer than the ratings-only audit."""
    idx = np.arange(len(d["is_theft"]))
    true_total = d["theft_bus_p"].sum(axis=(1, 2))
    _, st_cap = capacity_audit(d, idx)
    _, st_phys, _ = PhysicsInformedLocator(d).predict(idx)
    err_cap = np.abs(st_cap.sum(axis=1) - true_total).mean()
    err_phys = np.abs(st_phys.sum(axis=1) - true_total).mean()
    assert err_phys < err_cap


def test_metered_buses_found(d):
    score, _, _ = PhysicsInformedLocator(d).predict(np.arange(len(d["is_theft"])))
    mm = metered_mask(d)
    lab = d["is_theft"][:, mm].astype(bool)
    assert (score[:, mm][lab] > 0.1).all()
    assert (score[:, mm][~lab] < 0.1).all()

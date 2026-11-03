"""Checks on topology helpers, evaluation and baselines."""
import numpy as np
import pytest

from slgrid.baselines import PureML, energy_audit, metered_mask
from slgrid.dataset import DatasetConfig, generate
from slgrid.evaluation import evaluate, split_scenarios
from slgrid.topology import bus_features, path_matrix


@pytest.fixture(scope="module")
def d(tmp_path_factory):
    p = generate(DatasetConfig(n_scenarios=6, n_days=2), tmp_path_factory.mktemp("ds"),
                 progress=False)
    return dict(np.load(p))


def test_topology(d):
    t = bus_features(d["line_from"], d["line_to"], d["line_r_ohm"], d["line_x_ohm"])
    assert t.loc["bus_0", "n_downstream"] == 32
    assert t.loc["bus_17", "depth"] == 17
    A = path_matrix(d["line_from"], d["line_to"], d["bus_names"])
    assert A.shape == (32, 32)
    assert A[list(d["bus_names"]).index("bus_17")].sum() == 17


def test_customers_vary_between_scenarios(d):
    hh = d["households"]
    assert not np.allclose(hh[0], hh[1])
    assert np.allclose(hh.sum(axis=1), 12_000, atol=50)


def test_energy_audit_balances(d):
    idx = np.arange(len(d["is_theft"]))
    score, stolen = energy_audit(d, idx)
    assert score.shape == stolen.shape == d["is_theft"].shape
    assert (stolen >= 0).all() and (score >= 0).all() and (score <= 1).all()
    true_total = d["theft_bus_p"].sum(axis=(1, 2))
    assert np.allclose(stolen.sum(axis=1), true_total, rtol=0.25)


def test_pure_ml_and_evaluation(d):
    train, test = split_scenarios(len(d["is_theft"]), test_share=0.34)
    m = PureML().fit(d, train)
    s, st = m.predict(d, test)
    res = evaluate("ml", s, st, d, test, 0.5, metered_mask(d))
    assert set(res["buses"]) == {"all buses", "metered", "unmetered"}
    assert res["AUC"].between(0, 1).all()

"""Checks on placement and robustness helpers."""
import numpy as np
import pytest

from slgrid.dataset import DatasetConfig, generate
from slgrid.studies import perturb_assets, placement, remeter


@pytest.fixture(scope="module")
def d(tmp_path_factory):
    p = generate(DatasetConfig(n_scenarios=2, n_days=1), tmp_path_factory.mktemp("ds"),
                 progress=False)
    return dict(np.load(p))


@pytest.mark.parametrize("strategy", ["ends", "largest", "random"])
def test_placement_sizes(d, strategy):
    for n in (0, 3, 12):
        b = placement(d, n, strategy)
        assert len(b) == n and len(set(b)) == n


def test_ends_starts_with_far_ends(d):
    b = placement(d, 4, "ends")
    assert "bus_17" in b and "bus_32" in b


def test_remeter_shapes_and_dropout(d):
    dd = remeter(d, ["bus_3", "bus_17"], dropout=0.5)
    assert dd["meter_p"].shape[-1] == 2
    assert np.isnan(dd["meter_p"]).mean() == pytest.approx(0.5, abs=0.1)
    assert d["meter_p"].shape[-1] == 6          # original untouched


def test_perturb_assets_changes_only_model_inputs(d):
    dd = perturb_assets(d, cable_error=0.2, kva_error=0.3)
    assert not np.allclose(dd["line_r_ohm"], d["line_r_ohm"])
    assert not np.allclose(dd["transformer_kva"], d["transformer_kva"])
    assert np.array_equal(dd["true_bus_p"], d["true_bus_p"])

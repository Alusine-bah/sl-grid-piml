"""Physics checks on the base feeder."""
import numpy as np
import pytest

from slgrid.feeder import build_base_feeder, to_standard, run_powerflow, is_radial


@pytest.fixture(scope="module")
def solved():
    net = build_base_feeder()
    res = run_powerflow(net)
    return net, to_standard(net), res


def test_feeder_is_11kv_and_radial(solved):
    _, std, _ = solved
    assert (std["buses"]["vn_kv"] == 11.0).all()
    assert len(std["buses"]) == 33
    assert is_radial(std)


def test_voltages_in_plausible_range(solved):
    _, _, res = solved
    vm = res["buses"]["vm_pu"]
    assert vm.max() <= 1.0 + 1e-9
    assert vm.min() > 0.85


def test_power_balance_kcl(solved):
    """Power from the source = total load + total line losses (energy conservation)."""
    net, std, res = solved
    p_source = net.res_ext_grid["p_mw"].sum()
    p_load = std["buses"]["p_load_mw"].sum()
    p_loss = res["lines"]["loss_mw"].sum()
    assert np.isclose(p_source, p_load + p_loss, atol=1e-6)

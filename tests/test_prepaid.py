import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from slgrid.app_core import locate, meter_list, simulate_case
from slgrid.io import FeederDataError, frames_to_standard, standard_to_frames
from slgrid.prepaid import VendingConfig, rank_meters, simulate_vending, tokens_to_billing

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "example_prepaid"


@pytest.fixture(scope="module")
def case():
    d = simulate_case(0.25, 6, "ends", 0.2, False, 7, 7)
    c, t = simulate_vending(d, 0, VendingConfig(seed=1))
    return d, c, t


def test_vending_matches_feeder(case):
    d, c, t = case
    names = [str(b) for b in d["bus_names"]]
    legit = (d["true_bus_p"][0] - d["theft_bus_p"][0] - d["fault_bus_p"][0]).sum(axis=0)
    sold = tokens_to_billing(c, t, names, 7)[0]
    assert sold.shape == (32,)
    assert abs(sold.sum() / legit.sum() - 1) < 0.05            # feeder total within 5%
    assert np.median(np.abs(sold / legit - 1)) < 0.10           # typical area within 10%
    # bypassing customers only where there is theft, and about half the stolen energy
    theft = d["theft_bus_p"][0].sum(axis=0) * 1000 / 7          # kWh/day per bus
    by_bus = c.groupby("true_bus").stolen_kwh_day.sum().reindex(names).fillna(0).values
    assert (by_bus[theft == 0] == 0).all()
    assert abs(by_bus.sum() / theft.sum() - 0.5) < 0.05
    assert c.meter.is_unique and set(t.meter) <= set(c.meter)
    assert t.day.min() < -60 and t.day.max() <= 6


def test_mapping_error_moves_meters(case):
    d, _, _ = case
    c, _ = simulate_vending(d, 0, VendingConfig(seed=1, mapping_error=0.2))
    wrong = (c.bus != c.true_bus).mean()
    assert 0.15 < wrong < 0.25


def test_meter_ranking_finds_bypassers(case):
    d, c, t = case
    share = pd.Series(d["theft_bus_p"][0].sum(0) / d["true_bus_p"][0].sum(0),
                      index=[str(b) for b in d["bus_names"]])
    r = rank_meters(c, t, 7, share).merge(c[["meter", "is_bypass"]], on="meter")
    assert r.head(100).is_bypass.mean() > 2 * r.is_bypass.mean()


def test_app_prepaid_case_and_round_trip():
    d = simulate_case(0.25, 6, "ends", 0.2, False, 7, 7, billing="prepaid", mapping_error=0.1)
    assert d["billing"].shape == (1, 1, 32)
    res = locate(d)
    m = meter_list(d, res)
    assert len(m) == len(d["vending"]["customers"]) and "is_bypass" in m
    assert m.head(100).is_bypass.mean() > m.is_bypass.mean()
    fr = standard_to_frames(d)
    assert "billing" not in fr and {"customers", "tokens"} <= set(fr)
    d2, names = frames_to_standard(fr)
    assert np.allclose(d2["billing"], d["billing"])
    assert len(meter_list(d2, locate(d2))) == len(m)


def test_example_prepaid_files():
    fr = {k: pd.read_csv(EXAMPLE / f"{k}.csv")
          for k in ["lines", "transformers", "head", "meters", "customers", "tokens"]}
    d, names = frames_to_standard(fr)
    assert d["billing"].shape == (1, 1, 32) and "vending" in d


def test_token_dates_and_errors():
    fr = {k: pd.read_csv(EXAMPLE / f"{k}.csv")
          for k in ["lines", "transformers", "head", "meters", "customers", "tokens"]}
    d0, _ = frames_to_standard({k: v.copy() for k, v in fr.items()})
    start = pd.Timestamp("2026-03-02")
    dated = {k: v.copy() for k, v in fr.items()}
    dated["head"]["hour"] = (start + pd.to_timedelta(dated["head"]["hour"], "h")).astype(str)
    dated["meters"]["hour"] = (start + pd.to_timedelta(dated["meters"]["hour"], "h")).astype(str)
    dated["tokens"]["day"] = (start + pd.to_timedelta(dated["tokens"]["day"], "D")).dt.date.astype(str)
    d1, _ = frames_to_standard(dated)
    assert np.allclose(d1["billing"], d0["billing"])
    bad = {k: v.copy() for k, v in fr.items()}
    bad["tokens"]["day"] = (start + pd.to_timedelta(bad["tokens"]["day"], "D")).dt.date.astype(str)
    with pytest.raises(FeederDataError, match="dates"):
        frames_to_standard(bad)
    none = {k: v.copy() for k, v in fr.items() if k != "tokens"}
    with pytest.raises(FeederDataError, match="billing.csv"):
        frames_to_standard(none)

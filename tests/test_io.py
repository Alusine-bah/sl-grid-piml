"""Checks on reading a utility's own feeder files."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slgrid.app_core import locate, tree_layout
from slgrid.io import FeederDataError, frames_to_standard, standard_to_frames

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "example_feeder"
FILES = ["lines", "transformers", "head", "meters", "billing"]


def example():
    return {k: pd.read_csv(EXAMPLE / f"{k}.csv") for k in FILES}


def test_example_files_load_and_run():
    d, names = frames_to_standard(example())
    assert d["head"].shape == (1, 168, 3)
    assert d["billing"].shape == (1, 7, 32)
    assert len(d["meter_names"]) == 6
    assert names["bus_0"] == "0"
    res = locate(d)
    assert len(res) == 32 and (res.stolen_physics_mwh >= 0).all()
    nodes, edges = tree_layout(d)
    assert len(nodes) == 33


def test_round_trip_is_lossless():
    d, _ = frames_to_standard(example())
    d2, _ = frames_to_standard(standard_to_frames(d))
    for k in ["head", "billing", "transformer_kva", "line_r_ohm"]:
        assert np.allclose(d[k], d2[k])


def test_custom_bus_names_and_dates():
    fr = example()
    rename = {str(i): f"TX-{i:03d}" for i in range(1, 33)}
    rename["0"] = "KINGTOM-11kV"
    for k, cols in [("lines", ["from_bus", "to_bus"]), ("transformers", ["bus"]),
                    ("meters", ["bus"]), ("billing", ["bus"])]:
        for c in cols:
            fr[k][c] = fr[k][c].astype(str).map(rename)
    fr["head"]["hour"] = pd.date_range("2026-01-05", periods=168, freq="h").astype(str)
    fr["meters"]["hour"] = (pd.Timestamp("2026-01-05") + pd.to_timedelta(fr["meters"]["hour"], "h")).astype(str)
    fr["billing"]["day"] = (pd.Timestamp("2026-01-05") + pd.to_timedelta(fr["billing"]["day"], "D")).dt.date.astype(str)
    d, names = frames_to_standard(fr)
    assert names["bus_0"] == "KINGTOM-11kV"
    assert np.isfinite(d["meter_p"]).sum() == len(fr["meters"])


def test_no_meters_allowed():
    fr = example()
    fr["meters"] = fr["meters"].iloc[0:0]
    d, _ = frames_to_standard(fr)
    assert d["meter_p"].shape == (1, 168, 0)
    assert len(locate(d)) == 32


@pytest.mark.parametrize("breakage, message", [
    (lambda fr: fr["lines"].drop(columns="r_ohm", inplace=True), "missing column"),
    (lambda fr: fr.update(head=fr["head"].iloc[:100]), "whole days"),
    (lambda fr: fr.update(lines=pd.concat([fr["lines"], fr["lines"].iloc[[3]]])), "radial"),
    (lambda fr: fr.update(transformers=fr["transformers"].iloc[:-1]), "no kVA"),
    (lambda fr: fr["billing"].loc[fr["billing"].bus == 5, "mwh"].update(0) or
     fr.update(billing=fr["billing"].assign(mwh=np.where(fr["billing"].bus == 5, 0, fr["billing"].mwh))),
     "no billed energy"),
])
def test_clear_error_messages(breakage, message):
    fr = example()
    breakage(fr)
    with pytest.raises(FeederDataError, match=message):
        frames_to_standard(fr)


def test_period_billing_without_day_column():
    fr = example()
    fr["billing"] = fr["billing"].groupby("bus", as_index=False)["mwh"].sum()
    d, _ = frames_to_standard(fr)
    assert d["billing"].shape == (1, 1, 32)
    res = locate(d)
    assert len(res) == 32 and res.stolen_physics_mwh.sum() > 0

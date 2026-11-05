import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from slgrid.app_core import (district_params, district_plan, district_table, locate,
                             simulate_district)
from slgrid.io import FeederDataError, frames_to_standard, split_district_zip, standard_to_frames


@pytest.fixture(scope="module")
def district():
    return {k: (d, locate(d)) for k, d in simulate_district(3, 2, 4).items()}


def test_params_vary_and_repeat():
    a, b = district_params(8, 1), district_params(8, 1)
    assert a == b
    assert len({p["theft"] for p in a}) > 1 and len({p["n_meters"] for p in a}) > 1


def test_ranking_and_crew_plan(district):
    t = district_table(district)
    assert list(t["rank"]) == [1, 2, 3]
    assert (np.diff(t.stolen_month) <= 0).all()
    assert {"true_stolen_month", "true_rank"} <= set(t.columns)
    plan = district_plan(district, 10)
    assert len(plan) == 10 and (np.diff(plan.stolen_month) <= 0).all()
    every = pd.concat([r.assign(f=k) for k, (_, r) in district.items()])
    assert plan.stolen_month.iloc[0] == pytest.approx(every.stolen_physics_mwh.max() * 15)
    assert plan.attrs["total_stolen_month"] == pytest.approx(t.stolen_month.sum())


def test_district_zip_round_trip(district):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, (d, _) in district.items():
            for k, df in standard_to_frames(d).items():
                z.writestr(f"my district/{name}/{k}.csv", df.to_csv(index=False))
        z.writestr("__MACOSX/my district/Feeder 1/._head.csv", "junk")
    parts = split_district_zip(buf.getvalue())
    assert list(parts) == sorted(district)
    name = list(district)[0]
    d2, _ = frames_to_standard({k: pd.read_csv(io.BytesIO(v)) for k, v in parts[name].items()})
    assert np.allclose(d2["billing"], district[name][0]["billing"], rtol=1e-3, atol=1e-3)


def test_district_zip_errors():
    with pytest.raises(FeederDataError, match="zip"):
        split_district_zip(b"not a zip")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "hello")
    with pytest.raises(FeederDataError, match="No CSV"):
        split_district_zip(buf.getvalue())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("head.csv", "hour,p_mw\n0,1\n")
    assert list(split_district_zip(buf.getvalue())) == ["Feeder"]

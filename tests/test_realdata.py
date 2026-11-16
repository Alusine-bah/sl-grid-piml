import numpy as np

from slgrid.anomalies import ScenarioConfig, _pick_hotspots
from slgrid.realdata import feeder_loss_patterns, odisha_dt_losses


def test_odisha_data_is_consistent():
    d = odisha_dt_losses()
    assert len(d) == 1867 and d.feeder_id.nunique() == 468
    calc = 100 * (1 - d.billed_mu / d.input_mu)
    assert ((calc - d.loss_pct).abs() > 0.05).sum() <= 1
    assert 30 < d.loss_pct.median() < 37
    assert len(feeder_loss_patterns()) >= 100


def test_odisha_pattern_spreads_theft_and_hits_target():
    rng = np.random.default_rng(0)
    load = rng.uniform(0.05, 0.3, 32)
    cfg = ScenarioConfig(target_ntl=0.25, pattern="odisha")
    zero, total = [], []
    for _ in range(50):
        idx, sh = _pick_hotspots(load, list(range(32)), cfg, rng)
        assert len(idx) == 32 and (sh <= cfg.max_share + 1e-12).all()
        total.append((sh * load).sum() / load.sum())
        zero.append((sh < 0.05).mean())
    assert np.allclose(total, 0.25, atol=1e-6)
    assert np.mean(zero) < 0.35          # most areas lose something, unlike hotspots

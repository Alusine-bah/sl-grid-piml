"""Real utility data used to keep the simulation honest.

TP Central Odisha (India) publishes energy accounts for each metered distribution
transformer (data/real/odisha_dt_losses_q2fy25.csv, 1,867 transformers on 468 feeders).
They show how losses are spread across the transformer areas of a real feeder: in most
areas some energy is lost, and half the lost energy sits in about 3 in 10 areas.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "real" / "odisha_dt_losses_q2fy25.csv"


@lru_cache(maxsize=1)
def odisha_dt_losses() -> pd.DataFrame:
    """The Odisha transformer accounts, with loss as a fraction (0-1)."""
    d = pd.read_csv(DATA)
    d["loss"] = (1 - d["billed_mu"] / d["input_mu"]).clip(lower=0.0)
    return d


@lru_cache(maxsize=4)
def feeder_loss_patterns(min_transformers: int = 6) -> tuple[np.ndarray, ...]:
    """For every Odisha feeder with at least `min_transformers` metered transformers, the
    loss fraction of each of its transformer areas."""
    d = odisha_dt_losses()
    return tuple(g["loss"].values for _, g in d.groupby("feeder_id")
                 if len(g) >= min_transformers)

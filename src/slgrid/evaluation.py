"""Common evaluation for all methods.

Task: for every scenario and every load bus, output
  score      how likely the bus is a theft hotspot (higher = more likely)
  stolen     estimated stolen energy over the scenario, MWh

Metrics (computed on test scenarios only):
  AUC        ranking quality of the scores (0.5 = random, 1.0 = perfect)
  AP         average precision; rewards ranking true hotspots at the top
  F1         at the threshold that is best on the training scenarios
  top-k hit  share of the k highest-scored buses that are real hotspots, with k = the true
             number of hotspots in that scenario (what an inspection crew would visit)
  MAE        error of stolen energy per bus, MWh over the scenario
  feeder err error of total stolen energy for the whole feeder, %

Results are also split into buses that have a transformer meter and buses that do not,
since finding theft behind an unmetered transformer is the hard, useful case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def split_scenarios(n: int, test_share: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_share)))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    cands = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 49)))
    f1s = [f1_score(labels, scores >= c, zero_division=0) for c in cands]
    return float(cands[int(np.argmax(f1s))])


def _metrics(score, label, stolen_hat, stolen_true, thr):
    out = {"AUC": np.nan, "AP": np.nan}
    if 0 < label.sum() < len(label):
        out["AUC"] = roc_auc_score(label, score)
        out["AP"] = average_precision_score(label, score)
    out["F1"] = f1_score(label, score >= thr, zero_division=0)
    out["MAE_MWh"] = float(np.mean(np.abs(stolen_hat - stolen_true)))
    return out


def evaluate(method: str, scores: np.ndarray, stolen_hat: np.ndarray, d, test_idx,
             threshold: float, metered_mask: np.ndarray) -> pd.DataFrame:
    """scores, stolen_hat: [S_test, B]. Returns a one-row-per-group table."""
    labels = d["is_theft"][test_idx].astype(bool)
    stolen_true = d["theft_bus_p"][test_idx].sum(axis=1)   # MW*h = MWh over the scenario

    rows = []
    for group, mask in [("all buses", np.ones_like(metered_mask, dtype=bool)),
                        ("metered", metered_mask), ("unmetered", ~metered_mask)]:
        m = _metrics(scores[:, mask].ravel(), labels[:, mask].ravel(),
                     stolen_hat[:, mask].ravel(), stolen_true[:, mask].ravel(), threshold)
        m.update(method=method, buses=group)
        rows.append(m)

    hits = []
    for s in range(len(test_idx)):
        k = int(labels[s].sum())
        if k:
            top = np.argsort(-scores[s])[:k]
            hits.append(labels[s, top].mean())
    feeder_err = np.abs(stolen_hat.sum(axis=1) - stolen_true.sum(axis=1)) / stolen_true.sum(axis=1)
    rows[0]["top-k hit"] = float(np.mean(hits))
    rows[0]["feeder err %"] = float(100 * feeder_err.mean())
    cols = ["method", "buses", "AUC", "AP", "F1", "top-k hit", "MAE_MWh", "feeder err %"]
    return pd.DataFrame(rows).reindex(columns=cols)

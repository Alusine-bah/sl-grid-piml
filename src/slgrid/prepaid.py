"""Prepaid vending records: customers, token purchases and suspicious meters.

EDSA sells most electricity as prepaid tokens. What its vending system holds is a list of
purchases (meter number, date, kWh) and a customer database that says which transformer
each meter is connected to. This module

  * simulates such records on top of a simulated feeder period (`simulate_vending`),
  * turns them into energy sold per transformer area for the locator (`tokens_to_billing`),
  * ranks individual meters for inspection (`rank_meters`).

How the simulated customers behave
----------------------------------
Each transformer area has its households (from the feeder simulation) as prepaid accounts;
commercial areas get a fixed number of business accounts. Customer sizes vary widely
(log-normal). A few accounts are vacant and buy nothing. Customers buy tokens every few days
to every two weeks, each purchase covering roughly the energy they will use until the next.

Stolen energy at a theft hotspot comes from two kinds of theft:
  * illegal connections with no account at all (`illegal_share` of the stolen energy);
    vending records cannot show these, only the feeder-level physics can;
  * account holders who bypass their meter for part or all of their use. Their purchases
    fall below what they really use. Some started before the records begin, others during
    them, so their buying drops part-way.

The customer database can link some meters to the wrong transformer (`mapping_error`):
mostly a neighbouring one, sometimes one anywhere on the feeder.

Days are counted as in the feeder data: day 0 is the first day of the substation data, so
the analysis window is days 0 .. n_days-1 and earlier vending history has negative days.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VendingConfig:
    history_days: int = 90          # days of vending records, ending with the analysis window
    mapping_error: float = 0.0      # share of meters linked to the wrong transformer
    illegal_share: float = 0.5      # share of stolen energy taken by connections with no account
    vacant_share: float = 0.03      # accounts that are vacant and buy nothing
    size_sd: float = 0.7            # spread of customer consumption (log-normal sigma)
    bypass_fraction: tuple = (0.5, 1.0)   # share of own use a bypassing customer does not pay for
    started_before: float = 0.5     # share of bypassers already bypassing when records begin
    interval_days: tuple = (2.0, 14.0)    # mean days between token purchases (range over customers)
    commercial_accounts: int = 20   # business accounts at a commercial transformer area
    seed: int = 0


def _neighbours(d: dict) -> dict[str, list[str]]:
    buses = set(str(b) for b in d["bus_names"])
    nb: dict[str, list[str]] = {b: [] for b in buses}
    for f, t in zip(d["line_from"], d["line_to"]):
        f, t = str(f), str(t)
        if f in buses and t in buses:
            nb[f].append(t)
            nb[t].append(f)
    return nb


def simulate_vending(d: dict, s: int = 0, cfg: VendingConfig | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the customer list and token purchases for scenario `s` of a dataset.

    Returns (customers, tokens):
      customers  meter, bus (as listed in the customer database), kind, and truth columns
                 true_bus, use_kwh_day, stolen_kwh_day, is_bypass, is_vacant, bypass_start
      tokens     meter, day, kwh
    """
    cfg = cfg or VendingConfig()
    rng = np.random.default_rng(cfg.seed)
    buses = [str(b) for b in d["bus_names"]]
    T = d["true_bus_p"].shape[1]
    n_days = T // 24
    H = max(cfg.history_days, n_days)
    day0 = n_days - H                                     # first day of the records

    fault = d["fault_bus_p"][s] if "fault_bus_p" in d else 0.0
    use = (d["true_bus_p"][s] - fault).sum(axis=0) / n_days * 1000      # kWh/day per bus
    theft = d["theft_bus_p"][s].sum(axis=0) / n_days * 1000
    hh = d["households"][s]

    cust_parts, tok_parts = [], []
    serial = 0
    for j, b in enumerate(buses):
        n = int(hh[j]) if hh[j] > 0 else cfg.commercial_accounts
        kind = "residential" if hh[j] > 0 else "commercial"
        w = rng.lognormal(0.0, cfg.size_sd, n)
        vacant = rng.random(n) < cfg.vacant_share
        w[vacant] = 0.0
        w /= w.sum()
        # energy used by account holders (illegal connections are not account holders)
        illegal = cfg.illegal_share * theft[j]
        acc_use = (use[j] - illegal) * w
        to_bypass = theft[j] - illegal
        stolen = np.zeros(n)
        if to_bypass > 0:
            for i in rng.permutation(np.flatnonzero(~vacant)):
                take = min(rng.uniform(*cfg.bypass_fraction) * acc_use[i], to_bypass)
                stolen[i] = take
                to_bypass -= take
                if to_bypass <= 1e-9:
                    break
        bypass = stolen > 0
        start = np.full(n, day0, dtype=float)
        later = bypass & (rng.random(n) >= cfg.started_before)
        start[later] = rng.uniform(day0, 0, later.sum())        # started before the window

        meters = np.array([f"M{serial + i:06d}" for i in range(n)])
        serial += n
        cust_parts.append(pd.DataFrame({
            "meter": meters, "true_bus": b, "kind": kind, "use_kwh_day": acc_use,
            "stolen_kwh_day": stolen, "is_bypass": bypass, "is_vacant": vacant,
            "bypass_start": np.where(bypass, start, np.nan)}))

        # token purchases: renewal process per customer
        m = rng.uniform(*cfg.interval_days, n)
        K = int(np.ceil(H / (0.15 * cfg.interval_days[0]))) + 4
        gaps = rng.gamma(4.0, m[:, None] / 4.0, size=(n, K))
        t = day0 + rng.uniform(0, 1, n)[:, None] * m[:, None] + np.cumsum(gaps, axis=1) - gaps[:, :1]
        nxt = np.concatenate([gaps[:, 1:], gaps[:, -1:]], axis=1)          # time until next buy
        rate = np.where(t < start[:, None], acc_use[:, None], (acc_use - stolen)[:, None])
        kwh = rate * nxt * rng.lognormal(0.0, 0.1, size=(n, K))
        ok = (t < n_days) & (kwh > 0.05)
        ii, kk = np.nonzero(ok)
        tok_parts.append(pd.DataFrame({"meter": meters[ii],
                                       "day": np.floor(t[ii, kk]).astype(int),
                                       "kwh": np.round(kwh[ii, kk], 1)}))

    customers = pd.concat(cust_parts, ignore_index=True)
    tokens = pd.concat(tok_parts, ignore_index=True).sort_values(["day", "meter"], ignore_index=True)

    # customer database: which transformer each meter is listed under
    listed = customers["true_bus"].to_numpy().copy()
    if cfg.mapping_error > 0:
        nb = _neighbours(d)
        wrong = np.flatnonzero(rng.random(len(listed)) < cfg.mapping_error)
        for i in wrong:
            near = nb[listed[i]]
            if near and rng.random() < 0.6:
                listed[i] = near[rng.integers(len(near))]
            else:
                others = [b for b in buses if b != listed[i]]
                listed[i] = others[rng.integers(len(others))]
    customers.insert(1, "bus", listed)
    return customers, tokens


def _window(tokens: pd.DataFrame, n_days: int, lookback: int | None) -> tuple[pd.DataFrame, float]:
    """Tokens used to estimate sales for the analysis window, and the scale factor."""
    if lookback is None:                         # purchases inside the window only
        return tokens[(tokens.day >= 0) & (tokens.day < n_days)], 1.0
    first = int(tokens.day.min()) if len(tokens) else 0
    L = max(min(lookback, n_days - first), n_days)
    sel = tokens[(tokens.day >= n_days - L) & (tokens.day < n_days)]
    return sel, n_days / L


def tokens_to_billing(customers: pd.DataFrame, tokens: pd.DataFrame, bus_names, n_days: int,
                      lookback: int | None = 30) -> np.ndarray:
    """Energy sold per transformer area for the analysis window, in MWh, shape [1, B].

    lookback=None: sum of purchases made inside the window (simple, but a week of token
    sales can be far from a week of use). lookback=L: average daily purchases over the last
    L days (ending with the window) times the window length, which evens out when people
    happen to buy.
    """
    sel, scale = _window(tokens, n_days, lookback)
    bus_of = customers.set_index("meter")["bus"]
    per_bus = (sel.assign(bus=sel.meter.map(bus_of)).dropna(subset=["bus"])
               .groupby("bus")["kwh"].sum())
    out = per_bus.reindex([str(b) for b in bus_names]).fillna(0.0).to_numpy() * scale / 1000
    return out[None, :]


def rank_meters(customers: pd.DataFrame, tokens: pd.DataFrame, n_days: int,
                bus_share: dict | pd.Series | None = None, lookback: int = 30) -> pd.DataFrame:
    """Rank prepaid meters for inspection.

    For each meter:
      recent_kwh_day   average purchases over the last `lookback` days
      earlier_kwh_day  average purchases before that (NaN if there is too little history)
      peer_kwh_day     median recent purchases of meters of the same kind at the same transformer
      days_since_buy   days since the last purchase (or since records began)
      missing_kwh_day  how far recent buying is below the higher of what this meter used to
                       buy and what its peers buy
      priority         missing_kwh_day weighted by the estimated stolen share of the meter's
                       transformer area (from the feeder-level locator), so that low buyers in
                       areas where the physics says energy is going missing come first.
    """
    end = n_days
    first = int(tokens.day.min()) if len(tokens) else 0
    L = max(min(lookback, end - first), 1)
    early_len = (end - L) - first
    t = tokens.assign(recent=tokens.day >= end - L)
    rec = t[t.recent].groupby("meter")["kwh"].sum() / L
    out = customers[["meter", "bus", "kind"]].copy()
    out["recent_kwh_day"] = out.meter.map(rec).fillna(0.0)
    if early_len >= 14:
        ear = t[~t.recent].groupby("meter")["kwh"].sum() / early_len
        out["earlier_kwh_day"] = out.meter.map(ear).fillna(0.0)
    else:
        out["earlier_kwh_day"] = np.nan
    last = tokens.groupby("meter")["day"].max()
    out["days_since_buy"] = (end - 1 - out.meter.map(last)).fillna(end - first).astype(int)
    out["peer_kwh_day"] = out.groupby(["bus", "kind"])["recent_kwh_day"].transform("median")
    # Expected buying: the higher of what this meter used to buy and what similar meters on
    # the same transformer buy now. The first catches customers whose buying dropped; the
    # second catches those who were already bypassing when the records begin.
    ref = np.maximum(out["earlier_kwh_day"].fillna(0.0), out["peer_kwh_day"])
    out["missing_kwh_day"] = (ref - out["recent_kwh_day"]).clip(lower=0.0)
    w = pd.Series(1.0, index=out.index)
    if bus_share is not None:
        w = out.bus.map(pd.Series(bus_share)).fillna(0.0)
    out["area_stolen_share"] = w
    out["priority"] = out["missing_kwh_day"] * w
    return out.sort_values("priority", ascending=False, ignore_index=True)

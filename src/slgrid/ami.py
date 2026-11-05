"""Large customers on smart meters (AMI): hourly readings and tamper detection.

Large commercial and industrial customers are the first to get smart meters, which record
hourly energy, reactive energy and voltage and send them to the utility (AMI). This module

  1. simulates the large customers at the commercial transformer areas of a simulated
     feeder, with 28 days of hourly smart-meter history before the analysis period, and
     meter tampering by some of them (`simulate_ami`);
  2. turns their readings into inputs for the feeder model (`ami_inputs`): exact hourly
     sales at those transformer areas, and voltage measurements at those points of the
     feeder;
  3. ranks the large customers for inspection (`rank_large_customers`).

How tampering works in the simulation. A tampering customer's meter records only part of
the real use from a start hour onwards: either one of three phases bypassed (the meter sees
about two thirds of the load), or a shunted current transformer (it sees 10-50%). Some
started during the smart-meter history, so their readings drop; others were already
tampering when the history begins, so only a comparison with their contracted capacity and
with the physics of the feeder can point to them.

Times are hours counted from the start of the analysis period (hour 0); the history has
negative hours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

AMI_FILES = {"ami_customers": ["meter", "bus", "contract_kva"],
             "ami_readings": ["meter", "hour", "p_kw", "q_kvar", "v_pu"]}

# hourly shapes (per unit of the customer's peak) of typical large customers
_H = np.arange(24)
PROFILES = {
    "office/shop": np.where((_H >= 8) & (_H < 18), 1.0, np.where((_H == 7) | (_H == 18), 0.6, 0.25)),
    "hotel/restaurant": np.clip(0.45 + 0.55 * np.exp(-0.5 * ((_H - 20) / 3.0) ** 2)
                                + 0.25 * np.exp(-0.5 * ((_H - 8) / 1.5) ** 2), 0, 1),
    "telecom/water pumping": np.full(24, 0.85) + 0.1 * np.sin(2 * np.pi * (_H - 14) / 24),
    "workshop/factory": np.where((_H >= 7) & (_H < 17), 1.0, 0.12),
}
PROFILE_P = np.array([0.45, 0.2, 0.15, 0.2])


@dataclass
class AMIConfig:
    customers_per_bus: int = 12      # large customers at each commercial transformer area
    history_days: int = 28           # smart-meter history before the analysis period
    max_tamper: float = 0.6          # at most this share of large customers in an area tamper
    onset_in_history: float = 0.6    # share of tamperers who start during the history
    p_noise: float = 0.01            # relative noise of hourly use (behaviour, not the meter)
    day_sd: float = 0.15             # day-to-day variation of a large customer's energy
    honest_change: float = 0.15      # honest customers whose use changed for good during history
    v_noise: float = 0.002           # smart-meter voltage error, pu
    seed: int = 0


def large_customer_buses(d: dict, s: int = 0) -> list[int]:
    """Indices of transformer areas with large customers: those with no households."""
    return [j for j, h in enumerate(d["households"][s]) if h == 0]


def simulate_ami(d: dict, s: int = 0, cfg: AMIConfig | None = None
                 ) -> tuple[dict, pd.DataFrame, dict]:
    """Simulate large customers on smart meters for scenario `s`.

    Returns (d_new, customers, readings):
      d_new      copy of d where, at large-customer areas, stolen energy and billing are
                 what the tampered meters miss and record (the physical load is unchanged)
      customers  meter, bus, type, contract_kva and truth: tamper, tamper_kind, recorded
                 (share of use the meter records once tampering starts), onset_hour
      readings   arrays over hours -H*24 .. T-1:  p_kw, q_kvar [hours, N] as recorded,
                 true_p_kw, and v_pu [T, N] (analysis period only), plus "hours"
    """
    cfg = cfg or AMIConfig()
    rng = np.random.default_rng(cfg.seed)
    T = d["true_bus_p"].shape[1]
    D = T // 24
    Hh = cfg.history_days * 24
    hours = np.arange(-Hh, T)
    buses = [str(b) for b in d["bus_names"]]
    all_names = [str(b) for b in d["all_bus_names"]]
    on = d["supply"][s].astype(bool) if "supply" in d else np.ones(T, bool)

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    rows, P, Ptrue, Q, V = [], [], [], [], []
    serial = 0
    for j in large_customer_buses(d, s):
        bus = buses[j]
        fault = d["fault_bus_p"][s][:, j] if "fault_bus_p" in d else 0.0
        L = (d["true_bus_p"][s][:, j] - fault) * 1000                 # kW, all use incl. theft
        n = cfg.customers_per_bus
        kinds = rng.choice(list(PROFILES), n, p=PROFILE_P)
        w = rng.lognormal(0.0, 0.6, n)
        prof = np.array([PROFILES[k] for k in kinds])                # [n, 24]
        raw = w[:, None] * prof[:, np.tile(np.arange(24), D)]          # [n, T]
        raw *= rng.normal(1.0, cfg.p_noise * 5, raw.shape).clip(0.7, 1.3)
        share = raw / raw.sum(axis=0, keepdims=True)
        use = share * L[None, :]                                      # [n, T] kW
        # history: each customer's average day, with day-to-day variation, quieter Sundays
        # for businesses, and genuine lasting changes for some honest customers (a business
        # downsizing or adding equipment), which a tamper detector must not mistake for theft
        day_mean = use.reshape(n, D, 24).mean(axis=1)                 # [n, 24]
        hd = np.arange(-cfg.history_days, 0)
        dayf = rng.normal(1.0, cfg.day_sd, (n, cfg.history_days)).clip(0.4, 1.6)
        weekly = np.isin(kinds, ["office/shop", "workshop/factory"])
        dayf[np.ix_(weekly, (hd % 7) == 6)] *= 0.5
        change = rng.random(n) < cfg.honest_change
        change_day = rng.integers(-cfg.history_days + 3, 0, n)
        change_f = np.where(rng.random(n) < 0.5, rng.uniform(0.45, 0.8, n), rng.uniform(1.25, 1.8, n))
        before = change[:, None] & (hd[None, :] < change_day[:, None])
        dayf = np.where(before, dayf / change_f[:, None], dayf)   # history differs from now
        hist = (day_mean[:, None, :] * dayf[:, :, None]).reshape(n, Hh)
        hist *= rng.normal(1.0, cfg.p_noise * 3, hist.shape).clip(0.8, 1.2)
        full = np.concatenate([hist, use], axis=1)                    # [n, Hh+T]
        pf = rng.uniform(0.80, 0.95, n)
        peak = full.max(axis=1)
        # contracted capacity in the customer register: often generous or out of date
        contract = np.round(peak / pf * rng.lognormal(np.log(1.5), 0.3, n) / 5) * 5   # kVA

        # tampering, where the feeder simulation put theft at this area
        recorded = np.ones(n)
        kind = np.array([""] * n, dtype=object)
        theft_share = d["theft_bus_p"][s][:, j].sum() / max(L.sum() / 1000, 1e-9)
        if theft_share > 0:
            E = use.sum(axis=1)
            need, got = theft_share * E.sum(), 0.0
            for i in rng.permutation(n)[: int(np.ceil(cfg.max_tamper * n))]:
                if got >= need:
                    break
                if rng.random() < 0.4:
                    r, k = 2 / 3, "one phase bypassed"
                else:
                    r, k = rng.uniform(0.1, 0.5), "current transformer shunted"
                recorded[i], kind[i] = r, k
                got += (1 - r) * E[i]
        tamper = recorded < 1
        onset = np.full(n, -Hh - 1)
        late = tamper & (rng.random(n) < cfg.onset_in_history)
        onset[late] = rng.integers(-Hh + 72, 1, late.sum())
        frac = np.where(hours[None, :] >= onset[:, None], recorded[:, None], 1.0)
        rec = full * frac
        rec[:, Hh:] *= on[None, :]                                    # no supply, no reading
        full[:, Hh:] *= on[None, :]

        # bus voltage at the customers' connection point (smart-meter voltage)
        vb = d["vm_pu"][s][:, all_names.index(bus)]
        v = np.where(on[None, :], vb[None, :] + rng.normal(0, cfg.v_noise, (n, T)), np.nan)

        # the area's billing and stolen energy now come from the meters
        rec_w = rec[:, Hh:].sum(axis=0) / 1000                        # MW recorded, window
        out["theft_bus_p"][s][:, j] = (np.maximum(L / 1000 - rec_w, 0.0) if tamper.any()
                                       else 0.0)
        out["billing"][s][:, j] = _rebin(rec_w, out["billing"].shape[1])
        if "is_theft" in out:
            out["is_theft"][s][j] = bool(tamper.any())

        for i in range(n):
            serial += 1
            rows.append({"meter": f"L{serial:04d}", "bus": bus, "type": kinds[i],
                         "contract_kva": float(contract[i]), "tamper": bool(tamper[i]),
                         "tamper_kind": kind[i], "recorded": float(recorded[i]),
                         "onset_hour": int(onset[i]) if tamper[i] else None,
                         "honest_change": bool(change[i] and not tamper[i]),
                         "stolen_kwh": float((full[i, Hh:] - rec[i, Hh:]).sum())})
        P.append(rec); Ptrue.append(full); Q.append(rec * np.tan(np.arccos(pf))[:, None]); V.append(v)

    customers = pd.DataFrame(rows)
    readings = {"hours": hours, "p_kw": np.vstack(P).T, "true_p_kw": np.vstack(Ptrue).T,
                "q_kvar": np.vstack(Q).T, "v_pu": np.vstack(V).T}
    return out, customers, readings


def _rebin(x_mw: np.ndarray, n_bins: int) -> np.ndarray:
    """Hourly MW to energy per billing period (daily, or one total), MWh."""
    if n_bins == 1:
        return np.array([x_mw.sum()])
    return x_mw[: n_bins * 24].reshape(n_bins, 24).sum(axis=1)


def ami_inputs(d: dict, customers: pd.DataFrame, readings: dict, s: int = 0,
               use_voltage: bool = True) -> dict:
    """Add smart-meter data to dataset `d` (scenario `s`) for the feeder model.

    ami_p  [S, T, B]  recorded large-customer use per transformer area, MW (NaN elsewhere).
                      Where present it replaces billing spread over the hours, so the model
                      knows the exact hourly sales there.
    ami_v  [S, T, B]  average smart-meter voltage per transformer area, pu (NaN elsewhere),
                      used as extra voltage measurements along the feeder.
    """
    S, T, B = d["true_bus_p"].shape if "true_bus_p" in d else (d["head"].shape[0], d["head"].shape[1], len(d["bus_names"]))
    out = dict(d)
    ap = out.get("ami_p", np.full((S, T, B), np.nan)).copy()
    av = out.get("ami_v", np.full((S, T, B), np.nan)).copy()
    hrs = readings["hours"]
    w = (hrs >= 0) & (hrs < T)
    buses = [str(b) for b in d["bus_names"]]
    for bus, idx in customers.groupby("bus").groups.items():
        if bus not in buses:
            continue
        j = buses.index(bus)
        cols = np.asarray(list(idx))
        ap[s, :, j] = np.nansum(readings["p_kw"][w][:, cols], axis=1) / 1000
        if use_voltage:
            v = readings["v_pu"][:, cols]                      # analysis period only, [T, n]
            n_ok = np.isfinite(v).sum(axis=1)
            av[s, :, j] = np.where(n_ok > 0, np.nansum(v, axis=1) / np.maximum(n_ok, 1), np.nan)
    out["ami_p"] = ap
    if use_voltage:
        out["ami_v"] = av
    return out


def rank_large_customers(customers: pd.DataFrame, readings: dict, n_hours: int,
                         area_share: dict | pd.Series | None = None) -> pd.DataFrame:
    """Rank large customers on smart meters for inspection.

    For each meter, from its hourly readings:
      daily_kwh_recent   average daily energy over the last 7 days of data
      drop               the largest lasting fall in daily energy found anywhere in the history
                         (share lost, with its day), counting only falls well beyond normal
                         day-to-day variation
      use_of_contract    recent peak demand as a share of the contracted capacity, compared
                         with the typical large customer (a meter that records only part of
                         the load shows a low peak for the size of the connection)
      priority           the stronger of the two signals, weighted by the estimated stolen
                         share of the customer's transformer area from the feeder model and
                         (mildly) by the size of the connection
    """
    hrs = readings["hours"]
    keep = hrs < n_hours
    p = readings["p_kw"][keep]
    hrs = hrs[keep]
    first_day = int(np.floor(hrs.min() / 24))
    day = np.floor(hrs / 24).astype(int) - first_day
    n_days = day.max() + 1
    daily = np.zeros((n_days, p.shape[1]))
    np.add.at(daily, day, np.nan_to_num(p))
    # hours with no supply (load shedding: every meter reads nothing) are left out, and each
    # day is scaled to 24 energised hours, so outages do not look like a drop in use
    energised = ~(np.nan_to_num(p) <= 0).all(axis=1)
    counted = np.zeros(n_days)
    np.add.at(counted, day, energised.astype(float))
    good = counted >= 6
    dd = daily[good] / counted[good, None] * 24
    days_idx = np.flatnonzero(good) + first_day
    m = dd.shape[0]

    out = customers[["meter", "bus"]].copy()
    if "type" in customers:
        out["type"] = customers["type"].values
    out["contract_kva"] = customers["contract_kva"].values
    recent = dd[-7:].mean(axis=0)
    out["daily_kwh_recent"] = recent.round(1)

    # lasting step down: best split point, t-like statistic on log daily energy
    lg = np.log(np.maximum(dd, 1e-3))
    best_t = np.zeros(dd.shape[1]); best_k = np.full(dd.shape[1], -1); best_r = np.ones(dd.shape[1])
    for k in range(5, m - 3):
        a, b = lg[:k], lg[k:]
        diff = a.mean(axis=0) - b.mean(axis=0)
        sd = np.sqrt(a.var(axis=0, ddof=1) / k + b.var(axis=0, ddof=1) / (m - k)) + 1e-6
        t = diff / sd
        better = t > best_t
        best_t = np.where(better, t, best_t)
        best_k = np.where(better, k, best_k)
        best_r = np.where(better, np.exp(-diff), best_r)
    sig = best_t > 6.0
    out["drop_pct"] = np.where(sig, 100 * (1 - best_r), 0.0).round(0)
    out["drop_day"] = np.where(sig, days_idx[np.clip(best_k, 0, m - 1)], np.nan)

    # peak demand against contracted capacity (last 14 days)
    last = hrs >= hrs.max() - 14 * 24
    peak = np.nanmax(np.where(np.isfinite(p[last]), p[last], 0.0), axis=0)
    use_c = peak / np.maximum(customers["contract_kva"].values, 1e-6)
    out["use_of_contract_pct"] = (100 * use_c).round(0)
    typical = np.median(use_c)
    low = np.maximum(0.0, np.log(typical / np.maximum(use_c, 1e-3)) - 0.15)
    step = np.where(sig, -np.log(np.maximum(best_r, 1e-3)), 0.0)
    anomaly = np.maximum(step, low)
    share = (pd.Series(area_share).reindex(out["bus"].values).fillna(0.0).values
             if area_share is not None else np.ones(len(out)))
    out["area_stolen_share"] = np.round(share, 3)
    out["anomaly"] = anomaly.round(3)
    size = customers["contract_kva"].values / max(np.median(customers["contract_kva"].values), 1e-6)
    out["priority"] = (anomaly * (0.15 + share) * np.sqrt(size)).round(4)
    return out.sort_values("priority", ascending=False, ignore_index=True)

"""Detect and locate a new cable fault (constant resistive leak) on a feeder.

Why theft and faults can be told apart
--------------------------------------
Stolen power is ordinary household load: it draws real power P and reactive power Q in
the same ratio as billed load (power factor about 0.9). A failing cable joint or a
high-impedance earth fault is resistive: it draws P only. So at the substation

    z_t = P_t - Q_t / tan(phi)

is unaffected by legitimate or stolen load, and changes only through network losses
(which follow the load smoothly), cable charging (constant) and a fault (a step that
appears when the fault starts and then stays).

Detection
---------
For every possible start hour tau, fit
    z_t = a + b * P_t + c * P_t^2 + s * [t >= tau] + noise
by least squares. The largest t-statistic of s over all tau is the detection score; the
best tau is the estimated start time and s the estimated leak in MW.

Location
--------
When the leak starts, the voltage drop from the substation to transformer meter m rises by
about ZR[m, j] * s, where ZR[m, j] is the resistance of the cable shared by the routes to
meter m and to the fault bus j. A meter at the faulty transformer also sees the step in
its own P - Q/tan(phi). For each candidate bus, the measured steps are compared with the
predicted ones; the fit gives a probability for every bus.

Assumption and limit: loads keep a steady power factor. A fault already present at the
start of the data shows no step and cannot be detected this way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slgrid.topology import path_matrix

V_KV = 11.0


def energised_hours(head_p: np.ndarray) -> np.ndarray:
    """Hours when the feeder was supplied, inferred from the substation meter.

    During load shedding the feeder breaker is open and the meter reads (near) zero.
    An hour counts as energised when its power is above 2% of the typical energised
    level (and at least 2 kW). Missing readings count as off.
    """
    p = np.nan_to_num(np.asarray(head_p, dtype=float), nan=0.0)
    pos = p[p > 0]
    if not len(pos):
        return np.zeros(len(p), dtype=bool)
    return p > max(0.002, 0.02 * np.median(pos))


@dataclass
class FaultConfig:
    power_factor: float = 0.90
    min_segment: int = 12          # hours needed before and after a candidate start
    threshold: float = 4.0         # detection threshold on the step t-statistic (chosen on development data)


def _step_scan(y, X, min_seg, mask=None):
    """Best step in y given covariates X. Returns (t_stat, tau, step, se)."""
    T = len(y)
    if mask is None:
        mask = np.isfinite(y)
    best = (0.0, -1, 0.0, np.inf)
    idx = np.flatnonzero(mask)
    if len(idx) < 2 * min_seg + X.shape[1] + 2:
        return best
    yv, Xv = y[mask], X[mask]
    for tau in range(min_seg, T - min_seg + 1):
        step = (idx >= tau).astype(float)
        if step.sum() < min_seg / 2 or (1 - step).sum() < min_seg / 2:
            continue
        A = np.column_stack([Xv, step])
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
        resid = yv - A @ coef
        dof = len(yv) - A.shape[1]
        sigma2 = resid @ resid / dof
        try:
            cov = sigma2 * np.linalg.inv(A.T @ A)
        except np.linalg.LinAlgError:
            continue
        se = np.sqrt(cov[-1, -1])
        t = coef[-1] / se
        if t > best[0]:
            best = (t, tau, coef[-1], se)
    return best


def _step_at(y, X, tau, mask=None):
    """Step size and standard error at a known start hour."""
    if mask is None:
        mask = np.isfinite(y)
    T = len(y)
    step = (np.arange(T) >= tau).astype(float)
    m = mask
    if m.sum() < X.shape[1] + 3 or step[m].sum() < 3 or (1 - step[m]).sum() < 3:
        return np.nan, np.inf
    A = np.column_stack([X[m], step[m]])
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    resid = y[m] - A @ coef
    sigma2 = resid @ resid / max(len(resid) - A.shape[1], 1)
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    return coef[-1], np.sqrt(max(cov[-1, -1], 1e-18))


class FaultDetector:
    def __init__(self, d, cfg: FaultConfig | None = None):
        self.d = d
        self.cfg = cfg or FaultConfig()
        self.tan = np.tan(np.arccos(self.cfg.power_factor))
        self.buses = [str(b) for b in d["bus_names"]]
        self.meters = [str(m) for m in d["meter_names"]]
        self.m_idx = [self.buses.index(m) for m in self.meters]
        A = path_matrix(d["line_from"], d["line_to"], self.buses)
        self.ZR = (A * d["line_r_ohm"]) @ A.T / V_KV ** 2      # pu per MW

    def analyse(self, s: int) -> dict:
        cfg, d = self.cfg, self.d
        P, Q, Vh = d["head"][s, :, 0], d["head"][s, :, 1], d["head"][s, :, 2]
        T = len(P)
        on = energised_hours(P)
        Vh = np.where(np.isfinite(Vh), Vh, 1.0)
        X = np.column_stack([np.ones(T), P, P ** 2])
        z = P - Q / self.tan
        # Only energised hours: an outage is not a fault starting or stopping.
        t_stat, tau, step, se = _step_scan(z, X, cfg.min_segment, mask=on)
        out = {"score": t_stat, "detected": bool(t_stat >= cfg.threshold),
               "hours_on": int(on.sum()),
               "start_hour": int(tau) if tau >= 0 else None,
               "leak_kw": 1000 * step if tau >= 0 else 0.0,
               "bus_prob": np.zeros(len(self.buses)), "likely_section": []}
        if tau < 0 or step <= 0:
            return out

        # Steps seen by the transformer meters at the estimated start hour.
        obs, sd, pred_unit = [], [], []
        for k, m in enumerate(self.m_idx):
            v = d["meter_v"][s, :, k]
            mp_, mq_ = d["meter_p"][s, :, k], d["meter_q"][s, :, k]
            ok = np.isfinite(v) & np.isfinite(mp_) & np.isfinite(mq_) & on
            # Voltage drop to the meter, explained by feeder load, substation voltage and
            # the meter's own load; what is left steps up when the leak starts.
            Xv = np.column_stack([np.ones(T), P, Q, P ** 2, Vh,
                                  np.nan_to_num(mp_), np.nan_to_num(mq_)])
            dv, sv = _step_at(Vh - v, Xv, tau, ok)
            if np.isfinite(dv):
                obs.append(dv); sd.append(sv); pred_unit.append(self.ZR[m])   # per MW at each bus
            mp, mq = d["meter_p"][s, :, k], d["meter_q"][s, :, k]
            ok = np.isfinite(mp) & np.isfinite(mq) & on
            Xm = np.column_stack([np.ones(T), mp, mp ** 2])
            Xm[~ok] = 0
            dw, sw = _step_at(mp - mq / self.tan, Xm, tau, ok)
            if np.isfinite(dw):
                e = np.zeros(len(self.buses)); e[m] = 1.0
                obs.append(dw); sd.append(sw); pred_unit.append(e)
        if obs:
            obs, sd, U = np.array(obs), np.array(sd), np.array(pred_unit)   # U: [n_obs, B]
            # Fit the leak size per candidate bus (with a weak tie to the head estimate).
            w = 1 / sd ** 2
            chi = np.zeros(len(self.buses))
            for j in range(len(self.buses)):
                u = U[:, j]
                c_hat = (np.sum(w * u * obs) + step / se ** 2) / (np.sum(w * u * u) + 1 / se ** 2)
                chi[j] = np.sum(w * (obs - u * c_hat) ** 2) + (step - c_hat) ** 2 / se ** 2
            ll = -0.5 * (chi - chi.min())
            p = np.exp(ll)
            out["bus_prob"] = p / p.sum()
        else:
            out["bus_prob"] = np.full(len(self.buses), 1 / len(self.buses))
        order = np.argsort(-out["bus_prob"])
        cum = np.cumsum(out["bus_prob"][order])
        k = int(np.searchsorted(cum, 0.8) + 1)
        out["likely_section"] = [self.buses[i] for i in order[:k]]
        return out

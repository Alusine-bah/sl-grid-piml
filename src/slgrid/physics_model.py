"""Physics-informed theft localisation (label-free).

For each scenario independently, find the theft and fault pattern that best explains
the utility's measurements while obeying the feeder's power-flow physics. No theft
labels are used.

Unknowns per load bus j:
  theta_j >= 0   theft ratio: stolen power = theta_j * billed power at bus j
                 (stolen share of consumption = theta_j / (1 + theta_j))
  c_j     >= 0   constant leakage (a resistive cable fault), MW
Feeder-level unknowns:
  q0             constant reactive offset, MVAr (cable charging, capacitor banks: the
                 asset register rarely has reliable cable capacitance)
  v_m            constant voltage offset of each meter, pu (calibration error)

Load model at hour t:
  P_jt = b_jt * (1 + theta_j) + c_j
  Q_jt = b_jt * (1 + theta_j) * tan(phi)        (faults are resistive, no Q)
where b_jt is the billed energy of bus j spread over the day's hours using the
feeder-head load shape.

Physics used (all linear in theta and c once losses are fixed):
  1. Active power balance at the substation:
       head_P_t = sum_j P_jt + losses_t
  2. Reactive power balance:
       head_Q_t = sum_j Q_jt + reactive losses_t + q0
  3. Voltage drop to each meter (LinDistFlow):
       V_head_t - V_m_t = sum_j (ZR_mj * P_jt + ZX_mj * Q_jt) + correction_mt + v_m
     where ZR_mj (ZX_mj) is the resistance (reactance) of the path shared by the routes
     from the substation to meter m and to bus j, in pu per MW.
  4. Metered buses: measured P and Q equal the model's P_jt and Q_jt.

Losses and the LinDistFlow correction come from a full AC power flow run on the current
estimate, and the estimate is then refined (fixed-point iterations). So the final
answer is consistent with the full non-linear physics, not just the linear model.

Solved as bounded weighted least squares (scipy), a few milliseconds per scenario.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandapower as pp
from scipy.optimize import lsq_linear

from slgrid.topology import path_matrix

V_KV = 11.0


@dataclass
class PhysicsConfig:
    power_factor: float = 0.90
    sd_voltage: float = 0.004       # pu, meter + head voltage noise and model error
    sd_head_p: float = 0.04         # MW, hourly head balance error (billing spread is rough)
    sd_head_q: float = 0.04         # MVAr
    sd_meter: float = 0.02          # relative error of a transformer-meter reading
    sd_offset: float = 0.005        # pu, expected size of a meter's voltage calibration offset
    prior_weight: float = 0.5       # pull of theta toward the feeder average (no kVA data)
    capacity_sd: float = 0.35       # how loosely transformer loading follows its kVA rating
    fault_weight: float = 200.0     # penalty per MW of leakage, so faults are only used when needed
    max_theta: float = 9.0          # stolen share at most 90%
    max_fault_mw: float = 0.3
    ac_iterations: int = 2
    detect_faults: bool = False     # use the fault detector (slgrid.faults) to time faults
    window_hours: int | None = None # sum equations over windows of this length (None = whole run)
    agg_power: float = 1.0          # 1.0 = errors within a window treated as fully correlated


def _network(d):
    """Build a pandapower model of the feeder from the standard data arrays only."""
    net = pp.create_empty_network()
    names = [str(n) for n in d["all_bus_names"] if str(n) != "sub_33kV"]
    idx = {n: pp.create_bus(net, vn_kv=V_KV, name=n) for n in names}
    pp.create_ext_grid(net, idx["bus_0"], vm_pu=1.0)
    for f, t, r, x in zip(d["line_from"], d["line_to"], d["line_r_ohm"], d["line_x_ohm"]):
        pp.create_line_from_parameters(net, idx[str(f)], idx[str(t)], length_km=1.0,
                                       r_ohm_per_km=float(r), x_ohm_per_km=float(x),
                                       c_nf_per_km=0.0, max_i_ka=1.0)
    for b in d["bus_names"]:
        pp.create_load(net, idx[str(b)], p_mw=0.0, q_mvar=0.0, name=str(b))
    return net


def _aggregate(A, y, groups, sd, agg_power):
    """Sum equations within each group (e.g. day), then weight by the group's noise.

    Theft ratios and leakage are constant in time, so summing hourly equations over a
    day keeps all the information about them while averaging out the error in the
    hourly split of daily billing.
    """
    ug = np.unique(groups)
    A_out = np.stack([A[groups == g].sum(axis=0) for g in ug])
    y_out = np.array([y[groups == g].sum() for g in ug])
    cnt = np.array([(groups == g).sum() for g in ug], dtype=float)
    w = 1.0 / (sd * cnt ** agg_power)
    return A_out * w[:, None], y_out * w


class PhysicsInformedLocator:
    def __init__(self, d, cfg: PhysicsConfig | None = None):
        self.d = d
        self.cfg = cfg or PhysicsConfig()
        self.buses = [str(b) for b in d["bus_names"]]
        self.meters = [str(m) for m in d["meter_names"]]
        self.m_idx = [self.buses.index(m) for m in self.meters]
        A = path_matrix(d["line_from"], d["line_to"], self.buses)     # [B, L]
        self.ZR = (A * d["line_r_ohm"]) @ A.T / V_KV ** 2             # pu per MW
        self.ZX = (A * d["line_x_ohm"]) @ A.T / V_KV ** 2
        self.tan = np.tan(np.arccos(self.cfg.power_factor))
        self.net = _network(d)
        self.fd = None
        if self.cfg.detect_faults:
            from slgrid.faults import FaultConfig, FaultDetector
            self.fd = FaultDetector(d, FaultConfig(power_factor=self.cfg.power_factor))

    # --- helpers -------------------------------------------------------------
    def _hourly_billed(self, s):
        """Spread billed energy over the hours using the feeder-head load shape.

        Billing may be daily ([D, B] with D = hours / 24) or one total for the whole
        period ([1, B]), as with monthly bills or prepaid token sales.
        """
        bill = self.d["billing"][s]                      # [D, B] MWh
        from slgrid.faults import energised_hours
        head = self.d["head"][s, :, 0]
        head = np.where(energised_hours(head), np.nan_to_num(head), 0.0)   # [T]
        T = len(head)
        if bill.shape[0] == T // 24 and bill.shape[0] > 1:
            n_d = bill.shape[0]
            shape = head[: n_d * 24].reshape(n_d, 24)
            tot = shape.sum(axis=1, keepdims=True)
            shape = np.divide(shape, tot, out=np.zeros_like(shape), where=tot > 0)
            return (shape[:, :, None] * bill[:, None, :]).reshape(n_d * 24, -1)   # [T, B] MW
        shape = head / max(head.sum(), 1e-12)
        return shape[:, None] * bill.sum(axis=0)[None, :]

    def _ac_corrections(self, P, Q, v_head, on=None):
        """Losses and LinDistFlow voltage errors from a full AC power flow.
        Hours with the feeder switched off (on = False) are skipped: no flow, no losses."""
        T = P.shape[0]
        loss_p, loss_q = np.zeros(T), np.zeros(T)
        v_corr = np.zeros((T, len(self.meters)))
        load_bus = self.net.load["bus"].values
        bus_names = self.net.bus["name"].values
        m_rows = [int(np.flatnonzero(bus_names == m)[0]) for m in self.meters]
        for t in range(T):
            if on is not None and not on[t]:
                continue
            self.net.load["p_mw"] = P[t]
            self.net.load["q_mvar"] = Q[t]
            self.net.ext_grid["vm_pu"] = v_head[t]
            pp.runpp(self.net, numba=True)
            loss_p[t] = self.net.res_line["pl_mw"].sum()
            loss_q[t] = self.net.res_line["ql_mvar"].sum()
            v_ac = self.net.res_bus["vm_pu"].values[m_rows]
            drop_lin = self.ZR[self.m_idx] @ P[t] + self.ZX[self.m_idx] @ Q[t]
            v_corr[t] = (v_head[t] - v_ac) - drop_lin
        return loss_p, loss_q, v_corr

    # --- main ----------------------------------------------------------------
    def fit_scenario(self, s):
        cfg, d = self.cfg, self.d
        b = self._hourly_billed(s)                       # [T, B]
        T, B = b.shape
        M = len(self.meters)
        head_p, head_q, v_head = d["head"][s, :, 0], d["head"][s, :, 1], d["head"][s, :, 2]
        vm, mp, mq = d["meter_v"][s], d["meter_p"][s], d["meter_q"][s]
        # Hours the feeder was switched off (load shedding) carry no information:
        # loads, losses, cable charging and any leak are all zero then.
        from slgrid.faults import energised_hours
        on = energised_hours(head_p)
        head_p = np.where(on, np.nan_to_num(head_p), 0.0)
        head_q = np.where(on, np.nan_to_num(head_q), 0.0)
        v_head = np.where(on & np.isfinite(v_head), v_head, 1.0)

        # Unknown vector: [theta (B), c (B), q0 (1), v_off (M)]
        n = 2 * B + 1 + M
        i_th, i_c, i_q0, i_v = slice(0, B), slice(B, 2 * B), 2 * B, slice(2 * B + 1, n)

        # Prior from transformer ratings: physical demand at a bus is roughly
        # proportional to its installed kVA. Scale so the feeder total matches the head.
        theta_bar = max(head_p.sum() * 0.97 / b.sum() - 1, 0.0)
        if "transformer_kva" in d:
            kva = d["transformer_kva"][s]
            expected = kva / kva.sum() * head_p.sum() * 0.97
            theta_prior = np.clip(expected / b.sum(axis=0) - 1, 0, cfg.max_theta)
            prior_sd = cfg.capacity_sd * (1 + theta_prior) + 0.05
        else:
            theta_prior = np.full(B, theta_bar)
            prior_sd = np.full(B, 1.0 / cfg.prior_weight)
        loss_p = 0.03 * head_p
        loss_q = 0.03 * head_q
        v_corr = np.zeros((T, M))
        K = self.ZR + self.ZX * self.tan                  # voltage drop per MW of load at pf

        hours = np.arange(T)
        grp = hours // cfg.window_hours if cfg.window_hours else np.zeros(T, dtype=int)

        # Fault handling. Without the detector, a leak may be present at any bus all period.
        # With it: if a new fault is detected, leaks start at the detected hour, the period
        # is split into before/after so the step is visible, and the total leak and its
        # likely location come from the detector; if none is detected, no leak is fitted.
        fault_on = on.astype(float)
        fault_w = np.full(B, cfg.fault_weight)
        fault_ub = np.full(B, cfg.max_fault_mw)
        fault_sum = None
        fault = None
        if self.fd is not None:
            fault = self.fd.analyse(s)
            if fault["detected"]:
                fault_on = ((hours >= fault["start_hour"]) & on).astype(float)
                grp = grp * 2 + fault_on.astype(int)
                p = fault["bus_prob"]
                fault_w = cfg.fault_weight * (1.0 - p / p.max()) + 1.0
                c_hat = max(fault["leak_kw"], 0.0) / 1000
                fault_sum = (c_hat, max(0.2 * c_hat, 0.01))
            else:
                fault_ub = np.full(B, 1e-9)

        for it in range(cfg.ac_iterations + 1):
            blocks, rhs = [], []

            def add(A, y, g, sd):
                Ag, yg = _aggregate(A, y, g, sd, cfg.agg_power)
                blocks.append(Ag); rhs.append(yg)

            # 1. active power balance
            # (energised hours only: dead hours would dilute the weight of these equations)
            A = np.zeros((T, n)); A[:, i_th] = b; A[:, i_c] = fault_on[:, None]
            add(A[on], (head_p - b.sum(1) - loss_p)[on], grp[on], cfg.sd_head_p)
            # 2. reactive power balance
            A = np.zeros((T, n)); A[:, i_th] = self.tan * b; A[:, i_q0] = on
            add(A[on], (head_q - self.tan * b.sum(1) - loss_q)[on], grp[on], cfg.sd_head_q)
            # 3. voltage drop to each meter
            for k, m in enumerate(self.m_idx):
                ok = ~np.isnan(vm[:, k])
                A = np.zeros((ok.sum(), n))
                A[:, i_th] = K[m] * b[ok]
                A[:, i_c] = fault_on[ok, None] * self.ZR[m]
                A[:, 2 * B + 1 + k] = 1.0
                drop = v_head[ok] - vm[ok, k] - v_corr[ok, k] - b[ok] @ K[m]
                add(A, drop, grp[ok], cfg.sd_voltage)
            # 4. metered buses: P and Q
            for k, m in enumerate(self.m_idx):
                ok = ~np.isnan(mp[:, k])
                sdp = cfg.sd_meter * max(np.nanmean(mp[:, k]), 1e-3)
                A = np.zeros((ok.sum(), n)); A[:, m] = b[ok, m]; A[:, B + m] = fault_on[ok]
                add(A, mp[ok, k] - b[ok, m], grp[ok], sdp)
                sdq = cfg.sd_meter * max(np.nanmean(mq[:, k]), 1e-3)
                A = np.zeros((ok.sum(), n)); A[:, m] = self.tan * b[ok, m]
                add(A, mq[ok, k] - self.tan * b[ok, m], grp[ok], sdq)
            # priors: theta near the feeder average, leakage near zero, small offsets
            A = np.zeros((B, n)); A[:, i_th] = np.eye(B) / prior_sd[:, None]
            blocks.append(A); rhs.append(theta_prior / prior_sd)
            A = np.zeros((B, n)); A[:, i_c] = np.diag(fault_w)
            blocks.append(A); rhs.append(np.zeros(B))
            if fault_sum is not None:
                A = np.zeros((1, n)); A[0, i_c] = 1.0 / fault_sum[1]
                blocks.append(A); rhs.append(np.array([fault_sum[0] / fault_sum[1]]))
            A = np.zeros((M, n)); A[:, i_v] = np.eye(M) / cfg.sd_offset
            blocks.append(A); rhs.append(np.zeros(M))

            Amat, y = np.vstack(blocks), np.concatenate(rhs)
            lb = np.concatenate([np.zeros(2 * B), [-np.inf], np.full(M, -np.inf)])
            ub = np.concatenate([np.full(B, cfg.max_theta), fault_ub,
                                 [np.inf], np.full(M, np.inf)])
            sol = lsq_linear(Amat, y, bounds=(lb, ub), method="trf", lsmr_tol="auto")
            theta, c = sol.x[i_th], sol.x[i_c]

            if it < cfg.ac_iterations:
                P = b * (1 + theta) + fault_on[:, None] * c
                Q = b * (1 + theta) * self.tan
                loss_p, loss_q, v_corr = self._ac_corrections(P, Q, v_head, on)

        stolen = (b * theta).sum(axis=0)                 # MWh over the scenario
        score = theta / (1 + theta)
        return {"theta": theta, "fault_mw": c, "score": score, "stolen": stolen,
                "q0": sol.x[i_q0], "v_off": sol.x[i_v], "fault": fault}

    def predict(self, s_idx, n_jobs: int = 1):
        """Fit every scenario in s_idx. n_jobs > 1 runs scenarios in parallel processes."""
        if n_jobs == 1:
            out = [self.fit_scenario(s) for s in s_idx]
        else:
            from joblib import Parallel, delayed
            out = Parallel(n_jobs=n_jobs)(delayed(self.fit_scenario)(s) for s in s_idx)
        return (np.array([o["score"] for o in out]), np.array([o["stolen"] for o in out]),
                np.array([o["fault_mw"] for o in out]))

"""Read and write a single feeder in the utility CSV format.

A utility provides five CSV files for one 11 kV feeder:

  lines.csv         from_bus, to_bus, r_ohm, x_ohm
                    One row per line section. The substation 11 kV bus is the one bus
                    that never appears in to_bus. The feeder must be radial.
  transformers.csv  bus, kva
                    One row per load bus (a distribution transformer, or a group of them).
  head.csv          hour, p_mw, q_mvar, v_pu
                    Substation 11 kV breaker readings, one row per hour. `hour` may be an
                    integer (0, 1, 2, ...) or a timestamp. Hours when the feeder was switched
                    off (load shedding) may be left out or given as zero power; both are
                    treated as "supply off".
  meters.csv        hour, bus, p_mw, q_mvar, v_pu
                    Readings from transformer check meters. Any number of buses (zero is
                    allowed: an empty file with just the header). Missing hours are fine.
  billing.csv       [day,] bus, mwh
                    Energy sold per transformer area. Either daily (a `day` column with one
                    value per day of the period, as an integer or a date), or one total per
                    bus for the whole period (no `day` column), as with monthly bills or
                    prepaid token sales.

Prepaid utilities can give vending records instead of billing.csv:

  customers.csv     meter, bus [, kind]
                    Which transformer area (bus) each prepaid meter is listed under.
                    Optional `kind` (e.g. residential, commercial): meters are compared
                    with others of the same kind.
  tokens.csv        meter, day, kwh
                    Every token purchase. `day` is a date, or a day number counted like
                    head.csv (0 = first day of head.csv; earlier history has negative days).
                    Give a month or more of history before the period if possible: sales
                    are averaged over the last 30 days, which evens out when people buy.

Large customers on smart meters (AMI), optional, in addition to the files above:

  ami_customers.csv meter, bus, contract_kva [, type]
                    Each large customer's smart meter, its transformer area, and the
                    contracted capacity from the customer register.
  ami_readings.csv  meter, hour, p_kw, q_kvar, v_pu
                    Hourly smart-meter readings, numbered like head.csv (or timestamps).
                    Weeks of history before the period help find tampered meters.
                    Transformer areas whose customers are all on smart meters should list
                    them all: the model then uses their hourly sales and voltages.

The period is whole days. Hours with no supply (load shedding) may be left out of head.csv.

These are converted to the arrays the models use (see docs/data_format.md), with a
leading scenario dimension of 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = {
    "lines": ["from_bus", "to_bus", "r_ohm", "x_ohm"],
    "transformers": ["bus", "kva"],
    "head": ["hour", "p_mw", "q_mvar", "v_pu"],
    "meters": ["hour", "bus", "p_mw", "q_mvar", "v_pu"],
    "billing": ["bus", "mwh"],          # "day" is optional
}
PREPAID = {"customers": ["meter", "bus"], "tokens": ["meter", "day", "kwh"]}
AMI = {"ami_customers": ["meter", "bus", "contract_kva"],
       "ami_readings": ["meter", "hour", "p_kw", "q_kvar", "v_pu"]}
TOKEN_LOOKBACK = 30   # days of token sales averaged to estimate sales in the period


class FeederDataError(ValueError):
    """Raised with a message a utility engineer can act on."""


def _to_index(col: pd.Series, unit: str, t0: pd.Timestamp | None = None,
              allow_negative: bool = False) -> pd.Series:
    """Hour/day numbers counted from the start of the first day.

    Integers are used as they are (0 = first hour or day). Timestamps and dates are counted
    from midnight of `t0` (default: the first day in the column). Anchoring at midnight, not
    at the first reading, keeps the timeline right when outage hours at the start of the
    period are simply left out of the file.
    """
    if pd.api.types.is_numeric_dtype(col):
        if (col < 0).any() and not allow_negative:
            raise FeederDataError(f"The '{unit}' column has negative numbers; count from 0.")
        return col.astype(int)
    t = pd.to_datetime(col, errors="coerce")
    if t.isna().any():
        raise FeederDataError(f"Could not read some values in the '{unit}' column as numbers or dates.")
    start = (t.min() if t0 is None else t0).floor("D")
    step = pd.Timedelta(hours=1) if unit == "hour" else pd.Timedelta(days=1)
    return ((t - start) / step).round().astype(int)


def frames_to_standard(frames: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    """Convert the five utility tables to model arrays.

    Returns (d, names) where `names` maps internal bus names (bus_0, bus_1, ...) back to
    the utility's own bus identifiers.
    """
    prepaid = frames.get("billing") is None and all(frames.get(k) is not None for k in PREPAID)
    need = {k: v for k, v in REQUIRED.items() if not (prepaid and k == "billing")}
    if prepaid:
        need.update(PREPAID)
    for key, cols in need.items():
        if frames.get(key) is None:
            if key == "billing":
                raise FeederDataError("Missing file: billing.csv (or, for prepaid customers, "
                                      "customers.csv and tokens.csv)")
            raise FeederDataError(f"Missing file: {key}.csv")
        df = frames[key]
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise FeederDataError(f"{key}.csv is missing column(s): {', '.join(missing)}")

    lines = frames["lines"].copy()
    lines["from_bus"] = lines["from_bus"].astype(str).str.strip()
    lines["to_bus"] = lines["to_bus"].astype(str).str.strip()
    if lines[["r_ohm", "x_ohm"]].isna().any().any():
        raise FeederDataError("lines.csv has empty r_ohm or x_ohm values.")
    if lines["to_bus"].duplicated().any():
        dup = lines.loc[lines["to_bus"].duplicated(), "to_bus"].iloc[0]
        raise FeederDataError(f"Bus '{dup}' is fed by more than one line: the feeder must be radial.")
    roots = sorted(set(lines["from_bus"]) - set(lines["to_bus"]))
    if len(roots) != 1:
        raise FeederDataError(
            "Could not find a single substation bus (a bus that appears only in from_bus). "
            f"Found: {roots or 'none'}. Check that the feeder is one connected radial network.")
    root = roots[0]
    load_buses = list(dict.fromkeys(lines["to_bus"]))

    # internal names: substation = bus_0, others bus_1..bus_n in line order
    internal = {root: "bus_0"}
    for i, b in enumerate(load_buses, start=1):
        internal[b] = f"bus_{i}"
    names = {v: k for k, v in internal.items()}

    tr = frames["transformers"].copy()
    tr["bus"] = tr["bus"].astype(str).str.strip()
    unknown = sorted(set(tr["bus"]) - set(load_buses))
    if unknown:
        raise FeederDataError(f"transformers.csv lists bus(es) not in lines.csv: {', '.join(unknown[:5])}")
    kva = tr.groupby("bus")["kva"].sum().reindex(load_buses)
    if kva.isna().any():
        missing = list(kva[kva.isna()].index[:5])
        raise FeederDataError(f"transformers.csv has no kVA rating for bus(es): {', '.join(missing)}")

    head = frames["head"].copy()
    raw_hours = head["hour"]
    t0 = None if pd.api.types.is_numeric_dtype(raw_hours) else pd.to_datetime(raw_hours, errors="coerce").min()
    head["hour"] = _to_index(raw_hours, "hour")
    head = head.sort_values("hour").drop_duplicates("hour")
    last_hour = int(head["hour"].max())

    if prepaid:
        bill = pd.DataFrame(columns=["bus", "mwh"])
        daily = False
    else:
        bill = frames["billing"].copy()
        bill["bus"] = bill["bus"].astype(str).str.strip()
        daily = "day" in bill.columns and bill["day"].nunique() > 1
    if daily:
        bill["day"] = _to_index(bill["day"], "day", t0)
        n_d = int(bill["day"].max()) + 1
        if last_hour >= 24 * n_d:
            raise FeederDataError(f"head.csv runs to hour {last_hour} but billing.csv covers only {n_d} "
                                  "days. Give billing for every day, or remove the day column and give "
                                  "one total per bus for the whole period.")
        if last_hour < 24 * (n_d - 1):
            raise FeederDataError(f"head.csv stops at hour {last_hour} but billing.csv covers {n_d} days; "
                                  "head.csv must cover whole days (24, 48, ... hours). Hours with no "
                                  "supply may be left out, but not whole days at the end.")
    else:
        n_d = last_hour // 24 + 1
    n_t = 24 * n_d
    head = head.set_index("hour").reindex(range(n_t))
    # Missing hours = feeder switched off (load shedding) or meter gaps: no power flowed
    # as far as the model is concerned. Too few hours left means the file is unusable.
    off = head["p_mw"].isna()
    head.loc[off, ["p_mw", "q_mvar"]] = 0.0
    head.loc[off, "v_pu"] = np.nan
    if (~off).sum() < 12:
        raise FeederDataError("head.csv has fewer than 12 hours with readings; the model needs at "
                              "least 12 energised hours (a month or more is recommended).")
    if head.loc[~off, ["q_mvar", "v_pu"]].isna().any().any():
        raise FeederDataError("head.csv has hours with power but no q_mvar or v_pu reading.")

    unknown = sorted(set(bill["bus"]) - set(load_buses))
    if unknown:
        raise FeederDataError(f"billing.csv lists bus(es) not in lines.csv: {', '.join(unknown[:5])}")
    vending = None
    if prepaid:
        from slgrid.prepaid import tokens_to_billing
        cust, tok = _read_vending(frames, internal, t0)
        if (tok["day"] >= n_d).any():
            raise FeederDataError(f"tokens.csv has purchases after the last day of head.csv "
                                  f"(day {n_d - 1}). Cut it to end with the same day.")
        if not ((tok["day"] >= 0) & (tok["day"] < n_d)).any():
            raise FeederDataError("tokens.csv has no purchases during the period covered by head.csv.")
        sold = tokens_to_billing(cust, tok, [internal[b] for b in load_buses], n_d, TOKEN_LOOKBACK)
        billing = pd.DataFrame(sold, columns=load_buses)
        vending = {"customers": cust, "tokens": tok}
    elif daily:
        billing = (bill.pivot_table(index="day", columns="bus", values="mwh", aggfunc="sum")
                   .reindex(index=range(n_d), columns=load_buses).fillna(0.0))
    else:
        billing = bill.groupby("bus")["mwh"].sum().reindex(load_buses).fillna(0.0).to_frame().T
    if (billing.sum(axis=0) <= 0).any():
        zero = list(billing.columns[billing.sum(axis=0) <= 0][:5])
        src = "customers.csv/tokens.csv have" if prepaid else "billing.csv has"
        raise FeederDataError(f"{src} no billed energy for bus(es): {', '.join(zero)}. "
                              "Every load bus needs some billing.")

    met = frames["meters"].copy()
    if len(met):
        met["hour"] = _to_index(met["hour"], "hour", t0)
        met = met[(met["hour"] >= 0) & (met["hour"] < n_t)]
        met["bus"] = met["bus"].astype(str).str.strip()
        unknown = sorted(set(met["bus"]) - set(load_buses))
        if unknown:
            raise FeederDataError(f"meters.csv lists bus(es) not in lines.csv: {', '.join(unknown[:5])}")
    meter_buses = list(dict.fromkeys(met["bus"])) if len(met) else []

    def mpivot(col):
        if not meter_buses:
            return np.zeros((n_t, 0))
        return (met.pivot_table(index="hour", columns="bus", values=col, aggfunc="mean")
                .reindex(index=range(n_t), columns=meter_buses).values)

    d = {
        "bus_names": np.array([internal[b] for b in load_buses]),
        "all_bus_names": np.array(["bus_0"] + [internal[b] for b in load_buses]),
        "meter_names": np.array([internal[b] for b in meter_buses]),
        "line_from": np.array([internal[b] for b in lines["from_bus"]]),
        "line_to": np.array([internal[b] for b in lines["to_bus"]]),
        "line_r_ohm": lines["r_ohm"].astype(float).values,
        "line_x_ohm": lines["x_ohm"].astype(float).values,
        "transformer_kva": kva.values[None, :].astype(float),
        "head": head[["p_mw", "q_mvar", "v_pu"]].values[None].astype(float),
        "meter_p": mpivot("p_mw")[None],
        "meter_q": mpivot("q_mvar")[None],
        "meter_v": mpivot("v_pu")[None],
        "billing": billing.values[None].astype(float),
    }
    if vending is not None:
        d["vending"] = vending          # prepaid customer list and purchases, internal bus names
    if all(frames.get(k) is not None for k in AMI):
        d = _add_ami(d, frames, internal, t0, n_t)
    return d, names


def _add_ami(d: dict, frames: dict, internal: dict, t0, n_t: int) -> dict:
    """Large customers' smart-meter files -> d["ami"] and the model inputs ami_p / ami_v."""
    from slgrid.ami import ami_inputs
    for key, cols in AMI.items():
        df = frames[key]
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise FeederDataError(f"{key}.csv is missing column(s): {', '.join(missing)}")
    cust = frames["ami_customers"].copy()
    cust["meter"] = cust["meter"].astype(str).str.strip()
    cust["bus"] = cust["bus"].astype(str).str.strip().map(internal)
    if cust["bus"].isna().any():
        raise FeederDataError("ami_customers.csv lists bus(es) not in lines.csv.")
    cust = cust.drop_duplicates("meter", keep="last").reset_index(drop=True)
    if "type" not in cust.columns:
        cust["type"] = "large customer"
    rd = frames["ami_readings"].copy()
    rd["meter"] = rd["meter"].astype(str).str.strip()
    rd = rd[rd["meter"].isin(cust["meter"])]
    if rd.empty:
        raise FeederDataError("ami_readings.csv has no readings for the meters in ami_customers.csv.")
    if not pd.api.types.is_numeric_dtype(rd["hour"]) and t0 is None:
        raise FeederDataError("ami_readings.csv uses timestamps, so head.csv must use timestamps too.")
    rd["hour"] = _to_index(rd["hour"], "hour", t0, allow_negative=True)
    rd = rd[rd["hour"] < n_t]
    hours = np.arange(int(min(rd["hour"].min(), 0)), n_t)
    piv = lambda col: (rd.pivot_table(index="hour", columns="meter", values=col, aggfunc="mean")  # noqa: E731
                       .reindex(index=hours, columns=cust["meter"]).values.astype(float))
    readings = {"hours": hours, "p_kw": piv("p_kw"), "q_kvar": piv("q_kvar")}
    readings["v_pu"] = piv("v_pu")[hours >= 0]
    d = ami_inputs(d, cust, readings)
    d["ami"] = {"customers": cust, "readings": readings}
    return d


def _read_vending(frames: dict, internal: dict, t0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """customers.csv and tokens.csv with internal bus names and day numbers (0 = first day)."""
    cust = frames["customers"].copy()
    cust["meter"] = cust["meter"].astype(str).str.strip()
    cust["bus"] = cust["bus"].astype(str).str.strip().map(internal)
    if cust["bus"].isna().all():
        raise FeederDataError("None of the buses in customers.csv appear in lines.csv.")
    cust = cust.dropna(subset=["bus"]).drop_duplicates("meter", keep="last")
    if "kind" not in cust.columns:
        cust["kind"] = "customer"
    tok = frames["tokens"].copy()
    tok["meter"] = tok["meter"].astype(str).str.strip()
    if not pd.api.types.is_numeric_dtype(tok["day"]) and t0 is None:
        raise FeederDataError("tokens.csv uses dates, so head.csv must use timestamps too. "
                              "Or give day numbers in tokens.csv (0 = first day of head.csv).")
    tok["day"] = _to_index(tok["day"], "day", t0, allow_negative=True)
    tok["kwh"] = pd.to_numeric(tok["kwh"], errors="coerce")
    tok = tok.dropna(subset=["kwh"])
    return cust[["meter", "bus", "kind"]].reset_index(drop=True), tok[["meter", "day", "kwh"]]


def standard_to_frames(d: dict, s: int = 0) -> dict[str, pd.DataFrame]:
    """Export one scenario of a dataset as the five utility CSV tables."""
    buses = [str(b) for b in d["bus_names"]]
    short = lambda b: str(b).replace("bus_", "")   # noqa: E731
    lines = pd.DataFrame({"from_bus": [short(b) for b in d["line_from"]],
                          "to_bus": [short(b) for b in d["line_to"]],
                          "r_ohm": np.round(d["line_r_ohm"], 5),
                          "x_ohm": np.round(d["line_x_ohm"], 5)})
    transformers = pd.DataFrame({"bus": [short(b) for b in buses],
                                 "kva": d["transformer_kva"][s]})
    n_t = d["head"].shape[1]
    head = pd.DataFrame({"hour": range(n_t), "p_mw": d["head"][s, :, 0].round(5),
                         "q_mvar": d["head"][s, :, 1].round(5), "v_pu": d["head"][s, :, 2].round(5)})
    rows = []
    for k, m in enumerate(d["meter_names"]):
        for t in range(n_t):
            p = d["meter_p"][s, t, k]
            if np.isnan(p):
                continue
            rows.append({"hour": t, "bus": short(m), "p_mw": round(float(p), 5),
                         "q_mvar": round(float(d["meter_q"][s, t, k]), 5),
                         "v_pu": round(float(d["meter_v"][s, t, k]), 5)})
    meters = pd.DataFrame(rows, columns=REQUIRED["meters"])
    n_d = d["billing"].shape[1]
    if n_d == 1:
        billing = pd.DataFrame([{"bus": short(b), "mwh": round(float(d["billing"][s, 0, j]), 5)}
                                for j, b in enumerate(buses)])
    else:
        billing = pd.DataFrame([{"day": day, "bus": short(b), "mwh": round(float(d["billing"][s, day, j]), 5)}
                                for day in range(n_d) for j, b in enumerate(buses)])
    out = {"lines": lines, "transformers": transformers, "head": head, "meters": meters,
           "billing": billing}
    if "vending" in d:          # prepaid: vending records replace billing.csv
        v = d["vending"]
        out.pop("billing")
        out["customers"] = pd.DataFrame({"meter": v["customers"]["meter"],
                                         "bus": [short(b) for b in v["customers"]["bus"]],
                                         "kind": v["customers"].get("kind", "customer")})
        out["tokens"] = v["tokens"][["meter", "day", "kwh"]].reset_index(drop=True)
    if "ami" in d:              # large customers on smart meters
        a = d["ami"]
        c = a["customers"]
        out["ami_customers"] = pd.DataFrame({"meter": c["meter"],
                                             "bus": [short(b) for b in c["bus"]],
                                             "contract_kva": c["contract_kva"],
                                             "type": c.get("type", "large customer")})
        r = a["readings"]
        hrs = r["hours"]
        n = len(c)
        vfull = np.full((len(hrs), n), np.nan)
        vfull[hrs >= 0] = r["v_pu"]
        df = pd.DataFrame({"meter": np.tile(c["meter"].values, len(hrs)),
                           "hour": np.repeat(hrs, n),
                           "p_kw": np.round(r["p_kw"].ravel(), 2),
                           "q_kvar": np.round(r["q_kvar"].ravel(), 2),
                           "v_pu": np.round(vfull.ravel(), 4)})
        out["ami_readings"] = df.dropna(subset=["p_kw"]).reset_index(drop=True)
    return out

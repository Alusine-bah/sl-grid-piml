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


class FeederDataError(ValueError):
    """Raised with a message a utility engineer can act on."""


def _to_index(col: pd.Series, unit: str, t0: pd.Timestamp | None = None) -> pd.Series:
    """Hour/day numbers counted from the start of the first day.

    Integers are used as they are (0 = first hour or day). Timestamps and dates are counted
    from midnight of `t0` (default: the first day in the column). Anchoring at midnight, not
    at the first reading, keeps the timeline right when outage hours at the start of the
    period are simply left out of the file.
    """
    if pd.api.types.is_numeric_dtype(col):
        if (col < 0).any():
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
    for key, cols in REQUIRED.items():
        if key not in frames:
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
    if daily:
        billing = (bill.pivot_table(index="day", columns="bus", values="mwh", aggfunc="sum")
                   .reindex(index=range(n_d), columns=load_buses).fillna(0.0))
    else:
        billing = bill.groupby("bus")["mwh"].sum().reindex(load_buses).fillna(0.0).to_frame().T
    if (billing.sum(axis=0) <= 0).any():
        zero = list(billing.columns[billing.sum(axis=0) <= 0][:5])
        raise FeederDataError(f"billing.csv has no billed energy for bus(es): {', '.join(zero)}. "
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
    return d, names


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
    return {"lines": lines, "transformers": transformers, "head": head, "meters": meters,
            "billing": billing}

"""Feeder Loss Locator: interactive app.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from slgrid.app_core import (district_areas, district_params, district_plan,  # noqa: E402
                             district_table, feeder_summary, large_customer_list, locate,
                             meter_list, simulate_case, simulate_visits, tree_layout)
from slgrid.inspections import (OUTCOMES, POSITIVE, InspectionConfig,  # noqa: E402
                                area_posterior, clean_log, evaluate_log, plan_round,
                                recently_visited)  # noqa: E402
from slgrid.io import FeederDataError, frames_to_standard, split_district_zip, standard_to_frames  # noqa: E402
from slgrid.loads import SHEDDING_PATTERNS  # noqa: E402

st.set_page_config(page_title="Feeder Loss Locator", page_icon="⚡", layout="wide")

ORANGE = "#c2410c"
GREY = "#8a959e"
EXAMPLE_DIR = ROOT / "data" / "example_feeder"
FILES = ["lines", "transformers", "head", "meters", "billing"]
PREPAID_FILES = ["customers", "tokens"]
AMI_FILES = ["ami_customers", "ami_readings"]
EXAMPLE_PREPAID_DIR = ROOT / "data" / "example_prepaid"
CURRENCIES = {
    "USD": "US dollar", "SLE": "Sierra Leonean leone", "EUR": "Euro", "GBP": "British pound",
    "GHS": "Ghanaian cedi", "NGN": "Nigerian naira", "LRD": "Liberian dollar",
    "GNF": "Guinean franc", "XOF": "West African CFA franc", "GMD": "Gambian dalasi",
    "KES": "Kenyan shilling", "ZAR": "South African rand", "BDT": "Bangladeshi taka",
    "INR": "Indian rupee", "CNY": "Chinese yuan",
}


# ----------------------------------------------------------------------------- helpers
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def usd_rates():
    """Latest USD exchange rates from a free public service (open.er-api.com).
    Returns (rates, last_update) or (None, None) if the service cannot be reached."""
    try:
        with urllib.request.urlopen("https://open.er-api.com/v6/latest/USD", timeout=6) as r:
            data = json.load(r)
        if data.get("result") == "success":
            return data["rates"], data.get("time_last_update_utc", "")
    except Exception:
        pass
    return None, None

@st.cache_data(show_spinner=False)
def run_simulation(theft, n_meters, strategy, dropout, fault, days, seed, billing, bill_noise,
                   shedding, mapping_error, ami):
    d = simulate_case(theft, n_meters, strategy, dropout, fault, days, seed, billing, bill_noise,
                      shedding, mapping_error, ami)
    res = locate(d)
    return d, res, meter_list(d, res), large_customer_list(d, res)


@st.cache_data(show_spinner=False)
def run_uploaded(frames_bytes: dict, power_factor: float, tech_loss: float):
    frames = {k: pd.read_csv(io.BytesIO(v)) for k, v in frames_bytes.items()}
    d, names = frames_to_standard(frames)
    res = locate(d, power_factor=power_factor, tech_loss=tech_loss)
    return d, res, names, meter_list(d, res), large_customer_list(d, res)


def label(bus: str, names: dict | None) -> str:
    return names.get(bus, bus) if names else bus.replace("bus_", "")


def feeder_map(d: dict, res: pd.DataFrame, names: dict | None, show_truth: bool):
    nodes, edges = tree_layout(d)
    nodes = nodes.merge(res, on="bus", how="left")
    nodes["name"] = [label(b, names) for b in nodes.bus]
    nodes["kind"] = np.where(nodes.bus == "bus_0", "Substation",
                             np.where(nodes.metered.fillna(False).astype(bool), "Metered transformer",
                                      "Unmetered transformer"))
    nodes["share_pct"] = (100 * nodes.share_physics).round(0)
    nodes["stolen"] = nodes.stolen_physics_mwh.round(2)

    lines = alt.Chart(edges).mark_rule(color=GREY, strokeWidth=2).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), x2="x2:Q", y2="y2:Q")
    tip = [alt.Tooltip("name:N", title="Bus"), alt.Tooltip("kind:N", title="Type"),
           alt.Tooltip("kva:Q", title="Transformer kVA"),
           alt.Tooltip("billed_mwh:Q", title="Billed (MWh)", format=".2f"),
           alt.Tooltip("stolen:Q", title="Estimated stolen (MWh)"),
           alt.Tooltip("share_pct:Q", title="Estimated stolen share (%)")]
    if show_truth:
        nodes["true_pct"] = (100 * nodes.true_share).round(0)
        tip.append(alt.Tooltip("true_pct:Q", title="True stolen share (%)"))
    load = nodes[nodes.bus != "bus_0"]
    pts = alt.Chart(load).mark_point(filled=True, opacity=1).encode(
        x="x:Q", y="y:Q",
        color=alt.Color("share_pct:Q", title="Estimated stolen share (%)",
                        scale=alt.Scale(scheme="oranges", domain=[0, 80])),
        size=alt.Size("kva:Q", title="Transformer kVA", scale=alt.Scale(range=[80, 700])),
        shape=alt.Shape("kind:N", title="Type",
                        scale=alt.Scale(domain=["Unmetered transformer", "Metered transformer"],
                                        range=["circle", "square"])),
        stroke=alt.value("#333"), strokeWidth=alt.value(0.6), tooltip=tip)
    sub = alt.Chart(nodes[nodes.bus == "bus_0"]).mark_point(
        shape="triangle-up", size=400, filled=True, color="#1d2327").encode(
        x="x:Q", y="y:Q", tooltip=[alt.Tooltip("kind:N", title="")])
    text = alt.Chart(load).mark_text(dy=-16, fontSize=10, color="#555").encode(
        x="x:Q", y="y:Q", text="name:N")
    layers = [lines, pts, sub, text]
    fsec = load[load.in_fault_section.fillna(False).astype(bool)]
    if len(fsec):
        layers.append(alt.Chart(fsec).mark_point(shape="diamond", size=520, filled=False,
                                                 color="#1d4ed8", strokeWidth=2.5).encode(
            x="x:Q", y="y:Q"))
    if show_truth:
        ring = alt.Chart(load[load.true_hotspot.fillna(False).astype(bool)]).mark_point(
            shape="circle", size=900, filled=False, color="#1e6b52", strokeWidth=2).encode(
            x="x:Q", y="y:Q")
        layers.insert(1, ring)
    return alt.layer(*layers).properties(height=360).configure_view(strokeWidth=0)


def inspection_table(res: pd.DataFrame, names: dict | None, show_truth: bool) -> pd.DataFrame:
    t = pd.DataFrame({
        "Rank": res["rank"],
        "Bus": [label(b, names) for b in res.bus],
        "kVA": res.kva.round(0).astype(int),
        "Meter": np.where(res.metered, "yes", ""),
        "Billed (MWh)": res.billed_mwh.round(2),
        "Est. stolen (MWh)": res.stolen_physics_mwh.round(2),
        "Est. stolen share": (100 * res.share_physics).round(0).astype(int).astype(str) + "%",
    })
    # Only when a cable fault was detected: chance that the leak is at each bus.
    if res.attrs.get("fault", {}).get("detected"):
        t["Fault probability"] = np.where(res.fault_prob >= 0.005,
                                          (100 * res.fault_prob).round(0).astype(int).astype(str) + "%",
                                          "<1%")
    if show_truth:
        t["True stolen (MWh)"] = res.true_stolen_mwh.round(2)
        t["Truth"] = np.where(res.true_hotspot, "theft",
                              np.where(res.true_fault, "fault", "clean"))
    return t


def fault_alert(d, res, names, show_truth):
    f = res.attrs.get("fault", {})
    if f.get("detected"):
        h = f["start_hour"]
        sec = ", ".join(label(b, names) for b in f["likely_section"])
        st.warning(f"**Possible cable fault.** A constant resistive leak of about "
                   f"**{f['leak_kw']:.0f} kW** appeared at **hour {h}** (day {h // 24 + 1}, "
                   f"{h % 24:02d}:00) and stayed. Unlike theft, it draws no reactive power. "
                   f"Most likely section (80% confidence): buses **{sec}**. "
                   "Marked with blue diamonds on the map.", icon="⚠️")
    else:
        st.info("No new cable fault detected in this period (no lasting step in resistive load "
                "at the substation). Fault detection works best with a week or more of data; "
                "small faults in short periods can be missed.", icon="✅")
    if show_truth and "true_fault" in res:
        tf = res[res.true_fault]
        if len(tf):
            j = d["bus_names"].tolist().index(tf.bus.iloc[0])
            fp = d["fault_bus_p"][0, :, j]
            st.caption(f"Truth (simulation only): a {1000 * fp.max():.0f} kW fault at bus "
                       f"{label(tf.bus.iloc[0], names)} starting at hour {int(np.argmax(fp > 0))}.")
        else:
            st.caption("Truth (simulation only): no fault in this period.")


def meter_section(d, meters, names, show_truth, key):
    v = d["vending"]
    n_days = d["head"].shape[1] // 24
    st.subheader("Prepaid meters to check")
    st.caption("Each prepaid meter's recent token purchases (last 30 days) are compared with what "
               "it bought before and with similar meters on the same transformer. The shortfall "
               "is weighted by the estimated stolen share of its transformer area, so low buyers "
               "where the feeder physics says energy is going missing come first. Illegal "
               "connections with no meter cannot appear here: they are found by inspecting the "
               "top transformer areas above.")
    tok = v["tokens"]
    recent = tok[tok.day >= n_days - 30]
    c = st.columns(4)
    c[0].metric("Prepaid meters", f"{len(v['customers']):,}")
    c[1].metric("Purchases in the last 30 days", f"{len(recent):,}")
    c[2].metric("Meters with no purchase in 30 days",
                f"{int((meters.recent_kwh_day <= 0).sum()):,}")
    if show_truth and "is_bypass" in meters:
        top = meters.head(100)
        c[3].metric("Top 100 that really bypass their meter", f"{100 * top.is_bypass.mean():.0f}%",
                    f"vs {100 * meters.is_bypass.mean():.0f}% of all meters", delta_color="off")
    t = pd.DataFrame({
        "Rank": meters["rank"], "Meter": meters.meter,
        "Bus": [label(b, names) for b in meters.bus], "Kind": meters.kind,
        "Bought, last 30 days (kWh/day)": meters.recent_kwh_day.round(1),
        "Bought before (kWh/day)": meters.earlier_kwh_day.round(1),
        "Typical on this transformer (kWh/day)": meters.peer_kwh_day.round(1),
        "Days since last purchase": meters.days_since_buy,
        "Shortfall (kWh/day)": meters.missing_kwh_day.round(1),
        "Area stolen share": (100 * meters.area_stolen_share).round(0).astype(int).astype(str) + "%",
    })
    if show_truth and "is_bypass" in meters:
        t["Truth"] = np.where(meters.is_bypass, "bypass", np.where(meters.is_vacant, "vacant", "normal"))
        t["Listed under wrong transformer"] = np.where(meters.bus != meters.true_bus, "yes", "")
    st.dataframe(t.head(200), hide_index=True, width="stretch", height=380, key=f"{key}_mtbl")
    st.download_button("Download full meter list (CSV)", t.to_csv(index=False).encode(),
                       "prepaid_meter_list.csv", "text/csv", key=f"{key}_mdl")


def large_section(d, large, names, show_truth, key):
    st.subheader("Large customers on smart meters")
    st.caption("Each large customer's hourly smart-meter readings are checked for a lasting "
               "drop in daily energy (beyond normal day-to-day variation) and for a peak demand "
               "unusually low for its contracted capacity, which is how a meter that records "
               "only part of the load looks. Customers in transformer areas where the feeder "
               "physics finds missing energy come first. A drop can also be genuine (a business "
               "cutting back), so this is a list of meters to test, not of offenders.")
    c = st.columns(4)
    c[0].metric("Large customers on smart meters", f"{len(large)}")
    c[1].metric("With a lasting drop in use", f"{int((large.drop_pct > 0).sum())}")
    c[2].metric("Peak below half the typical share of contract",
                f"{int((large.use_of_contract_pct < 0.5 * large.use_of_contract_pct.median()).sum())}")
    if show_truth and "tamper" in large:
        k = int(large.tamper.sum())
        if k:
            c[3].metric(f"Tampered meters in the top {k}", f"{int(large.head(k).tamper.sum())} of {k}")
        else:
            c[3].metric("Tampered meters", "none in this case")
    t = pd.DataFrame({
        "Rank": large["rank"], "Meter": large.meter, "Bus": [label(b, names) for b in large.bus],
        "Type": large["type"] if "type" in large else "",
        "Contract (kVA)": large.contract_kva.round(0).astype(int),
        "Daily use, last 7 days (kWh)": large.daily_kwh_recent.round(0),
        "Lasting drop": np.where(large.drop_pct > 0, large.drop_pct.astype(int).astype(str) + "%", ""),
        "Drop from day": large.drop_day.map(lambda v: "" if pd.isna(v) else f"{int(v)}"),
        "Peak / contract": large.use_of_contract_pct.astype(int).astype(str) + "%",
        "Area stolen share": (100 * large.area_stolen_share).round(0).astype(int).astype(str) + "%",
    })
    if show_truth and "tamper" in large:
        t["Truth"] = np.where(large.tamper, large.tamper_kind.fillna(""), "honest")
        t["Tampering since day"] = large.onset_hour.map(
            lambda v: "" if pd.isna(v) else ("before the records" if v < -28 * 24 else f"{int(v // 24)}"))
    st.dataframe(t, hide_index=True, width="stretch", height=320, key=f"{key}_ltbl")
    st.download_button("Download large-customer list (CSV)", t.to_csv(index=False).encode(),
                       "large_customer_list.csv", "text/csv", key=f"{key}_ldl")
    # daily energy of the top five, to see the drops
    a = d["ami"]
    r = a["readings"]
    T = d["head"].shape[1]
    keep = r["hours"] < T
    p, hrs = r["p_kw"][keep], r["hours"][keep]
    top = list(large.meter.head(5))
    cols = [list(a["customers"].meter).index(m) for m in top]
    day = np.floor(hrs / 24).astype(int)
    on = ~(np.nan_to_num(p) <= 0).all(axis=1)
    df = pd.DataFrame(np.nan_to_num(p[:, cols]), columns=top)
    df["day"], df["on"] = day, on
    g = df.groupby("day")
    daily = g[top].sum().div(g["on"].sum().clip(lower=1), axis=0) * 24
    daily = daily[g["on"].sum() >= 6].reset_index().melt("day", var_name="meter", value_name="kWh")
    st.altair_chart(alt.Chart(daily).mark_line(point=True).encode(
        x=alt.X("day:Q", title="Day (0 = first day of the period; earlier = smart-meter history)",
                scale=alt.Scale(domain=[int(daily.day.min()), int(daily.day.max())], nice=False)),
        y=alt.Y("kWh:Q", title="Daily energy recorded (kWh)"),
        color=alt.Color("meter:N", title="Top 5 meters")).properties(height=260),
        width="stretch", key=f"{key}_lchart")


def results_view(d, res, names, show_truth, key, meters=None, large=None):
    s = feeder_summary(d, res)
    c = st.columns(4)
    sup = (f"{s['days']} days" if s["hours_on"] == s["hours"]
           else f"{s['days']} days, supplied {s['hours_on']} of {s['hours']} h")
    c[0].metric("Energy into feeder", f"{s['head_mwh']:.0f} MWh", sup, delta_color="off")
    c[1].metric("Unbilled energy", f"{s['unbilled_pct']:.0f}%",
                f"{s['head_mwh'] - s['billed_mwh']:.0f} MWh", delta_color="off")
    if show_truth:
        err = 100 * (s["stolen_physics"] - s["stolen_true"]) / s["stolen_true"]
        c[2].metric("Estimated stolen", f"{s['stolen_physics']:.1f} MWh",
                    f"{err:+.1f}% vs true {s['stolen_true']:.1f}", delta_color="off")
        c[3].metric("Inspection hit rate",
                    f"{100 * s['topk_physics']:.0f}%",
                    f"top {s['n_hotspots']} buses visited", delta_color="off")
    else:
        c[2].metric("Estimated stolen", f"{s['stolen_physics']:.1f} MWh",
                    f"{100 * s['stolen_physics'] / s['head_mwh']:.0f}% of energy in", delta_color="off")
        c[3].metric("Transformer meters", f"{s['n_meters']} of {s['n_buses']}")

    fault_alert(d, res, names, show_truth)

    st.subheader("Feeder map")
    cap = ("Colour = estimated share of consumption stolen. Size = transformer kVA. "
           "Squares have a transformer meter. Triangle = substation. "
           "Blue diamonds = likely section of a detected cable fault.")
    if show_truth:
        cap += " Green rings = true theft hotspots (known only in simulation)."
    st.caption(cap)
    st.altair_chart(feeder_map(d, res, names, show_truth), width="stretch", key=f"{key}_map")

    st.subheader("Inspection list")
    st.caption("Transformer areas ranked by estimated stolen energy. Send inspection crews "
               "down the list. 'Fault probability' shows where a detected cable fault most "
               "likely is (for the maintenance team).")
    table = inspection_table(res, names, show_truth)
    st.dataframe(table, hide_index=True, width="stretch", height=380, key=f"{key}_tbl")
    st.download_button("Download inspection list (CSV)", table.to_csv(index=False).encode(),
                       "inspection_list.csv", "text/csv", key=f"{key}_dl")
    if meters is not None:
        meter_section(d, meters, names, show_truth, key)
    if large is not None:
        large_section(d, large, names, show_truth, key)

    st.subheader("Inspection plan and money at stake")
    c1, c2 = st.columns([2, 1])
    tariff = c2.number_input("Revenue per kWh sold (US$)", 0.0, 2.0, 0.20, 0.01, key=f"{key}_tariff",
                             help="Example value. Enter your own average tariff in US dollars.")
    cur = c2.selectbox("Show amounts in", list(CURRENCIES), key=f"{key}_cur",
                       format_func=lambda c: f"{c} - {CURRENCIES[c]}")
    rate = 1.0
    if cur != "USD":
        rates, updated = usd_rates()
        live = rates.get(cur) if rates else None
        rate = c2.number_input(
            f"Exchange rate: 1 US$ = ? {cur}", min_value=0.0,
            value=float(live) if live else None, format="%.4f", key=f"{key}_rate_{cur}",
            help=(f"Live rate from open.er-api.com, updated {updated}. You can type your own."
                  if live else "Live rates could not be loaded. Type the current rate."))
    days = d["head"].shape[1] / 24
    per_month = 30 / days
    stolen_month = res.stolen_physics_mwh.sum() * per_month
    usd_month = stolen_month * 1000 * tariff
    c2.metric("Estimated stolen per month", f"{stolen_month:,.0f} MWh")
    if cur == "USD":
        c2.metric("Estimated revenue lost per month", f"US$ {usd_month:,.0f}")
    elif rate:
        c2.metric("Estimated revenue lost per month", f"{cur} {usd_month * rate:,.0f}",
                  f"US$ {usd_month:,.0f}", delta_color="off")
    else:
        c2.metric("Estimated revenue lost per month", f"US$ {usd_month:,.0f}")
        c2.caption(f"Enter an exchange rate to see the amount in {cur}.")
    k_half = int(np.searchsorted(res.stolen_physics_mwh.cumsum() / res.stolen_physics_mwh.sum(), 0.5) + 1)
    c2.caption(f"Inspecting the top **{k_half}** of {len(res)} transformer areas targets about half "
               "of the estimated stolen energy.")
    plan = pd.DataFrame({"inspected": np.arange(1, len(res) + 1),
                         "Estimated": 100 * res.stolen_physics_mwh.cumsum() / res.stolen_physics_mwh.sum()})
    if show_truth:
        plan["Actually found (simulation)"] = 100 * res.true_stolen_mwh.cumsum() / res.true_stolen_mwh.sum()
        plan["Random order"] = 100 * plan.inspected / len(res)
    plan = plan.melt("inspected", var_name="series", value_name="pct")
    c1.altair_chart(alt.Chart(plan).mark_line(point=True).encode(
        x=alt.X("inspected:Q", title="Transformer areas inspected, in ranked order"),
        y=alt.Y("pct:Q", title="Share of stolen energy covered (%)",
                scale=alt.Scale(domain=[-3, 103], nice=False),
                axis=alt.Axis(values=[0, 20, 40, 60, 80, 100])),
        color=alt.Color("series:N", title="", scale=alt.Scale(
            domain=["Estimated", "Actually found (simulation)", "Random order"],
            range=[ORANGE, "#1e6b52", GREY]))).properties(height=260),
        width="stretch", key=f"{key}_plan")

    with st.expander("Compare methods for each transformer area"):
        long = pd.concat([
            pd.DataFrame({"bus": [label(b, names) for b in res.bus], "MWh": res[col], "method": m})
            for col, m in [("stolen_audit_mwh", "Energy audit"),
                           ("stolen_rules_mwh", "Combined rules"),
                           ("stolen_physics_mwh", "Physics-informed")]
            + ([("true_stolen_mwh", "True (simulation)")] if show_truth else [])])
        order = [label(b, names) for b in res.bus]
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("bus:N", sort=order, title="Bus (ranked by physics-informed estimate)"),
            xOffset="method:N",
            y=alt.Y("MWh:Q", title="Stolen energy over the period (MWh)"),
            color=alt.Color("method:N", scale=alt.Scale(
                domain=["Energy audit", "Combined rules", "Physics-informed", "True (simulation)"],
                range=["#b8c0c7", "#6b7f8f", ORANGE, "#1e6b52"])),
            tooltip=["bus", "method", alt.Tooltip("MWh:Q", format=".2f")]).properties(height=300)
        st.altair_chart(chart, width="stretch", key=f"{key}_cmp")

    with st.expander("Substation meter versus billing, hour by hour"):
        T = d["head"].shape[1]
        n_b = d["billing"].shape[1]
        head_p = d["head"][0, :, 0]
        if n_b == T // 24 and n_b > 1:
            shape = head_p.reshape(n_b, 24)
            billed_h = (shape / shape.sum(axis=1, keepdims=True)
                        * d["billing"][0].sum(axis=1)[:, None]).ravel()
            bl = "Billed (spread over each day)"
        else:
            billed_h = head_p / head_p.sum() * d["billing"][0].sum()
            bl = "Billed (period total spread over the hours)"
        ts = pd.DataFrame({"hour": np.arange(T), "Substation meter": head_p, bl: billed_h})
        if show_truth:
            ts["True legitimate use"] = (d["true_bus_p"][0] - d["theft_bus_p"][0]
                                         - d["fault_bus_p"][0]).sum(axis=1)
        ts = ts.melt("hour", var_name="series", value_name="MW")
        st.altair_chart(alt.Chart(ts).mark_line().encode(
            x=alt.X("hour:Q", title="Hour"), y=alt.Y("MW:Q", title="MW"),
            color=alt.Color("series:N", title="")).properties(height=280),
            width="stretch", key=f"{key}_ts")


def money_inputs(col, key):
    """Tariff and currency inputs; returns (tariff in US$/kWh, currency code, rate per US$)."""
    tariff = col.number_input("Revenue per kWh sold (US$)", 0.0, 2.0, 0.20, 0.01, key=f"{key}_tariff",
                              help="Example value. Enter your own average tariff in US dollars.")
    cur = col.selectbox("Show amounts in", list(CURRENCIES), key=f"{key}_cur",
                        format_func=lambda c: f"{c} - {CURRENCIES[c]}")
    rate = 1.0
    if cur != "USD":
        rates, updated = usd_rates()
        live = rates.get(cur) if rates else None
        rate = col.number_input(f"Exchange rate: 1 US$ = ? {cur}", min_value=0.0,
                                value=float(live) if live else None, format="%.4f",
                                key=f"{key}_rate_{cur}")
        if not rate:
            cur, rate = "USD", 1.0
    return tariff, cur, rate


def district_view(feeders, names, show_truth, key, extras=None):
    """Ranking of several feeders, a district-wide crew plan, and each feeder in detail.

    feeders: {name: (d, res)}; names: {name: bus-name map or None}."""
    tab = district_table(feeders)
    c1, c2 = st.columns([3, 1])
    tariff, cur, rate = money_inputs(c2, key)
    money = lambda mwh: mwh * 1000 * tariff * rate  # noqa: E731
    sym = "US$" if cur == "USD" else cur
    m = c1.columns(4)
    m[0].metric("Feeders", len(tab))
    m[1].metric("Energy in per month", f"{tab.energy_in_month.sum():,.0f} MWh")
    m[2].metric("Estimated stolen per month", f"{tab.stolen_month.sum():,.0f} MWh",
                f"{100 * tab.stolen_month.sum() / tab.energy_in_month.sum():.0f}% of energy in",
                delta_color="off")
    m[3].metric("Revenue lost per month", f"{sym} {money(tab.stolen_month.sum()):,.0f}")

    c1.subheader("Feeders ranked by estimated theft")
    c1.caption("Estimated stolen energy and revenue per month, from the physics-informed "
               "locator on each feeder. Unbilled energy is shown for comparison: it also "
               "contains technical losses and cable-fault leakage, so it can rank feeders "
               "wrongly.")
    t = pd.DataFrame({
        "Rank": tab["rank"], "Feeder": tab.feeder,
        "Stolen per month (MWh)": tab.stolen_month.round(0).astype(int),
        f"Revenue lost per month ({sym})": tab.stolen_month.map(lambda v: f"{money(v):,.0f}"),
        "Stolen share of energy in": tab.stolen_pct.round(0).astype(int).astype(str) + "%",
        "Unbilled energy": tab.unbilled_pct.round(0).astype(int).astype(str) + "%",
        "Theft in top 5 areas": tab.top5_share.round(0).astype(int).astype(str) + "%",
        "Transformer meters": tab.meters.astype(str) + " of " + tab.areas.astype(str),
        "Supply": np.where(tab.hours_on < tab.hours,
                           (100 * tab.hours_on / tab.hours).round(0).astype(int).astype(str) + "% of hours",
                           "all day"),
        "Cable fault": np.where(tab.fault, tab.fault_kw.round(0).astype(int).astype(str) + " kW leak", ""),
    })
    if show_truth and "true_stolen_month" in tab:
        t["True stolen per month (MWh)"] = tab.true_stolen_month.round(0).astype(int)
        t["True rank"] = tab.true_rank
        t["True fault"] = np.where(tab.true_fault, "yes", "")
    st.dataframe(t, hide_index=True, width="stretch", key=f"{key}_rank")
    bars = pd.DataFrame({"Feeder": tab.feeder, "MWh": tab.stolen_month, "series": "Estimated"})
    if show_truth and "true_stolen_month" in tab:
        bars = pd.concat([bars, pd.DataFrame({"Feeder": tab.feeder, "MWh": tab.true_stolen_month,
                                              "series": "True (simulation)"})])
    st.altair_chart(alt.Chart(bars).mark_bar().encode(
        x=alt.X("Feeder:N", sort=list(tab.feeder), title=None),
        xOffset="series:N",
        y=alt.Y("MWh:Q", title="Stolen energy per month (MWh)"),
        color=alt.Color("series:N", title="", scale=alt.Scale(
            domain=["Estimated", "True (simulation)"], range=[ORANGE, "#1e6b52"])),
        tooltip=["Feeder", "series", alt.Tooltip("MWh:Q", format=",.0f")]).properties(height=260),
        width="stretch", key=f"{key}_bars")

    st.subheader("Where to send a limited number of crews")
    n_total = int(tab.areas.sum())
    n = st.slider("Transformer areas your crews can inspect", 1, n_total,
                  max(1, round(0.1 * n_total)), key=f"{key}_budget",
                  help="The plan picks the areas with the highest estimated theft across all "
                       "feeders, instead of giving every feeder the same number of crews.")
    plan = district_plan(feeders, n)
    share = 100 * plan.stolen_month.sum() / max(plan.attrs["total_stolen_month"], 1e-12)
    m = st.columns(3)
    m[0].metric("Estimated theft in these areas", f"{plan.stolen_month.sum():,.0f} MWh/month",
                f"{share:.0f}% of the district's", delta_color="off")
    m[1].metric("Revenue at stake", f"{sym} {money(plan.stolen_month.sum()):,.0f} per month")
    if show_truth and "true_stolen_month" in plan:
        got = 100 * plan.true_stolen_month.sum() / plan.attrs["true_total_month"]
        per = n // len(tab)
        eq = 0.0
        for name, (d, res) in feeders.items():
            pm = 30.0 / (d["head"].shape[1] / 24)
            eq += res.head(per).true_stolen_mwh.sum() * pm
        eq = 100 * eq / plan.attrs["true_total_month"]
        m[2].metric("True theft reached (simulation)", f"{got:.0f}%",
                    f"equal crews per feeder: {eq:.0f}%" if per else "", delta_color="off")
    by_f = plan.feeder.value_counts().reindex(tab.feeder).fillna(0).astype(int)
    st.caption("Areas per feeder in this plan: " + ", ".join(f"{f} {k}" for f, k in by_f.items()))
    pt = pd.DataFrame({
        "Order": plan.order, "Feeder": plan.feeder,
        "Bus": [label(b, names.get(f)) for f, b in zip(plan.feeder, plan.bus)],
        "Estimated stolen (MWh/month)": plan.stolen_month.round(1),
        f"Revenue (per month, {sym})": plan.stolen_month.map(lambda v: f"{money(v):,.0f}"),
        "Stolen share": (100 * plan.share).round(0).astype(int).astype(str) + "%"})
    if show_truth and "true_hotspot" in plan:
        pt["True theft (MWh/month)"] = plan.true_stolen_month.round(1)
    st.dataframe(pt, hide_index=True, width="stretch", height=320, key=f"{key}_plan")
    c = st.columns(2)
    c[0].download_button("Download feeder ranking (CSV)", t.to_csv(index=False).encode(),
                         "feeder_ranking.csv", "text/csv", key=f"{key}_dl1")
    c[1].download_button("Download crew plan (CSV)", pt.to_csv(index=False).encode(),
                         "crew_plan.csv", "text/csv", key=f"{key}_dl2")

    st.subheader("One feeder in detail")
    pick = st.selectbox("Feeder", list(tab.feeder), key=f"{key}_pick")
    d, res = feeders[pick]
    ex = (extras or {}).get(pick, (None, None))
    results_view(d, res, names.get(pick), show_truth, key=f"{key}_{list(feeders).index(pick)}",
                 meters=ex[0], large=ex[1])


def inspection_report(log, areas, cfg, show_truth, key):
    """What the inspections so far show, honestly, and the next round's probabilities."""
    ev = evaluate_log(log, n_areas=len(areas), sensitivity=cfg.sensitivity)
    st.subheader("What the inspections show")
    m = st.columns(4)
    m[0].metric("Inspections logged", ev["inspections"])
    m[1].metric("Valid outcomes", ev["inspections"] - ev["missing"])
    m[2].metric("No access / not found / aborted", ev["missing"],
                f"{ev['refused']} refused", delta_color="off")
    m[3].metric("Cable or network faults found", ev["network_faults"])
    rows = []
    names = {"model": "Model (top of the list)", "random": "Random audit",
             "tip-off": "Tip-off", "routine": "Routine", "follow-up": "Follow-up"}
    for arm, a in ev["arms"].items():
        rows.append({"Selection": names.get(arm, arm), "Valid visits": a["n"],
                     "Non-technical loss found": a["positives"],
                     "Hit rate": f"{100 * a['hit_rate']:.0f}%",
                     "95% interval": f"{100 * a['lo']:.0f}% to {100 * a['hi']:.0f}%",
                     "kWh recovered (back-billed)": f"{a['kwh_recovered']:,.0f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", key=f"{key}_arms")
    if ev["arms"]:
        bars = pd.DataFrame([{"arm": names.get(k, k), "rate": 100 * a["hit_rate"],
                              "lo": 100 * a["lo"], "hi": 100 * a["hi"]} for k, a in ev["arms"].items()])
        base = alt.Chart(bars).encode(y=alt.Y("arm:N", title=None))
        st.altair_chart((base.mark_bar(color=ORANGE, opacity=0.8).encode(
            x=alt.X("rate:Q", title="Hit rate with 95% interval (%)", scale=alt.Scale(domain=[0, 100])))
            + base.mark_rule(strokeWidth=2).encode(x="lo:Q", x2="hi:Q")).properties(height=40 + 40 * len(bars)),
            width="stretch", key=f"{key}_armchart")
    if "evidence" in ev["verdict"] or "no clear" in ev["verdict"]:
        st.info(f"**Verdict: {ev['verdict']}.** Until then, the model's hit rate is not proof "
                "that it beats random visits.")
    else:
        st.success(f"**Verdict: {ev['verdict']}.**")
    if ev.get("lift"):
        L = ev["lift"]
        st.write(f"Lift (model hit rate / random hit rate): **{L['value']:.1f}x**, "
                 f"95% interval {L['lo']:.1f}x to {L['hi']:.1f}x.")
    if "ht_theft_share" in ev:
        lo, hi = ev["ht_theft_interval"]
        st.write(f"Share of all transformer areas with non-technical loss, estimated from every "
                 f"valid visit weighted by its selection probability: about "
                 f"**{100 * ev['ht_theft_share']:.0f}%** (95% interval {100 * lo:.0f}% to "
                 f"{100 * hi:.0f}%), allowing for theft that crews miss (crew sensitivity "
                 f"{cfg.sensitivity:.0%}, an assumption). The interval is wide while there are few "
                 f"random-audit visits; it narrows round by round.")
        if show_truth and "true_theft" in areas:
            st.caption(f"Truth (simulation only): {100 * areas.true_theft.mean():.0f}% of areas have theft.")
    post = area_posterior(areas[["feeder", "bus", "model_score", "weight"]], log, cfg)
    cal = post.attrs["calibration"]
    st.caption(("Probabilities below are calibrated on the logged outcomes "
                f"({cal['n']} valid visits): P(theft) = 1 / (1 + exp(-({cal['a']:.2f} + {cal['b']:.2f} x estimated stolen share))).")
               if cal["fitted"] else
               f"Probabilities below use the default map until {cfg.calibrate_after} valid "
               f"outcomes are logged (now {cal['n']}).")
    st.caption("These numbers do not show how much theft was missed (recall cannot be measured "
               "from inspections), a CLEAN result is not proof of no theft, and no-access visits "
               "are left out of every rate. A finding is the crew's record with its evidence, not "
               "the model's ranking.")
    return post


def inspection_planner(areas, post, cfg, key, round_id, show_truth, log=None):
    budget = st.slider("Transformer areas to visit in the next round", 5, min(200, len(areas)),
                       min(30, len(areas)), key=f"{key}_budget")
    base = post if post is not None else areas
    a = base[["feeder", "bus", "model_score"]].copy()
    a["weight"] = base["next_weight"].values if (post is not None and st.session_state.get(f"{key}_use")) \
        else areas["weight"].values
    recent = recently_visited(log, cfg.cooldown_rounds)
    if recent:
        keep = [(f, b) not in recent for f, b in zip(a.feeder, a.bus)]
        st.caption(f"{len(keep) - sum(keep)} areas visited in the last {cfg.cooldown_rounds} rounds "
                   "are left out of this round (they were just dealt with); their chance is 0.")
        a = a[keep]
    plan = plan_round(a, budget, cfg, seed=zlib.crc32(round_id.encode()) % 10_000,
                      round_id=round_id)
    if post is not None:
        plan = plan.merge(post[["feeder", "bus", "p_post", "visits", "positives"]], on=["feeder", "bus"], how="left")
    show = pd.DataFrame({
        "Inspection": plan.inspection_id, "Feeder": plan.feeder,
        "Area (bus)": plan.bus.str.replace("bus_", ""), "Selection": plan.selection_arm,
        "Chance of selection": (100 * plan.pi).round(1).astype(str) + "%",
        "Estimated stolen (MWh/month)": plan.weight.round(1)})
    if "p_post" in plan:
        show["P(theft) now"] = (100 * plan.p_post).round(0).astype(int).astype(str) + "%"
        show["Earlier visits"] = plan.visits.fillna(0).astype(int)
    st.dataframe(show, hide_index=True, width="stretch", height=300, key=f"{key}_plan")
    cols = [c for c in ["inspection_id", "round_id", "feeder", "bus", "selection_arm", "pi",
                        "model_score", "model_rank", "model_version", "date", "crew", "premises",
                        "premises_checked", "outcome", "kwh_per_day_found", "kwh_recovered",
                        "amount_billed", "amount_collected", "notes"] if c in plan]
    st.download_button("Download the work list for crews (CSV)",
                       plan[cols].to_csv(index=False).encode(), f"work_list_{round_id}.csv",
                       "text/csv", key=f"{key}_wl")
    st.caption("Crews fill in date, crew, premises, premises_checked, outcome (one of the codes "
               "below), kwh_per_day_found and kwh_recovered, then upload the file back here. Keep "
               "the pi and selection_arm columns as they are: they make the results measurable.")
    return plan


# ----------------------------------------------------------------------------- layout
st.title("⚡ Feeder Loss Locator")
st.write("Find **where** electricity is being lost on an 11 kV distribution feeder, using the "
         "substation meter, a few transformer meters, billing records and the transformer "
         "register. No customer smart meters and no records of past theft are needed.")

tab_sim, tab_own, tab_dist, tab_insp, tab_study, tab_how = st.tabs(
    ["Simulate a feeder", "Analyse your own data", "Compare feeders", "Inspection rounds",
     "Study results", "How it works"])

# --- 1. Simulation ------------------------------------------------------------------
with tab_sim:
    st.write("A Freetown-style 11 kV feeder: 32 transformer areas, about 12,000 households, "
             "real Freetown cable data. Choose a situation, and the app simulates the feeder "
             "hour by hour, then tries to find the theft without being told where it is.")
    with st.form("sim"):
        c1, c2, c3 = st.columns(3)
        theft = c1.slider("Share of energy stolen", 0.05, 0.45, 0.25, 0.05, format="%.2f",
                          help="Sierra Leone's commercial losses are reported at 18-40%.")
        days = c1.select_slider("Days of data", [1, 2, 3, 5, 7], value=3)
        n_meters = c2.slider("Transformer meters installed", 0, 16, 6)
        strategy = c2.selectbox("Where meters are placed", ["ends", "largest", "random"],
                                format_func={"ends": "Far ends of branches",
                                             "largest": "Largest transformers",
                                             "random": "Random"}.get)
        billing = c2.selectbox("Billing data available",
                               ["daily", "period", "prepaid"],
                               format_func={"daily": "Daily per transformer area",
                                            "period": "One total for the period (monthly bills)",
                                            "prepaid": "Prepaid token sales per meter (90 days)"}.get,
                               help="Prepaid: every customer's token purchases are simulated, "
                                    "and sales per transformer area come from the customer "
                                    "list. Individual meters are then ranked for inspection.")
        mapping = c2.slider("Meters listed under the wrong transformer (prepaid)", 0.0, 0.3, 0.05,
                            0.05, format="%.2f",
                            help="Share of prepaid customers that the customer database links "
                                 "to the wrong transformer. Only used with prepaid token sales.")
        bill_noise = c1.slider("Billing error", 0.0, 0.3, 0.03, 0.01, format="%.2f",
                               help="How far each billed value is from what was really used. "
                                    "Prepaid token sales can be 10-20% off, because people buy "
                                    "tokens when they have money, not when they use power.")
        dropout = c3.slider("Meter readings missing", 0.0, 0.8, 0.2, 0.1, format="%.1f")
        shedding = c3.selectbox("Load shedding", list(SHEDDING_PATTERNS),
                                format_func=SHEDDING_PATTERNS.get,
                                help="Feeder switched off on a schedule. 'Rotation' is the 2008 "
                                     "load-sharing programme for domestic feeders (JICA report). "
                                     "Demand surges for two hours when supply returns.")
        fault = c3.checkbox("Include a cable fault", value=False)
        ami = c3.checkbox("Large customers on smart meters (AMI)", value=False,
                          help="The 36 large customers at the three commercial transformer "
                               "areas have smart meters with 28 days of hourly history. Some "
                               "tamper with their meters where there is theft.")
        seed = c3.number_input("Random seed (changes the neighbourhoods and theft spots)",
                               0, 9999, 7)
        go = st.form_submit_button("Run simulation and locate losses", type="primary")
    if go or "sim_done" in st.session_state:
        st.session_state["sim_done"] = True
        with st.spinner("Simulating the feeder and locating losses..."):
            d, res, meters, large = run_simulation(theft, n_meters, strategy, dropout, fault, days,
                                                   int(seed), billing, bill_noise, shedding,
                                                   mapping, ami)
        results_view(d, res, None, show_truth=True, key="sim", meters=meters, large=large)
        with st.expander("Download this simulated feeder as utility CSV files"):
            st.write("These are the same files a utility would upload in the next tab "
                     "(with prepaid sales: customers.csv and tokens.csv instead of billing.csv). "
                     "Useful as a filled-in example.")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for k, df in standard_to_frames(d).items():
                    z.writestr(f"{k}.csv", df.to_csv(index=False))
            st.download_button("Download feeder files (.zip)", buf.getvalue(),
                               "simulated_feeder.zip", "application/zip", key="sim_zip")

# --- 2. Own data ----------------------------------------------------------------------
with tab_own:
    st.write("Upload CSV files describing one feeder: four network and meter files, plus either "
             "billing per transformer area or, for prepaid customers, the customer list and "
             "token sales. Start from the example files and replace the numbers with your own.")
    with st.expander("What each file contains", expanded=False):
        st.markdown("""
| File | Columns | What it is |
|---|---|---|
| `lines.csv` | from_bus, to_bus, r_ohm, x_ohm | Each 11 kV line section and its resistance and reactance in ohms. The substation bus is the one that never appears in `to_bus`. Must be radial. |
| `transformers.csv` | bus, kva | Installed transformer capacity at each load bus (from the asset register). |
| `head.csv` | hour, p_mw, q_mvar, v_pu | Substation 11 kV breaker readings, whole days. `hour` can be 0,1,2… or a timestamp. Hours when the feeder was switched off (load shedding) can be left out or given as zero. |
| `meters.csv` | hour, bus, p_mw, q_mvar, v_pu | Any transformer check meters. Gaps are fine; can be empty (header only). |
| **Either** `billing.csv` | day, bus, mwh — or just bus, mwh | Energy sold per transformer area: either per day, or one total per bus for the whole period (monthly bills). |
| **or** `customers.csv` | meter, bus, (kind) | Prepaid customer list: the transformer area each meter is listed under. Optional `kind` (residential, commercial…). |
| *optional* `ami_customers.csv` | meter, bus, contract_kva, (type) | Large customers on smart meters: transformer area and contracted capacity from the customer register. |
| *optional* `ami_readings.csv` | meter, hour, p_kw, q_kvar, v_pu | Their hourly smart-meter readings, numbered like `head.csv` (history before the period has negative hours) or timestamps. |
| **and** `tokens.csv` | meter, day, kwh | Every token purchase from the vending system. `day` is a date, or a day number (0 = first day of `head.csv`, earlier history negative). A month or more of history before the period is best. |
""")
    ex_sets = {"billing": (EXAMPLE_DIR, FILES),
               "prepaid": (EXAMPLE_PREPAID_DIR, FILES[:4] + PREPAID_FILES
                           + [k for k in AMI_FILES if (EXAMPLE_PREPAID_DIR / f"{k}.csv").exists()])}
    c1, c2 = st.columns(2)
    for col, (kind, (folder, names_)) in zip((c1, c2), ex_sets.items()):
        paths = {k: folder / f"{k}.csv" for k in names_}
        if all(p.exists() for p in paths.values()):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for k, p in paths.items():
                    z.write(p, f"{k}.csv")
            col.download_button(
                "Download example files (.zip)" if kind == "billing"
                else "Download prepaid + smart-meter example files (.zip)",
                buf.getvalue(), f"example_feeder{'_prepaid' if kind == 'prepaid' else ''}.zip",
                "application/zip", key=f"ex_zip_{kind}")

    use_example = st.radio("Data", ["Upload my files", "Example: billing per area",
                                    "Example: prepaid sales + smart meters"], horizontal=True)
    uploads = {}
    if use_example == "Upload my files":
        cols = st.columns(4)
        for i, k in enumerate(FILES[:4]):
            f = cols[i].file_uploader(f"{k}.csv", type="csv", key=f"up_{k}")
            if f is not None:
                uploads[k] = f.getvalue()
        cols = st.columns(3)
        for c, k, lab in zip(cols, ["billing"] + PREPAID_FILES,
                             ["billing.csv", "or prepaid: customers.csv", "and tokens.csv"]):
            f = c.file_uploader(lab, type="csv", key=f"up_{k}")
            if f is not None:
                uploads[k] = f.getvalue()
        with st.expander("Optional: large customers on smart meters (AMI)"):
            cols = st.columns(2)
            for c, k in zip(cols, AMI_FILES):
                f = c.file_uploader(f"{k}.csv", type="csv", key=f"up_{k}")
                if f is not None:
                    uploads[k] = f.getvalue()
    else:
        folder, names_ = ex_sets["billing" if "billing" in use_example else "prepaid"]
        uploads = {k: (folder / f"{k}.csv").read_bytes() for k in names_}

    c1, c2 = st.columns(2)
    pf = c1.slider("Typical power factor of loads", 0.75, 1.0, 0.90, 0.01)
    tl = c2.slider("Expected technical losses on the 11 kV network", 0.0, 0.10, 0.03, 0.01,
                   format="%.2f")
    have_sales = "billing" in uploads or all(k in uploads for k in PREPAID_FILES)
    if all(k in uploads for k in FILES[:4]) and have_sales:
        if "billing" in uploads:
            uploads = {k: v for k, v in uploads.items() if k not in PREPAID_FILES}
        try:
            with st.spinner("Checking the files and locating losses..."):
                d, res, names, meters, large = run_uploaded(uploads, pf, tl)
            nb = d["billing"].shape[1]
            sales = ("prepaid token sales" if "vending" in d
                     else "daily billing" if nb > 1 else "one billing total per bus")
            if "ami" in d:
                sales += f", {len(d['ami']['customers'])} large customers on smart meters"
            st.success(f"Read a feeder with {len(res)} load buses, {int(res.metered.sum())} "
                       f"transformer meters and {d['head'].shape[1] // 24} days of data "
                       f"({sales}).")
            results_view(d, res, names, show_truth=False, key="own", meters=meters, large=large)
        except FeederDataError as e:
            st.error(f"There is a problem with the files: {e}")
    else:
        missing = [k + ".csv" for k in FILES[:4] if k not in uploads]
        if not have_sales:
            missing.append("billing.csv (or customers.csv + tokens.csv)")
        st.info(f"Waiting for files: {', '.join(missing)}")

with tab_dist:
    st.write("A utility runs many feeders. Rank them by estimated theft, and plan where a "
             "limited number of inspection crews should go across all of them.")
    mode = st.radio("Feeders", ["Simulate a district", "Upload several feeders (.zip)"],
                    horizontal=True, key="dist_mode")
    if mode == "Simulate a district":
        with st.form("dist_form"):
            c1, c2, c3 = st.columns(3)
            n_f = c1.slider("Number of feeders", 3, 10, 6)
            n_days = c2.slider("Days of data", 3, 7, 7)
            dseed = c3.number_input("Random seed", 0, 10_000, 3, step=1)
            st.caption("Each simulated feeder gets its own theft level (5-45%), number of "
                       "transformer meters (0-8), load shedding and, on some, a cable fault.")
            go = st.form_submit_button("Simulate and rank feeders", type="primary")
        if go:
            st.session_state["dist_args"] = (n_f, n_days, int(dseed))
        if "dist_args" in st.session_state:
            n_f, n_days, dseed = st.session_state["dist_args"]
            feeders = {}
            bar = st.progress(0.0, text="Simulating and analysing feeders...")
            for i, p in enumerate(district_params(n_f, dseed)):
                d, res, _, _ = run_simulation(p["theft"], p["n_meters"], "ends", 0.2, p["fault"],
                                              n_days, p["seed"], "daily", 0.03, p["shedding"],
                                              0.05, False)
                feeders[f"Feeder {i + 1}"] = (d, res)
                bar.progress((i + 1) / n_f, text=f"Analysed feeder {i + 1} of {n_f}")
            bar.empty()
            district_view(feeders, {}, show_truth=True, key="dsim")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for name, (d, _) in feeders.items():
                    for k, df in standard_to_frames(d).items():
                        z.writestr(f"{name.replace(' ', '_')}/{k}.csv", df.to_csv(index=False))
            st.download_button("Download these feeders as a district .zip (to try the upload)",
                               buf.getvalue(), "district_example.zip", "application/zip",
                               key="dsim_zip")
    else:
        st.markdown("Upload **one .zip** with **one folder per feeder**. Each folder holds the "
                    "same CSV files as in *Analyse your own data* (lines, transformers, head, "
                    "meters, and billing or prepaid customers + tokens; smart-meter files "
                    "optional). The folder name is used as the feeder's name. To see the layout, "
                    "simulate a district and download it as a .zip.")
        up = st.file_uploader("District .zip", type="zip", key="dist_zip")
        c1, c2 = st.columns(2)
        pf = c1.slider("Typical power factor of loads", 0.75, 1.0, 0.90, 0.01, key="dist_pf")
        tl = c2.slider("Expected technical losses on the 11 kV network", 0.0, 0.10, 0.03, 0.01,
                       format="%.2f", key="dist_tl")
        if up is not None:
            try:
                parts = split_district_zip(up.getvalue())
            except FeederDataError as e:
                st.error(str(e))
                parts = {}
            feeders, names, extras, problems = {}, {}, {}, []
            bar = st.progress(0.0, text="Analysing feeders...") if parts else None
            for i, (name, files) in enumerate(parts.items()):
                try:
                    if "billing" in files:
                        files = {k: v for k, v in files.items() if k not in PREPAID_FILES}
                    d, res, nm, meters, large = run_uploaded(files, pf, tl)
                    feeders[name], names[name], extras[name] = (d, res), nm, (meters, large)
                except FeederDataError as e:
                    problems.append(f"**{name}**: {e}")
                except Exception as e:  # noqa: BLE001 - report and carry on with the others
                    problems.append(f"**{name}**: could not be read ({type(e).__name__}).")
                bar.progress((i + 1) / len(parts), text=f"Analysed {i + 1} of {len(parts)} feeders")
            if bar:
                bar.empty()
            if problems:
                st.warning("Some feeders were skipped:\n\n" + "\n\n".join(problems))
            if feeders:
                st.success(f"Analysed {len(feeders)} feeder(s).")
                district_view(feeders, names, show_truth=False, key="dup", extras=extras)


with tab_insp:
    st.write("Plan inspection rounds, log what crews find, and measure honestly whether the "
             "targeting works. Each round sends most crews to the top of the list and about 15% "
             "to randomly chosen areas: the random visits are the yardstick that makes every other "
             "number credible. Every area's chance of selection is written on the work list.")
    with st.expander("Outcome codes for the log"):
        st.dataframe(pd.DataFrame({"Code": list(OUTCOMES), "Meaning": list(OUTCOMES.values()),
                                   "Counts as": ["non-technical loss found" if k in POSITIVE else
                                                 "nothing found (may miss theft)" if k == "CLEAN" else
                                                 "a finding, not theft" if k == "CABLE_OR_NETWORK_FAULT" else
                                                 "missing (not counted as clean)" for k in OUTCOMES]}),
                     hide_index=True, width="stretch")
    c1, c2, c3 = st.columns(3)
    share = c1.slider("Random audit share", 0.10, 0.20, 0.15, 0.01,
                      help="Share of each round sent to randomly chosen areas (10-20%).")
    sens = c2.slider("Crew sensitivity (assumption)", 0.5, 1.0, 0.8, 0.05,
                     help="Chance a crew finds theft that is there. No published figure exists; "
                          "0.8 is an assumption.")
    use = c3.checkbox("Use inspection results in the ranking", value=False, key="insp_use",
                      help="Off by default: in the closed-loop study this did not improve the theft "
                           "reached. The results are always used to calibrate the probabilities.")
    st.session_state["isim_use"] = st.session_state["iown_use"] = use
    cfg = InspectionConfig(random_share=share, sensitivity=sens)
    mode = st.radio("Data", ["Practice with a simulated district", "My feeders and inspection log"],
                    horizontal=True, key="insp_mode")
    if mode == "Practice with a simulated district":
        with st.form("insp_form"):
            c1, c2, c3 = st.columns(3)
            nf = c1.slider("Number of feeders", 3, 10, 6, key="insp_nf")
            rounds = c2.slider("Rounds already done", 0, 12, 4)
            budget = c3.slider("Areas visited per round", 10, 60, 30)
            iseed = st.number_input("Random seed", 0, 10_000, 5, step=1, key="insp_seed")
            go = st.form_submit_button("Simulate rounds", type="primary")
        if go:
            st.session_state["insp_args"] = (nf, rounds, budget, int(iseed))
        if "insp_args" in st.session_state:
            nf, rounds, budget, iseed = st.session_state["insp_args"]
            feeders = {}
            bar = st.progress(0.0, text="Analysing feeders...")
            for i, p in enumerate(district_params(nf, iseed)):
                d, res, _, _ = run_simulation(p["theft"], p["n_meters"], "ends", 0.2, p["fault"], 7,
                                              p["seed"], "daily", 0.03, p["shedding"], 0.05, False)
                feeders[f"Feeder {i + 1}"] = (d, res)
                bar.progress((i + 1) / nf)
            bar.empty()
            areas = district_areas(feeders)
            logs = []
            for r in range(rounds):
                a = areas[["feeder", "bus", "model_score", "weight"]].copy()
                if logs:
                    recent = recently_visited(clean_log(pd.concat(logs)), cfg.cooldown_rounds)
                    a = a[[(f, b) not in recent for f, b in zip(a.feeder, a.bus)]]
                if logs and use:
                    a = a.assign(weight=area_posterior(a, clean_log(pd.concat(logs)), cfg)["next_weight"].values)
                plan = plan_round(a, budget, cfg, seed=iseed * 100 + r, round_id=f"R{r + 1}")
                logs.append(simulate_visits(plan, areas, seed=iseed * 100 + r, sensitivity=sens))
            post = None
            if logs:
                raw = pd.concat(logs, ignore_index=True)
                log = clean_log(raw)
                post = inspection_report(log, areas, cfg, True, "isim")
                st.download_button("Download the simulated inspection log (CSV)",
                                   raw.to_csv(index=False).encode(), "inspection_log.csv",
                                   "text/csv", key="isim_log")
            st.subheader(f"Next round: R{rounds + 1}")
            inspection_planner(areas, post, cfg, "isim", f"R{rounds + 1}", True,
                               log if logs else None)
    else:
        st.markdown("Upload your feeders as a **district .zip** (one folder per feeder, as in "
                    "*Compare feeders*). Add the **inspection log** (the work lists filled in by "
                    "crews, all rounds in one CSV) once you have one.")
        c1, c2 = st.columns(2)
        zf = c1.file_uploader("District .zip", type="zip", key="insp_zip")
        lf = c2.file_uploader("Inspection log (CSV, optional)", type="csv", key="insp_log")
        if zf is not None:
            feeders, names = {}, {}
            try:
                parts = split_district_zip(zf.getvalue())
                for name, files in parts.items():
                    if "billing" in files:
                        files = {k: v for k, v in files.items() if k not in PREPAID_FILES}
                    d, res, nm, _, _ = run_uploaded(files, 0.90, 0.03)
                    feeders[name], names[name] = (d, res), nm
            except FeederDataError as e:
                st.error(f"There is a problem with the files: {e}")
            if feeders:
                areas = district_areas(feeders)
                # utility bus names in the work list
                areas["bus"] = [(names[f] or {}).get(b, b.replace("bus_", ""))
                                for f, b in zip(areas.feeder, areas.bus)]
                post = None
                if lf is not None:
                    try:
                        log = clean_log(pd.read_csv(lf))
                        log["bus"] = log["bus"].astype(str)
                        post = inspection_report(log, areas, cfg, False, "iown")
                    except ValueError as e:
                        st.error(f"There is a problem with the inspection log: {e}")
                rid = st.text_input("Name of the next round", "R1", key="insp_rid")
                st.subheader(f"Next round: {rid}")
                inspection_planner(areas, post, cfg, "iown", rid, False,
                                   log if post is not None else None)


# --- 3. Study results ---------------------------------------------------------------
with tab_study:
    st.write("Results from 100 simulated weeks (25 held back for testing). "
             "Full details and a corrected earlier claim are in `docs/results.md` on GitHub.")
    exp_path = ROOT / "results" / "experiments.csv"
    if exp_path.exists():
        e = pd.read_csv(exp_path)
        nice = {"B1 energy audit": "Energy audit (today's method)",
                "B1b audit + ratings": "Audit + transformer ratings",
                "B1c combined rules": "Combined rules",
                "B2 pure ML (3 weeks)": "Machine learning, 3 labelled weeks",
                "B2 pure ML (75 weeks)": "Machine learning, 75 labelled weeks",
                "P1 physics-informed": "Physics-informed (this project)"}
        e["method"] = e["method"].map(lambda m: nice.get(m, m))
        allb = e[e.buses == "all buses"].set_index("method")
        unm = e[e.buses == "unmetered"].set_index("method")
        tbl = pd.DataFrame({
            "Theft labels needed": allb["theft labels"],
            "Locating at unmetered buses (AUC)": unm["AUC"].round(3),
            "Error per transformer (MWh/week)": allb["MAE_MWh"].round(2),
            "Error in total stolen (%)": allb["feeder err %"].round(1),
        })
        st.dataframe(tbl, width="stretch")
        plot = unm.reset_index()[["method", "AUC"]]
        plot["group"] = np.where(plot.method.str.contains("this project"), "This project",
                                 np.where(plot.method.str.startswith("Machine"), "Needs labelled theft",
                                          "No labels needed"))
        plot["label"] = plot.AUC.round(2).map(lambda v: f"{v:.2f}")
        base = alt.Chart(plot).encode(
            x=alt.X("AUC:Q", scale=alt.Scale(domain=[0.5, 1.0]),
                    title="Finding theft at unmetered transformers (AUC: 0.5 = guessing, 1 = perfect)"),
            y=alt.Y("method:N", sort=list(plot.sort_values("AUC", ascending=False).method),
                    title="", axis=alt.Axis(labelLimit=320)))
        bars = base.mark_bar(clip=True).encode(color=alt.Color(
            "group:N", title="", legend=alt.Legend(orient="bottom"), scale=alt.Scale(
                domain=["No labels needed", "Needs labelled theft", "This project"],
                range=[GREY, "#9fb3c8", ORANGE])))
        text = base.mark_text(align="left", dx=4, fontSize=12).encode(text="label:N")
        st.altair_chart((bars + text).properties(height=260), width="stretch")
    pl_path, rb_path = ROOT / "results" / "placement.csv", ROOT / "results" / "robustness.csv"
    c1, c2 = st.columns(2)
    if pl_path.exists():
        pl = pd.read_csv(pl_path)
        pl["strategy"] = pl.strategy.map({"ends": "Far ends", "largest": "Largest", "random": "Random"})
        c1.markdown("**Meter placement** (AUC at unmetered buses)")
        c1.altair_chart(alt.Chart(pl).mark_line(point=True).encode(
            x=alt.X("meters:Q", title="Transformer meters"),
            y=alt.Y("AUC unmetered:Q", scale=alt.Scale(domain=[0.88, 0.97]), title="AUC"),
            color=alt.Color("strategy:N", title="Placement")).properties(height=260),
            width="stretch")
    if rb_path.exists():
        rb = pd.read_csv(rb_path)
        c2.markdown("**Robustness** (AUC at unmetered buses)")
        long = pd.concat([rb.assign(method="Physics-informed", AUC=rb["AUC unmetered"]),
                          rb.assign(method="Combined rules", AUC=rb["B1c AUC unmetered"])])
        long["test"] = long["test"].str.capitalize()
        c2.altair_chart(alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("level:Q", title="Stress level"),
            y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.8, 0.96])),
            color=alt.Color("method:N", scale=alt.Scale(
                domain=["Combined rules", "Physics-informed"], range=[GREY, ORANGE]),
                title="", legend=alt.Legend(orient="bottom")),
            facet=alt.Facet("test:N", columns=3, title="")).properties(height=200, width=140)
            .resolve_scale(x="independent"))

    sh_path, pp_path = ROOT / "results" / "shedding.csv", ROOT / "results" / "prepaid.csv"
    if sh_path.exists():
        sh = pd.read_csv(sh_path)
        st.markdown("**Load shedding** (40 feeder-weeks per pattern, half with a cable fault)")
        st.dataframe(pd.DataFrame({
            "Pattern": sh.shedding.map({"4h": "4 h off a day", "8h": "8 h off a day",
                                        "rotation": "Supply one day in three"}),
            "Hours supplied per week": sh["hours supplied per week"].round(0).astype(int),
            "Locating theft (AUC)": sh["P1 AUC unmetered"].round(3),
            "Error in total stolen (%)": sh["P1 total err %"].round(1),
            "Cable faults detected": sh["faults detected"],
            "False alarms": sh["false alarms"]}), hide_index=True, width="stretch")
    if pp_path.exists():
        pp = pd.read_csv(pp_path)
        st.markdown("**Prepaid token sales** (40 feeder-weeks, about 12,000 prepaid meters each, "
                    "90 days of vending records)")
        cols = {"case": "Sales data", "P1 AUC unmetered": "Locating theft (AUC)",
                "P1 total err %": "Error in total stolen (%)",
                "bypassers in top 100": "Top 100 meters that really bypass",
                "bypassers among all meters": "Share of all meters that bypass"}
        t = pp[[c for c in cols if c in pp]].rename(columns=cols)
        for c in ["Top 100 meters that really bypass", "Share of all meters that bypass"]:
            if c in t:
                t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{100 * v:.0f}%")
        st.dataframe(t.round(3), hide_index=True, width="stretch")

    am_path = ROOT / "results" / "ami.csv"
    if am_path.exists():
        am = pd.read_csv(am_path)
        st.markdown("**Large customers on smart meters** (40 feeder-weeks, 36 large customers "
                    "each, 28 days of hourly history)")
        st.dataframe(pd.DataFrame({
            "Data": am["case"].str.capitalize(),
            "Locating theft (AUC)": am["P1 AUC unmetered"].round(3),
            "Error in stolen energy at large-customer areas (%)":
                am["large-customer areas: theft error % (median)"].round(1),
            "Tampered meters in top k (%)": am["hits in top k (k = number of tamperers) %"].round(0),
            "Same, readings alone (%)": am["same, smart-meter data alone %"].round(0)}),
            hide_index=True, width="stretch")

    di_path = ROOT / "results" / "district.csv"
    if di_path.exists():
        di = pd.read_csv(di_path).iloc[0]
        st.markdown(f"**Crews across a district** ({int(di.districts)} districts of "
                    f"{int(di['feeders per district'])} feeders; crews for "
                    f"{int(di['crew budget (areas)'])} transformer areas per district)")
        st.dataframe(pd.DataFrame({
            "Where the crews go": ["District-wide, highest estimated theft",
                                   "Equal crews on every feeder",
                                   "Feeders with most unbilled energy (usual practice)",
                                   "Random areas"],
            "True stolen energy reached (%)": [round(di[c], 1) for c in [
                "crews: district-wide", "crews: equal per feeder",
                "crews: worst feeders by unbilled", "crews: random"]]}),
            hide_index=True, width="stretch")

    in_path = ROOT / "results" / "inspections.csv"
    if in_path.exists():
        ins = pd.read_csv(in_path)
        ins = ins[ins.scenario == "accurate physics"]
        st.markdown("**Inspection rounds** (1,920 simulated areas, 8 rounds of 60 visits, "
                    "30 repetitions; true share of areas with theft 36.7%)")
        st.dataframe(pd.DataFrame({
            "Policy": ins.policy.map({"greedy": "Top of the list only", "top+random": "Top + 15% random (app)",
                                      "top+random+learn": "Top + random + re-ranking",
                                      "spread": "Risk-tilted random draw", "random": "Random"}),
            "Theft reached (MWh/week)": ins["theft reached (MWh/week)"].round(0).astype(int),
            "Share of areas with theft, as reported (%)": (100 * ins["estimated theft share of areas"]).round(1),
            "Proven better than random (% of runs)": (100 * ins["verdict model>random"]).round(0)}),
            hide_index=True, width="stretch")

# --- 4. How it works ------------------------------------------------------------------
with tab_how:
    st.markdown("""
**The problem.** Sierra Leone's distribution utility loses 30-53% of the electricity it buys.
Comparing the substation meter with billing shows *how much* is missing on a feeder, but not
*where*, because most distribution transformers have no meter.

**The method.** For each feeder, the locator finds the theft pattern that best explains
everything the utility measures, while obeying the physics of the network:

1. **Power balance** at the substation: power in = everything consumed (billed + stolen) + line losses.
2. **Reactive power balance**: stolen load draws reactive power like normal load; a resistive cable fault does not.
3. **Voltage drop** from the substation to each transformer meter, through the real cable impedances.
   Extra unbilled load upstream of a meter shows up as extra voltage drop.
4. **Transformer meters** where they exist.
5. **Transformer ratings** from the asset register as a starting guess for where demand should be.

Losses and voltage corrections come from a full AC power flow, so the answer is consistent
with the non-linear network physics. It needs no records of past theft.

**Theft or cable fault?** Stolen power is ordinary household load: it draws real and reactive
power in the usual ratio. A failing cable joint draws real power only. So the substation's
"real power minus what reactive power implies" is blind to theft but steps up when a fault
starts. The app scans for that step to find the start hour and leak size, then uses the step
it causes in each transformer meter's voltage to estimate where the fault is. In tests it
caught 28 of 30 faults with no false alarms in 30 clean weeks.

**Load shedding.** Hours when the feeder was switched off are recognised from the substation
meter (zero or missing readings) and left out, so outages are not mistaken for theft or
faults. With very little supply (for example 24 hours a week), use a month or more of data:
fault detection in particular needs energised hours before and after a fault starts.

**Prepaid customers.** Most customers buy prepaid tokens, so the vending system holds every
purchase (meter, date, kWh) and the customer database says which transformer each meter is on.
The app adds up purchases per transformer area, averaging the last 30 days because people buy
tokens when they have money, not when they use power. It then ranks individual meters: a meter
whose buying dropped, or is far below similar meters on the same transformer, in an area where
the physics says energy is going missing, comes first. Meters listed under the wrong
transformer blur the picture; the app still works with 10-20% of them wrong.

**Large customers on smart meters.** Hourly smart-meter readings give the exact sales of
the areas where every customer has one, and extra voltage points along the feeder. Each
large customer is checked for a lasting drop in daily use and for a peak far below its
contracted capacity, which is how a meter that records only part of the load looks. A check
meter on each large-customer transformer then makes the energy balance of those areas exact.

**Inspection rounds.** Most crews go to the top of the list, and about 15% to randomly
chosen areas, with every area's chance of selection written on the work list. The random
visits are the yardstick: they show whether the targeting beats chance, and, weighted by
the chances of selection, how many areas really have theft. Outcomes then calibrate the
probabilities the app shows. The physics estimate itself never learns from inspection
labels, so it cannot inherit the inspectors' blind spots.

**Several feeders.** The *Compare feeders* tab runs the same analysis on every feeder of a
district, ranks the feeders by estimated theft and revenue lost, and plans where a limited
number of crews should go: the areas with the highest estimated theft across all feeders,
not the same number on every feeder.

**Honest limits.** All results so far come from simulation calibrated to published Freetown
data. Fault location is to a section of the feeder (typically 8 of 32 buses), not one exact
transformer. A fault already present at the start of the data shows no step and is not
detected. An out-of-date transformer register reduces accuracy for every method.

**Code and sources:** [github.com/Alusine-bah/sl-grid-piml](https://github.com/Alusine-bah/sl-grid-piml)
""")

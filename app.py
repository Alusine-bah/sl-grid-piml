"""Feeder Loss Locator: interactive app.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from slgrid.app_core import feeder_summary, locate, simulate_case, tree_layout  # noqa: E402
from slgrid.io import FeederDataError, frames_to_standard, standard_to_frames  # noqa: E402

st.set_page_config(page_title="Feeder Loss Locator", page_icon="⚡", layout="wide")

ORANGE = "#c2410c"
GREY = "#8a959e"
EXAMPLE_DIR = ROOT / "data" / "example_feeder"
FILES = ["lines", "transformers", "head", "meters", "billing"]
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
def run_simulation(theft, n_meters, strategy, dropout, fault, days, seed, billing, bill_noise):
    d = simulate_case(theft, n_meters, strategy, dropout, fault, days, seed, billing, bill_noise)
    return d, locate(d)


@st.cache_data(show_spinner=False)
def run_uploaded(frames_bytes: dict, power_factor: float, tech_loss: float):
    frames = {k: pd.read_csv(io.BytesIO(v)) for k, v in frames_bytes.items()}
    d, names = frames_to_standard(frames)
    res = locate(d, power_factor=power_factor, tech_loss=tech_loss)
    return d, res, names


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
        "Fault probability": np.where(res.fault_prob > 0.005,
                                      (100 * res.fault_prob).round(0).astype(int).astype(str) + "%", ""),
    })
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


def results_view(d, res, names, show_truth, key):
    s = feeder_summary(d, res)
    c = st.columns(4)
    c[0].metric("Energy into feeder", f"{s['head_mwh']:.0f} MWh", f"{s['days']} days", delta_color="off")
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
        y=alt.Y("pct:Q", title="Share of stolen energy covered (%)", scale=alt.Scale(domain=[0, 100])),
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


# ----------------------------------------------------------------------------- layout
st.title("⚡ Feeder Loss Locator")
st.write("Find **where** electricity is being lost on an 11 kV distribution feeder, using the "
         "substation meter, a few transformer meters, billing records and the transformer "
         "register. No customer smart meters and no records of past theft are needed.")

tab_sim, tab_own, tab_study, tab_how = st.tabs(
    ["Simulate a feeder", "Analyse your own data", "Study results", "How it works"])

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
                               ["daily", "period"],
                               format_func={"daily": "Daily per transformer area",
                                            "period": "One total for the period (monthly bills / prepaid sales)"}.get)
        bill_noise = c1.slider("Billing error", 0.0, 0.3, 0.03, 0.01, format="%.2f",
                               help="How far each billed value is from what was really used. "
                                    "Prepaid token sales can be 10-20% off, because people buy "
                                    "tokens when they have money, not when they use power.")
        dropout = c3.slider("Meter readings missing", 0.0, 0.8, 0.2, 0.1, format="%.1f")
        fault = c3.checkbox("Include a cable fault", value=False)
        seed = c3.number_input("Random seed (changes the neighbourhoods and theft spots)",
                               0, 9999, 7)
        go = st.form_submit_button("Run simulation and locate losses", type="primary")
    if go or "sim_done" in st.session_state:
        st.session_state["sim_done"] = True
        with st.spinner("Simulating the feeder and locating losses..."):
            d, res = run_simulation(theft, n_meters, strategy, dropout, fault, days, int(seed),
                                    billing, bill_noise)
        results_view(d, res, None, show_truth=True, key="sim")
        with st.expander("Download this simulated feeder as utility CSV files"):
            st.write("These are the same five files a utility would upload in the next tab. "
                     "Useful as a filled-in example.")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for k, df in standard_to_frames(d).items():
                    z.writestr(f"{k}.csv", df.to_csv(index=False))
            st.download_button("Download feeder files (.zip)", buf.getvalue(),
                               "simulated_feeder.zip", "application/zip", key="sim_zip")

# --- 2. Own data ----------------------------------------------------------------------
with tab_own:
    st.write("Upload five CSV files describing one feeder. Start from the example files and "
             "replace the numbers with your own.")
    with st.expander("What each file contains", expanded=False):
        st.markdown("""
| File | Columns | What it is |
|---|---|---|
| `lines.csv` | from_bus, to_bus, r_ohm, x_ohm | Each 11 kV line section and its resistance and reactance in ohms. The substation bus is the one that never appears in `to_bus`. Must be radial. |
| `transformers.csv` | bus, kva | Installed transformer capacity at each load bus (from the asset register). |
| `head.csv` | hour, p_mw, q_mvar, v_pu | Substation 11 kV breaker readings, every hour, whole days. `hour` can be 0,1,2… or a timestamp. |
| `meters.csv` | hour, bus, p_mw, q_mvar, v_pu | Any transformer check meters. Gaps are fine; can be empty (header only). |
| `billing.csv` | day, bus, mwh — or just bus, mwh | Energy sold per transformer area: either per day, or one total per bus for the whole period (monthly bills or prepaid token sales). Monthly totals work almost as well as daily figures. |
""")
    ex = {k: (EXAMPLE_DIR / f"{k}.csv") for k in FILES}
    if all(p.exists() for p in ex.values()):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for k, p in ex.items():
                z.write(p, f"{k}.csv")
        st.download_button("Download example files (.zip)", buf.getvalue(),
                           "example_feeder.zip", "application/zip", key="ex_zip")

    use_example = st.toggle("Use the example files instead of uploading", value=False)
    uploads = {}
    if not use_example:
        cols = st.columns(5)
        for c, k in zip(cols, FILES):
            f = c.file_uploader(f"{k}.csv", type="csv", key=f"up_{k}")
            if f is not None:
                uploads[k] = f.getvalue()
    else:
        uploads = {k: p.read_bytes() for k, p in ex.items()}

    c1, c2 = st.columns(2)
    pf = c1.slider("Typical power factor of loads", 0.75, 1.0, 0.90, 0.01)
    tl = c2.slider("Expected technical losses on the 11 kV network", 0.0, 0.10, 0.03, 0.01,
                   format="%.2f")
    if len(uploads) == 5:
        try:
            with st.spinner("Checking the files and locating losses..."):
                d, res, names = run_uploaded(uploads, pf, tl)
            nb = d["billing"].shape[1]
            st.success(f"Read a feeder with {len(res)} load buses, {int(res.metered.sum())} "
                       f"transformer meters and {d['head'].shape[1] // 24} days of data "
                       f"({'daily billing' if nb > 1 else 'one billing total per bus'}).")
            results_view(d, res, names, show_truth=False, key="own")
        except FeederDataError as e:
            st.error(f"There is a problem with the files: {e}")
    else:
        st.info(f"Waiting for files: {', '.join(k + '.csv' for k in FILES if k not in uploads)}")

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

**Honest limits.** All results so far come from simulation calibrated to published Freetown
data. Fault location is to a section of the feeder (typically 8 of 32 buses), not one exact
transformer. A fault already present at the start of the data shows no step and is not
detected. An out-of-date transformer register reduces accuracy for every method.

**Code and sources:** [github.com/Alusine-bah/sl-grid-piml](https://github.com/Alusine-bah/sl-grid-piml)
""")

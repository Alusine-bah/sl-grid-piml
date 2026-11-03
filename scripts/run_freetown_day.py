"""Simulate one day on the Freetown-style feeder and save summary plots.

Outputs (in outputs/):
  freetown_day_load.png      feeder load and technical losses over 24 hours
  freetown_peak_voltage.png  voltage along the feeder at the evening peak
  household_profile.png      demand of one Freetown household (from the survey)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slgrid.freetown import build_freetown_feeder, feeder_summary
from slgrid.loads import assign_customers, generate_profiles, household_profile_kw
from slgrid.simulate import run_timeseries

out = Path("outputs")
out.mkdir(exist_ok=True)
V_LIMIT = 0.94  # Sierra Leone grid code, -6%

net = build_freetown_feeder()
cust = assign_customers(net)
prof = generate_profiles(cust, n_days=1, seed=1)
res = run_timeseries(net, prof["p_mw"], prof["q_mvar"])
s = res["summary"]

fs = feeder_summary(net)
loss = s.p_loss_line_mw + s.p_loss_trafo_mw
t_peak = s.p_load_mw.idxmax()

print(f"Feeder: {fs['line_km']:.0f} km of line, {100*fs['underground_share']:.0f}% underground, "
      f"{fs['load_buses']} load buses ({fs['commercial_buses']} commercial)")
print(f"Customers: {cust['households'].sum():,} households on ~{cust['transformers'].sum()} "
      f"transformers + {cust['commercial_peak_mw'].sum():.1f} MW commercial")
print(f"Daily energy delivered: {s.p_load_mw.sum():.1f} MWh")
print(f"Peak load: {s.p_load_mw.max():.2f} MW at {t_peak:02d}:00")
print(f"Technical losses: {100*loss.sum()/s.p_source_mw.sum():.1f}% of energy "
      f"({100*loss[t_peak]/s.p_source_mw[t_peak]:.1f}% at peak)")
print(f"Lowest voltage: {s.vm_min_pu.min():.3f} pu at {s.loc[s.vm_min_pu.idxmin(), 'bus_vm_min']}")
hrs_low = (s.vm_min_pu < V_LIMIT).sum()
print(f"Hours with some bus below {V_LIMIT} pu: {hrs_low} of 24")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(s.index, s.p_load_mw, marker="o", label="Customer load")
ax.plot(s.index, s.p_source_mw, linestyle="--", label="Supplied from substation")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Power (MW)")
ax.set_title("Freetown-style feeder: one day")
ax.set_xticks(range(0, 24, 2))
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(out / "freetown_day_load.png", dpi=150)

vm = res["vm_pu"].loc[t_peak]
branches = {
    "Main line (0-17)": list(range(0, 18)),
    "Branch at bus 1 (18-21)": [1] + list(range(18, 22)),
    "Branch at bus 2 (22-24)": [2] + list(range(22, 25)),
    "Branch at bus 5 (25-32)": [5] + list(range(25, 33)),
}
fig, ax = plt.subplots(figsize=(8, 4))
for label, buses in branches.items():
    ax.plot(buses, [vm[f"bus_{b}"] for b in buses], marker="o", label=label)
ax.axhline(V_LIMIT, color="red", linestyle="--", label=f"{V_LIMIT} pu grid-code limit")
ax.set_xlabel("Feeder bus number")
ax.set_ylabel("Voltage (pu)")
ax.set_title(f"Voltage along the feeder at peak ({t_peak:02d}:00)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "freetown_peak_voltage.png", dpi=150)

d = household_profile_kw()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(d.index, d.values * 1000)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Demand per household (W)")
ax.set_title("Average Freetown household demand (Rogers et al., 2024 survey)")
ax.set_xticks(range(0, 24, 2))
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(out / "household_profile.png", dpi=150)

print(f"Saved 3 plots to {out}/")

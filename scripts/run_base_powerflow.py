"""Run a power flow on the base 11 kV feeder and save a voltage-profile plot."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slgrid.feeder import build_base_feeder, to_standard, run_powerflow

out = Path("outputs")
out.mkdir(exist_ok=True)

net = build_base_feeder()
std = to_standard(net)
res = run_powerflow(net)

p_load = std["buses"]["p_load_mw"].sum()
p_loss = res["lines"]["loss_mw"].sum()
p_src = net.res_ext_grid["p_mw"].sum()
vm = res["buses"]["vm_pu"]

print(f"Buses: {len(std['buses'])}   Lines: {len(std['lines'])}")
print(f"Total load:        {p_load*1000:8.1f} kW")
print(f"Line losses:       {p_loss*1000:8.1f} kW  ({100*p_loss/p_src:.1f}% of supply)")
print(f"Supplied by source:{p_src*1000:8.1f} kW")
print(f"Lowest voltage:    {vm.min():.4f} pu at bus {vm.idxmin()}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(res["buses"]["bus_id"], vm, marker="o")
ax.axhline(0.95, color="red", linestyle="--", label="0.95 pu limit")
ax.set_xlabel("Bus number")
ax.set_ylabel("Voltage (pu)")
ax.set_title("Base 11 kV feeder: voltage profile")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "base_voltage_profile.png", dpi=150)
print(f"Saved plot to {out / 'base_voltage_profile.png'}")

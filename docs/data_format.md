# Data format

The models in this project read one standard set of arrays. Today these come from the
simulator; later they can come from a real utility. This page lists what a utility such
as EDSA would need to provide for one 11 kV feeder, and how it maps to the arrays.

## What a utility would provide

| Data | Typical source | Resolution | Array |
|---|---|---|---|
| Feeder-head P, Q, V | Substation 11 kV breaker meter or SCADA | hourly (or 15 min) | `head` |
| Transformer meter P, Q, V (a few transformers) | Check meters or smart meters on distribution transformers | hourly, gaps allowed | `meter_p`, `meter_q`, `meter_v` |
| Energy sold per transformer area | Prepaid vending records or billing system, grouped by transformer | daily (monthly also works) | `billing` |
| Feeder layout | Network map: which transformer connects where | static | `line_from`, `line_to`, `bus_names` |
| Line types and lengths | Asset register | static | `line_r_ohm`, `line_x_ohm` |
| Transformer kVA rating at each bus | Asset register | static | `transformer_kva` |

Nothing in this list needs customer smart meters. Missing readings are allowed and are
stored as NaN.

Load shedding: hours when the feeder was switched off can be left out of the substation
file or given as zero. The models treat them as dead hours (no flow, no readings) and use
only energised hours. Hour numbers and timestamps are counted from the start (midnight) of
the first day.

## Arrays (shapes)

S = scenarios, T = hours, B = load buses (32), N = all buses (34), M = metered buses,
D = days, L = lines (32).

Observed:
- `head` [S, T, 3]: P (MW), Q (MVAr), V (pu) at the 11 kV side of the substation
- `meter_p`, `meter_q`, `meter_v` [S, T, M]: transformer meters, NaN where missing
- `billing` [S, D, B]: billed energy, MWh per day per bus

Ground truth (simulation only, used for training targets and scoring):
- `true_bus_p`, `theft_bus_p`, `fault_bus_p` [S, T, B]: MW
- `vm_pu` [S, T, N]: voltage at every bus
- `line_loss` [S, T, L]: technical loss per line, MW
- `is_theft`, `is_fault` [S, B]: per-bus labels

Asset data:
- `transformer_kva` [S, B]: installed transformer capacity behind each bus, kVA

Network description:
- `bus_names`, `all_bus_names`, `meter_names`
- `line_from`, `line_to`, `line_r_ohm`, `line_x_ohm`

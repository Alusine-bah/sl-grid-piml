# Real data

## odisha_dt_losses_q2fy25.csv

Distribution-transformer (DT) energy accounts of TP Central Odisha Distribution Ltd
(TPCODL), India, for Q2 FY2024-25 (July-September 2024): 1,867 metered DTs on 468 11 kV
feeders. For each DT: the 11 kV feeder code, DT code, capacity (kVA), energy received at
the DT (`input_mu`, million kWh), energy billed to its customers (`billed_mu`), and the
loss in percent (technical losses of the low-voltage network plus non-technical losses).

Source: TPCODL, *Energy Audit Report Q2 FY24-25*, "Details of DT-wise losses"
(mandatory periodic energy accounting under the Bureau of Energy Efficiency regulations),
https://www.tpcentralodisha.com/pdf/energyaudit/Periodic_EA/EA_Report_Q2_FY24-25.pdf

Extracted from the published PDF. Rows 1-61, 694 and 1318 of the report's table could not
be read and are missing. Every row passes the check loss = 1 - billed / received (to 0.05
points; one row with tiny energies differs by rounding).

Used by `src/slgrid/realdata.py` to give the simulator a realistic spread of losses across
the transformer areas of a feeder (see docs/results.md, Phase 14).

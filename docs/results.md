# Results log

## Phase 4: baselines (dataset freetown_v2: 100 scenarios x 7 days, 75 train / 25 test)

Six of 32 buses have a transformer meter. About 41% of buses are theft hotspots in a
typical scenario, so a random guess scores AUC 0.5 and average precision about 0.41.

| Method | AUC metered | AUC unmetered | Top-k hit | Error in total stolen energy |
|---|---|---|---|---|
| B1 energy audit (no training) | 1.00 | 0.63 | 0.34 | 2.8% |
| B2 pure ML (75 labelled weeks) | 1.00 | 0.94 | 0.87 | 12.9% |
| B2 pure ML (3 labelled weeks) | — | 0.85 | — | 20.0% |

### What this shows
1. **Behind a meter, theft is easy.** Comparing a transformer meter with billing finds it
   almost perfectly. The hard and useful problem is the 26 unmetered buses.
2. **The energy audit gets the total right but cannot say where.** It measures total
   stolen energy within about 3%, but at unmetered buses it is barely better than chance
   (AUC 0.63), because it has no information to locate the loss.
3. **Pure ML locates theft well, but only with labelled history from the same feeder,
   and it gets the amounts wrong.** With 75 labelled weeks it ranks unmetered hotspots
   well (AUC 0.94), mainly by learning what normal billing looks like at each bus. But:
   - it needs scenarios where the true theft locations are known, which a utility like
     EDSA does not have (theft data is rarely labelled; see S3, S4);
   - what it learns is specific to this feeder's layout;
   - its estimate of total stolen energy is off by 13-25%, because nothing ties its
     output to the energy balance of the feeder.

### Target for the physics-informed model (Phase 5)
- Work **without theft labels**.
- Locate theft at unmetered buses clearly better than the audit (AUC 0.63), aiming to
  approach supervised ML (0.94).
- Keep the total stolen energy accurate (about 3%, like the audit) by obeying the
  feeder's power-flow physics.

## Phase 5: physics-informed model (dataset freetown_v3: 100 scenarios x 7 days, 75 train / 25 test)

Dataset v3 adds transformer kVA ratings (asset register data), which every method may use.

| Method | Theft labels | AUC unmetered | Top-k hit | MAE per bus (MWh) | Error in total stolen |
|---|---|---|---|---|---|
| B1 energy audit | none | 0.65 | 0.37 | 4.30 | 3.4% |
| B1b audit + transformer ratings | none | 0.90 | 0.82 | 3.72 | 47.7% |
| B1c combined rules (ratings + meters + feeder gap) | none | 0.90 | 0.85 | 2.44 | 3.4% |
| B2 pure ML, 3 labelled weeks | 3 weeks | 0.92 | 0.83 | 3.39 | 23.6% |
| B2 pure ML, 75 labelled weeks | 75 weeks | 0.97 | 0.90 | 1.79 | 12.2% |
| **P1 physics-informed** | **none** | **0.92** | **0.85** | **2.11** | **2.7%** |

Fault localisation by P1's leakage estimate: AUC 0.68 (7 test scenarios with a fault).
P1 runtime: 5-20 seconds per feeder-week on a laptop, depending on the machine.

### How P1 works
For each feeder, separately and with no theft labels, P1 finds the theft ratio and
fault leakage at every bus that best explain the measurements while obeying:
1. active and reactive power balance at the substation,
2. LinDistFlow voltage drop from the substation to each meter, through the real cable
   impedances, corrected by full AC power flow (fixed-point iterations),
3. transformer meter readings at the metered buses,
4. a starting guess from transformer ratings, held loosely (35% uncertainty).
Equations are summed over the whole week, because theft is a steady share of
consumption while the hourly split of daily billing is rough. Unknowns for cable
charging and meter calibration offsets are estimated alongside.

### What this shows
1. **Most of the gain over today's audit comes from combining two data sources EDSA
   already has:** transformer ratings (where demand should be) and the feeder-head meter
   (how much is missing). A simple rule doing this (B1c) raises AUC at unmetered buses
   from 0.65 to 0.90 and keeps the total within 3.4%, with no new hardware and no training.
2. **Physics adds a modest, consistent improvement on top:** AUC 0.92 vs 0.90 at unmetered
   buses, 14% lower per-bus error (2.11 vs 2.44 MWh), and a more accurate total
   (2.7% vs 3.4%). It matches pure ML trained on 3 labelled weeks at locating theft,
   with far better amounts.
3. **Physics alone cannot locate theft:** without ratings, six voltage equations cannot
   separate 26 unmetered buses (AUC about 0.6 in development runs).
4. **ML with a long labelled history still locates best** (0.97 with 75 weeks), but such
   labels do not exist at EDSA today.
5. **Limitation:** small cable faults (40–150 kW) are hard to separate from large-scale
   theft (AUC 0.68). Weekly sums discard the moment a fault starts; a change detector is a
   natural next step.

### Correction
An earlier version of this page claimed P1 was "4-17 times more accurate on the stolen
amount than any method that can locate theft". That compared against B1b, whose 48% error
comes from a weak rule (it never rescales to the feeder energy gap), not from missing
physics. The fairer rule-based baseline B1c was added, and the claim was withdrawn.

### Development notes
- First version fitted hour by hour: AUC 0.75, worse than ratings alone, because the
  hourly split of daily billing is about 17% wrong and the solver bent theft estimates to
  fit that noise. Summing equations per day, then per week, fixed it.
- First version ignored cable charging (about 0.17 MVAr from underground cables), which
  broke the reactive power balance. Now estimated as an unknown offset.
- Settings were chosen on training scenarios only.

## Phase 6: meter placement and robustness (first 12 test scenarios of freetown_v3)

With only 12 scenarios, differences of about 0.01 in AUC are within noise.

### Meter placement: AUC at unmetered buses / error in total stolen energy
| Meters | P1, far ends of branches | P1, largest transformers | P1, random | B1c (any placement) |
|---|---|---|---|---|
| 0 | 0.926 / 2.4% | 0.926 / 2.4% | 0.926 / 2.4% | 0.918 / 3.6% |
| 3 | 0.940 / 2.2% | 0.937 / 1.8% | 0.937 / 2.1% | 0.917–0.927 / 3.6% |
| 6 | 0.941 / 1.9% | 0.929 / 1.9% | 0.942 / 2.5% | 0.918–0.925 / 3.6% |
| 12 | 0.941 / 1.6% | 0.935 / 2.2% | 0.949 / 4.5% | 0.926–0.934 / 3.6% |

- **Transformer meters help only a little** for unmetered buses: from 0.926 with none to
  about 0.94 with 3-6. The feeder-head meter plus ratings carries most of the information.
- **Physics turns meters into information about unmetered buses**, through the voltage
  drop they measure. B1c gains almost nothing at unmetered buses from more meters.
- **Far-end placement gives the steadiest improvement in the total** (2.4% → 1.6%).
  No placement strategy clearly wins on locating, given the noise.

### Robustness (6 meters at default positions; AUC over all buses)
| Stress | Level | P1 AUC | P1 total error |
|---|---|---|---|
| Missing meter readings | 20% / 50% / 80% | 0.951 / 0.950 / 0.947 | 1.3% / 1.4% / 1.9% |
| Cable impedance error | 0 / 10% / 25% | 0.951 / 0.950 / 0.950 | 1.3% / 1.7% / 1.8% |
| Transformer rating error | 0 / 0.2 / 0.4 (log sd) | 0.951 / 0.947 / 0.897 | 1.3% / 1.8% / 2.2% |

For comparison, B1c's AUC at unmetered buses falls from 0.91 to 0.85 at the largest
rating error.
- **Robust to missing data and wrong cable data:** losing 80% of meter readings or having
  25% cable errors barely changes the result.
- **Sensitive to the asset register:** a badly out-of-date transformer register (log sd 0.4,
  ratings often off by 50%) costs about 0.05 AUC, for P1 and the rule-based method alike.
  **Keeping the transformer register accurate matters more than adding meters.**

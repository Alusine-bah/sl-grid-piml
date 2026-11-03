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

| Method | Theft labels needed | AUC unmetered | Top-k hit | MAE per bus (MWh) | Error in total stolen |
|---|---|---|---|---|---|
| B1 energy audit | none | 0.65 | 0.37 | 4.3 | 3.4% |
| B1b audit + transformer ratings | none | 0.90 | 0.82 | 3.7 | 47.7% |
| B2 pure ML, 3 labelled weeks | 3 weeks | 0.92 | 0.83 | 3.4 | 23.6% |
| B2 pure ML, 75 labelled weeks | 75 weeks | 0.97 | 0.90 | 1.8 | 12.2% |
| **P1 physics-informed (ours)** | **none** | **0.92** | **0.85** | **2.1** | **2.7%** |

Fault localisation by P1's leakage estimate: AUC 0.68 (7 test scenarios with a fault).

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
1. **With no labels, P1 locates theft as well as ML trained on 3 labelled weeks**
   (AUC 0.92 at unmetered buses).
2. **P1 gives the most accurate amounts:** total stolen energy within 2.7%, against
   12-48% for every other method that can locate theft. Only the plain audit is as accurate
   on the total, and it cannot say where the theft is.
3. **What each part contributes:**
   - Transformer ratings alone locate theft well (0.90) but get the total badly wrong (48%).
   - Physics alone (6 meters, no ratings) cannot locate it: six voltage equations cannot
     separate 26 unmetered buses (AUC about 0.6 in development runs).
   - Together: the location quality of the ratings, the total accuracy of the audit, plus a
     small gain in AUC.
4. **ML with a long labelled history still locates better** (0.97 with 75 weeks), but such
   labels do not exist at EDSA today.
5. **Limitation:** small cable faults (40–150 kW) are hard to separate from large-scale theft
   (AUC 0.68). Weekly sums discard the moment a fault starts; a dedicated change detector
   is a natural next step.
6. P1 takes about 5 seconds per feeder-week on a laptop, using NumPy, SciPy and pandapower.

### Development notes (kept for honesty)
- First version fitted hour by hour: AUC 0.75, worse than ratings alone, because the
  hourly split of daily billing is about 17% wrong and the solver bent theft estimates to
  fit that noise. Summing equations per day, then per week, fixed it.
- First version ignored cable charging (about 0.17 MVAr from underground cables), which
  broke the reactive power balance. Now estimated as an unknown offset.
- Settings were chosen on training scenarios only; the test scenarios were run once.

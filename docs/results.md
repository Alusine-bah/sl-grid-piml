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

### Robustness (6 meters at default positions; AUC at unmetered buses / total error)
| Stress | Level | P1 physics-informed | B1c combined rules |
|---|---|---|---|
| Missing meter readings | 20% | 0.937 / 1.3% | 0.910 / 3.6% |
| | 50% | 0.936 / 1.4% | 0.910 / 3.6% |
| | 80% | 0.934 / 1.9% | 0.909 / 3.6% |
| Cable impedance error | 10% | 0.934 / 1.7% | 0.910 / 3.6% |
| | 25% | 0.934 / 1.8% | 0.910 / 3.6% |
| Transformer rating error | log sd 0.2 | 0.931 / 1.8% | 0.900 / 3.6% |
| | log sd 0.4 | 0.872 / 2.2% | 0.847 / 3.6% |

P1 keeps its lead over the rule-based method under every stress.
- **Robust to missing data and wrong cable data:** losing 80% of meter readings or having
  25% cable errors barely changes the result.
- **Sensitive to the asset register:** a badly out-of-date transformer register (log sd 0.4,
  ratings often off by 50%) costs about 0.06 AUC, for P1 and the rule-based method alike.
  **Keeping the transformer register accurate matters more than adding meters.**

## Phase 7: monthly bills and prepaid token sales (25 test scenarios of freetown_v3)

EDSA has prepaid token sales and monthly bills rather than daily readings. Billing was
re-created as one total per transformer area for the whole period, with 3-20% error
(prepaid tokens are bought when people have money, not when they use power).
Reproduce with `python scripts/run_billing_study.py`.

| Billing data | Error | P1 AUC unmetered | P1 total error | B1c AUC unmetered | B1c total error |
|---|---|---|---|---|---|
| Daily | 3% | 0.916 | 3.1% | 0.901 | 3.4% |
| One total per period | 3% | 0.915 | 4.1% | 0.898 | 4.5% |
| One total per period | 10% | 0.910 | 7.9% | 0.891 | 8.8% |
| One total per period | 20% | 0.894 | 14.6% | 0.869 | 16.0% |

(The daily row differs slightly from Phase 5 because billing noise was redrawn.)

- **Daily billing is not needed.** One total per transformer area for the period locates theft
  as well as daily values.
- **Billing error mainly hurts the estimate of how much is stolen**, not where. With 20% error
  per area, localisation is still 0.89 AUC.
- **Practical advice:** analyse a month or more at a time. Token-timing error shrinks as the
  period gets longer.
- The physics-informed model keeps its lead over the rule-based method at every level.

## Phase 8: telling cable faults apart from theft

Earlier, small cable faults (40-150 kW) were hard to separate from theft (fault-location AUC
0.57-0.68). A dedicated detector was added (`src/slgrid/faults.py`).

**Idea.** Stolen power is household load: it draws real and reactive power in the usual
ratio (power factor about 0.9). A failing cable joint draws real power only. So at the
substation, z = P - Q / tan(phi) is blind to legitimate and stolen load, and steps up when
a fault starts. The detector scans every possible start hour for a lasting step in z
(allowing for network losses), then locates the fault from the step it causes in each
transformer meter's voltage drop, through the real cable resistances.

**Data.** Two new datasets of 60 feeder-weeks, half with a fault at a random bus and hour:
development (seed 101) and a held-out test set (seed 202). The detection threshold (4.0)
was chosen on development data, where clean weeks never scored above 3.4. Reproduce with
`python scripts/run_fault_study.py`.

| | Development | Test (held out) |
|---|---|---|
| Faults detected | 20 / 23 | **28 / 30** |
| False alarms in clean weeks | 0 / 37 | **0 / 30** |
| Detection AUC | 0.974 | 0.998 |
| Start-hour error (median) | 0 h | 0 h |
| Leak-size error (median) | 13% | 7% |
| Fault-location AUC | 0.873 | **0.911** |
| True bus ranked first / in top 3 | 30% / 50% | 46% / 61% |
| True bus inside the reported 80% section | 80% | 79% |
| Section size (median) | 10.5 buses | 7.5 buses |
| Theft total error without / with detector | 4.3% / 1.8% | 4.9% / **2.1%** |
| Theft AUC at unmetered buses without / with | 0.927 / 0.930 | 0.935 / 0.939 |

- **Faults are now detected reliably and timed exactly**, with no false alarms in 67 clean weeks.
- **Location is to a section, not a single transformer:** six meters cannot separate
  neighbouring buses, whose voltage signatures differ by about as much as meter noise.
  The reported 80% section contains the true bus 79-80% of the time, so its confidence
  is honest.
- **Theft estimates improve too:** knowing when a leak started stops it being counted as
  theft, halving the error in total stolen energy on fault weeks.
- **Missed faults** start in the first or last few hours of the data, where there is too
  little before/after data to see a step. A fault already present when the data starts
  cannot be detected this way. Small faults in short periods (for example 50 kW in 3 days)
  can also be missed.
- **Assumption:** loads keep a steady power factor. A real feeder whose load mix changes
  through the day would need hour-of-day terms in the detector; to be checked on real data.

The main experiments (Phases 5-7) use the model without the detector, so their numbers are
unchanged. The app uses the detector.

## Phase 9: load shedding

Sierra Leone feeders are often switched off on a schedule. The simulator now supports
three supply patterns plus "none" (`shedding` in `DatasetConfig`, and in the app):

- **4h:** one random 4-hour outage per day.
- **8h:** one random 8-hour outage per day.
- **rotation:** supply 19:00-07:00 one day in three, as in the 2008 rotation programme
  (JICA 2009). About 26 supplied hours a week.

When supply returns, demand is higher for a while because fridges, freezers and water
heaters all restart together (cold load pickup): +30% in the first hour, +15% in the second.
During an outage the feeder is dead: no flows, no losses, no meter readings, and a cable
fault draws nothing. The models never see the schedule; they infer the off hours from the
substation meter.

**Data.** Three datasets of 40 feeder-weeks each (seeds 301, 302, 303), half with a cable
fault. Reproduce with `python scripts/run_shedding_study.py`.

| Pattern | Hours supplied per week | P1 AUC unmetered | P1 total error | B1c AUC unmetered | B1c total error | Faults detected | False alarms | Fault location AUC |
|---|---|---|---|---|---|---|---|---|
| 4h | 140 | 0.928 | 2.1% | 0.913 | 4.0% | 17 / 20 | 0 / 20 | 0.84 |
| 8h | 114 | 0.910 | **1.6%** | 0.901 | 3.8% | 22 / 24 | 0 / 16 | 0.87 |
| rotation | 26 | **0.938** | 3.6% | 0.926 | 3.9% | 4 / 18 | 0 / 22 | 0.51 |

### What this shows
- **Theft location and size survive load shedding.** Even with only 26 supplied hours a
  week, the physics-informed model locates theft at unmetered transformers with 0.94 AUC
  and estimates the total stolen within 4%.
- **Honest note:** before any changes, the model already coped with 4-8 hours a day of
  shedding (8h: 1.9% total error, 22/24 faults). The work mainly fixed the heavy-rationing case.
- **Two bugs found and fixed:**
  1. The combined-rules method (B1c) averaged meter readings over all hours, including
     dead ones, which broke it under rotation (461% total error, AUC 0.5). It now uses
     energised hours only.
  2. The physics model let dead hours (all zeros) dilute the weight of the substation
     balance equations, so it overestimated theft by 12-17% even in clean weeks under
     rotation. It now uses energised hours only: total error under rotation fell from 10.5%
     to 3.6%, and a false fault alarm disappeared.
- **Cable faults under heavy rationing are mostly missed** (4/18 detected), because a week
  of rotation gives only a few energised hours before and after a fault starts. No false
  alarms in any pattern.
- **Practical advice:** with heavy rationing, analyse a month or more at a time.
- Results without shedding are unchanged (identical to the earlier phases).
- **Upload format:** hours when a feeder was off can be left out of `head.csv` or given as
  zero. Timestamps are counted from midnight of the first day, so outages at the start of
  the period do not shift the timeline.

## Phase 10: prepaid token sales and individual meters

Most EDSA customers are prepaid. The vending system holds every token purchase (meter, date,
kWh), and the customer database links each meter to a transformer. Both can now be used
directly (`src/slgrid/prepaid.py`; `customers.csv` + `tokens.csv` in the app).

**Simulation.** For each feeder-week, 90 days of vending records for about 12,000 customers:
- Customer sizes vary widely (log-normal); 3% of accounts are vacant and buy nothing.
- Each customer buys tokens on their own rhythm (every 2-14 days on average, irregular gaps),
  each purchase covering roughly their use until the next one.
- At a theft hotspot, half the stolen energy is taken by illegal connections with no account,
  and half by account holders who bypass their meter for 50-100% of their use. Half of these
  were already bypassing when the records begin; the rest start during them.
- The customer database can list some meters under the wrong transformer (mostly a
  neighbouring one).

**Data.** 40 feeder-weeks (seed 404). Reproduce with `python scripts/run_prepaid_study.py`.

| Sales data | Sales error per area (median) | P1 AUC unmetered | P1 total error | B1c AUC unmetered | B1c total error |
|---|---|---|---|---|---|
| Daily billing (reference) | 0.8% | 0.930 | 1.3% | 0.919 | 3.5% |
| Token sales in the 7 days | 4.4% | 0.931 | 10.7% | 0.914 | 11.2% |
| 30-day token average | 1.7% | 0.926 | 5.0% | 0.910 | 6.5% |
| 30-day average, 5% of meters mis-mapped | 3.6% | 0.914 | 5.2% | 0.889 | 6.5% |
| 30-day average, 10% mis-mapped | 6.0% | 0.904 | 5.3% | 0.874 | 6.5% |
| 30-day average, 20% mis-mapped | 11.2% | 0.869 | 5.3% | 0.834 | 6.5% |

**Ranking individual meters** (30-day average). A meter's shortfall is how far its last 30
days of buying fall below the higher of what it bought before and what similar meters on the
same transformer buy now. The shortfall is weighted by the physics-informed estimate of the
stolen share in the meter's area.

| | Correct mapping | 5% mis-mapped | 10% | 20% |
|---|---|---|---|---|
| Meters that really bypass, among the top 100 | **70%** | 67% | 65% | 60% |
| Same, ranking by buying alone (no area weight) | 31% | 31% | 30% | 31% |
| Meter AUC (with area weight) | 0.84 | 0.83 | 0.82 | 0.80 |
| Share of all meters that bypass | 21% | 21% | 21% | 21% |

### What this shows
- **Vending records work as billing**, if sales are averaged over about a month. Using only
  the purchases made during the analysed week still locates theft (AUC 0.93), but the total
  stolen is 11% off, because a week of token sales is far from a week of use.
- The 30-day average estimates the total stolen within 5%, not 1.3% as with true daily
  billing. Customers who began bypassing during the month still show part of their earlier
  buying, so sales look a little higher than real paid use. A small error in sales becomes a
  larger relative error in theft, because theft is only part of the energy.
- **Physics plus vending records find individual meters.** Of the 100 meters ranked first,
  70% really bypass, against 21% of all meters (3.4 times better than random). Ranking by
  buying alone gets only 31%, because many honest customers simply use little. The feeder
  physics tells the crew which areas to trust a low buyer in.
- **The customer database matters.** With 20% of meters listed under the wrong transformer,
  area localisation drops from 0.93 to 0.87 AUC and the meter list from 70% to 60% hits.
  Cleaning the customer-to-transformer mapping is worth doing alongside meter audits. The
  physics-informed model stays ahead of the rule-based method at every level.
- **Limits.** Illegal connections with no account cannot appear on the meter list; they are
  found by inspecting the top-ranked areas. Customer behaviour is simulated and must be
  checked against real vending data.


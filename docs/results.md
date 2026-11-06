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

## Phase 11: large customers on smart meters (AMI)

Large commercial and industrial customers are the first to get smart meters with hourly
readings. The model can now use them (`src/slgrid/ami.py`; `ami_customers.csv` +
`ami_readings.csv` in the app).

**Simulation.** The three commercial transformer areas get 12 large customers each (offices
and shops, hotels, telecom and water pumping, workshops), with 28 days of hourly history:
- day-to-day variation of 15% and quieter Sundays;
- 15% of honest customers change their use for good during the history (a business cutting
  back or expanding), which a detector must not mistake for theft;
- contracted capacities in the register are generous and uneven (typically 1.5 times the
  real peak).

Where the feeder simulation put theft at a commercial area, up to 60% of its large customers
tamper. Either one phase is bypassed (the meter records two thirds of the load), or the
current transformer is shunted (it records 10-50%). 60% of tamperers start during the
history; the rest were already tampering when it begins.

**Data.** 40 feeder-weeks (seed 505); 35 have tampering large customers. Reproduce with
`python scripts/run_ami_study.py`.

| | No smart-meter data | Smart meters | Smart meters + check meter on each large-customer transformer |
|---|---|---|---|
| Theft at unmetered transformers (AUC) | 0.945 | 0.944 | 0.948 |
| Error in total stolen | 1.3% | 1.2% | 1.2% |
| Error in stolen energy at the large-customer areas (median) | 23.5% | 20.1% | **0.7%** |
| Tampered meters in the top k (k = number of tamperers) | 85% | 85% | 85% |
| Same, smart-meter readings alone (no feeder physics) | 80% | 80% | 80% |
| Meter AUC (alone: 0.879) | 0.888 | 0.888 | 0.890 |

Tampered meters found in the top k: 88% of those that started during the history, and 81% of
those already tampering when it began (found from their low peak for the contracted capacity
and from the feeder physics).

### What this shows
- **Smart-meter analytics find tampered large meters.** A lasting drop in daily energy, or a
  peak far below the contracted capacity, puts 80% of tampered meters at the top of the list
  on its own. Weighting by the feeder model's estimate for each area raises this to 85%.
- **Honest note:** on their own, the smart meters of 36 large customers do not improve where
  theft is found on the rest of the feeder (AUC 0.944 vs 0.945). Three extra voltage points
  and the exact hourly sales of three areas are too little to change the picture for 29
  other areas.
- **The cheap win is a check meter on each large-customer transformer.** With every customer
  behind a transformer on a smart meter, a meter on the transformer itself closes the energy
  balance exactly. The stolen energy in those areas is then known to within 1% instead of
  about 20%: three transformer meters turn the smart-meter programme into an exact audit of
  its areas.
- **Limits.** A careful tamperer who keeps the same drop from the first day of the records and
  is sized well within their contract is hard to find from readings alone; the transformer
  check meter catches them in the area total. Customer behaviour is simulated and must be
  checked against real smart-meter data.

## Phase 12: ranking feeders and sharing crews across a district

A utility manages many feeders and has a limited number of inspection crews. The app's
**Compare feeders** tab ranks several feeders by estimated theft and plans where crews
should go across all of them (`district_table`, `district_plan` in
`src/slgrid/app_core.py`). Feeders can be uploaded as one .zip with a folder per feeder.

**Data.** 60 feeder-weeks (seed 606) with theft from 5% to 45% of consumption and a cable
fault on about 30%, grouped into 10 districts of 6 feeders. Crews can inspect 10% of a
district's transformer areas (19 of 192). Reproduce with `python scripts/run_district_study.py`.

| Question | Physics-informed | Unbilled energy (usual practice) |
|---|---|---|
| Rank correlation of feeders with true theft (Spearman) | 0.97 | 0.97 |
| Feeder with the most theft ranked first | 9 of 10 districts | 10 of 10 |
| Error in each feeder's total stolen (median) | 2.3% | (not an estimate of theft) |

| Where the 19 crews go | Share of the district's true stolen energy reached |
|---|---|
| **District-wide: the 19 areas with the highest estimated theft** | **46.7%** |
| Equal crews: the top 3 areas on every feeder (+1 district-wide) | 39.9% |
| Usual practice: the feeders with most unbilled energy, areas by combined rules | 29.6% |
| Random areas | 9.9% |

### What this shows
- **Honest note:** ranking whole feeders does not need the physics model. When feeders are
  of the same design, unbilled energy already orders them correctly (both 0.97). The
  physics model adds a reliable estimate of how much each feeder loses (within about 2%),
  which unbilled energy cannot give because it also holds technical losses and leaks.
- **Where the method pays off is the crew plan.** Sending crews to the highest-theft areas
  across the whole district reaches 47% of the stolen energy, against 40% with equal crews
  per feeder and 30% with the usual approach of concentrating on the worst feeders. That is
  1.6 times the usual practice for the same number of crews.
- **Limits.** All simulated feeders share one network design; real feeders differ in length,
  load and cable type, which would make unbilled energy a weaker guide and the physics
  estimate more useful. This should be checked on real feeders.

## Phase 13: inspection rounds with honest measurement

**Research first.** Before building, we reviewed what is publicly known (sources in the
research report [Inspection feedback data and methods](research/inspection_feedback_data_and_methods.md)):
- **No open dataset joins real feeder data with real inspection outcomes.** The State Grid
  of China theft dataset has real labels but no feeder layout. The Odisha (India) energy
  audits give real losses for 800 feeders but no inspection outcomes. The large labelled
  sets (Brazil, Endesa, Jamaica) are private.
- **Published hit rates:** untargeted inspections find theft in about 16-25% of visits
  (Jamaica 16%, Madhya Pradesh 25%, Endesa all visits 17%). Data-targeted campaigns reach
  28-50% (Rio 28%, Naturgy 36%, Endesa about 50%), and hotspot sweeps 54-80%. Ghana ECG
  (Ashanti West) surcharged 27% of the customers it visited.
- **Selection bias:** utilities that train on their own inspections learn the inspectors'
  map, not the thieves'. In a Brazilian utility, inspected customers differed measurably
  from the rest (Matthews correlation 0.22-0.27). Greedy "inspect the top" loops can
  converge on the wrong answer, and a small random share fixes this (IRS audit and
  insurance-fraud studies).
- **No published figures** exist for crew sensitivity, cost per inspection or repeat-theft
  rates, so these are user settings with labelled defaults.

**Built** (`src/slgrid/inspections.py`, the app's *Inspection rounds* tab):
- **Planner.** Each round sends 85% of visits to the areas with the highest estimated
  stolen energy and 15% (adjustable 10-20%) to a random sample of all other areas. Every
  area's chance of selection is written on the work list. Areas visited in the last two
  rounds are left out.
- **Log.** Ten outcome codes. Five count as non-technical loss found (bypass, tamper,
  illegal connection, non-fraud meter fault, billing error). CLEAN is a noisy negative.
  No access, not found and aborted for safety are missing, never clean. Cable faults are
  recorded separately.
- **Measurement.** Hit rate per arm with Wilson 95% intervals, and lift with a bootstrap
  interval. The share of all areas with theft is estimated by weighting every visit with
  1/(its chance of selection), allowing for theft crews miss. The verdict stays
  "insufficient evidence" until the sample size needed to tell model from random (80%
  power) is reached.
- **Learning.** The physics estimate stays label-free. Inspections are used to:
  - calibrate the displayed probability of theft (a two-parameter map fitted on outcomes
    weighted by 1/chance of selection, allowing for missed theft);
  - keep a Beta record per area, pooled per feeder;
  - turn measured theft into a lower bound on the area's stolen energy in the physics
    model.

**Closed-loop test.** 1,920 simulated transformer areas (60 feeders, 37% with theft), 8
monthly rounds of 60 visits, 30 repetitions:
- crews find theft that is there with probability 0.8 x share of premises checked (60-100%);
- no access on 8% of visits to clean areas and 15% to theft areas;
- a second situation where 15 feeders' substation meters read 3x high, fooling the physics
  model there.

Reproduce with `python scripts/run_inspection_study.py`.

| Accurate physics | Greedy (top only) | **Top + 15% random** | Top + random + learning | Spread (random, risk-tilted) | Random |
|---|---|---|---|---|---|
| True theft reached (MWh/week) | 1,041 | **950** | 945 | 467 | 202 |
| Hit rate of model visits | 61% | 61% | 61% | 38% | - |
| Hit rate of random visits | - | 21% | 19% | 22% | 22% |
| Share of areas with theft, as each policy would report it (true 36.7%) | 95% | **36.6%** | 33.6% | 34.8% | 34.5% |
| Model proven better than random after 8 rounds | never | **every run** | every run | 47% of runs | - |

With 15 substation meters reading 3x high, theft reached falls for every model policy
(greedy 902, top + random 837 MWh/week), and the conclusions are unchanged.

| Displayed probabilities (Brier score, lower is better) | Before | After 8 rounds |
|---|---|---|
| Starting from the default map | 0.076 | 0.074 |
| Starting from a wrong map (as if copied from another utility) | 0.146 | **0.072** |

### What this shows
- **The random 15% costs about 9% of the theft reached and buys honest numbers.** Top +
  random reaches 91% of the greedy policy's theft. Its estimate of how many areas have
  theft is right on average (36.6% against 36.7%, spread about 6.6 points). Its advantage
  over random visits is proven statistically in every run.
- **Greedy targeting cannot measure itself.** Its 61% hit rate, read as the state of the
  network, suggests theft in almost every area (95% after allowing for missed theft)
  against a true 37%. This is the trap the research warned about.
- **Spreading visits by risk (our first design) gave up half the theft reached.** The
  closed-loop test showed this, so the planner was changed to top + random. This is a
  deliberate change from the research report's suggested design, made on evidence.
- **Honest note: learning from inspections did not improve the ranking** (945 against 950
  MWh/week), even when 15 feeders' meters misled the model. The physics estimate already
  ranks areas well, and an area just inspected is not revisited soon. Re-ranking is
  therefore off by default in the app.
- **Where learning does help: honest probabilities.** Starting from a wrong probability map,
  calibration on the logged outcomes halves the error (Brier 0.146 to 0.072). This is what
  a utility needs when it takes the tool to a new country.
- **The interval shown in the app is deliberately cautious.** It comes from the Wilson
  interval of the random visits (the outcomes of the visited top areas are known). In 40
  test runs on a 6-feeder district with 4 rounds of 30 visits, it contained the true share
  every time, and the estimate averaged 28% against a true 31%.
- **Refusals bias every estimate slightly low** (about 2 points here), because theft areas
  refuse access more often. The app reports refusals beside every rate.
- **Limits.** Crew sensitivity (0.8) and refusal behaviour are assumptions. All numbers
  are from simulation; the module is built to produce the first real measurement when
  crews use it.

# Data sources

Every number used to build the simulated Sierra Leone feeder is traced here.

## S1. Rogers, Diawuo & Domfeh (2024) — Freetown household appliances
J. K. Rogers, F. A. Diawuo, M. K. Domfeh, "Assessing Peak Load Reduction Potential on
Distribution Network: A Case Study of Residential Households in Freetown, Sierra Leone,"
*International Journal of Energy Research*, 2024, Article ID 5587946.
https://doi.org/10.1155/2024/5587946 (open access, CC BY)

Survey of 250 Freetown households, May–June 2023.

Used for:
- `data/sources/rogers2024_appliances.csv` — appliance saturation, units per household,
  maximum diversified demand per customer (Table 2), utilisation factors (Fig. 4, read from chart).
- `data/sources/rogers2024_hourly_lvf.csv` — hourly load variation factor, % of each
  appliance's peak (Fig. 5). **Digitised by eye from the figure; approximate to ±5%.**

Key facts:
- Mean household size 5.65 persons.
- Diversified peak demand ≈ 0.30 kW per household (sum of Table 2 MDD values).
- Two daily peaks: morning 06:00–08:00, evening 18:00–21:00 (highest around 20:00).
- Lighting is the largest contributor to both peaks; then freezers, air conditioners, TVs, irons.
- Freetown measured city load (Sept 2023, 161 primary substations) ran about 55–70 MW,
  suppressed by load shedding; electrified households estimated at 114,220.

Not covered: commercial loads, outage schedule, theft methods.

## S2. Kamara, Kamara, Redwood-Sawyerr & Kallon (2021) — Freetown energy audit
B. Kamara, S. Kamara, J. A. S. Redwood-Sawyerr, D. V. V. Kallon, "Towards the Development of an
Energy Audit Program for Residents in Freetown," *Proc. 11th IEOM International Conference*,
Singapore, March 7–11, 2021, pp. 5998–6008.

Mostly a survey on awareness of energy audits; limited technical data. Used for:
- Freetown distribution/transmission losses of 34.5% (citing a 2017 government report).
- Theft and fraud types named by EDSA: illegal connections, wrong tariffs, multiple connections
  from a single meter, and multiple feeders on a single meter.
- About 90% of EDSA customers are in Freetown.

## S3. Trimble, Kojima, Perez Arroyo & Mohammadzadeh (2016) — World Bank
C. Trimble et al., "Financial Viability of Electricity Sectors in Sub-Saharan Africa: Quasi-Fiscal
Deficits and Hidden Costs," World Bank Policy Research Working Paper 7788, August 2016.

Used for:
- Sierra Leone T&D network losses of 33% in the study's reference year (~2014), with about
  80,894 customers.
- International benchmark used by the World Bank: combined losses of 10% count as good performance
  (technical losses under 10%, non-technical close to zero).

## S4. Kourouma & Camara (2026) — West African losses
M. L. Kourouma, Y. Camara, "Technical and Commercial Losses in the West-African Interconnected
Electricity Network: Challenges and Prospects," *Physical Science International Journal*, 30(3),
1–9, 2026. https://doi.org/10.9734/psij/2026/v30i3940

Used for:
- Sierra Leone: 12% technical + 18% commercial = 30% total (Table 3, WAPP/AfDB data).
- Regional averages: 9.1% technical, ~21% commercial.
- Neighbours for comparison: Liberia 37%, Guinea 35%, Ghana 31%.
- Theft forms: direct connection without a meter (common in dense urban and peri-urban areas),
  meter tampering, falsified manual readings.

## Summary: Sierra Leone loss levels across sources
| Source | Year of data | Technical | Commercial | Total |
|---|---|---|---|---|
| World Bank WP 7788 (S3) | ~2014 | — | — | 33% |
| Freetown gov't report via S2 | 2017 | — | — | 34.5% |
| Kourouma & Camara (S4) | ~2022 | 12% | 18% | 30% |
| Africa Energy Portal | Feb 2021 | — | — | 44% |
| World Bank Mission 300 compact | ~2024 | — | — | 53.2% |

Figures vary widely by year and by what is counted (some include unpaid bills).
**Simulation choice:** run scenarios with total feeder losses of 30%, 40% and 50%, with technical
losses of 10–15% coming from the physics and the rest injected as theft.

## S5. Mansaray, Diawuo & Bantinge (2024) — Freetown distribution network model
M. O. Mansaray, F. A. Diawuo, B. Bantinge, "Integration of multiple distributed solar PV (DSP) into
the grid: The case of the distribution network in Freetown, Sierra Leone," *Solar Compass*, 10,
100075, 2024. https://doi.org/10.1016/j.solcom.2024.100075 (open access, CC BY-NC-ND)

The only published power-flow model of the real Freetown network we have found (built in ETAP
from EDSA data). Used for:
- **Voltage levels:** 161 / 33 / 11 / 0.415 kV and 220 V; power is distributed at **11 kV**.
  Bulk supply point at Kingtom 161 kV.
- **Size:** 9 primary substations, **511 distribution transformers**, 30-bus ETAP model.
- **Load:** peak demand 95.8 MW (2023 model); average peak 2010–2022 about 60 MW; demand 115 MW
  vs generation 92 MW, so EDSA runs load shedding.
- **Losses:** technical + commercial losses 46% (World Bank 2020); technical losses of the modelled
  MV network alone 5.5 MW, about 4.8% of generation.
- **Voltages:** weakest 11 kV buses are Wellington (9.56 kV, 0.869 pu) and Roportee (9.91 kV,
  0.901 pu). Sierra Leone grid-code voltage limit is ±6% (0.94–1.06 pu).
- **Reliability:** average MV outage duration 22.64 hours per year (2020).
- Named areas: Kingtom, Wellington, Goderich, Roportee, Falcon Bridge, Cline Town, Aberdeen, Jui,
  Waterloo, York, Tombo.

Derived checks for our model:
- ~114,220 electrified households (S1) ÷ 511 transformers ≈ **220 households per transformer**.
- 0.30 kW diversified peak per household (S1) × 114,220 ≈ 34 MW residential peak, consistent with
  S1's modelled 31 MW evening peak.
- Our base feeder's lowest voltage (0.913 pu) sits between Roportee and Wellington, so it is a
  realistic stand-in for a weak Freetown feeder. Use **0.94 pu** as the voltage limit.

Not provided: line lengths, conductor types or impedances per feeder.

## Checked, not used
- Anerua (2021), B.Sc. thesis, Obafemi Awolowo University: Nigerian community demand response;
  no Sierra Leone network data. Points to Conteh et al. (2020) on Freetown demand response.

## S6. JICA / Yachiyo Engineering (2009) — Western Area power supply master plan
JICA & Yachiyo Engineering Co., "Final Report on the Master Plan Study on Power Supply in Western
Area in the Republic of Sierra Leone," National Power Authority, September 2009.
https://openjicareport.jica.go.jp/pdf/11961901_01.pdf (the copy we have covers chapters 1–5)

Data is from 2008, before EDSA was formed (the utility was then the NPA). Network has grown since
(S5 counts 511 transformers in 2023), but the physical character of the old 11 kV network remains.

Used for:
- **11 kV line lengths by area (Table 3.4-8), total 287 km, 35% overhead / 65% underground:**
  Kingtom 27 km (3 OH / 24 UG), Falcon Bridge 31 (1/30), Blackhall Road 45 (16/29),
  Congo Cross 22 (20/2), Brookfield 24 (2/22), Wilberforce 62 (23/39), Wellington 76 (36/40).
- **Distribution transformers (2008):** about 210 sets, 91 MVA total, about 10 overloaded.
- **Feeder capacity:** each 11 kV main line carries about 5 MW; area peak loads 3–9 MW
  (2008 preliminary study, total 45.3 MW of which 26.7 MW could not be supplied).
- **Power factor:** measured 0.79 on 11 kV feeders at Blackhall Road; 0.90 used for planning.
- **Source voltage:** delivery voltage at the Kingtom 11 kV switchgear was only 10–10.5 kV
  (0.91–0.95 pu), so feeders start below nominal voltage.
- **Low-voltage network:** some LV lines over 1 km long, unbalanced loading, many temporary
  connections; illegal connections named as a cause of LV losses.
- **Losses:** monthly T&D losses in 2008 ranged 37–52% (average 44%); ~71% in 2007.
- **Faults:** 214 faults on 11 kV lines in 2008, 67% on underground cables (mostly straight joints).
- **Load shedding (Table 3.4-17):** domestic zones supplied 7 pm–7 am on a 1-day-on / 2-days-off
  rotation; commercial district, banks and large customers 12 h daytime; hotels and essential
  services 24 h.
- **Daily load shape (2002):** evening peak around 10 pm, higher than the daytime peak.
- About 910 registered captive (backup) generators totalling 40 MW.

Not in this copy: line constants (Table 6.4-1, chapter 6), i.e. cable impedances per km.

## Simulation choices informed by S1–S6
- Source voltage: test 1.00 pu and 0.95 pu at the feeder head.
- Power factor: 0.90 baseline, 0.80 as a stress case.
- Households: ~220 per distribution transformer; diversified peak ~0.30 kW per household.
- Daily shape: from S1 hourly factors (morning and evening peaks).
- Supply: scenarios with full supply, and with zone rotation (evening-only supply on 1 day in 3).
- Losses: total 30% / 40% / 50%; technical part from the power flow, the rest injected as theft.
- Voltage limit: 0.94 pu (Sierra Leone grid code, ±6%).

## S7. JICA / Yachiyo Engineering (2009) — master plan, chapter 6
Same report as S6, second part (chapter 6, power flow analysis).

Used for:
- `data/sources/jica2009_conductors.csv` — **line constants of real Freetown cables and lines**
  (Table 6.4-1), converted from µΩ/m to Ω/km and pF/m to nF/km. Key 11 kV types:
  - Underground: Cu 185 mm² PILC (0.118 + j0.080 Ω/km, 355 A), Cu 185 mm² XLPE
    (0.128 + j0.093 Ω/km, 430 A), Al 240 mm² XLPE (0.161 + j0.109 Ω/km, 405 A).
  - Overhead: AAAC 120 mm² (0.270 + j0.354 Ω/km, 351 A), ACSR 240 mm² (0.120 + j0.323 Ω/km, 593 A).
- Typical 11 kV link lengths between substations: 1.5–3.4 km.
- **33/11 kV substation transformers (Table 6.4-2):** 10–20 MVA each, impedance about 9.5%
  (Regent 5 MVA, 8.0%).

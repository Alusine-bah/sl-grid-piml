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

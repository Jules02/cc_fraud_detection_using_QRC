# Arm B without input reset at L = 8 and 10

Same as `results/armb/sweep1` at L = 8 and 10 (1 input + 3 memory qubits, depth 4, `XYZ+XYZ-2body`, exact,
seeds 42–44, same circuits), but **without `--reset-inputs`**. L = 5 without reset is in `results/armb/noreset_l5`.
Jobs in `jobs.txt`, all rows in `summary.csv`.

Validation customer AP (mean ± std over 3 seeds), with amount AP and test customer AP:

| L | reset: val customer | no reset: val customer | no reset: test customer | reset: val amount | no reset: val amount | logistic: val customer |
|---|---|---|---|---|---|---|
| 5 | 0.247 ± 0.005 | 0.241 ± 0.005 | 0.241 | 0.986 | 0.985 | 0.209 |
| 8 | 0.314 ± 0.006 | 0.211 ± 0.042 | 0.213 | 0.982 | 0.915 | 0.287 |
| 10 | 0.330 ± 0.037 | 0.185 ± 0.029 | 0.190 | 0.982 | 0.845 | 0.338 |

Per seed (42 / 43 / 44), no reset, val customer: L = 8: 0.210 / 0.254 / 0.169; L = 10: 0.179 / 0.159 / 0.217.

## Findings

- **Without reset, the QRC is best at L = 5 and gets worse with longer windows** (0.241 → 0.211 → 0.185), while with reset
  it keeps improving (0.247 → 0.314 → 0.330). The crossover is between L = 5 and 8, earlier than `results/memcap` suggested.
- **No-reset falls below the linear baseline from L = 8** (0.211 vs 0.287; 0.185 vs 0.338 at L = 10).
- **The current transaction is lost:** amount AP drops from 0.98 to 0.92 (L = 8) and 0.85 (L = 10). With unitary
  dynamics the latest input is drowned by the earlier ones (the C(0) loss seen in `results/memcap`).
- Seed spread grows a lot without reset (std up to 0.04 vs ≤ 0.006 with reset at L = 8).

## Consequence

A dissipation mechanism inside the window (input reset, or swapping in fresh ancillae) is required for the windows
where the customer signal lives (L ≥ 8). Measurement-free operation with a single initialization per shot is only viable
up to L ≈ 5.

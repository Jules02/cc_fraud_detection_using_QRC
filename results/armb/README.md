# Arm B: reset-window QRC on raw log amounts

Arm B feeds a window of the customer's last L transactions, one per step, into the reservoir. The question is whether
the reservoir's memory helps on customer-compromise fraud (scenario 3), which is invisible in the current transaction
alone. Details of each experiment are in its `ANALYSIS.md`, with jobs in `jobs.txt` and all rows in `summary.csv`.

Common setup:
- Data: `data/sc3_c1000_x6` (1000 customers, 6 compromised per day), `--feature-set 1` (raw log amount).
- QRC: 1 input + 3 memory qubits, random circuit depth 4, `XYZ+XYZ-2body` (39 observables), seeds 42–44.
- Baselines: logistic regression on the flattened raw window; LSTM (hidden 32) on the same window.
- Metric: average precision, overall and per fraud category. Customer AP is the memory signal and amount AP the
  current-transaction signal. Terminal AP is ~0.006 everywhere (a customer's amounts can't see terminal compromise).

| folder | question | runs |
|---|---|---|
| `sweep1/` | Does the reset QRC gain from longer windows, and does it beat the linear baseline? | L ∈ {1, 2, 3, 5, 8, 10}, reset, exact, + logistic |
| `noreset_l5/` | Is the input reset needed at L = 5, and how much readout noise can the edge take? | L = 5, no reset, exact / gaussian 0.01 / 0.05 |
| `noreset_l8_l10/` | Is the input reset needed at longer windows? | L = 8, 10, no reset, exact |
| `lstm/` | How much signal does a trained nonlinear recurrence find in the same windows? | L ∈ {1, 2, 3, 5, 8, 10}, LSTM, seeds 42–44 |
| `plots/` | Figures for the above (`plot.py`) | |

## Results

Customer AP, mean over 3 seeds, validation / test:

| L | logistic | QRC, reset | QRC, no reset | LSTM |
|---|---|---|---|---|
| 1 | .147 / .158 | .134 / .145 | | .147 / .158 |
| 2 | .147 / .156 | .133 / .148 | | .233 / .243 |
| 3 | .171 / .175 | .176 / .185 | | .317 / .335 |
| 5 | .209 / .228 | .247 / .249 | .241 / .241 | .494 / .491 |
| 8 | .287 / .291 | .314 / .316 | .211 / .213 | .658 / .645 |
| 10 | .338 / .332 | .330 / .335 | .185 / .190 | .713 / .701 |

At L = 5 without reset, gaussian readout noise: precision 0.01 → .216 / .216, precision 0.05 → .134 / .140.

## Findings

1. **The reservoir uses its memory, and with reset it tracks or beats the linear baseline.** Customer AP rises from
   .13 (L ≤ 2) to .33 (L = 10). The QRC is ahead at L = 5 and 8 by ~.03 (several seed stds). At L = 10 seed 43 degrades
   (.288 val) while seeds 42 and 44 beat the baseline, so L = 10 needs ~10 seeds. Most of the overall-AP gain is amount
   AP: the QRC stays at .98 while logistic falls to .95.
2. **In-window dissipation is required for L ≥ 8** (`plots/dissipation.png`). Without reset, customer AP peaks at L = 5
   and then falls below logistic (.211 at L = 8, .185 at L = 10), while with reset it keeps rising. Amount AP without
   reset drops from .98 to .85: under unitary dynamics the current transaction is drowned by the earlier ones (the C(0)
   loss of `results/memcap`). The seed spread also grows (std up to .04 vs ≤ .006 with reset).
   Measurement-free operation with one initialization per shot is only viable up to L ≈ 5.
3. **The memory edge does not survive realistic readout noise** (`plots/noise_l5.png`). Precision 0.01 (~10⁴ shots per
   observable) brings the L = 5 QRC down to the logistic level (just above it on validation, below it on test).
   Precision 0.05 erases the memory entirely (at or below the L = 1 no-memory baseline).
4. **The QRC extracts only a small part of the signal in the window** (`lstm/`). The LSTM roughly doubles its customer
   AP from L = 3 (.49 vs .25 at L = 5, .71 vs .33 at L = 10), gains from L = 2 already, keeps amount AP ≥ .996 at every
   L, and is stable across seeds (std ≤ .005). The QRC's ~.03 edge over logistic regression is small next to this.

## Caveats and next steps

- The LSTM trains its recurrence, the QRC only a linear readout on fixed dynamics, so the LSTM is an upper bound rather
  than a like-for-like control. Next: a classical ESN (fixed random recurrence, same readout and width) to tell whether
  the gap comes from the reservoir paradigm or from the quantum reservoir; equal-width random features and
  HistGradientBoosting as further controls.
- Gaussian noise is readout-only. It doesn't test whether channel dissipation (amplitude damping, or fresh ancillae
  swapped in, as on silicon spin qubits) can replace the reset. That is the open question behind finding 2.
- Raise the feature signal relative to the noise (fewer memory qubits, encoding scaling, observable choice).
- One reservoir configuration only, no validation-based tuning yet.
- Splits: these are development experiments, so design decisions should be read off validation. Test is shown for
  reference; keep it untouched for the final comparison against the classical controls.

## Figures

- `plots/dissipation.png`: customer and amount AP vs L, reset vs no reset vs logistic.
- `plots/sweep1_categories.png`: AP per fraud category vs L, reset QRC vs logistic and LSTM.
- `plots/noise_l5.png`: L = 5 no-reset customer AP vs readout precision.

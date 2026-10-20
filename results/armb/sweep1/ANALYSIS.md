# Arm B sweep 1: window length L

Data: `data/sc3_c1000_x6` (1000 customers, 6 compromised per day). Input: `--feature-set 1` (raw log amount).
QRC: 1 input + 3 memory qubits, random circuit depth 4, input reset before each step, `XYZ+XYZ-2body`
(39 observables), exact density matrix, seeds 42–44. Baseline: logistic regression on the flattened raw window.
Jobs in `jobs.txt`, all rows in `summary.csv`.

Average precision (QRC: mean ± std over 3 seeds):

| L | model | val AP | val customer | val amount | test AP | test customer | reservoir |
|---|---|---|---|---|---|---|---|
| 1 | baseline | 0.588 | 0.147 | 1.000 | 0.613 | 0.158 | |
| 1 | qrc | 0.585 | 0.134 ± 0.001 | 1.000 | 0.609 | 0.145 ± 0.001 | 3.3 min |
| 2 | baseline | 0.587 | 0.147 | 0.987 | 0.610 | 0.156 | |
| 2 | qrc | 0.594 | 0.133 ± 0.002 | 0.996 | 0.618 | 0.148 ± 0.001 | 3.9 min |
| 3 | baseline | 0.587 | 0.171 | 0.972 | 0.609 | 0.175 | |
| 3 | qrc | 0.611 | 0.176 ± 0.001 | 0.993 | 0.632 | 0.185 ± 0.000 | 4.4 min |
| 5 | baseline | 0.596 | 0.209 | 0.956 | 0.618 | 0.228 | |
| 5 | qrc | 0.635 | 0.247 ± 0.005 | 0.986 | 0.652 | 0.249 ± 0.004 | 5.9 min |
| 8 | baseline | 0.617 | 0.287 | 0.951 | 0.630 | 0.291 | |
| 8 | qrc | 0.655 ± 0.003 | 0.314 ± 0.006 | 0.982 | 0.669 | 0.316 ± 0.009 | 8.2 min |
| 10 | baseline | 0.630 | 0.338 | 0.952 | 0.641 | 0.332 | |
| 10 | qrc | 0.660 ± 0.011 | 0.330 ± 0.037 | 0.982 | 0.674 | 0.335 ± 0.036 | 8.5 min |

Terminal AP is ~0.006 everywhere (a single customer's amounts can't see terminal compromise).

## Findings

- **The reservoir uses its memory.** Customer AP rises from 0.134 (L ≤ 2) to 0.31–0.33 (L = 8–10).
- **It beats the linear baseline on the same window at L = 5 and 8.** Customer AP gains are +0.038 (val) / +0.021 (test)
  at L = 5 and +0.027 / +0.025 at L = 8, each several seed stds. Bootstrap CIs are not computed yet.
- **At L = 10 the mean gain disappears, because one seed degrades.** Customer AP per seed, L = 8 → 10:
  seed 42: 0.317 → 0.345 (val), 0.310 → 0.354 (test); seed 43: 0.307 → 0.288, 0.311 → 0.294;
  seed 44: 0.318 → 0.357, 0.327 → 0.358. Seeds 42 and 44 keep improving at about the baseline's rate and beat it
  at L = 10 (baseline: 0.338 val, 0.332 test). Seed 43 loses its older inputs. The seed spread grows from ≤ 0.006
  (L ≤ 8) to 0.037. Long-window memory depends on the random circuit drawn (as in `results/memcap`), so L = 10 needs
  more seeds (~10).
- **At L ≤ 2 the QRC is below the baseline on customer AP** (0.134 vs 0.147). The encoding/readout loses some resolution
  on the current amount.
- **Most of the overall-AP gain is the amount category.** The QRC stays at 0.98–0.99 while the linear baseline drops to 0.95
  as L grows. A threshold on the current amount is easy for a nonlinear map and hard for a linear readout diluted over L inputs.

## Caveats

- The baseline is linear. The QRC's advantage over it may be plain nonlinearity, not quantum memory. Next controls on
  the same raw windows: equal-width random features (tanh/cos), a classical ESN, HistGradientBoosting (balanced).
- Only one reservoir configuration (3 memory qubits, depth 4). Validation-based tuning is not done yet.

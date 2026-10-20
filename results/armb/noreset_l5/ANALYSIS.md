# Arm B at L = 5 without input reset: exact vs sampling noise

Data: `data/sc3_c1000_x6`, `--feature-set 1` (raw log amount), window length 5.
QRC: same as `results/armb/sweep1` at L = 5 (1 input + 3 memory qubits, depth 4, `XYZ+XYZ-2body`, seeds 42–44,
same circuits), but **without `--reset-inputs`**. Backends: exact, and `gaussian` (exact expectation values plus
N(0, precision²) noise on each one, at the final readout only) at precision 0.01 (~10⁴ shots per observable) and 0.05 (~400).
Jobs in `jobs.txt`, all rows in `summary.csv`.

Average precision (mean ± std over 3 seeds):

| model | val AP | val customer | val amount | test AP | test customer |
|---|---|---|---|---|---|
| no reset, exact | 0.632 | 0.241 ± 0.005 | 0.985 | 0.647 | 0.241 ± 0.005 |
| no reset, gaussian 0.01 | 0.621 | 0.216 ± 0.010 | 0.980 | 0.637 | 0.216 ± 0.004 |
| no reset, gaussian 0.05 | 0.565 | 0.134 ± 0.028 | 0.924 | 0.587 | 0.140 ± 0.024 |
| reset, exact (sweep1) | 0.635 | 0.247 ± 0.005 | 0.986 | 0.652 | 0.249 ± 0.004 |
| logistic, raw window L = 5 | 0.596 | 0.209 | 0.956 | 0.618 | 0.228 |
| logistic, L = 1 (no memory) | 0.588 | 0.147 | 1.000 | 0.613 | 0.158 |

Reservoir time ~4 min per run (vs ~6 min with reset).

## Findings

- **Without noise, dropping the reset costs little at L = 5:** customer AP 0.241 vs 0.247 (about one seed std).
  Consistent with `results/memcap` and sweep 2: the reset only matters from L ≈ 8.
- **Precision 0.01 removes the advantage over the linear baseline.** Customer AP 0.216 vs 0.209 (val) and
  0.216 vs 0.228 (test). The amount category is still fine (0.98).
- **Precision 0.05 removes the memory.** Customer AP 0.134 / 0.140 is at the L = 1 (no-memory) level, and amount AP
  (0.924) falls below the linear baseline. Seed 44 is the worst at both noise levels (val customer 0.102 at 0.05).
- The QRC's ~+0.03 customer-AP edge at L = 5 is smaller than the cost of ~10⁴ shots per observable per transaction.
  With 39 observables and ~690k transactions, that budget is already far beyond realistic hardware.

## Caveats

- Gaussian noise is readout noise only (added once after the last step). It is not a channel in the dynamics, so it
  says nothing about dissipation-based forgetting.
- Reset + noise is not measured here. With reset the features were smaller in `results/memcap`, so it is expected to
  do no better.
- Next: raise the feature signal relative to the noise (fewer memory qubits, encoding scaling, observable choice), and
  the nonlinear classical controls on the same windows.

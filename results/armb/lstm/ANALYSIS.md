# Arm B LSTM control

Same windows as `results/armb/sweep1`: data `data/sc3_c1000_x6`, `--feature-set 1` (raw log amount),
L ∈ {1, 2, 3, 5, 8, 10}, seeds 42–44. Model: `lstm` CLI defaults (1-layer LSTM, hidden 32, linear head on the last
step, ≤ 50 epochs, early stopping on validation loss, patience 8). Jobs in `jobs.txt`, all rows in `summary.csv`.

Average precision (mean over 3 seeds; seed std ≤ .005 everywhere), with the sweep 1 numbers for reference:

| L | val AP | val customer | val amount | test AP | test customer | QRC val / test customer | logistic val / test customer | train time |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.588 | 0.147 | 1.000 | 0.613 | 0.158 | .134 / .145 | .147 / .158 | 0.4 min |
| 2 | 0.624 | 0.233 | 0.999 | 0.646 | 0.243 | .133 / .148 | .147 / .156 | 0.8 min |
| 3 | 0.657 | 0.317 | 0.998 | 0.676 | 0.335 | .176 / .185 | .171 / .175 | 0.9 min |
| 5 | 0.711 | 0.494 | 0.997 | 0.722 | 0.491 | .247 / .249 | .209 / .228 | 1.5 min |
| 8 | 0.758 | 0.658 | 0.996 | 0.763 | 0.645 | .314 / .316 | .287 / .291 | 1.9 min |
| 10 | 0.772 | 0.713 | 0.997 | 0.777 | 0.701 | .330 / .335 | .338 / .332 | 2.2 min |

## Findings

- **The LSTM roughly doubles the QRC's customer AP from L = 3** (.49 vs .25 at L = 5, .71 vs .33 at L = 10), and gains
  from L = 2 already, where the QRC and logistic regression gain nothing. The QRC's ~.03 edge over logistic regression is
  a small fraction of the signal in the window.
- **It keeps the current transaction at every L** (amount AP ≥ .996), where logistic regression falls to .95 and the
  QRC to .98.
- **Very stable across seeds** (std ≤ .005), unlike the QRC at L = 10 (std .037).
- At L = 1 it equals logistic regression (.147): with a single transaction there is nothing nonlinear to exploit.

## Caveats

- Not a like-for-like control: the LSTM trains its recurrence, the QRC only a linear readout on fixed dynamics. It is
  an upper bound on what the window holds. The equal-footing control is a classical ESN (fixed random recurrence,
  same readout, same width): it tells whether the gap is the reservoir paradigm or the quantum reservoir.
- Default hyperparameters, no tuning.

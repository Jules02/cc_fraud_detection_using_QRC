# Arm A: Amount Features

Compares a classical random feature map with a QELM on the same
transaction-level features (`L=1`), custom data, and paired seeds.

- **Data:** `data/sc3_c1000_x6`
- **Feature set 20:** `CUSTOMER_AMOUNT_DEVIATION_30D`, `CUSTOMER_AMOUNT_RATIO_90D`,
  `CUSTOMER_AMOUNT_SHIFT_7D_30D`, `CUSTOMER_ACTIVITY_AMOUNT_30D`, `TX_AMOUNT_LOG`
- **Random map:** select `tanh` (default) or `rff` with
  `--random-map-method`; width 5 matches QELM's five Z observables
- **QELM:** 5 input qubits, no memory qubits, depth 4, exact backend
- **Metrics:** average precision overall and by fraud category;

Run jobs from the repository root with `python -m experiments.sweep
results/arma/jobs.txt`. Each run is saved under `tanh/` or `qelm/`.

Generate validation and test comparison plots with
`python results/arma/plot.py`; figures are saved in `plots/`.

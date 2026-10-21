# sc3_c1000_x6: scenario-3-rich dataset for arm B

Generated 2026-10-16 (commit b564a3a + the `n_compromised_customers` parameter of `add_frauds`):

```python
from qrc.datasets.create_dataset import run_simulation
from qrc.datasets.split_dataset import main as split_and_save

customers, terminals, tx = run_simulation(
    n_customers=1000, n_terminals=10000, nb_days=365, start_date="2025-01-01", r=5,
    n_compromised_customers=6, output_dir="data/sc3_c1000_x6/raw",
)
split_and_save(tx, validation_start="2025-09-13", test_start="2025-11-06", output_dir="data/sc3_c1000_x6")
```

- Same simulator settings as the shared dataset, except 1000 customers (instead of 5000) and 6 customers
  compromised per day in scenario 3 (instead of 3): the same per-customer rate as 30/day for 5000 customers.
- The genuine transactions and the scenario-2 frauds are exactly those of customers 0–999 in the shared dataset
  (checked). Only scenario 3 differs.
- Split dates are the shared dataset's (`split_dataset`'s month-based defaults differ).

Frauds per category:

| split      | transactions | amount | terminal | customer |
| ---------- | ------------ | ------ | -------- | -------- |
| train      | 483,807      | 7,122  | 2,495    | 6,176    |
| validation | 102,003      | 1,536  | 575      | 1,336    |
| test       | 106,153      | 1,680  | 581      | 1,401    |

Load with `python -m experiments.cli ... --data load --data-dir data/sc3_c1000_x6`.

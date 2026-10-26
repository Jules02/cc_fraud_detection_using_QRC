# IPC Benchmarks

This directory contains Information Processing Capacity (IPC) benchmark logs,
plotting scripts, and generated figures. Benchmark logs are grouped by
experiment and Hamiltonian.

## Experiments

- **Precision** (`precision/`): compares IPC capacities across shot precision
  values. The job definitions use precisions `0.01`, `0.05`, and `0.1`, with
  seeds 1–50.
- **Reservoir depth** (`depth/`): compares IPC capacities across reservoir
  depths. The job definitions use depths 1, 3, 8, and 15, with seeds 1–10.
- **Window length** (`window/`): compares IPC capacities across window lengths
  5, 10, 15, and 20, with seeds 1–10. The current job definitions include
  `random`, `tfim`, and `native-exchange`; they use different reservoir depths
  for `random` (4) and the other Hamiltonians (2).

The benchmark job definitions are in [`jobs.txt`](jobs.txt). They invoke
`experiments.run_benchmarks` with the `ipc` task and write each run's console
output to a `.txt` log under the corresponding experiment and Hamiltonian
directory.

## Metrics

The plot scripts read these IPC values from the JSON block in each log:

- `degree_1`
- `degree_2`
- `degree_3`
- `total_capacity`

Plots show the mean across available seeds and standard-deviation error bars.
Logs without a valid IPC JSON block are skipped.

## Generate plots

Run these commands from the repository root:

```bash
python results/benchmarks/ipc/precision/plot.py
python results/benchmarks/ipc/depth/plot.py
python results/benchmarks/ipc/window/plot.py
```

All generated figures are saved in `results/benchmarks/ipc/plots/`:

- `ipc_capacity_all_hamiltonians.png` — precision comparison
- `ipc_random_depth.png` and `ipc_native-exchange_depth.png` — depth comparisons
- `ipc_capacity_vs_window_length_all_hamiltonians.png` — window-length
  comparison for the Hamiltonians configured in `window/plot.py`

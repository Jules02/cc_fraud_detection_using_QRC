# Observable sets under shot and gate noise

What measuring more observables buys a quantum reservoir (memory, fraud AP), and what it costs on current spin-qubit
hardware. This folder has everything needed to reproduce the figures and numbers in `report/`.

## Question

With exact expectation values, more observables always help: each measurement setting gives every Pauli string that
commutes with it for free, and the reservoir's memory sits largely in high-weight correlators. On hardware, each
setting takes its own share of the shot budget, and gate noise shrinks the features until shot noise drowns them.
Which observable set should we measure at a given fidelity and shot budget?

## Setup (common to all phases)

- **Reservoir:** the CLI's TFIM (`--hamiltonian tfim`): H = Σ J_ij X_iX_j + Σ Z_i, J_ij ~ U(0, 1) on a chain,
  evolution time 1, 2 Trotter steps.
- **Input:** 1 input qubit with RY(π u) encoding, **reset before every step**. Windows of L = 10 steps, seeds 42–51.
- **Measurement:** the hardware only reads Z, so a measurement setting is a single-qubit frame change followed by Z
  on every qubit. Every Z-type string of the frame comes free.
- **Observable sets:**

  | Set | Settings | What it measures |
  |---|---|---|
  | local Z (`Z-w1`) | 1 | single-qubit Z |
  | `Z-strings` | 1 | every Z string |
  | `frames-3` | 3 | Z-strings in the Z frame and in 2 fixed random single-qubit frames |
  | `XYZ-strings` | 3 | every single-letter X…, Y…, Z… string |
  | `XYZ+XYZ-2body` | 9 | local X/Y/Z plus nearest-neighbour two-body products |
  | `all-Pauli` | 3ⁿ | full tomography, n = 4 only |

- **Shots:** a budget of B shots per window is split evenly over a set's settings. Every window and setting is sampled
  from its exact outcome distribution, for train and test alike. "Exact" means exact expectation values.

## Phases

| Script | Phase | Task | Noise model | Output |
|---|---|---|---|---|
| `observables.py` | P2 | short-term memory, n = 4, 5 | depolarizing after every native gate (rx/ry/rz/rzz) at 2-qubit fidelity 100 / 99.5 / 99 / 98 %, 1-qubit error = 2-qubit error / 5, 0.07 % readout and reset error; 98 % with 1 % readout as a check | `data/p2/` |
| `spinpulse_p3.py` | P3 | short-term memory, n = 5 | SpinPulse pulse-level 1/f dephasing, T2* calibrated (`--calibrate`) so RZZ(π/2) is 99 / 98 % accurate; basis changes applied as pulses | `data/p3/` |
| `fraud_p4.py` | P4 | arm B fraud (`sc3_c1000_x6`, feature set 1), n = 4 | P2's model at 100 / 99 / 98 %, applied as one noisy channel per step; CLI readout and metrics; logistic regression on the raw window as reference | `data/p4/` |
| `report_figures.py` | — | — | — | `report/` figures, CSVs and `facts.md` |

Each script has a `--check` that validates its simulator:
- P2 against `NumpyBackend`;
- P3 against `SpinPulseBackend` on the same noise traces;
- P4's per-step channel against the gate-by-gate simulation.

## Reproduce

1. **Data (CI on the fork, about 15–30 minutes per workflow).**

   ```
   gh workflow run observable_sets_memory.yml -f phase=p2                  # P2, n = 4,5
   gh workflow run observable_sets_memory.yml -f phase=p3 -f n_qubits=5    # P3
   gh workflow run observable_sets_fraud.yml                               # P4 (regenerates the dataset)
   gh run download <run-id> -n p2-data -D results/observable_sets/data/p2  # likewise p3-data, p4-data
   ```

   Or run locally, one call per seed (and n), from the repository root:

   ```
   python results/observable_sets/observables.py --n 5 --seed 42
   python results/observable_sets/spinpulse_p3.py --n 5 --seed 42
   python results/observable_sets/fraud_p4.py --seed 42   # needs data/sc3_c1000_x6
   ```

2. **Figures and numbers.** Run `python results/observable_sets/report_figures.py`. It writes `report/fig1`–`fig4`
   (PNG, PDF, CSV) and `report/facts.md`.

The scripts are tracked in git; `data/` and the generated files in `report/` are not.

## Not included

These were run during the study but are not used in the report:
- runs without input reset;
- the parity task;
- the other observable sets (`Z-w2`–`Z-w4`, `frames-2`, local `XYZ`, a mixing layer before the Z readout).

They are in the history of `results/observables/` on the `observables-sweep` branch.

"""Plot IPC capacities against window length for each Hamiltonian."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks" / "ipc" / "window"
PLOTS_DIR = PROJECT_ROOT / "results" / "benchmarks" / "ipc" / "plots"

HAMILTONIANS = ["random", "native-exchange", "tfim"]
WINDOW_LENGTHS = [5, 10, 15, 20]
SEEDS = range(1, 11)

CAPACITY_KEYS = {
    "degree_1": "1st-degree IPC capacity",
    "degree_2": "2nd-degree IPC capacity",
    "degree_3": "3rd-degree IPC capacity",
    "total_capacity": "Total IPC capacity",
}

LOG_PATTERN = re.compile(
    r"shots_precision0\.01_window(?P<window>\d+)_seed(?P<seed>\d+)\.txt$"
)


def read_ipc_capacity(log_path):
    """Extract IPC capacity metrics from one log file."""
    text = log_path.read_text(errors="replace")
    marker = "Information Processing Capacity (IPC)"
    marker_index = text.find(marker)

    if marker_index == -1:
        print(f"Warning: IPC results not found in {log_path}")
        return None

    json_start = text.find("{", marker_index + len(marker))
    if json_start == -1:
        print(f"Warning: IPC JSON not found in {log_path}")
        return None

    try:
        data, _ = json.JSONDecoder().raw_decode(text[json_start:])
    except json.JSONDecodeError:
        print(f"Warning: Could not parse IPC JSON in {log_path}")
        return None

    if not all(key in data for key in CAPACITY_KEYS):
        print(f"Warning: Missing capacity metrics in {log_path}")
        return None

    return {key: float(data[key]) for key in CAPACITY_KEYS}


def collect_results():
    """Collect capacities by Hamiltonian, window length, and seed."""
    results = {
        hamiltonian: {window: {} for window in WINDOW_LENGTHS}
        for hamiltonian in HAMILTONIANS
    }

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    for hamiltonian in HAMILTONIANS:
        hamiltonian_dir = RESULTS_DIR / hamiltonian
        if not hamiltonian_dir.is_dir():
            print(f"Warning: Hamiltonian directory not found: {hamiltonian_dir}")
            continue

        for log_path in hamiltonian_dir.glob("*.txt"):
            match = LOG_PATTERN.search(log_path.name)
            if not match:
                continue

            window = int(match.group("window"))
            seed = int(match.group("seed"))

            if window not in WINDOW_LENGTHS or seed not in SEEDS:
                continue

            capacities = read_ipc_capacity(log_path)
            if capacities is not None:
                results[hamiltonian][window][seed] = capacities

    return results


def plot_capacities(results):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for ax, (metric, ylabel) in zip(axes, CAPACITY_KEYS.items()):
        for hamiltonian in HAMILTONIANS:
            windows = []
            means = []
            stds = []

            for window in WINDOW_LENGTHS:
                seed_results = results[hamiltonian][window]
                values = [
                    seed_results[seed][metric] for seed in SEEDS if seed in seed_results
                ]

                if not values:
                    print(
                        f"Warning: No results for {hamiltonian}, "
                        f"window length {window}, {metric}"
                    )
                    continue

                windows.append(window)
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)

                if len(values) < len(SEEDS):
                    print(
                        f"Note: {hamiltonian}, window={window}: "
                        f"{len(values)}/{len(SEEDS)} valid seeds"
                    )

            if windows:
                ax.errorbar(
                    windows,
                    means,
                    yerr=stds,
                    marker="o",
                    capsize=4,
                    linewidth=2,
                    label=hamiltonian,
                )

        ax.set_xlabel("IPC window length")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(WINDOW_LENGTHS)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("IPC Capacity vs Window Length by Hamiltonian")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = PLOTS_DIR / "ipc_capacity_vs_window_length_all_hamiltonians.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    results = collect_results()
    plot_capacities(results)


if __name__ == "__main__":
    main()

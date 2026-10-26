"""Plot IPC capacities against reservoir depth for each Hamiltonian."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks" / "ipc" / "depth"
PLOTS_DIR = PROJECT_ROOT / "results" / "benchmarks" / "ipc" / "plots"

HAMILTONIANS = ["random", "native-exchange"]
SEEDS = range(1, 11)

CAPACITY_KEYS = {
    "degree_1": "1st-degree IPC capacity",
    "degree_2": "2nd-degree IPC capacity",
    "degree_3": "3rd-degree IPC capacity",
    "total_capacity": "Total IPC capacity",
}

# Matches files such as:
# shots_precision0.01_depth1_seed1.txt
LOG_PATTERN = re.compile(
    r"shots_precision0\.01_depth(?P<depth>\d+)_seed(?P<seed>\d+)\.txt$"
)


def read_ipc_capacity(log_path):
    """Extract IPC capacity values from one benchmark log."""
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
        results, _ = json.JSONDecoder().raw_decode(text[json_start:])
    except json.JSONDecodeError:
        print(f"Warning: Could not parse IPC JSON in {log_path}")
        return None

    if not all(key in results for key in CAPACITY_KEYS):
        print(f"Warning: Missing capacity metrics in {log_path}")
        return None

    return {key: float(results[key]) for key in CAPACITY_KEYS}


def collect_results():
    """Collect capacities, grouped by Hamiltonian, depth, and seed."""
    results = {hamiltonian: {} for hamiltonian in HAMILTONIANS}

    for hamiltonian in HAMILTONIANS:
        directory = RESULTS_DIR / hamiltonian

        if not directory.exists():
            print(f"Warning: Directory not found: {directory}")
            continue

        for log_path in directory.glob("*.txt"):
            match = LOG_PATTERN.search(log_path.name)
            if not match:
                continue

            depth = int(match.group("depth"))
            seed = int(match.group("seed"))

            if seed not in SEEDS:
                continue

            capacities = read_ipc_capacity(log_path)
            if capacities is not None:
                results[hamiltonian].setdefault(depth, {})[seed] = capacities

    return results


def plot_capacities(results):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for hamiltonian in HAMILTONIANS:
        available_depths = sorted(results[hamiltonian].keys())
        if not available_depths:
            print(f"Warning: No IPC depth results found for {hamiltonian}")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        for ax, (metric, ylabel) in zip(axes, CAPACITY_KEYS.items()):
            for depth in available_depths:
                seed_results = results[hamiltonian][depth]
                values = [
                    seed_results[seed][metric] for seed in SEEDS if seed in seed_results
                ]

                if not values:
                    print(
                        f"Warning: No data for {hamiltonian}, "
                        f"depth={depth}, metric={metric}"
                    )
                    continue

                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0.0

                ax.errorbar(
                    depth,
                    mean,
                    yerr=std,
                    marker="o",
                    capsize=4,
                    linestyle="none",
                )

            depths_with_data = []
            means = []
            for depth in available_depths:
                seed_results = results[hamiltonian][depth]
                values = [
                    seed_results[seed][metric] for seed in SEEDS if seed in seed_results
                ]
                if values:
                    depths_with_data.append(depth)
                    means.append(np.mean(values))

            if depths_with_data:
                ax.plot(depths_with_data, means, marker="o")

            label = ""
            if hamiltonian == "random":
                label = "reservoir circuit depth"
            else:
                label = "number of trotter steps"

            ax.set_xlabel(f"{label}")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.set_xticks(available_depths)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"IPC Capacity vs {label} — {hamiltonian}")
        fig.tight_layout()

        output_path = PLOTS_DIR / f"ipc_{hamiltonian}_depth.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")


def main():
    results = collect_results()
    plot_capacities(results)


if __name__ == "__main__":
    main()

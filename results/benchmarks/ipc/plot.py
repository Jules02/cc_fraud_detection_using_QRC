"""Plot IPC benchmark results: four subplots per Hamiltonian.

Expected log filenames:
    <backend>_precision<precision>_seed<seed>.txt

Expected log contents:
    Information Processing Capacity (IPC)
    {
        "degree_1": ...,
        "degree_2": ...,
        "degree_3": ...,
        "total_capacity": ...
    }


"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Directory containing this script and the Hamiltonian result directories.
RESULTS_DIR = Path(__file__).resolve().parent

HAMILTONIANS = [
    "random",
    "tfim",
    "tfim-longitudinal",
    "tfim-native",
    "native-exchange",
]

CAPACITY_KEYS = {
    "degree_1": "1st-degree IPC capacity",
    "degree_2": "2nd-degree IPC capacity",
    "degree_3": "3rd-degree IPC capacity",
    "total_capacity": "Total IPC capacity",
}

LOG_PATTERN = re.compile(
    r"(?P<backend>.+)_precision(?P<precision>[-+0-9.eE]+)" r"_seed(?P<seed>\d+)\.txt$"
)


def read_ipc_capacities(log_path):
    """Extract individual degree and total IPC capacities from a log."""
    text = log_path.read_text(errors="replace")

    marker = "Information Processing Capacity (IPC)"
    marker_pos = text.find(marker)

    if marker_pos == -1:
        return None

    json_text = text[marker_pos + len(marker) :].strip()

    try:
        result, _ = json.JSONDecoder().raw_decode(json_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(result, dict):
        return None

    capacities = {}

    for key in CAPACITY_KEYS:
        value = result.get(key)

        if isinstance(value, (int, float)) and np.isfinite(value):
            capacities[key] = float(value)

    return capacities if capacities else None


def collect_results(hamiltonian_dir):
    """Group capacities by metric, backend, precision, and seed."""
    results = {key: {} for key in CAPACITY_KEYS}

    for log_path in hamiltonian_dir.glob("*.txt"):
        match = LOG_PATTERN.match(log_path.name)

        if not match:
            continue

        backend = match.group("backend")
        precision = float(match.group("precision"))
        seed = int(match.group("seed"))

        capacities = read_ipc_capacities(log_path)

        if capacities is None:
            print(f"Skipping unreadable or incomplete log: {log_path}")
            continue

        group_key = (backend, precision)

        for metric, capacity in capacities.items():
            results[metric].setdefault(group_key, {})[seed] = capacity

    return results


def plot_hamiltonian(hamiltonian):
    """Create one PNG with four IPC plots for a Hamiltonian."""
    hamiltonian_dir = RESULTS_DIR / hamiltonian

    if not hamiltonian_dir.is_dir():
        print(f"No directory found for {hamiltonian}: {hamiltonian_dir}")
        return

    results = collect_results(hamiltonian_dir)

    if not any(results.values()):
        print(f"No valid IPC results found for {hamiltonian}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, CAPACITY_KEYS.items()):
        metric_results = results[metric]
        plotted = False

        # Plot each backend as a separate connected series.
        backends = sorted({backend for backend, _ in metric_results})

        for backend in backends:
            if backend == "estimator-exact":
                continue

            points = []

            for (result_backend, precision), seed_results in metric_results.items():
                if result_backend != backend:
                    continue

                values = np.array(
                    list(seed_results.values()),
                    dtype=float,
                )

                if len(values) == 0:
                    continue

                mean = values.mean()
                std = values.std(ddof=1) if len(values) > 1 else 0.0

                points.append((precision, mean, std, len(values)))

            # Connect points from lowest to highest precision.
            points.sort(key=lambda point: point[0])

            if not points:
                continue

            precisions = [point[0] for point in points]
            means = [point[1] for point in points]
            stds = [point[2] for point in points]

            ax.errorbar(
                precisions,
                means,
                yerr=stds,
                fmt="o-",
                capsize=4,
                label=f"{backend} estimator",
            )

            plotted = True

        # Plot exact estimator results as a horizontal reference line.
        exact_samples = [
            value
            for (backend, _), seed_results in metric_results.items()
            if backend == "estimator-exact"
            for value in seed_results.values()
        ]

        if exact_samples:
            exact_mean = float(np.mean(exact_samples))
            exact_n = len(exact_samples)

            ax.axhline(
                exact_mean,
                linestyle="--",
                label=(f"estimator-exact " f"(mean={exact_mean:.3f}, n={exact_n})"),
            )

            plotted = True

        ax.set_title(title)
        ax.set_xlabel("Precision")
        ax.set_ylabel("IPC capacity")
        ax.grid(True, alpha=0.3)

        if plotted:
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No valid results",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    fig.suptitle(f"IPC capacity — {hamiltonian}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = hamiltonian_dir / "ipc_capacity.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    for hamiltonian in HAMILTONIANS:
        plot_hamiltonian(hamiltonian)


if __name__ == "__main__":
    main()

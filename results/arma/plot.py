"""Plot Arm A Tanh, RFF, and QELM results found under results/arma.

Reads completed metrics.json/config.json run pairs recursively.
Does not run experiments.

Usage from the repository root:
    python results/arma/plot.py
    python results/arma/plot.py --split validation
    python results/arma/plot.py --split test
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent
PLOTS_DIR = RESULTS_DIR / "plots"

METRICS = (
    ("average_precision", "All fraud"),
    ("average_precision_amount", "Amount"),
    ("average_precision_terminal", "Terminal"),
    ("average_precision_customer", "Customer"),
)

METHODS = (
    ("classical", "Tanh random map"),
    ("rff", "Random Fourier features (RFF)"),
    ("qrc", "QELM"),
)

COLORS = {
    "classical": "#e07a32",
    "rff": "#2a9d8f",
    "qrc": "#3478b9",
}


def get_method(args):
    """Identify Tanh, RFF, or QELM from saved run configuration."""
    command = args.get("command")

    if command == "qrc":
        return "qrc"

    if command == "rff":
        return "rff"

    if command == "classical":
        # Newer configurations may store the map type separately.
        map_method = args.get(
            "random_map_method",
            args.get("classical_model", "tanh"),
        )

        if map_method == "rff":
            return "rff"
        if map_method in ("tanh", "classical"):
            return "classical"

    return None


def load_runs():
    """Load complete result pairs; skip incomplete or unrelated folders."""
    runs = []

    for metrics_path in sorted(RESULTS_DIR.rglob("metrics.json")):
        try:
            config_path = metrics_path.with_name("config.json")

            if not config_path.is_file():
                print(f"Skipping {metrics_path.parent}: config.json is missing")
                continue

            metrics = json.loads(metrics_path.read_text())
            config = json.loads(config_path.read_text())
            args = config.get("args", {})

            method = get_method(args)
            if method is None:
                continue

            seed = args.get("seed")
            if seed is None:
                print(f"Skipping {metrics_path.parent}: run seed is missing")
                continue

            # Require both evaluation splits to be present.
            if not all(split in metrics for split in ("validation", "test")):
                print(
                    f"Skipping {metrics_path.parent}: "
                    "validation or test metrics are incomplete"
                )
                continue

            runs.append(
                {
                    "command": method,
                    "seed": int(seed),
                    "metrics": metrics,
                    "path": metrics_path.parent,
                }
            )

        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"Skipping unreadable run {metrics_path.parent}: {exc}")

    return runs


def plot_split(runs, split):
    """Plot mean AP, standard deviation, and individual seeds."""
    available_metrics = [
        (key, label)
        for key, label in METRICS
        if any(key in run["metrics"][split] for run in runs)
    ]

    if not available_metrics:
        print(f"No AP metrics found for the {split} split; no plot created.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(available_metrics), dtype=float)

    methods = [
        (command, label)
        for command, label in METHODS
        if any(run["command"] == command for run in runs)
    ]

    # Three distinct positions around each fraud category.
    offsets = np.linspace(-0.24, 0.24, len(methods))
    bar_width = 0.20

    for offset, (command, label) in zip(offsets, methods):
        for index, (metric, _) in enumerate(available_metrics):
            values = [
                (run["seed"], run["metrics"][split][metric])
                for run in runs
                if run["command"] == command
                and metric in run["metrics"][split]
                and isinstance(run["metrics"][split][metric], (int, float))
                and np.isfinite(run["metrics"][split][metric])
            ]

            if not values:
                continue

            scores = np.asarray([score for _, score in values], dtype=float)
            position = x[index] + offset

            mean = float(scores.mean())
            std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0

            ax.bar(
                position,
                mean,
                width=bar_width,
                color=COLORS[command],
                alpha=0.8,
                label=label if index == 0 else None,
                zorder=2,
            )

            ax.errorbar(
                position,
                mean,
                yerr=std,
                color="#222222",
                capsize=3,
                linewidth=1,
                zorder=3,
            )

            # Plot each seed's score and label it.
            point_offsets = np.linspace(-0.035, 0.035, len(values))

            for (seed, score), point_offset in zip(values, point_offsets):
                point_x = position + point_offset

                ax.scatter(
                    point_x,
                    score,
                    color=COLORS[command],
                    edgecolor="white",
                    linewidth=0.6,
                    s=34,
                    zorder=4,
                )

                ax.annotate(
                    str(seed),
                    (point_x, score),
                    xytext=(3, 2),
                    textcoords="offset points",
                    fontsize=7,
                    color="#444444",
                )

    ax.set_xticks(x, [label for _, label in available_metrics])
    ax.set_ylabel("Average precision")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Arm A: {split.capitalize()} AP by model and fraud category")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
    ax.legend(title="Model", frameon=False)

    fig.text(
        0.01,
        0.01,
        "Bars show the mean; error bars show ±1 standard deviation; "
        "point labels are seeds.",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output = PLOTS_DIR / f"ap_{split}.png"

    fig.savefig(output, dpi=160)
    plt.close(fig)

    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("both", "validation", "test"),
        default="both",
        help="which evaluation split(s) to plot",
    )

    args = parser.parse_args()
    runs = load_runs()

    if not runs:
        raise SystemExit(
            f"No complete Tanh, RFF, or QELM runs found under {RESULTS_DIR}"
        )

    for command, label in METHODS:
        seeds = sorted({run["seed"] for run in runs if run["command"] == command})
        if seeds:
            print(f"{label}: loaded seeds {seeds}")

    splits = ("validation", "test") if args.split == "both" else (args.split,)

    for split in splits:
        plot_split(runs, split)


if __name__ == "__main__":
    main()

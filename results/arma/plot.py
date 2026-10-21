"""Plot Arm A tanh-random-map vs QELM results found under results/arma.

The script reads completed ``metrics.json``/``config.json`` run pairs
recursively, so results may be stored in model subfolders. It does not run
experiments.

Usage from the repository root:
    python results/arma/plot.py
    python results/arma/plot.py --split validation
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
COLORS = {"classical": "#e07a32", "qrc": "#3478b9"}


def load_runs():
    """Load complete result pairs; incomplete or unrelated folders are skipped."""
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
            command = args.get("command")
            if command not in ("classical", "qrc"):
                continue

            seed = args.get("seed")
            if seed is None:
                print(f"Skipping {metrics_path.parent}: run seed is missing")
                continue

            for split in ("validation", "test"):
                if split not in metrics:
                    print(
                        f"Skipping {metrics_path.parent}: "
                        f"{split} metrics are incomplete"
                    )
                    break
            else:
                runs.append(
                    {
                        "command": command,
                        "seed": int(seed),
                        "metrics": metrics,
                        "path": metrics_path.parent,
                    }
                )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"Skipping unreadable run {metrics_path.parent}: {exc}")

    return runs


def plot_split(runs, split):
    """Plot mean AP with per-seed points for all fraud and each fraud category."""
    available_metrics = [
        (key, label)
        for key, label in METRICS
        if any(key in run["metrics"][split] for run in runs)
    ]
    if not available_metrics:
        print(f"No AP metrics found for the {split} split; no plot created.")
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(available_metrics), dtype=float)
    methods = [
        (command, label)
        for command, label in (("classical", "Tanh random map"), ("qrc", "QELM"))
        if any(run["command"] == command for run in runs)
    ]
    offsets = np.linspace(-0.13, 0.13, max(1, len(methods)))

    for offset, (command, label) in zip(offsets, methods):
        for index, (metric, _) in enumerate(available_metrics):
            values = [
                (run["seed"], run["metrics"][split][metric])
                for run in runs
                if run["command"] == command
                and metric in run["metrics"][split]
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
                width=0.22,
                color=COLORS[command],
                alpha=0.78,
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
            point_offsets = np.linspace(-0.045, 0.045, len(values))
            for (seed, score), point_offset in zip(values, point_offsets):
                ax.scatter(
                    position + point_offset,
                    score,
                    color=COLORS[command],
                    edgecolor="white",
                    linewidth=0.6,
                    s=34,
                    zorder=4,
                )
                ax.annotate(
                    str(seed),
                    (position + point_offset, score),
                    xytext=(4, 2),
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
        "Bars show the mean; error bars show ±1 standard deviation; point labels are seeds.",
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
        raise SystemExit(f"No complete tanh or QELM runs found under {RESULTS_DIR}")

    for command, label in (("classical", "tanh"), ("qrc", "QELM")):
        seeds = sorted({run["seed"] for run in runs if run["command"] == command})
        if seeds:
            print(f"{label}: loaded seeds {seeds}")

    splits = ("validation", "test") if args.split == "both" else (args.split,)
    for split in splits:
        plot_split(runs, split)


if __name__ == "__main__":
    main()

"""Merge the per-job CSVs of observables.py and write tables (tables.md) and figures (capacity_n*_reset*.png).

Totals: MC / PC = sum over delays of the capacity, mean and standard error over seeds. The best set per
(noise, budget) is the one with the highest mean total, marked with * when it is not within one standard error of
the runner-up.

Usage: python results/observables/analyze.py [--runs results/observables/runs] [--out results/observables]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASK_LABEL = {"stm": "memory capacity", "parity": "parity capacity"}
LEVELS = ["ideal", "2q0.5", "2q1", "2q2", "2q2-ro1"]  # observables.NOISE order
PLOT_LEVELS = ("ideal", "2q1", "2q2")
# fixed set -> colour (reference palette slots 1-4, light mode); colour follows the set in every panel
PLOT_SETS = {
    "Z-w1": "#2a78d6",
    "Z-strings": "#eb6834",
    "XYZ-strings": "#1baf7a",
    "XYZ+XYZ-2body": "#eda100",
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(runs):
    cap = pd.concat([pd.read_csv(f) for f in sorted(runs.glob("capacity_*.csv"))], ignore_index=True)
    feat = pd.concat([pd.read_csv(f) for f in sorted(runs.glob("features_*.csv"))], ignore_index=True)
    return cap, feat


def totals(cap):
    keys = ["task", "n", "reset", "noise", "observables", "n_settings", "n_features", "budget"]
    per_seed = cap.groupby(keys + ["seed"], as_index=False)[["capacity", "chance"]].sum()
    g = per_seed.groupby(keys)
    out = g.capacity.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"].fillna(0) / np.sqrt(out["count"])
    out["chance"] = g.chance.mean().values
    out["noise"] = pd.Categorical(out["noise"], [lv for lv in LEVELS if lv in set(out["noise"])], ordered=True)
    return out.sort_values(["task", "n", "reset", "noise", "observables", "budget"])


def md(df):
    """Markdown table without the optional tabulate dependency."""
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def budget_label(b):
    return "exact" if np.isinf(b) else f"{b:.0e}".replace("e+0", "e")


def tables(tot, feat, out):
    lines = ["# Observable sets: tables\n", "Mean ± s.e. over seeds of the total capacity (sum over delays).\n"]
    for (task, n, reset), d in tot.groupby(["task", "n", "reset"]):
        seeds = int(d["count"].max())
        lines.append(f"\n## {TASK_LABEL[task]}, n = {n}, reset = {bool(reset)} ({seeds} seeds)\n")
        for level, dl in d.groupby("noise", observed=True):
            cell = dl.assign(v=[f"{m:.2f} ± {s:.2f}" for m, s in zip(dl["mean"], dl["sem"])])
            t = cell.pivot_table(index=["observables", "n_settings", "n_features"], columns="budget", values="v",
                                 aggfunc="first")
            t.columns = [budget_label(b) for b in t.columns]
            lines.append(f"\n### noise {level}\n\n" + md(t.reset_index()) + "\n")
        # best set per (noise, budget)
        rows = []
        for (level, budget), db in d.groupby(["noise", "budget"], observed=True):
            db = db.sort_values("mean", ascending=False)
            best, second = db.iloc[0], db.iloc[1]
            clear = best["mean"] - best["sem"] > second["mean"] + second["sem"]
            rows.append(dict(noise=level, budget=budget, best=best["observables"] + ("*" if clear else "")))
        best = pd.DataFrame(rows).pivot(index="noise", columns="budget", values="best")
        best = best.reindex([lv for lv in LEVELS if lv in best.index])
        best.columns = [budget_label(b) for b in best.columns]
        lines.append("\n### best set (* = clear of the runner-up by one s.e.)\n\n" + md(best.reset_index()) + "\n")
    # feature size per Pauli weight: what gate noise does to high-weight strings
    f = feat[feat.observables.isin(["XYZ-strings"])].groupby(["task", "n", "reset", "noise", "weight"]).median_std.mean()
    t = f.unstack("weight")
    lines.append("\n## Median feature std per Pauli weight (XYZ-strings, exact)\n\n" + md(t.round(4).reset_index()) + "\n")
    (out / "tables.md").write_text("".join(lines))


def figures(tot, out):
    for (n, reset), d in tot.groupby(["n", "reset"]):
        levels = [lv for lv in PLOT_LEVELS if lv in set(d.noise)]
        fig, axes = plt.subplots(2, len(levels), figsize=(4.2 * len(levels), 6.6), sharex=True, sharey="row",
                                 squeeze=False)
        for i, task in enumerate(("stm", "parity")):
            for j, level in enumerate(levels):
                ax = axes[i, j]
                dl = d[(d.task == task) & (d.noise == level)]
                finite = dl[np.isfinite(dl.budget)]
                xmax = finite.budget.max()
                for name, colour in PLOT_SETS.items():
                    s = finite[finite.observables == name].sort_values("budget")
                    ex = dl[(dl.observables == name) & ~np.isfinite(dl.budget)]
                    if s.empty:
                        continue
                    ax.fill_between(s.budget, s["mean"] - s["sem"], s["mean"] + s["sem"], color=colour, alpha=0.15,
                                    lw=0)
                    ax.plot(s.budget, s["mean"], color=colour, lw=2, marker="o", ms=4, label=name)
                    if not ex.empty:  # exact value as a marker past the last budget
                        ax.plot([xmax * 4], ex["mean"], color=colour, marker="D", ms=6, ls="none")
                ax.set_xscale("log")
                ax.grid(color=GRID, lw=0.8)
                ax.set_axisbelow(True)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.tick_params(colors=MUTED, labelsize=9)
                if i == 0:
                    ax.set_title(f"noise {level}", color=TEXT, fontsize=11)
                if j == 0:
                    ax.set_ylabel(TASK_LABEL[task], color=TEXT)
                if i == 1:
                    ax.set_xlabel("shots per window (◆ = exact)", color=MUTED)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   ncol=min(len(labels), 2 * len(levels)), frameon=False, fontsize=10)
        fig.suptitle(f"n = {n} qubits, input reset {'on' if reset else 'off'}: capacity vs shot budget",
                     color=TEXT, y=1.03 + 0.05 * (1 + (len(labels) > 2 * len(levels))))
        fig.savefig(out / f"capacity_n{n}_reset{int(reset)}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="results/observables/runs")
    parser.add_argument("--out", default="results/observables")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap, feat = load(Path(args.runs))
    tot = totals(cap)
    tot.to_csv(out / "totals.csv", index=False)
    tables(tot, feat, out)
    figures(tot, out)
    print(f"analyzed {cap.seed.nunique()} seeds, {len(tot)} cells -> {out}")

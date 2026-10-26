"""Merge the per-seed CSVs of fraud_p4.py: tables (tables.md, summary.csv) and a figure (fraud_p4.png).

Per (noise, observable set, budget): mean ± s.e. over seeds of validation and test AP (customer, amount, overall)
and Card Precision@K, the paired difference to logistic regression on the raw window (deterministic, so the
difference is QRC − its value), and the paired difference XYZ-strings − Z-strings with a t-test over seeds.

Usage: python results/observables/analyze_p4.py [--runs results/observables/runs_p4] [--out results/observables/p4]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LEVELS = ["ideal", "2q0.5", "2q1", "2q2", "2q2-ro1"]
SETS = ["Z-w1", "Z-strings", "frames-3", "XYZ-strings", "XYZ+XYZ-2body"]
# colour follows the set (reference palette slots 1-4 for the sets in the figure)
PLOT_SETS = {"Z-w1": "#2a78d6", "Z-strings": "#eb6834", "XYZ-strings": "#1baf7a", "XYZ+XYZ-2body": "#eda100"}
METRICS = {
    "average_precision_customer": "customer AP",
    "average_precision_amount": "amount AP",
    "average_precision": "AP",
    "card_precision_top_k": "CP@K",
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def md(df):
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def budget_label(b):
    return "exact" if np.isinf(b) else f"{b:.0e}".replace("e+0", "e")


def summarize(df):
    qrc = df[df.model == "qrc"]
    base = df[df.model == "logistic"].groupby("seed").first()
    rows = []
    for (level, s, b), d in qrc.groupby(["noise", "observables", "budget"]):
        row = dict(noise=level, observables=s, budget=b, n_settings=int(d.n_settings.iloc[0]), seeds=len(d))
        for split in ("validation", "test"):
            for key in METRICS:
                col = f"{split}_{key}"
                if col not in d:
                    continue
                v = d.set_index("seed")[col]
                row[f"{col}_mean"] = v.mean()
                row[f"{col}_sem"] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                row[f"{col}_minus_logistic"] = (v - base.loc[v.index, col]).mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out["noise"] = pd.Categorical(out.noise, [lv for lv in LEVELS if lv in set(out.noise)], ordered=True)
    return out.sort_values(["noise", "observables", "budget"]), base


def tables(summary, base, df, out):
    lines = ["# P4: fraud AP of the observable sets under shot and gate noise\n",
             f"\nL = 10, 1 input + 3 memory qubits, input reset, {summary.seeds.max()} seeds. Mean ± s.e. over seeds.\n"]
    lines.append("\n## Logistic regression on the raw window (reference)\n\n")
    lines.append(md(pd.DataFrame([{f"{split} {name}": f"{base[f'{split}_{key}'].iloc[0]:.3f}"
                                   for split in ("validation", "test") for key, name in METRICS.items()}])) + "\n")
    for split in ("test", "validation"):
        col = f"{split}_average_precision_customer"
        for level, d in summary.groupby("noise", observed=True):
            cell = d.assign(v=[f"{m:.3f} ± {s:.3f} ({x:+.3f})" for m, s, x in
                               zip(d[f"{col}_mean"], d[f"{col}_sem"], d[f"{col}_minus_logistic"])])
            t = cell.pivot_table(index=["observables", "n_settings"], columns="budget", values="v", aggfunc="first")
            t = t.reindex([s for s in SETS if s in t.index.get_level_values(0)], level=0)
            t.columns = [budget_label(b) for b in t.columns]
            lines.append(f"\n## {split} customer AP, noise {level} (in brackets: minus logistic)\n\n"
                         + md(t.reset_index()) + "\n")
    # paired XYZ-strings - Z-strings
    qrc = df[df.model == "qrc"]
    rows = []
    for (level, b), d in qrc.groupby(["noise", "budget"]):
        p = d.pivot_table(index="seed", columns="observables", values="test_average_precision_customer")
        if {"XYZ-strings", "Z-strings"} <= set(p.columns):
            diff = p["XYZ-strings"] - p["Z-strings"]
            rows.append(dict(noise=level, budget=budget_label(b), diff=f"{diff.mean():+.3f}",
                             sem=f"{diff.std(ddof=1) / np.sqrt(len(diff)):.3f}",
                             p=f"{stats.ttest_1samp(diff, 0).pvalue:.3f}", xyz_wins=f"{int((diff > 0).sum())}/{len(diff)}"))
    order = {lv: i for i, lv in enumerate(LEVELS)}
    paired = pd.DataFrame(rows).sort_values("noise", key=lambda s: s.map(order), kind="stable")
    lines.append("\n## Test customer AP, XYZ-strings − Z-strings (paired over seeds, t-test)\n\n" + md(paired) + "\n")
    # other metrics at the two settings-count extremes
    for key in ("average_precision_amount", "average_precision", "card_precision_top_k"):
        col = f"test_{key}"
        t = summary.assign(v=[f"{m:.3f}" for m in summary[f"{col}_mean"]]).pivot_table(
            index=["noise", "observables"], columns="budget", values="v", aggfunc="first", observed=True)
        t.columns = [budget_label(b) for b in t.columns]
        lines.append(f"\n## test {METRICS[key]} (logistic {base[col].iloc[0]:.3f})\n\n" + md(t.reset_index()) + "\n")
    (out / "tables.md").write_text("".join(lines))


def figure(summary, base, out):
    levels = list(summary.noise.cat.categories)
    fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 4.2), sharey=True, squeeze=False)
    col = "test_average_precision_customer"
    for j, level in enumerate(levels):
        ax = axes[0, j]
        d = summary[summary.noise == level]
        finite = d[np.isfinite(d.budget)]
        xmax = finite.budget.max()
        ax.axhline(base[col].iloc[0], color=MUTED, lw=1.2, ls="--", label="logistic (raw window)")
        for name, colour in PLOT_SETS.items():
            s = finite[finite.observables == name].sort_values("budget")
            ex = d[(d.observables == name) & ~np.isfinite(d.budget)]
            if s.empty:
                continue
            ax.fill_between(s.budget, s[f"{col}_mean"] - s[f"{col}_sem"], s[f"{col}_mean"] + s[f"{col}_sem"],
                            color=colour, alpha=0.15, lw=0)
            ax.plot(s.budget, s[f"{col}_mean"], color=colour, lw=2, marker="o", ms=4, label=name)
            if not ex.empty:
                ax.plot([xmax * 4], ex[f"{col}_mean"], color=colour, marker="D", ms=6, ls="none")
        ax.set_xscale("log")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_title(f"noise {level}", color=TEXT, fontsize=11)
        ax.set_xlabel("shots per window (◆ = exact)", color=MUTED)
        if j == 0:
            ax.set_ylabel("test customer AP", color=TEXT)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels), frameon=False,
               fontsize=10)
    fig.suptitle("Fraud (scenario 3, L = 10): customer AP vs shot budget", color=TEXT, y=1.1)
    fig.savefig(out / "fraud_p4.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="results/observables/runs_p4")
    parser.add_argument("--out", default="results/observables/p4")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(Path(args.runs).glob("fraud_seed*.csv"))], ignore_index=True)
    summary, base = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    tables(summary, base, df, out)
    figure(summary, base, out)
    print(f"analyzed {df.seed.nunique()} seeds -> {out}")

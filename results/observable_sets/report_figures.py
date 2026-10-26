"""Report figures for the observable-set section (input reset on throughout).

    fig1_memory_vs_shots       memory capacity vs shot budget, ideal / 99 % / 98 % two-qubit fidelity (P2, n = 5)
    fig2_xyz_gain              gain of measuring X and Y too (XYZ-strings − Z-strings) vs shot budget, one line per
                               fidelity: memory capacity (P2, n = 5) and fraud customer AP (P4)
    fig3_fraud_vs_shots        test customer AP vs shot budget, against logistic regression on the raw window (P4)
    fig4_feature_size          feature spread per Pauli weight vs the shot-noise level (P2, n = 5, memory task)

Each figure is written as PNG and PDF to report/, with the plotted numbers as CSV next to it. report/facts.md
gets every number quoted in the captions (crossover budgets, readout and frame checks, tomography, SpinPulse, fraud).
Inputs: data/p2, data/p3, data/p4 (per-seed CSVs of observables.py, spinpulse_p3.py, fraud_p4.py).

Usage: python results/observable_sets/report_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "report"
N = 5  # register size of the memory figures (1 input + 4 memory qubits)

TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# observable set -> (label, colour); colour follows the set in every figure (reference palette slots 1-4)
SETS = {
    "Z-w1": ("local Z (1 setting)", "#2a78d6"),
    "Z-strings": ("Z-strings (1 setting)", "#eb6834"),
    "XYZ-strings": ("XYZ-strings (3 settings)", "#1baf7a"),
    "XYZ+XYZ-2body": ("XYZ + 2-body (9 settings)", "#eda100"),
}
# noise level -> (label, colour): one-hue ordinal ramp, darker = noisier (validated with --ordinal)
LEVELS = {
    "ideal": ("noiseless", "#86b6ef"),
    "2q0.5": ("99.5 %", "#3987e5"),
    "2q1": ("99 %", "#256abf"),
    "2q2": ("98 % (Quobly today)", "#0d366b"),
}
PANEL_LEVELS = ("ideal", "2q1", "2q2")


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def save(fig, name, table):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    table.to_csv(OUT / f"{name}.csv", index=False)


def label_end(ax, x, y, text, colour, dy=0.0):
    """Direct label at the right end of a line, in text ink with a colour swatch."""
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=8.5, color=TEXT)
    ax.plot([x], [y], marker="o", ms=4, color=colour)


def exact_axis(ax, budgets):
    """Log x axis over the finite budgets plus an 'exact' tick to the right."""
    xmax = max(budgets)
    x_exact = xmax * 6
    ax.set_xscale("log")
    ticks = [b for b in budgets if np.isclose(np.log10(b), round(np.log10(b)))]  # powers of ten only
    ax.set_xticks(ticks + [x_exact])
    ax.set_xticklabels([f"$10^{{{int(np.log10(b))}}}$" for b in ticks] + ["exact"])
    ax.minorticks_off()
    ax.axvspan(xmax * 2.5, x_exact * 2, color="#f3f2ef", lw=0, zorder=0)
    ax.set_xlim(min(budgets) / 1.6, x_exact * 1.8)
    return x_exact


# ---------------------------------------------------------------- data

def p2_per_seed():
    return memory_capacity("p2", N)


def memory_capacity(phase, n):
    """Per-seed memory capacity (sum over delays) of one phase's data at register size n."""
    cap = pd.concat([pd.read_csv(f) for f in sorted((HERE / "data" / phase).glob(f"capacity_n{n}_seed*.csv"))])
    return cap.groupby(["noise", "observables", "n_settings", "budget", "seed"], as_index=False).capacity.sum()


def p4_runs():
    return pd.concat([pd.read_csv(f) for f in sorted((HERE / "data" / "p4").glob("fraud_seed*.csv"))])


def mean_sem(d, value, keys):
    g = d.groupby(keys)[value]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"])
    return out.drop(columns="std")


# ---------------------------------------------------------------- figures

def fig1(mc):
    sets = ("Z-strings", "XYZ-strings", "XYZ+XYZ-2body")
    t = mean_sem(mc[mc.observables.isin(sets) & mc.noise.isin(PANEL_LEVELS)], "capacity",
                 ["noise", "observables", "budget"])
    budgets = sorted(b for b in t.budget.unique() if np.isfinite(b))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, level in zip(axes, PANEL_LEVELS):
        style(ax)
        x_exact = exact_axis(ax, budgets)
        for i, s in enumerate(sets):
            label, colour = SETS[s]
            d = t[(t.noise == level) & (t.observables == s)].sort_values("budget")
            f, e = d[np.isfinite(d.budget)], d[~np.isfinite(d.budget)]
            ax.fill_between(f.budget, f["mean"] - f["sem"], f["mean"] + f["sem"], color=colour, alpha=0.15, lw=0)
            ax.plot(f.budget, f["mean"], color=colour, lw=2, marker="o", ms=4, label=label)
            ax.plot([x_exact], e["mean"], color=colour, marker="D", ms=6, ls="none")
        ax.set_title(f"two-qubit fidelity: {LEVELS[level][0]}", color=TEXT, fontsize=10.5)
        ax.set_xlabel("shots per window", color=MUTED)
    axes[0].set_ylabel("memory capacity (of 10)", color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Measuring more observables pays off only above a shot budget that grows with gate noise",
                 color=TEXT, y=1.17, fontsize=12)
    save(fig, "fig1_memory_vs_shots", t)


def gains(mc, fraud):
    """Per-seed paired XYZ-strings − Z-strings: memory capacity (P2) and test customer AP (P4)."""
    p = mc.pivot_table(index=["noise", "budget", "seed"], columns="observables", values="capacity").reset_index()
    p["gain"] = p["XYZ-strings"] - p["Z-strings"]
    q = fraud[fraud.model == "qrc"].pivot_table(index=["noise", "budget", "seed"], columns="observables",
                                                values="test_average_precision_customer").reset_index()
    q["gain"] = q["XYZ-strings"] - q["Z-strings"]
    return mean_sem(p, "gain", ["noise", "budget"]), mean_sem(q, "gain", ["noise", "budget"])


def fig2(mc, fraud):
    g_mc, g_fr = gains(mc, fraud)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    panels = ((axes[0], g_mc, "memory capacity gain", "memory task (n = 5)"),
              (axes[1], g_fr, "test customer AP gain", "fraud, scenario 3 (n = 4, L = 10)"))
    for ax, g, ylabel, title in panels:
        style(ax)
        budgets = sorted(b for b in g.budget.unique() if np.isfinite(b))
        x_exact = exact_axis(ax, budgets)
        ax.axhline(0, color=MUTED, lw=1)
        for level, (label, colour) in LEVELS.items():
            d = g[g.noise == level].sort_values("budget")
            if d.empty:
                continue
            f, e = d[np.isfinite(d.budget)], d[~np.isfinite(d.budget)]
            ax.fill_between(f.budget, f["mean"] - f["sem"], f["mean"] + f["sem"], color=colour, alpha=0.18, lw=0)
            ax.plot(f.budget, f["mean"], color=colour, lw=2, marker="o", ms=4, label=label)
            ax.plot([x_exact], e["mean"], color=colour, marker="D", ms=6, ls="none")
        ax.set_title(title, color=TEXT, fontsize=10.5)
        ax.set_xlabel("shots per window", color=MUTED)
        ax.set_ylabel(ylabel, color=TEXT)
        ax.annotate("X/Y helps ↑", (budgets[-1], 0), xytext=(-4, 4), textcoords="offset points", ha="right",
                    va="bottom", fontsize=8, color=MUTED)
        ax.annotate("Z-only better ↓", (budgets[-1], 0), xytext=(-4, -4), textcoords="offset points", ha="right",
                    va="top", fontsize=8, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=10,
               title="two-qubit gate fidelity", title_fontsize=9)
    fig.suptitle("Gain of measuring X and Y as well as Z (XYZ-strings − Z-strings, paired over 10 seeds)",
                 color=TEXT, y=1.24, fontsize=12)
    g_mc.insert(0, "metric", "memory capacity")
    g_fr.insert(0, "metric", "test customer AP")
    save(fig, "fig2_xyz_gain", pd.concat([g_mc, g_fr]))


def fig3(fraud):
    col = "test_average_precision_customer"
    qrc = fraud[(fraud.model == "qrc") & fraud.observables.isin(SETS)]
    t = mean_sem(qrc, col, ["noise", "observables", "budget"])
    logistic = fraud[fraud.model == "logistic"][col].iloc[0]
    budgets = sorted(b for b in t.budget.unique() if np.isfinite(b))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, level in zip(axes, PANEL_LEVELS):
        style(ax)
        x_exact = exact_axis(ax, budgets)
        ax.axhline(logistic, color=MUTED, lw=1.3, ls="--", label="logistic regression, raw window")
        for s, (label, colour) in SETS.items():
            d = t[(t.noise == level) & (t.observables == s)].sort_values("budget")
            f, e = d[np.isfinite(d.budget)], d[~np.isfinite(d.budget)]
            ax.fill_between(f.budget, f["mean"] - f["sem"], f["mean"] + f["sem"], color=colour, alpha=0.15, lw=0)
            ax.plot(f.budget, f["mean"], color=colour, lw=2, marker="o", ms=4, label=label)
            ax.plot([x_exact], e["mean"], color=colour, marker="D", ms=6, ls="none")
        ax.set_title(f"two-qubit fidelity: {LEVELS[level][0]}", color=TEXT, fontsize=10.5)
        ax.set_xlabel("shots per window", color=MUTED)
    axes[0].set_ylabel("test customer AP (scenario 3)", color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Fraud detection: no finite shot budget reaches the classical baseline", color=TEXT, y=1.24,
                 fontsize=12)
    t["logistic"] = logistic
    save(fig, "fig3_fraud_vs_shots", t)


def fig4():
    feat = pd.concat([pd.read_csv(f) for f in sorted((HERE / "data" / "p2").glob(f"features_n{N}_seed*.csv"))])
    feat = feat[feat.observables == "XYZ-strings"]
    t = mean_sem(feat, "median_std", ["noise", "weight"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    style(ax)
    ax.set_yscale("log")
    for level in PANEL_LEVELS:
        label, colour = LEVELS[level]
        d = t[t.noise == level].sort_values("weight")
        ax.plot(d.weight, d["mean"], color=colour, lw=2, marker="o", ms=5, label=label)
    # standard error of one Pauli estimate, B shots per window split over the 3 settings: ~ sqrt(3 / B)
    for B in (1e3, 1e4, 1e5):
        se = np.sqrt(3 / B)
        ax.axhline(se, color=MUTED, lw=1, ls=":")
        ax.text(N + 1.85, se * 1.06, f"shot noise at $10^{{{int(np.log10(B))}}}$ shots", va="bottom", ha="right",
                fontsize=8, color=MUTED)
    ax.set_xticks(range(1, N + 1))
    ax.set_xlim(0.8, N + 1.9)
    ax.set_xlabel("Pauli weight of the observable", color=MUTED)
    ax.set_ylabel("feature spread across windows\n(median std, XYZ-strings)", color=TEXT)
    ax.legend(frameon=False, fontsize=9, title="two-qubit gate fidelity", title_fontsize=9, loc="lower left")
    ax.set_title("Gate noise shrinks the features below the shot noise,\nhigh-weight correlators first",
                 color=TEXT, fontsize=11)
    t["shot_noise_1e3"], t["shot_noise_1e4"], t["shot_noise_1e5"] = np.sqrt(3e-3), np.sqrt(3e-4), np.sqrt(3e-5)
    save(fig, "fig4_feature_size", t)


# ---------------------------------------------------------------- numbers for the text

def md(df):
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def blabel(b):
    return "exact" if not np.isfinite(b) else f"1e{np.log10(b):.1f}".replace(".0", "")


def crossover(gain):
    """Per noise level: first budget where the paired gain is > 0 beyond one s.e., and where it reaches 0.5."""
    rows = []
    for level, d in gain.groupby("noise", sort=False):
        d = d[np.isfinite(d.budget)].sort_values("budget")
        pos = d[d["mean"] - d["sem"] > 0].budget
        half = d[d["mean"] >= 0.5].budget
        rows.append(dict(noise=level, gain_positive_from=blabel(pos.iloc[0]) if len(pos) else "never",
                         gain_half_unit_from=blabel(half.iloc[0]) if len(half) else "never"))
    return pd.DataFrame(rows)


def paired_gain(mc):
    p = mc.pivot_table(index=["noise", "budget", "seed"], columns="observables", values="capacity").reset_index()
    p["gain"] = p["XYZ-strings"] - p["Z-strings"]
    g = mean_sem(p, "gain", ["noise", "budget"])
    return g.round(3)


def facts(mc, fraud):
    from scipy import stats
    lines = ["# Numbers quoted in the report text\n\nGenerated by report_figures.py. Memory capacity (MC) is the sum "
             "over 10 delays, mean over seeds; gains are paired over seeds (XYZ-strings − Z-strings).\n"]
    order = {lv: i for i, lv in enumerate(["ideal", "2q0.5", "2q1", "2q2", "2q2-ro1",
                                           "sp-ideal", "sp-2q99", "sp-2q98"])}

    g = paired_gain(mc)
    g = g.sort_values("noise", key=lambda c: c.map(order), kind="stable")
    lines.append("\n## P2, n = 5: where measuring X/Y starts to pay\n\n" + md(crossover(g)) + "\n")
    sel = mc[mc.noise.isin(["ideal", "2q2"]) & mc.observables.isin(["Z-strings", "XYZ-strings"])
             & mc.budget.isin([1e3, 1e4, 1e5, np.inf])]
    t = sel.groupby(["noise", "observables", "budget"]).capacity.mean().unstack("budget").round(2)
    t = t.sort_index(level="noise", key=lambda i: i.map(order))
    t.columns = [blabel(b) for b in t.columns]
    lines.append("\n## P2, n = 5: MC of Z-strings vs XYZ-strings\n\n" + md(t.reset_index()) + "\n")

    both = pd.concat([memory_capacity("p2", n).assign(n=n) for n in (4, 5)])
    m = both.groupby(["n", "noise", "observables", "budget"]).capacity.mean()
    ro = (m.xs("2q2-ro1", level="noise") - m.xs("2q2", level="noise")).abs()
    lines.append(f"\n## Readout error\n\nmax |MC(1 % readout) − MC(0.07 % readout)| at 98 % fidelity, over n = 4, 5, "
                 f"all sets and budgets: {ro.max():.2f} (mean {ro.mean():.2f})\n")
    fr = (m.xs("frames-3", level="observables") - m.xs("XYZ-strings", level="observables")).loc[5]
    fr = fr.groupby("noise").agg(lambda v: v.abs().max()).round(2).sort_index(key=lambda i: i.map(order))
    lines.append("\n## Native or not: |MC(frames-3) − MC(XYZ-strings)|, n = 5, max over budgets\n\n"
                 + md(fr.reset_index().rename(columns={"capacity": "max abs diff"})) + "\n")
    tomo = m.loc[4, "2q2"].unstack("observables")[["Z-strings", "XYZ-strings", "XYZ+XYZ-2body", "all-Pauli"]]
    tomo.index = [blabel(b) for b in tomo.index]
    lines.append("\n## Full tomography, n = 4, 98 % fidelity: MC by budget\n\n"
                 + md(tomo.round(2).reset_index().rename(columns={"index": "budget"})) + "\n")

    feat = pd.concat([pd.read_csv(f) for f in sorted((HERE / "data" / "p2").glob(f"features_n{N}_seed*.csv"))])
    fs = feat[feat.observables == "XYZ-strings"].groupby(["noise", "weight"]).median_std.mean().unstack("noise")
    shrink = (fs["ideal"] / fs["2q2"]).round(1)
    lines.append("\n## Feature spread shrink factor, noiseless / 98 %, n = 5, by Pauli weight\n\n"
                 + md(shrink.reset_index().rename(columns={0: "shrink"})) + "\n")

    if any((HERE / "data" / "p3").glob(f"capacity_n{N}_seed*.csv")):
        p3 = memory_capacity("p3", N)
        g3 = paired_gain(p3).sort_values("noise", key=lambda c: c.map(order), kind="stable")
        lines.append("\n## P3 (SpinPulse), n = 5: where measuring X/Y starts to pay\n\n" + md(crossover(g3)) + "\n")
        fs3 = pd.concat([pd.read_csv(f) for f in sorted((HERE / "data" / "p3").glob(f"features_n{N}_seed*.csv"))])
        fs3 = fs3[fs3.observables == "XYZ-strings"].groupby(["noise", "weight"]).median_std.mean().unstack("noise")
        lines.append("\n## P3 feature spread by Pauli weight (XYZ-strings, n = 5)\n\n"
                     + md(fs3.round(4).reset_index()) + "\n")

    col = "test_average_precision_customer"
    logistic = fraud[fraud.model == "logistic"][col].iloc[0]
    q = fraud[fraud.model == "qrc"]
    at5 = q[q.budget == 1e5].groupby(["noise", "observables"])[col].mean()
    best = pd.DataFrame({"best set": at5.groupby("noise").idxmax().str[1], "AP": at5.groupby("noise").max().round(3)})
    best = best.sort_index(key=lambda i: i.map(order))
    exact = q[~np.isfinite(q.budget)].groupby(["noise", "observables"])[col].mean().unstack("observables").round(3)
    exact = exact.sort_index(key=lambda i: i.map(order))
    lines.append(f"\n## Fraud (P4)\n\nlogistic regression on the raw window: test customer AP {logistic:.3f}\n\n"
                 "best set at 1e5 shots:\n\n" + md(best.reset_index()) + "\n\nexact:\n\n" + md(exact.reset_index()) + "\n")
    rows = []
    p = q.pivot_table(index=["noise", "budget", "seed"], columns="observables", values=col)
    for (level, b), d in p.groupby(level=["noise", "budget"]):
        diff = d["XYZ-strings"] - d["Z-strings"]
        rows.append(dict(noise=level, budget=blabel(b), gain=f"{diff.mean():+.3f}",
                         sem=f"{diff.std(ddof=1) / np.sqrt(len(diff)):.3f}", p=f"{stats.ttest_1samp(diff, 0).pvalue:.3f}",
                         xyz_wins=f"{int((diff > 0).sum())}/{len(diff)}"))
    rows = sorted(rows, key=lambda r: order[r["noise"]])
    lines.append("\nXYZ-strings − Z-strings, test customer AP (paired t-test):\n\n" + md(pd.DataFrame(rows)) + "\n")
    (OUT / "facts.md").write_text("".join(lines))


if __name__ == "__main__":
    mc = p2_per_seed()
    fraud = p4_runs()
    fig1(mc)
    fig2(mc, fraud)
    fig3(fraud)
    fig4()
    facts(mc, fraud)
    print(f"figures written to {OUT}")

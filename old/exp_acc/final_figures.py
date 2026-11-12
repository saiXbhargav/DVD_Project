"""Figures for the final (GPX) results -> figures/final/ (new folder; old figures untouched).

part 1 (needs only saved models / CSVs):
  pred_vs_spice_<case>.png    test set, 4 currents, final models
  size_study.png              accuracy vs training rows (gates, parity)
  gate_cases_before_after.png per gate case: official (before) vs final
part 2 (needs exp_acc/out/mlnodes_gate_ml.csv from ml_nodes.py):
  gate_parity.png, circuit_parity.png        version A (SPICE node voltages)
  gate_parity_ml_nodes.png, circuit_parity_ml_nodes.png   version B (ML node voltages)
  error_vs_temperature.png    signed error vs temperature, gates and parity (version A)
Usage: python3 final_figures.py 1|2
"""
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import OUT, C, G  # noqa: E402
from common2 import T  # noqa: E402
from common3 import parity_sum, vec_counts  # noqa: E402

FIG = C.FIG_DIR / "final"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID, AXIS, SURF = "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ON = ("nmos_on", "pmos_on")
NAMES = {"nmos_on": "NMOS ON", "nmos_off": "NMOS OFF", "pmos_on": "PMOS ON", "pmos_off": "PMOS OFF"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    print("saved figures/final/" + name, flush=True)


def models():
    return {c: joblib.load(OUT / "models" / f"{'GPBag87X' if c in ON else 'GPBag87'}_{c}.joblib") for c in C.CASES}


def parity_plot(x, y, color_by, title, name, cbar_label):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(x, y, c=color_by, s=6, cmap="Blues", vmin=np.min(color_by) - 0.3 * np.ptp(color_by),
                    edgecolors="none", alpha=0.8)
    lo, hi = min(x.min(), y.min()) * 0.7, max(x.max(), y.max()) * 1.4
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="SPICE leakage (A)",
           ylabel="ML leakage (A)", title=title)
    fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)
    save(fig, name)


def part1():
    ms = models()
    for case in C.CASES:
        _, _, te = T.split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
        P, Y = ms[case].predict(te), te[C.CURRENTS].to_numpy()
        fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
        for j, (ax, cur) in enumerate(zip(axs, C.CURRENTS)):
            y, p = np.abs(Y[:, j]), np.abs(P[:, j])
            ok = y > 0
            e = 100 * np.abs(P[ok, j] - Y[ok, j]) / y[ok]
            ax.scatter(y[ok], np.maximum(p[ok], 1e-20), s=4, color=BLUE, alpha=0.5, edgecolors="none")
            lo, hi = y[ok].min() * 0.5, y[ok].max() * 2
            ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
            ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel=f"SPICE |{cur}| (A)",
                   title=f"{cur}: median {np.median(e):.2f} %")
            if j == 0:
                ax.set_ylabel("ML |I| (A)")
        fig.suptitle(f"{NAMES[case]}: final model vs SPICE, test set (1,500 samples)", color=INK)
        save(fig, f"pred_vs_spice_{case}.png")

    s = pd.read_csv(OUT / "size_study.csv")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for col, lab, colr in (("gate_acc", "Gates", BLUE), ("parity_acc", "Parity checker", ORANGE)):
        ax.plot(s.rows_nominal, s[col], color=colr, lw=2, marker="o", ms=5)
        ax.annotate(f"{lab} {s[col].iloc[-1]:.2f} %", (s.rows_nominal.iloc[-1], s[col].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.set(xscale="log", xlabel="Training rows per transistor case", ylabel="Mean accuracy (%)",
           title="Accuracy vs training-set size (official corners)", xlim=(1500, 2.6e5))
    ax.set_xticks(s.rows_nominal)
    ax.set_xticklabels([f"{int(r / 1000)}k" for r in s.rows_nominal])
    ax.legend(["Gates", "Parity checker"], loc="lower right")
    save(fig, "size_study.png")

    a = pd.read_csv(OUT / "gate_results_GP_7k_rule.csv", dtype={"pattern": str})
    b = pd.read_csv(OUT / "round2_gate_results_GPX.csv", dtype={"pattern": str})
    m = a[["gate", "pattern", "accuracy_pct_test"]].merge(b[["gate", "pattern", "accuracy_pct_test"]],
                                                          on=["gate", "pattern"], suffixes=("_before", "_final"))
    lab = (m.gate + " " + m.pattern).to_numpy()
    yy = np.arange(len(m))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.hlines(yy, m.accuracy_pct_test_before, m.accuracy_pct_test_final, color=GRID, lw=2)
    ax.scatter(m.accuracy_pct_test_before, yy, color=GRAY, s=28, zorder=3, label="Before (official GP, 7k rows)")
    ax.scatter(m.accuracy_pct_test_final, yy, color=BLUE, s=28, zorder=3, label="Final (averaged GP, 87k rows)")
    ax.axvline(99, color=AXIS, lw=1, ls=":")
    ax.set_yticks(yy)
    ax.set_yticklabels(lab)
    ax.set(xlabel="Mean accuracy (%)", title="Gate accuracy per case, official corners", xlim=(72, 100.5))
    ax.legend(loc="lower left")
    save(fig, "gate_cases_before_after.png")


def part2():
    cor = G.load_corners().set_index("corner")
    g = pd.read_csv(OUT / "mlnodes_gate_ml.csv", dtype={"pattern": str})
    csp = pd.read_csv(C.RESULTS_DIR / "circuit_spice.csv", dtype={"vector": str})
    cnt = vec_counts()
    temp_g = cor.loc[g.corner, "temp"].to_numpy()
    for ver, suffix, txt in (("A", "", "SPICE node voltages"), ("B", "_ml_nodes", "ML-predicted node voltages")):
        e = (g[f"leak_{ver}"] - g.leak_spice) / g.leak_spice
        parity_plot(g.leak_spice.to_numpy(), g[f"leak_{ver}"].to_numpy(), temp_g,
                    f"Gates, {txt}: accuracy {100 - 100 * e.abs().mean():.2f} %, median {100 * e.abs().median():.2f} %",
                    f"gate_parity{suffix}.png", "Temperature (°C)")
        nand = g[g.gate == "NAND2"].pivot(index="corner", columns="pattern", values=f"leak_{ver}")
        p = parity_sum(nand, cnt, csp)
        ep = (p.leak_ml - p.leak_spice) / p.leak_spice
        parity_plot(p.leak_spice.to_numpy(), p.leak_ml.to_numpy(), cor.loc[p.corner, "temp"].to_numpy(),
                    f"Parity checker, {txt}: accuracy {100 - 100 * ep.abs().mean():.2f} %, bias {100 * ep.median():+.2f} %",
                    f"circuit_parity{suffix}.png", "Temperature (°C)")
        if ver == "A":
            p_a = p.assign(temp=cor.loc[p.corner, "temp"].to_numpy(), err=100 * ep)
            g_a = g.assign(temp=temp_g, err=100 * e)
            sp_n = g[g.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_spice")
            comp = parity_sum(sp_n, cnt, csp)
            comp = comp.assign(temp=cor.loc[comp.corner, "temp"].to_numpy(),
                               err=100 * (comp.leak_ml - comp.leak_spice) / comp.leak_spice)
    bins = np.arange(-40, 126, 15)
    mid = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for d, lab, colr in ((g_a, "Gates (ML)", BLUE), (p_a, "Parity checker (ML)", ORANGE),
                         (comp, "Parity: sum of exact SPICE gates", AQUA)):
        med = d.groupby(pd.cut(d.temp, bins), observed=False).err.median().to_numpy()
        ax.plot(mid, med, color=colr, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set(xlabel="Temperature (°C)", ylabel="Median signed error (%)",
           title="Bias vs temperature (official corners, SPICE node voltages)")
    ax.legend(loc="lower left")
    save(fig, "error_vs_temperature.png")


if __name__ == "__main__":
    part1() if sys.argv[1] == "1" else part2()

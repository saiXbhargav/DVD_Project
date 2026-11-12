"""Step 9: figures of the final results (figures/; the trend plots come from step 2).

  pred_vs_spice_<case>.png    Part A: ML vs SPICE terminal currents, test split, per current
  gate_parity.png             Part B: ML vs SPICE gate leakage, version A and version B
  gate_accuracy_by_case.png   Part B: accuracy of each of the 18 gate cases, A and B
  circuit_parity.png          Part C: ML vs SPICE parity-checker leakage, A and B
  error_vs_conditions.png     Part B: gate error vs temperature, VDD, vth0 (NMOS, PMOS), Wn, Wp
                              (mean |error| per decile of each variable)
Reads only results of steps 3, 6, 8 (and the models for the Part A plots).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from final_method import FIG, RES, C, corners, load_models, split, transistor_metrics  # noqa: E402

BLUE, ORANGE, GRAY, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#898781", "#1f1e1c", "#6b6964", "#fcfcfb"
VER = {"A": ("A: SPICE node voltages", BLUE), "B": ("B: ML node voltages", ORANGE)}
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 9, "grid.color": "#e4e2dd", "grid.linewidth": 0.6})


def err(ml, sp):
    return 100 * (ml - sp) / sp


def loglog(ax, sp, ml, color, title):
    ax.scatter(sp, ml, s=6, alpha=0.45, color=color, linewidths=0)
    lo, hi = min(sp.min(), ml.min()), max(sp.max(), ml.max())
    ax.plot([lo, hi], [lo, hi], color=GRAY, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.set_title(title, color=INK)


def stats(e):
    return f"accuracy {100 - np.mean(np.abs(e)):.2f} %, median |err| {np.median(np.abs(e)):.2f} %"


def part_a():
    models = load_models()
    for case in C.CASES:
        _, _, te = split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
        _, P = transistor_metrics(models[case], te)
        T = te[C.CURRENTS].to_numpy()
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
        for j, (c, ax) in enumerate(zip(C.CURRENTS, axs)):
            ok = np.abs(T[:, j]) > 0
            e = 100 * np.abs(P[ok, j] - T[ok, j]) / np.abs(T[ok, j])
            loglog(ax, np.abs(T[ok, j]), np.abs(P[ok, j]), BLUE,
                   f"|{c}|: median {np.median(e):.2f} %, p95 {np.quantile(e, .95):.1f} %")
            ax.set_xlabel(f"SPICE |{c}| (A)")
            ax.set_ylabel(f"ML |{c}| (A)")
        fig.suptitle(f"{case}: final model vs SPICE, test split (n = {len(te)})", color=INK)
        fig.tight_layout()
        fig.savefig(FIG / f"pred_vs_spice_{case}.png", dpi=150)
        plt.close(fig)


def gates(g):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, v in zip(axs, "AB"):
        name, col = VER[v]
        loglog(ax, g.leak_spice, g[f"leak_{v}"], col, f"{name}\n{stats(err(g[f'leak_{v}'], g.leak_spice))}")
        ax.set_xlabel("SPICE gate leakage (A)")
    axs[0].set_ylabel("ML gate leakage (A)")
    fig.suptitle("Gates: 18 cases x 200 corners, varying widths", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "gate_parity.png", dpi=150)
    plt.close(fig)

    t = pd.read_csv(RES / "gate_results.csv", dtype={"pattern": str})
    lab = t.gate + " " + t.pattern
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7, 6.5))
    h = 0.38
    for k, v in enumerate("AB"):
        name, col = VER[v]
        acc = t[f"accuracy_pct_{v}"]
        ax.barh(y + (k - 0.5) * h, acc - 90, left=90, height=h - 0.04, color=col, label=name)
    ax.set_yticks(y)
    ax.set_yticklabels(lab)
    ax.invert_yaxis()
    ax.set_xlim(90, 100)
    ax.set_xlabel("accuracy (%) = 100 - mean |relative error|")
    ax.grid(True, axis="x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.set_title("Accuracy per gate case (varying widths)", color=INK, pad=28)
    fig.tight_layout()
    fig.savefig(FIG / "gate_accuracy_by_case.png", dpi=150)
    plt.close(fig)


def circuit():
    p = pd.read_csv(RES / "circuit_ml.csv", dtype={"vector": str})
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, v in zip(axs, "AB"):
        name, col = VER[v]
        loglog(ax, p.leak_spice, p[f"leak_{v}"], col, f"{name}\n{stats(err(p[f'leak_{v}'], p.leak_spice))}")
        ax.set_xlabel("SPICE parity-checker leakage (A)")
    axs[0].set_ylabel("ML parity-checker leakage (A)")
    fig.suptitle("8-bit parity checker: 256 vectors x 200 corners, prescribed sum of 28 NAND2", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "circuit_parity.png", dpi=150)
    plt.close(fig)


def conditions(g):
    cor = corners("varying").set_index("corner")
    d = g.join(cor[["vdd", "temp", "wn", "wp", "n_vth0", "p_vth0"]], on="corner")
    d["p_vth0"] = d["p_vth0"].abs()
    xs = [("temp", "temperature (C)", 1), ("vdd", "VDD (V)", 1), ("n_vth0", "NMOS vth0 (V)", 1),
          ("p_vth0", "PMOS |vth0| (V)", 1), ("wn", "NMOS width Wn (nm)", 1e9), ("wp", "PMOS width Wp (nm)", 1e9)]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharey=True)
    axs = axs.ravel()
    for ax, (c, label, sc) in zip(axs, xs):
        bins = np.quantile(d[c], np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(d[c], bins[1:-1]), 0, 9)
        mid = [sc * d[c][idx == k].median() for k in range(10)]
        for v in "AB":
            name, col = VER[v]
            e = np.abs(err(d[f"leak_{v}"], d.leak_spice))
            ax.plot(mid, [e[idx == k].mean() for k in range(10)], color=col, lw=2, marker="o", ms=4, label=name)
        ax.set_xlabel(label)
        ax.grid(True)
    axs[0].set_ylabel("mean |error| of the gates (%)")
    axs[3].set_ylabel("mean |error| of the gates (%)")
    axs[0].legend(frameon=False)
    fig.suptitle("Gate error vs operating conditions (deciles of each variable; varying widths)", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "error_vs_conditions.png", dpi=150)
    plt.close(fig)


def main():
    part_a()
    g = pd.read_csv(RES / "gate_ml.csv", dtype={"pattern": str})
    gates(g)
    circuit()
    conditions(g)
    print(f"saved figures in {FIG.relative_to(C.ROOT)}/", flush=True)
    print("STEP 9 DONE", flush=True)


if __name__ == "__main__":
    main()

"""Part C steps 7-8: hierarchical ML leakage of the 8-bit parity checker vs SPICE.

For each input vector a logic simulation gives the input pattern of each of the
28 NAND2 instances; the NAND2 gate model of Part B (ML leakage for that pattern at
that corner) is called for every instance and the 28 values are summed.
Inputs are assumed ideal (0 / VDD) and inter-gate loading is ignored.
"""
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import circuits as G
import config as C
from importlib import import_module

summary = import_module("06_gate_ml").summary


def main():
    gml = pd.read_csv(C.RESULTS_DIR / "gate_ml.csv", dtype={"pattern": str})
    nand = gml[gml.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_ml")
    sp = pd.read_csv(C.RESULTS_DIR / "circuit_spice.csv", dtype={"vector": str})

    vectors = ["".join(v) for v in product("01", repeat=8)]
    # count of each NAND2 pattern per vector
    counts = {}
    for v in vectors:
        _, pats = G.parity_logic(v)
        c = pd.Series([p for _, p in pats]).value_counts()
        counts[v] = c.reindex(["00", "01", "10", "11"], fill_value=0)
    cnt = pd.DataFrame(counts).T                                        # vector x pattern

    leak = nand[["00", "01", "10", "11"]].to_numpy() @ cnt.to_numpy().T  # corner x vector
    ml = pd.DataFrame(leak, index=nand.index, columns=cnt.index).stack().rename("leak_ml").reset_index()
    ml.columns = ["corner", "vector", "leak_ml"]
    out = sp.merge(ml, on=["corner", "vector"])
    out["err_pct"] = 100 * (out.leak_ml - out.leak_spice) / out.leak_spice
    out.to_csv(C.RESULTS_DIR / "circuit_ml.csv", index=False, float_format="%.6e")

    tab = summary(out.assign(circuit="parity8"), ["circuit"])
    tab.to_csv(C.RESULTS_DIR / "circuit_results.csv", index=False, float_format="%.4g")
    by_ones = summary(out.assign(ones=out.vector.str.count("1")), ["ones"]).sort_values("ones")
    by_ones.to_csv(C.RESULTS_DIR / "circuit_results_by_ones.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(tab.to_string(index=False))
        print(by_ones.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(out.leak_spice, out.leak_ml, s=3, alpha=0.4, c=out.corner.map(G.load_corners().set_index("corner").temp), cmap="coolwarm")
    lo, hi = out[["leak_spice", "leak_ml"]].min().min() * 0.8, out[["leak_spice", "leak_ml"]].max().max() * 1.25
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("SPICE leakage (A)"); ax.set_ylabel("ML leakage (A)")
    a = tab.iloc[0]
    ax.set_title(f"8-bit parity checker, 256 vectors x 200 corners\nmedian err {a.median_err_pct:.2f}%, "
                 f"p95 {a.p95_err_pct:.2f}%, mean accuracy {a.accuracy_pct:.2f}%", fontsize=10)
    fig.colorbar(sc, label="temperature (°C)")
    ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "circuit_parity.png", dpi=150); plt.close(fig)
    print("saved results/circuit_ml.csv, results/circuit_results.csv, figures/circuit_parity.png")


if __name__ == "__main__":
    main()

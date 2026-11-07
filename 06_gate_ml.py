"""Part B steps 4-5: ML gate leakage and comparison with SPICE.

For every gate, input pattern and corner, the SPICE-recorded output / internal node
voltages give each transistor's terminal voltages; the trained transistor models
(chosen by each device's gate level: ON or OFF) predict the terminal currents and
the cut-set rule in circuits.py turns them into the gate leakage.

Outputs: results/gate_ml.csv (per simulation, with SPICE and ML internal node
voltages), results/gate_results.csv (the 18-row table: SPICE median, ML median,
bias, median error, p95 error), results/gate_node_voltages.csv and
figures/gate_parity.png.
ML internal node voltages (x, y, nd, nr) are solved from KCL with the transistor
models (circuits.ml_solve_nodes); the ML leakage itself uses the SPICE node
voltages, as the spec prescribes.
Error = (ML - SPICE) / SPICE; bias = median signed error.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import circuits as G
import config as C


def load_models():
    return {case: joblib.load(C.MODEL_DIR / f"{case}.joblib") for case in C.CASES}


def summary(df, keys):
    rows = []
    for k, d in df.groupby(keys, sort=False):
        e = (d.leak_ml - d.leak_spice) / d.leak_spice
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)),
                         n=len(d), spice_median_A=d.leak_spice.median(), ml_median_A=d.leak_ml.median(),
                         bias_pct=100 * e.median(), mean_abs_err_pct=100 * e.abs().mean(),
                         median_err_pct=100 * e.abs().median(), p95_err_pct=100 * e.abs().quantile(0.95),
                         accuracy_pct=100 - 100 * e.abs().mean()))
    return pd.DataFrame(rows)


def main():
    models = load_models()
    corners = G.load_corners().set_index("corner", drop=False)
    sp = pd.read_csv(C.RESULTS_DIR / "gate_spice.csv", dtype={"pattern": str})
    parts = []
    for (gate, pat), d in sp.groupby(["gate", "pattern"], sort=False):
        d = d.sort_values("corner")
        cs = corners.loc[d.corner.to_numpy()]
        node_v = {n: d[f"v_{n}"].to_numpy() for n in G.GATES[gate]["nodes"]}
        d = d.copy()
        d["leak_ml"] = G.ml_gate_leakage(gate, pat, cs, node_v, models)
        for node, v in G.ml_solve_nodes(gate, pat, cs, node_v, models).items():
            d[f"v_{node}_ml"] = v
        parts.append(d)
        print(f"  {gate:5s} {pat}: done", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out["err_pct"] = 100 * (out.leak_ml - out.leak_spice) / out.leak_spice
    out = out.rename(columns={f"v_{k}": f"v_{k}_spice" for k in ("x", "y", "nd", "nr", "out")})
    out.to_csv(C.RESULTS_DIR / "gate_ml.csv", index=False, float_format="%.6e")
    nv = []
    for k in ("x", "y", "nd", "nr"):
        if f"v_{k}_ml" in out:
            dd = out.dropna(subset=[f"v_{k}_ml"])
            for (g_, p_), q in dd.groupby(["gate", "pattern"], sort=False):
                nv.append({"gate": g_, "pattern": p_, "node": k,
                           "spice_median_V": q[f"v_{k}_spice"].median(), "ml_median_V": q[f"v_{k}_ml"].median(),
                           "median_abs_diff_mV": 1e3 * (q[f"v_{k}_ml"] - q[f"v_{k}_spice"]).abs().median()})
    pd.DataFrame(nv).to_csv(C.RESULTS_DIR / "gate_node_voltages.csv", index=False, float_format="%.5g")

    tab = summary(out, ["gate", "pattern"])
    tab.to_csv(C.RESULTS_DIR / "gate_results.csv", index=False, float_format="%.4g")
    allrow = summary(out.assign(all="all"), ["all"])
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(tab.to_string(index=False))
        print(allrow.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for gate, d in out.groupby("gate", sort=False):
        ax.scatter(d.leak_spice, d.leak_ml, s=6, alpha=0.6, label=gate)
    lo, hi = out[["leak_spice", "leak_ml"]].min().min() * 0.7, out[["leak_spice", "leak_ml"]].max().max() * 1.4
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("SPICE leakage (A)"); ax.set_ylabel("ML leakage (A)")
    a = allrow.iloc[0]
    ax.set_title(f"Gate leakage, 18 cases x 200 corners\nmedian err {a.median_err_pct:.2f}%, "
                 f"p95 {a.p95_err_pct:.2f}%, mean accuracy {a.accuracy_pct:.2f}%", fontsize=10)
    ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "gate_parity.png", dpi=150); plt.close(fig)
    print("saved results/gate_ml.csv, results/gate_results.csv, figures/gate_parity.png")


if __name__ == "__main__":
    main()

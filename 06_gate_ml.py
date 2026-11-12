"""Step 6 (Part B): ML leakage of the 18 gate cases at the 200 corners, from the transistor models.

Main setup (varying widths; data/corners_w.csv, results/gate_spice_w.csv for evaluation):
  version A  node voltages (internal nodes, output) taken from the gate's SPICE simulation
  version B  node voltages solved from KCL with the transistor models (no SPICE input at all)
  every transistor gets its real gate voltage and its corner width; cut-set leakage rule
Reference (fixed sizes; data/corners.csv, results/gate_spice.csv): version A.
Outputs (results/):
  gate_ml.csv            per gate case and corner: SPICE leakage, leak_A, leak_B, ML node voltages
                         (full precision, read by step 8)
  gate_results.csv       accuracy / median / p95 / bias per gate case, versions A and B
  node_voltages.csv      ML vs SPICE node voltages per gate case (version B)
  gate_ml_fixed.csv, gate_results_fixed.csv   fixed-size reference, version A
Runtime ~2 h on 4 cores (version B: 32 bisection steps x up to 8 sweeps per node).
Threads are pinned to 1 per process (as in the runs that produced the reported numbers), so the
results are bit-for-bit reproducible.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from final_method import RES, G, corners, gate_leakage, load_models, solve_nodes, summary  # noqa: E402

MODELS, COR, GSP = None, None, None


def _case(key):
    gate, pat = key
    t0 = time.time()
    d = GSP[(GSP.gate == gate) & (GSP.pattern == pat)].sort_values("corner").copy()
    cs = COR.loc[d.corner.to_numpy()]
    spice_v = {n: d[f"v_{n}"].to_numpy() for n in G.GATES[gate]["nodes"]}
    d["leak_A"] = gate_leakage(gate, pat, cs, spice_v, MODELS)
    ml_v, sweeps = solve_nodes(gate, pat, cs, MODELS)
    d["leak_B"] = gate_leakage(gate, pat, cs, ml_v, MODELS)
    nv = []
    for n in ml_v:
        d[f"v_{n}_ml"] = ml_v[n]
        diff = 1e3 * np.abs(ml_v[n] - spice_v[n])
        nv.append(dict(gate=gate, pattern=pat, node=n, spice_median_V=np.median(spice_v[n]),
                       ml_median_V=np.median(ml_v[n]), median_abs_diff_mV=np.median(diff),
                       p95_abs_diff_mV=np.quantile(diff, .95), max_abs_diff_mV=diff.max()))
    print(f"  {gate:5s} {pat}: {sweeps} sweeps, {time.time() - t0:.0f} s", flush=True)
    return d, nv


def table(df, col):
    s = summary(df.rename(columns={col: "leak_ml"}), ["gate", "pattern"])
    med = df.groupby(["gate", "pattern"], sort=False).agg(spice_median_amp=("leak_spice", "median"),
                                                          ml_median_amp=(col, "median"))
    return s.merge(med.reset_index(), on=["gate", "pattern"])


def fixed_sizes(models):
    cor = corners("fixed").set_index("corner", drop=False)
    sp = pd.read_csv(RES / "gate_spice.csv", dtype={"pattern": str})
    sp = sp[sp.corner.isin(cor.corner)]
    parts = []
    for (gate, pat), d in sp.groupby(["gate", "pattern"], sort=False):
        d = d.sort_values("corner").copy()
        cs = cor.loc[d.corner.to_numpy()]
        d["leak_ml"] = gate_leakage(gate, pat, cs, {n: d[f"v_{n}"].to_numpy() for n in G.GATES[gate]["nodes"]},
                                    models)
        parts.append(d)
    g = pd.concat(parts, ignore_index=True)
    g.to_csv(RES / "gate_ml_fixed.csv", index=False, float_format="%.17g")
    summary(g, ["gate", "pattern"]).to_csv(RES / "gate_results_fixed.csv", index=False, float_format="%.4g")
    a = summary(g.assign(all=1), ["all"]).iloc[0]
    print(f"FIXED SIZES (reference) version A: gates {a.accuracy_pct:.2f}% (median {a.median_err_pct:.2f}%)",
          flush=True)


def main():
    global MODELS, COR, GSP
    MODELS = load_models()
    COR = corners("varying").set_index("corner", drop=False)
    GSP = pd.read_csv(RES / "gate_spice_w.csv", dtype={"pattern": str})
    GSP = GSP[GSP.corner.isin(COR.corner)]
    keys = list(GSP.groupby(["gate", "pattern"], sort=False).groups)
    t0 = time.time()
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_case, keys, chunksize=1)
    g = pd.concat([r[0] for r in res], ignore_index=True)
    print(f"varying-width gates done in {time.time() - t0:.0f} s", flush=True)
    g.to_csv(RES / "gate_ml.csv", index=False, float_format="%.17g")
    pd.DataFrame([row for r in res for row in r[1]]).to_csv(RES / "node_voltages.csv", index=False,
                                                           float_format="%.5g")
    tab = table(g, "leak_A").merge(table(g, "leak_B"), on=["gate", "pattern", "n", "spice_median_amp"],
                                   suffixes=("_A", "_B"))
    tab.to_csv(RES / "gate_results.csv", index=False, float_format="%.4g")
    for ver in ("A", "B"):
        a = summary(g.rename(columns={f"leak_{ver}": "leak_ml"}).assign(all=1), ["all"]).iloc[0]
        print(f"VARYING WIDTHS version {ver}: gates {a.accuracy_pct:.2f}% (median {a.median_err_pct:.2f}%, "
              f"p95 {a.p95_err_pct:.2f}%, bias {a.bias_pct:+.2f}%)", flush=True)
    with pd.option_context("display.width", 250):
        print(tab[["gate", "pattern", "accuracy_pct_A", "accuracy_pct_B", "bias_pct_A", "bias_pct_B"]]
              .round(2).to_string(index=False))
    fixed_sizes(MODELS)
    print("STEP 6 DONE", flush=True)


if __name__ == "__main__":
    main()

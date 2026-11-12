"""Varying widths, step 2: ML prediction vs SPICE, version A and version B.

  version A : node voltages from the gate SPICE runs (as the project document prescribes)
  version B : node voltages predicted with the transistor models (exp_acc/ml_nodes.solve)
Final GPX transistor models, official cut rule, per-corner widths (common_w).
Version B runs the 18 gate cases in parallel (4 processes, 1 thread each).
Parity checker: sum of the 28 NAND2 leakages (A and B); composition = sum of the
SPICE NAND2 leakages vs SPICE of the full circuit.
Outputs (out/): w_gate_ml.csv, w_node_voltages.csv, w_gate_results.csv, w_summary.csv,
w_parity_by_ones.csv. SMOKE=1 -> first 3 corners (_smoke files).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_w import OUT, SMOKE, C, G, corners_w, load_models  # noqa: E402  (also installs the width hook)
from common import gate_leakage, summary  # noqa: E402
from common3 import parity_sum, vec_counts  # noqa: E402
import ml_nodes  # noqa: E402

SUF = "_smoke" if SMOKE else ""
MODELS, COR, GSP = None, None, None


def _init():
    global MODELS, COR, GSP
    MODELS = load_models()
    COR = corners_w().set_index("corner", drop=False)
    GSP = pd.read_csv(OUT / f"gate_spice_w{SUF}.csv", dtype={"pattern": str})


def _case(key):
    gate, pat = key
    t0 = time.time()
    d = GSP[(GSP.gate == gate) & (GSP.pattern == pat)].sort_values("corner").copy()
    cs = COR.loc[d.corner.to_numpy()]
    spice_v = {n: d[f"v_{n}"].to_numpy() for n in G.GATES[gate]["nodes"]}
    d["leak_A"] = gate_leakage(gate, pat, cs, spice_v, MODELS, "rule")
    ml_v, sweeps = ml_nodes.solve(gate, pat, cs, MODELS)
    d["leak_B"] = gate_leakage(gate, pat, cs, ml_v, MODELS, "rule")
    nv = []
    for n in ml_v:
        d[f"v_{n}_ml"] = ml_v[n]
        diff = 1e3 * np.abs(ml_v[n] - spice_v[n])
        nv.append(dict(gate=gate, pattern=pat, node=n, spice_median_V=np.median(spice_v[n]),
                       ml_median_V=np.median(ml_v[n]), median_abs_diff_mV=np.median(diff),
                       p95_abs_diff_mV=np.quantile(diff, .95), max_abs_diff_mV=diff.max()))
    print(f"  {gate:5s} {pat}: {sweeps} sweeps, {time.time() - t0:.0f} s", flush=True)
    return d, nv


def table(df, keys, col):
    s = summary(df.rename(columns={col: "leak_ml"}), keys)
    med = df.groupby(keys, sort=False).agg(spice_median_amp=("leak_spice", "median"), ml_median_amp=(col, "median"))
    return s.merge(med.reset_index(), on=keys)


def main():
    _init()
    keys = list(GSP.groupby(["gate", "pattern"], sort=False).groups)
    t0 = time.time()
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_case, keys, chunksize=1)
    g = pd.concat([r[0] for r in res], ignore_index=True)
    nv = pd.DataFrame([row for r in res for row in r[1]])
    print(f"gates done in {time.time() - t0:.0f} s", flush=True)
    g.to_csv(OUT / f"w_gate_ml{SUF}.csv", index=False, float_format="%.6e")
    nv.to_csv(OUT / f"w_node_voltages{SUF}.csv", index=False, float_format="%.5g")
    tab = table(g, ["gate", "pattern"], "leak_A").merge(table(g, ["gate", "pattern"], "leak_B"),
                                                        on=["gate", "pattern", "n", "spice_median_amp"],
                                                        suffixes=("_A", "_B"))
    tab.to_csv(OUT / f"w_gate_results{SUF}.csv", index=False, float_format="%.4g")

    psp = pd.read_csv(OUT / f"circuit_spice_w{SUF}.csv", dtype={"vector": str})
    cnt = vec_counts()
    sp_nand = g[g.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_spice")
    comp = summary(parity_sum(sp_nand, cnt, psp).assign(all=1), ["all"]).iloc[0]
    rows, ones = [], []
    for ver in ("A", "B"):
        ag = summary(g.rename(columns={f"leak_{ver}": "leak_ml"}).assign(all=1), ["all"]).iloc[0]
        nand = g[g.gate == "NAND2"].pivot(index="corner", columns="pattern", values=f"leak_{ver}")
        p = parity_sum(nand, cnt, psp)
        ap = summary(p.assign(all=1), ["all"]).iloc[0]
        rows.append(dict(version="A: SPICE node voltages" if ver == "A" else "B: ML node voltages",
                         gate_acc=ag.accuracy_pct, gate_median=ag.median_err_pct, gate_p95=ag.p95_err_pct,
                         gate_bias=ag.bias_pct, parity_acc=ap.accuracy_pct, parity_median=ap.median_err_pct,
                         parity_p95=ap.p95_err_pct, parity_bias=ap.bias_pct,
                         composition_acc=comp.accuracy_pct, composition_bias=comp.bias_pct))
        ones.append(summary(p.assign(ones=p.vector.str.count("1"), version=ver), ["version", "ones"]))
        print(f"VARYING WIDTHS version {ver}: gates {ag.accuracy_pct:.2f}% (med {ag.median_err_pct:.2f}%, "
              f"p95 {ag.p95_err_pct:.2f}%, bias {ag.bias_pct:+.2f}%) | parity {ap.accuracy_pct:.2f}% "
              f"(med {ap.median_err_pct:.2f}%, bias {ap.bias_pct:+.2f}%)", flush=True)
    print(f"composition (sum of SPICE NAND2 vs SPICE circuit): accuracy {comp.accuracy_pct:.2f}%, "
          f"bias {comp.bias_pct:+.2f}%", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"w_summary{SUF}.csv", index=False, float_format="%.4g")
    pd.concat(ones).to_csv(OUT / f"w_parity_by_ones{SUF}.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250):
        print(tab[["gate", "pattern", "accuracy_pct_A", "accuracy_pct_B", "bias_pct_A", "bias_pct_B"]]
              .round(2).to_string(index=False))
    print("EVAL_W DONE", flush=True)


if __name__ == "__main__":
    main()

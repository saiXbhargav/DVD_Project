"""Version B on the fresh corners, parallel (4 processes, 1 thread each).

Same solver and outputs as ml_nodes.py (ml_nodes.solve, exp_acc/fresh/mlnodes_*.csv); only
the 18 gate cases run in parallel instead of one after another. SMOKE=1 -> 3 corners,
outputs in exp_acc/smoke_mlnodes_par/.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import HERE, G, gate_leakage, summary  # noqa: E402
from common3 import parity_sum, vec_counts  # noqa: E402
import ml_nodes  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
MODELS, COR, GSP = None, None, None


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
    print(f"  fresh {gate:5s} {pat}: {sweeps} sweeps, {time.time() - t0:.0f} s", flush=True)
    return d, nv


def main():
    global MODELS, COR, GSP
    fresh = HERE / "fresh"
    outdir = HERE / "smoke_mlnodes_par" if SMOKE else fresh
    outdir.mkdir(exist_ok=True)
    MODELS = ml_nodes.models_gpx()
    COR = pd.read_csv(fresh / "corners.csv")
    GSP = pd.read_csv(fresh / "gate_spice.csv", dtype={"pattern": str})
    psp = pd.read_csv(fresh / "circuit_spice.csv", dtype={"vector": str})
    if SMOKE:
        keep = COR.corner.iloc[:3]
        COR, GSP, psp = COR[COR.corner.isin(keep)], GSP[GSP.corner.isin(keep)], psp[psp.corner.isin(keep)]
    COR = COR.set_index("corner", drop=False)
    keys = list(GSP.groupby(["gate", "pattern"], sort=False).groups)
    t0 = time.time()
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_case, keys, chunksize=1)
    g = pd.concat([r[0] for r in res], ignore_index=True)
    nv = pd.DataFrame([row for r in res for row in r[1]])
    print(f"fresh gates done in {time.time() - t0:.0f} s", flush=True)
    g.to_csv(outdir / "mlnodes_gate_ml.csv", index=False, float_format="%.6e")
    nv.to_csv(outdir / "mlnodes_node_voltages.csv", index=False, float_format="%.5g")
    sA = summary(g.rename(columns={"leak_A": "leak_ml"}), ["gate", "pattern"])
    sB = summary(g.rename(columns={"leak_B": "leak_ml"}), ["gate", "pattern"])
    sA.merge(sB, on=["gate", "pattern", "n"], suffixes=("_A_spice_nodes", "_B_ml_nodes")).to_csv(
        outdir / "mlnodes_gate_results.csv", index=False, float_format="%.4g")
    cnt, rows = vec_counts(), []
    for ver in ("A", "B"):
        ag = summary(g.rename(columns={f"leak_{ver}": "leak_ml"}).assign(all=1), ["all"]).iloc[0]
        nand = g[g.gate == "NAND2"].pivot(index="corner", columns="pattern", values=f"leak_{ver}")
        ap = summary(parity_sum(nand, cnt, psp).assign(all=1), ["all"]).iloc[0]
        rows.append(dict(corners="fresh", version="A: SPICE node voltages" if ver == "A" else "B: ML node voltages",
                         gate_acc=ag.accuracy_pct, gate_median=ag.median_err_pct, gate_p95=ag.p95_err_pct,
                         gate_bias=ag.bias_pct, parity_acc=ap.accuracy_pct, parity_median=ap.median_err_pct,
                         parity_p95=ap.p95_err_pct, parity_bias=ap.bias_pct))
        print(f"fresh version {ver}: gates {ag.accuracy_pct:.2f}% (med {ag.median_err_pct:.2f}%, p95 "
              f"{ag.p95_err_pct:.2f}%, bias {ag.bias_pct:+.2f}%) | parity {ap.accuracy_pct:.2f}% "
              f"(med {ap.median_err_pct:.2f}%, bias {ap.bias_pct:+.2f}%)", flush=True)
    pd.DataFrame(rows).to_csv(outdir / "mlnodes_summary.csv", index=False, float_format="%.4g")
    print("ML NODES PAR DONE", flush=True)


if __name__ == "__main__":
    main()

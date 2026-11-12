"""Fix 1, step 3: gates and parity checker with the gate-voltage-aware models (VG) and each
transistor's real gate voltage.

  varying widths (main setup): exp_width corners / gate SPICE / parity SPICE (read-only)
      version A: node voltages from SPICE; version B: node voltages predicted (ml_nodes.solve)
  fixed sizes (reference): official corners and SPICE, version A only
Parity checker: prescribed sum of the 28 NAND2 leakages (its NAND2 inputs are exact 0 / VDD, so
fix 1 mainly changes the two-stage gates; the parity composition limit is unchanged).
Outputs: out/vg_gate_ml.csv, out/vg_node_voltages.csv, out/vg_gate_results.csv,
out/vg_summary.csv, out/vg_fixed_gate_results.csv. SMOKE=1 -> 3 corners, _smoke files.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_vg import OUT, SMOKE, C, G  # noqa: E402  (installs the real-Vg hook)
from common import gate_leakage, summary  # noqa: E402
from common3 import parity_sum, vec_counts  # noqa: E402
import common_w  # noqa: E402
import ml_nodes  # noqa: E402

MODELSET = os.environ.get("MODELSET", "VG")      # VG = new models; GPX = current final models + real Vg
SUF = ("_smoke" if SMOKE else "") + ("_gpx" if MODELSET == "GPX" else "")
MODELS, COR, GSP = None, None, None


def load_vg():
    if MODELSET == "GPX":
        d = OUT.parent.parent / "exp_acc" / "out" / "models"
        return {c: joblib.load(d / f"{'GPBag87X' if c in ('nmos_on', 'pmos_on') else 'GPBag87'}_{c}.joblib")
                for c in C.CASES}
    return {c: joblib.load(OUT / "models" / f"VG_{c}{'_smoke' if SMOKE else ''}.joblib") for c in C.CASES}


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


def table(df, col):
    s = summary(df.rename(columns={col: "leak_ml"}), ["gate", "pattern"])
    med = df.groupby(["gate", "pattern"], sort=False).agg(spice_median_amp=("leak_spice", "median"),
                                                          ml_median_amp=(col, "median"))
    return s.merge(med.reset_index(), on=["gate", "pattern"])


def fixed_sizes(models):
    """Reference: fixed gate sizing, official corners, version A only."""
    cor = G.load_corners().set_index("corner", drop=False)
    sp = pd.read_csv(C.RESULTS_DIR / "gate_spice.csv", dtype={"pattern": str})
    if SMOKE:
        sp = sp[sp.corner < 3]
    parts = []
    for (gate, pat), d in sp.groupby(["gate", "pattern"], sort=False):
        d = d.sort_values("corner").copy()
        cs = cor.loc[d.corner.to_numpy()]
        d["leak_ml"] = gate_leakage(gate, pat, cs, {n: d[f"v_{n}"].to_numpy() for n in G.GATES[gate]["nodes"]},
                                    models, "rule")
        parts.append(d)
    g = pd.concat(parts, ignore_index=True)
    summary(g, ["gate", "pattern"]).to_csv(OUT / f"vg_fixed_gate_results{SUF}.csv", index=False, float_format="%.4g")
    psp = pd.read_csv(C.RESULTS_DIR / "circuit_spice.csv", dtype={"vector": str})
    if SMOKE:
        psp = psp[psp.corner < 3]
    nand = g[g.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_ml")
    ag = summary(g.assign(all=1), ["all"]).iloc[0]
    ap = summary(parity_sum(nand, vec_counts(), psp).assign(all=1), ["all"]).iloc[0]
    print(f"FIXED SIZES (reference) version A: gates {ag.accuracy_pct:.2f}% (med {ag.median_err_pct:.2f}%) | "
          f"parity {ap.accuracy_pct:.2f}% (bias {ap.bias_pct:+.2f}%)", flush=True)
    return dict(setup="fixed sizes", version="A: SPICE node voltages", gate_acc=ag.accuracy_pct,
                gate_median=ag.median_err_pct, gate_p95=ag.p95_err_pct, gate_bias=ag.bias_pct,
                parity_acc=ap.accuracy_pct, parity_median=ap.median_err_pct, parity_p95=ap.p95_err_pct,
                parity_bias=ap.bias_pct)


def main():
    global MODELS, COR, GSP
    MODELS = load_vg()
    COR = common_w.corners_w().set_index("corner", drop=False)
    GSP = pd.read_csv(common_w.OUT / "gate_spice_w.csv", dtype={"pattern": str})
    psp = pd.read_csv(common_w.OUT / "circuit_spice_w.csv", dtype={"vector": str})
    if SMOKE:
        GSP, psp = GSP[GSP.corner < 3], psp[psp.corner < 3]
    keys = list(GSP.groupby(["gate", "pattern"], sort=False).groups)
    t0 = time.time()
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_case, keys, chunksize=1)
    g = pd.concat([r[0] for r in res], ignore_index=True)
    print(f"varying-width gates done in {time.time() - t0:.0f} s", flush=True)
    g.to_csv(OUT / f"vg_gate_ml{SUF}.csv", index=False, float_format="%.6e")
    pd.DataFrame([row for r in res for row in r[1]]).to_csv(OUT / f"vg_node_voltages{SUF}.csv", index=False,
                                                           float_format="%.5g")
    tab = table(g, "leak_A").merge(table(g, "leak_B"), on=["gate", "pattern", "n", "spice_median_amp"],
                                   suffixes=("_A", "_B"))
    tab.to_csv(OUT / f"vg_gate_results{SUF}.csv", index=False, float_format="%.4g")
    cnt, rows = vec_counts(), []
    for ver in ("A", "B"):
        ag = summary(g.rename(columns={f"leak_{ver}": "leak_ml"}).assign(all=1), ["all"]).iloc[0]
        nand = g[g.gate == "NAND2"].pivot(index="corner", columns="pattern", values=f"leak_{ver}")
        ap = summary(parity_sum(nand, cnt, psp).assign(all=1), ["all"]).iloc[0]
        rows.append(dict(setup="varying widths", version="A: SPICE node voltages" if ver == "A" else
                         "B: ML node voltages", gate_acc=ag.accuracy_pct, gate_median=ag.median_err_pct,
                         gate_p95=ag.p95_err_pct, gate_bias=ag.bias_pct, parity_acc=ap.accuracy_pct,
                         parity_median=ap.median_err_pct, parity_p95=ap.p95_err_pct, parity_bias=ap.bias_pct))
        print(f"VARYING WIDTHS + VG version {ver}: gates {ag.accuracy_pct:.2f}% (med {ag.median_err_pct:.2f}%, "
              f"p95 {ag.p95_err_pct:.2f}%, bias {ag.bias_pct:+.2f}%) | parity {ap.accuracy_pct:.2f}% "
              f"(med {ap.median_err_pct:.2f}%, bias {ap.bias_pct:+.2f}%)", flush=True)
    rows.append(fixed_sizes(MODELS))
    pd.DataFrame(rows).to_csv(OUT / f"vg_summary{SUF}.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250):
        print(tab[["gate", "pattern", "accuracy_pct_A", "accuracy_pct_B", "bias_pct_A", "bias_pct_B"]]
              .round(2).to_string(index=False))
    print("EVAL_VG DONE", flush=True)


if __name__ == "__main__":
    main()

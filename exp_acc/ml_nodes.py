"""Version B (TA request): gate and parity leakage with ML-PREDICTED node voltages.

Version A (the prescribed one) takes every internal node voltage and the output voltage
from the gate's SPICE simulation. Version B uses no SPICE information at all:
  1. every node (x / y, the NAND2 / NOR2 output nd / nr, and the gate output) starts at
     its logic level (0 or VDD; VDD/2 for a stack node between two OFF transistors);
  2. each node voltage is solved from KCL with the four transistor models: bisection on
     [0, VDD] for the voltage where the model currents into the node sum to zero,
     Gauss-Seidel over the nodes until no node moves by more than 0.1 uV;
  3. the gate leakage is computed from those voltages with the same cut-set rule as A.
Parity checker: the sum of the 28 NAND2 leakages, each from version B.
Models: GPX (GPBag87 for OFF, GPBag87X for ON). Corners: official 200 and fresh 100.
Output: exp_acc/out/mlnodes_*.csv, exp_acc/fresh/mlnodes_*.csv. SMOKE=1 -> 3 corners only.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd

from common import HERE, OUT, C, G, gate_leakage, summary
from common3 import parity_sum, vec_counts

SMOKE = os.environ.get("SMOKE") == "1"
ON = ("nmos_on", "pmos_on")
ITERS, SWEEPS, TOL = 32, 8, 1e-7


def models_gpx():
    return {c: joblib.load(OUT / "models" / f"{'GPBag87X' if c in ON else 'GPBag87'}_{c}.joblib") for c in C.CASES}


def logic_levels(gate, pat):
    """Logic value (0 / 1) of every node, None for a stack node between two OFF devices."""
    a = int(pat[0])
    b = int(pat[1]) if len(pat) > 1 else None
    if gate == "INV":
        return {"out": 1 - a}
    if gate in ("NAND2", "AND2"):
        n = 1 - (a & b)
        x = 0 if b == 1 else (n if a == 1 else None)     # MNB (gate b) to ground, MNA (gate a) to the output
        return {"x": x, "out": n} if gate == "NAND2" else {"x": x, "nd": n, "out": 1 - n}
    n = 1 - (a | b)
    y = 1 if a == 0 else (n if b == 0 else None)         # MPA (gate a) to VDD, MPB (gate b) to the output
    return {"y": y, "out": n} if gate == "NOR2" else {"y": y, "nr": n, "out": 1 - n}


def solve(gate, pat, cs, models):
    vdd = cs["vdd"].to_numpy(float)
    lv = logic_levels(gate, pat)
    nodes = G.GATES[gate]["nodes"]
    init = {n: vdd * (0.5 if lv[n] is None else lv[n]) for n in nodes}
    ctx = G._Ctx(gate, pat, cs, init)                    # logic sides come from the logic levels
    for sweep in range(1, SWEEPS + 1):
        moved = 0.0
        for node in nodes:
            devs = [dv for dv in ctx.g["devices"] if node in dv[2:6]]
            lo, hi = np.zeros(ctx.n), vdd.copy()
            old = ctx.node_v[node].copy()
            for _ in range(ITERS):
                mid = 0.5 * (lo + hi)
                ctx.node_v[node] = mid
                f = np.zeros(ctx.n)                      # model current flowing out of the devices into the node
                for dv in devs:
                    I = G._device_currents(ctx, dv, models)
                    for col, term in enumerate(dv[2:6]):
                        if term == node:
                            f += I[:, col]
                lo = np.where(f > 0, mid, lo)
                hi = np.where(f > 0, hi, mid)
            ctx.node_v[node] = 0.5 * (lo + hi)
            moved = max(moved, float(np.max(np.abs(ctx.node_v[node] - old))))
        if moved < TOL:
            break
    return {n: ctx.node_v[n] for n in nodes}, sweep


def run(label, cor, gsp, psp, models, outdir):
    cor = cor.set_index("corner", drop=False)
    parts, nv_rows = [], []
    t0 = time.time()
    for (gate, pat), d in gsp.groupby(["gate", "pattern"], sort=False):
        d = d.sort_values("corner").copy()
        cs = cor.loc[d.corner.to_numpy()]
        spice_v = {n: d[f"v_{n}"].to_numpy() for n in G.GATES[gate]["nodes"]}
        d["leak_A"] = gate_leakage(gate, pat, cs, spice_v, models, "rule")
        ml_v, sweeps = solve(gate, pat, cs, models)
        d["leak_B"] = gate_leakage(gate, pat, cs, ml_v, models, "rule")
        for n in ml_v:
            d[f"v_{n}_ml"] = ml_v[n]
            diff = 1e3 * np.abs(ml_v[n] - spice_v[n])
            nv_rows.append(dict(gate=gate, pattern=pat, node=n, spice_median_V=np.median(spice_v[n]),
                                ml_median_V=np.median(ml_v[n]), median_abs_diff_mV=np.median(diff),
                                p95_abs_diff_mV=np.quantile(diff, .95), max_abs_diff_mV=diff.max()))
        parts.append(d)
        print(f"  {label} {gate:5s} {pat}: {sweeps} sweeps, {time.time() - t0:.0f} s", flush=True)
    g = pd.concat(parts, ignore_index=True)
    g.to_csv(outdir / "mlnodes_gate_ml.csv", index=False, float_format="%.6e")
    pd.DataFrame(nv_rows).to_csv(outdir / "mlnodes_node_voltages.csv", index=False, float_format="%.5g")
    sA = summary(g.rename(columns={"leak_A": "leak_ml"}), ["gate", "pattern"])
    sB = summary(g.rename(columns={"leak_B": "leak_ml"}), ["gate", "pattern"])
    tab = sA.merge(sB, on=["gate", "pattern", "n"], suffixes=("_A_spice_nodes", "_B_ml_nodes"))
    tab.to_csv(outdir / "mlnodes_gate_results.csv", index=False, float_format="%.4g")
    cnt = vec_counts()
    rows = []
    for ver in ("A", "B"):
        ag = summary(g.rename(columns={f"leak_{ver}": "leak_ml"}).assign(all=1), ["all"]).iloc[0]
        nand = g[g.gate == "NAND2"].pivot(index="corner", columns="pattern", values=f"leak_{ver}")
        ap = summary(parity_sum(nand, cnt, psp).assign(all=1), ["all"]).iloc[0]
        rows.append(dict(corners=label, version="A: SPICE node voltages" if ver == "A" else "B: ML node voltages",
                         gate_acc=ag.accuracy_pct, gate_median=ag.median_err_pct, gate_p95=ag.p95_err_pct,
                         gate_bias=ag.bias_pct, parity_acc=ap.accuracy_pct, parity_median=ap.median_err_pct,
                         parity_p95=ap.p95_err_pct, parity_bias=ap.bias_pct))
        print(f"{label} version {ver}: gates {ag.accuracy_pct:.2f}% (med {ag.median_err_pct:.2f}%, p95 "
              f"{ag.p95_err_pct:.2f}%, bias {ag.bias_pct:+.2f}%) | parity {ap.accuracy_pct:.2f}% "
              f"(med {ap.median_err_pct:.2f}%, bias {ap.bias_pct:+.2f}%)", flush=True)
    pd.DataFrame(rows).to_csv(outdir / "mlnodes_summary.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250):
        print(tab[["gate", "pattern", "accuracy_pct_A_spice_nodes", "accuracy_pct_B_ml_nodes",
                   "bias_pct_A_spice_nodes", "bias_pct_B_ml_nodes"]].round(2).to_string(index=False))
        print(pd.DataFrame(nv_rows)[["gate", "pattern", "node", "spice_median_V", "ml_median_V",
                                     "median_abs_diff_mV", "max_abs_diff_mV"]].to_string(
            index=False, float_format=lambda x: f"{x:.4g}"))


def main():
    models = models_gpx()
    sets = [("official", G.load_corners(), pd.read_csv(C.RESULTS_DIR / "gate_spice.csv", dtype={"pattern": str}),
             pd.read_csv(C.RESULTS_DIR / "circuit_spice.csv", dtype={"vector": str}), OUT),
            ("fresh", pd.read_csv(HERE / "fresh" / "corners.csv"),
             pd.read_csv(HERE / "fresh" / "gate_spice.csv", dtype={"pattern": str}),
             pd.read_csv(HERE / "fresh" / "circuit_spice.csv", dtype={"vector": str}), HERE / "fresh")]
    for label, cor, gsp, psp, outdir in sets:
        if SMOKE:
            keep = cor.corner.iloc[:3]
            cor, gsp, psp = cor[cor.corner.isin(keep)], gsp[gsp.corner.isin(keep)], psp[psp.corner.isin(keep)]
            outdir = HERE / "smoke_mlnodes"
            outdir.mkdir(exist_ok=True)
        run(label, cor, gsp, psp, models, outdir)
        if SMOKE:
            break
    print("ML NODES DONE", flush=True)


if __name__ == "__main__":
    main()

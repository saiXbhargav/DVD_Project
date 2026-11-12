"""Interim: varying-width gates, version A only (SPICE node voltages). Read-only except a CSV."""
import numpy as np
import pandas as pd

from common_w import OUT, G, corners_w, load_models
from common import gate_leakage, summary

models = load_models()
cor = corners_w().set_index("corner", drop=False)
sp = pd.read_csv(OUT / "gate_spice_w.csv", dtype={"pattern": str})
parts = []
for (gate, pat), d in sp.groupby(["gate", "pattern"], sort=False):
    d = d.sort_values("corner").copy()
    cs = cor.loc[d.corner.to_numpy()]
    d["leak_ml"] = gate_leakage(gate, pat, cs, {n: d[f"v_{n}"].to_numpy() for n in G.GATES[gate]["nodes"]}, models, "rule")
    parts.append(d)
g = pd.concat(parts, ignore_index=True)
t = summary(g, ["gate", "pattern"])
med = g.groupby(["gate", "pattern"], sort=False).agg(spice_median=("leak_spice", "median"), ml_median=("leak_ml", "median"))
t = t.merge(med.reset_index(), on=["gate", "pattern"])
t.to_csv(OUT / "quick_gate_results_A.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 200)
print(t[["gate", "pattern", "spice_median", "ml_median", "bias_pct", "median_err_pct", "p95_err_pct",
         "accuracy_pct"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
a = summary(g.assign(all=1), ["all"]).iloc[0]
print(f"ALL: accuracy {a.accuracy_pct:.2f}%, median {a.median_err_pct:.2f}%, p95 {a.p95_err_pct:.2f}%, bias {a.bias_pct:+.2f}%")
vdd = g.corner.map(cor.vdd)
logic = {"INV": lambda a, b: 1 - a, "NAND2": lambda a, b: 1 - (a & b), "NOR2": lambda a, b: 1 - (a | b),
         "AND2": lambda a, b: a & b, "OR2": lambda a, b: a | b}
exp = np.array([logic[x](int(p[0]), int(p[-1])) for x, p in zip(g.gate, g.pattern)])
off = np.abs(g.v_out - exp * vdd) > 0.05 * vdd
print("outputs off their logic level by > 5 % of VDD:")
print(g[off].assign(wn_nm=g.corner.map(cor.wn) * 1e9, wp_nm=g.corner.map(cor.wp) * 1e9, temp=g.corner.map(cor.temp),
                    v_out=g.v_out, err_pct=100 * (g.leak_ml - g.leak_spice) / g.leak_spice)[
    ["gate", "pattern", "corner", "temp", "wn_nm", "wp_nm", "v_out", "err_pct"]].round(3).to_string(index=False))

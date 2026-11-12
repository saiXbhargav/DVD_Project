"""Where do the varying-width errors come from? Reads saved results only."""
import numpy as np
import pandas as pd

from common_w import OUT

g = pd.read_csv(OUT / "w_gate_ml.csv", dtype={"pattern": str})
cor = pd.read_csv(OUT / "corners_w.csv").set_index("corner")
g["wn"], g["wp"] = g.corner.map(cor.wn) * 1e9, g.corner.map(cor.wp) * 1e9
g["temp"], g["vdd"] = g.corner.map(cor.temp), g.corner.map(cor.vdd)
g["eA"] = 100 * (g.leak_A - g.leak_spice) / g.leak_spice
g["eB"] = 100 * (g.leak_B - g.leak_spice) / g.leak_spice
g["case"] = g.gate + " " + g.pattern
pd.set_option("display.width", 220)

print("=== mean |error| (version A) by Wp and Wn bins, all gates")
bins = [16, 40, 64, 88, 112, 128]
for w in ("wp", "wn"):
    t = g.groupby(pd.cut(g[w], bins), observed=False).eA.agg(lambda e: e.abs().mean()).round(2)
    print(w, dict(zip([str(i) for i in t.index], t.values)))

print("\n=== per case: mean |err| A for Wp < 40 nm vs Wp >= 40 nm, and Wn/Wp ratio > 3 vs rest")
rows = []
for c, d in g.groupby("case", sort=False):
    narrow = d.wp < 40
    skew = (d.wn / d.wp) > 3
    rows.append(dict(case=c, all=d.eA.abs().mean(), wp_lt40=d.eA[narrow].abs().mean(), wp_ge40=d.eA[~narrow].abs().mean(),
                     wn_over_wp_gt3=d.eA[skew].abs().mean(), rest=d.eA[~skew].abs().mean(), n_skew=int(skew.sum())))
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n=== two-stage gates: signed error A vs sag of the internal node (mV from its rail)")
for c, node, hi in (("OR2 00", "nr", True), ("AND2 11", "nd", False), ("AND2 01", "nd", True), ("OR2 01", "nr", False)):
    d = g[g.case == c].copy()
    sag = 1e3 * ((d.vdd - d[f"v_{node}"]) if hi else d[f"v_{node}"])
    d["sag_mV"] = sag
    t = d.groupby(pd.qcut(d.sag_mV, 4), observed=False).agg(sag=("sag_mV", "median"), bias=("eA", "median"),
                                                             wp=("wp", "median"), wn=("wn", "median"))
    print(c, f"(node {node}; corr(sag, error) = {np.corrcoef(d.sag_mV, d.eA)[0, 1]:+.2f})")
    print(t.round(2).to_string())

print("\n=== worst 10 simulations (version A)")
print(g.reindex(g.eA.abs().sort_values(ascending=False).index)[["case", "corner", "temp", "vdd", "wn", "wp", "eA", "eB"]]
      .head(10).round(2).to_string(index=False))

print("\n=== NOR2 11 by temperature (A)")
d = g[g.case == "NOR2 11"]
print(d.groupby(pd.cut(d.temp, [-40, 0, 40, 80, 125]), observed=False).eA.agg(["median", lambda e: e.abs().mean()]).round(2))

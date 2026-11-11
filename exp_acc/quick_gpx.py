"""Interim: gates + parity with GPX (GPBag87, ON cases -> GPBag87X); no MLPs needed."""
from itertools import product

import pandas as pd

from common import DATA, C, G, summary
from evaluate2 import ON, gates, ld

base = {c: ld("GPBag87", c) for c in C.CASES}
gpx = {c: ld("GPBag87X", c) if c in ON else base[c] for c in C.CASES}
vcor = pd.read_csv(DATA / "val_corners.csv").set_index("corner", drop=False)
vsp = pd.read_csv("out/gate_spice_val.csv", dtype={"pattern": str})
tcor = G.load_corners().set_index("corner", drop=False)
tsp = pd.read_csv(C.RESULTS_DIR / "gate_spice.csv", dtype={"pattern": str})
csp = pd.read_csv(C.RESULTS_DIR / "circuit_spice.csv", dtype={"vector": str})
vectors = ["".join(v) for v in product("01", repeat=8)]
cnt = pd.DataFrame({v: pd.Series([p for _, p in G.parity_logic(v)[1]]).value_counts()
                    .reindex(["00", "01", "10", "11"], fill_value=0) for v in vectors}).T
gv, gt = gates(vsp, vcor, gpx), gates(tsp, tcor, gpx)
nand = gt[gt.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_ml")[["00", "01", "10", "11"]]
ml = pd.DataFrame(nand.to_numpy() @ cnt.to_numpy().T, index=nand.index, columns=cnt.index).stack().rename("leak_ml").reset_index()
ml.columns = ["corner", "vector", "leak_ml"]
p = csp.merge(ml, on=["corner", "vector"])
for lab, d in (("VAL gates", gv), ("TEST gates", gt), ("parity", p)):
    a = summary(d.assign(all=1), ["all"]).iloc[0]
    print(f"{lab:10s} accuracy {a.accuracy_pct:.2f}%  median {a.median_err_pct:.2f}%  p95 {a.p95_err_pct:.2f}%  bias {a.bias_pct:+.2f}%")
t = summary(gt, ["gate", "pattern"])[["gate", "pattern", "accuracy_pct", "median_err_pct", "bias_pct"]]
print(t.round(2).to_string(index=False))
print("cases >= 99 %:", (t.accuracy_pct >= 99).sum())

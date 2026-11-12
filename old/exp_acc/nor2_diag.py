"""NOR2 11 diagnosis: GPBag87 everywhere, swapping one case model at a time."""
import sys

import joblib
import numpy as np
import pandas as pd

from common import HERE, OUT, C, G, GPCap, gate_leakage

sys.modules["__main__"].GPCap = GPCap


def load(name, case):
    if name == "GP_7k":
        return joblib.load(C.MODEL_DIR / f"{case}.joblib")
    if name == "GP_cap":
        return joblib.load(HERE.parent / "exp_extra" / "out" / "models" / f"GP_cap_{case}.joblib")
    return joblib.load(OUT / "models" / f"{name}_{case}.joblib")


cor = G.load_corners().set_index("corner", drop=False)
sp = pd.read_csv(C.RESULTS_DIR / "gate_spice.csv", dtype={"pattern": str})
d = sp[(sp.gate == "NOR2") & (sp.pattern == "11")].sort_values("corner")
cs = cor.loc[d.corner.to_numpy()]
nv = {n: d[f"v_{n}"].to_numpy() for n in G.GATES["NOR2"]["nodes"]}
print("NOR2 11 node voltages: out median %.2e V, y median %.3f V" % (np.median(nv["out"]), np.median(nv["y"])))
base = {c: load("GPBag87", c) for c in C.CASES}
for case in C.CASES:
    for alt in ("GP_7k", "GP_cap", "GPBag47"):
        m = dict(base)
        m[case] = load(alt, case)
        e = (gate_leakage("NOR2", "11", cs, nv, m) - d.leak_spice.to_numpy()) / d.leak_spice.to_numpy()
        print(f"  {case:9s} <- {alt:8s}: accuracy {100 - 100*np.abs(e).mean():6.2f}%  median {100*np.median(np.abs(e)):5.2f}%  "
              f"worst {100*np.abs(e).max():7.1f}%")
e = (gate_leakage("NOR2", "11", cs, nv, base) - d.leak_spice.to_numpy()) / d.leak_spice.to_numpy()
print(f"  all GPBag87: accuracy {100 - 100*np.abs(e).mean():.2f}%, worst {100*np.abs(e).max():.1f}%")
w = np.argsort(-np.abs(e))[:5]
print(cs.iloc[w][["vdd", "temp", "n_vth0", "p_vth0"]].assign(err_pct=100 * e[w], out_V=nv["out"][w], y_V=nv["y"][w]).round(4))

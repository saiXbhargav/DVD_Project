"""NMOS ON at Vd = Vs = 0 (the NOR2 11 operating point): model vs single-transistor SPICE."""
import sys
from importlib import import_module

import joblib
import numpy as np
import pandas as pd

from common import HERE, OUT, C, G, GPCap

sys.modules["__main__"].GPCap = GPCap
gen = import_module("01_gen_transistor_data")
import spice_utils as S  # noqa: E402

cor = G.load_corners()
df = pd.DataFrame({"vdd": cor.vdd, "temp": cor.temp, "w": G.WU, "vd": 0.0, "vs": 0.0, "vg": cor.vdd, "vb": 0.0})
for p in C.PROC_PARAMS:
    df[p] = cor[f"n_{p}"]
I = S.run_op([gen.netlist(r, "n", "nmos_on") for r in df.to_dict("records")], ["i(vd)", "i(vg)", "i(vs)", "i(vb)"])
cold = df.temp < 0
print("SPICE ig at Vds=0: median %.2e A; share of |ig| in (|id|+|is|+|ib|+|ig|)/2: %.3f" %
      (np.median(np.abs(I[:, 1])), np.median(np.abs(I[:, 1]) / (np.abs(I).sum(1) / 2))))
for name, f in (("GP_7k", C.MODEL_DIR / "nmos_on.joblib"),
                ("GP_cap", HERE.parent / "exp_extra/out/models/GP_cap_nmos_on.joblib"),
                ("GPBag47", OUT / "models/GPBag47_nmos_on.joblib"),
                ("GPBag87", OUT / "models/GPBag87_nmos_on.joblib")):
    P = joblib.load(f).predict(df)
    for j, c in ((1, "ig"), (3, "ib")):
        e = 100 * (P[:, j] - I[:, j]) / np.abs(I[:, j])
        print(f"  {name:8s} {c}: median |err| {np.median(np.abs(e)):6.2f}%   cold (<0 C) median {np.median(np.abs(e[cold])):6.2f}%"
              f"  worst {e[np.argmax(np.abs(e))]:+8.1f}% at T = {df.temp[np.argmax(np.abs(e))]:.0f} C")
# training-data coverage near this point
tr = pd.read_csv(C.DATA_DIR / "nmos_on.csv")
near = (np.maximum(tr.vd, tr.vs) < 0.05)
print(f"training rows with both Vd, Vs < 50 mV: {near.sum()} of {len(tr)}; of those below 0 C: {(near & (tr.temp < 0)).sum()}")

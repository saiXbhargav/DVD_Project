"""exp_acc round 2, step 1: GPBag87 with plain-voltage features (FEATS_X) for the ON cases,
then a direct single-transistor SPICE check at the gates' ON operating point
(NMOS ON: Vd = Vs = 0; PMOS ON: Vd = Vs = VDD) on the 200 official corners' PVT values.
(The check only uses PVT values and fresh SPICE of single transistors, no gate results.)
"""
import time
from importlib import import_module

import joblib
import numpy as np
import pandas as pd

from common import OUT, C, G, GPBag
from common2 import TMX, pool87
import spice_utils as S

gen = import_module("01_gen_transistor_data")


def check(case):
    dev = C.CASES[case][0]
    cor = G.load_corners()
    rail = 0.0 if dev == "n" else cor.vdd
    df = pd.DataFrame({"vdd": cor.vdd, "temp": cor.temp, "w": G.WU, "vd": rail, "vs": rail,
                       "vg": cor.vdd if dev == "n" else 0.0, "vb": 0.0 if dev == "n" else cor.vdd})
    for p in C.PROC_PARAMS:
        df[p] = cor[f"{dev}_{p}"]
    I = S.run_op([gen.netlist(r, dev, case) for r in df.to_dict("records")], ["i(vd)", "i(vg)", "i(vs)", "i(vb)"])
    cold = (df.temp < 0).to_numpy()
    for name in ("GPBag87", "GPBag87X"):
        P = joblib.load(OUT / "models" / f"{name}_{case}.joblib").predict(df)
        for j, c in ((1, "ig"), (3, "ib")):
            e = 100 * np.abs(P[:, j] - I[:, j]) / np.abs(I[:, j])
            print(f"  {case} {name:9s} {c} at Vds=0: median {np.median(e):6.2f}%  cold median {np.median(e[cold]):6.2f}%  "
                  f"max {e.max():7.1f}%", flush=True)


def main():
    for case in ("nmos_on", "pmos_on"):
        f = OUT / "models" / f"GPBag87X_{case}.joblib"
        if not f.exists():
            t0 = time.time()
            pool = pool87(case)
            joblib.dump(TMX(GPBag(), case).fit(pool), f)
            print(f"{case} GPBag87X: {len(pool)} rows, {time.time() - t0:.0f} s", flush=True)
        check(case)


if __name__ == "__main__":
    main()

"""Overnight job B: dataset-size study for the final GP model type.

Model type = averaged GP (GPBag), plain-voltage features (FEATS_X) for the ON cases and the
official features for the OFF cases (the "GPX" setup). Training rows:
  2k / 4k / 7k  : first rows of the original (random) training split
  27k / 47k     : training split + 2 / 4 blocks of exp_extra data
  87k           : + the 4 exp_acc blocks (existing models reused)
Scored on the transistor validation split, the 100 validation corners (gates), the official
corners (gates) and the parity checker. Output: exp_acc/out/size_study.csv (SMOKE=1 -> tiny run).
"""
import glob
import os
import time

import joblib
import numpy as np
import pandas as pd

from common import DATA, HERE, OUT, C, G, M, GPBag, summary
from common2 import T, TMX
from common3 import ON, gates, parity_sum, vec_counts

SMOKE = os.environ.get("SMOKE") == "1"
SIZES = [400] if SMOKE else [2000, 4000, 7000, 27000, 47000, 87000]
MDIR = OUT / ("models_smoke" if SMOKE else "models")
MDIR.mkdir(exist_ok=True)


def pool(case, n):
    tr, _, _ = T.split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
    if n <= 7000:
        return tr.iloc[:n]
    blocks = sorted(glob.glob(str(HERE.parent / "exp_extra" / "data" / f"{case}_extra_*.csv"))) \
        + sorted(glob.glob(str(DATA / f"{case}_more_*.csv")))
    k = (n - 7000) // 10000
    return pd.concat([tr] + [pd.read_csv(f) for f in blocks[:k]], ignore_index=True)


def model(case, n):
    reuse = {(87000, True): "GPBag87X", (87000, False): "GPBag87", (47000, False): "GPBag47"}
    name = reuse.get((n, case in ON))
    if name and not SMOKE:
        return joblib.load(OUT / "models" / f"{name}_{case}.joblib")
    f = MDIR / f"size_{n}_{case}.joblib"
    if f.exists():
        return joblib.load(f)
    t0 = time.time()
    reg = GPBag(slice_n=300, n_sub=100) if SMOKE else GPBag()
    p = pool(case, n)
    m = (TMX(reg, case) if case in ON else M.TransistorModel(reg, case)).fit(p)
    joblib.dump(m, f)
    print(f"  trained {case} on {len(p)} rows in {time.time() - t0:.0f} s", flush=True)
    return m


def main():
    vcor = pd.read_csv(DATA / "val_corners.csv").set_index("corner", drop=False)
    vsp = pd.read_csv(OUT / "gate_spice_val.csv", dtype={"pattern": str})
    tcor = G.load_corners().set_index("corner", drop=False)
    tsp = pd.read_csv(C.RESULTS_DIR / "gate_spice.csv", dtype={"pattern": str})
    psp = pd.read_csv(C.RESULTS_DIR / "circuit_spice.csv", dtype={"vector": str})
    if SMOKE:
        vsp, tsp = vsp[vsp.corner < 3], tsp[tsp.corner < 3]
        psp = psp[psp.corner < 3]
    cnt = vec_counts()
    rows = []
    for n in SIZES:
        print(f"===== {n} rows", flush=True)
        models = {c: model(c, n) for c in C.CASES}
        row = dict(rows_nominal=n)
        for case in C.CASES:
            _, va, _ = T.split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
            P, Y = models[case].predict(va), va[C.CURRENTS].to_numpy()
            r = T.metrics(P, Y)
            e = 100 * np.abs(P - Y) / np.abs(Y)
            row[f"{case}_val_mean_median"] = r["mean_med%"]
            row[f"{case}_val_median_ge_1e11"] = np.median(e[np.abs(Y) >= 1e-11])
        gv, gt = gates(vsp, vcor, models), gates(tsp, tcor, models)
        nand = gt[gt.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_ml")
        av = summary(gv.assign(all=1), ["all"]).iloc[0]
        at = summary(gt.assign(all=1), ["all"]).iloc[0]
        ap = summary(parity_sum(nand, cnt, psp).assign(all=1), ["all"]).iloc[0]
        row.update(val_gate_acc=av.accuracy_pct, val_gate_median=av.median_err_pct, gate_acc=at.accuracy_pct,
                   gate_median=at.median_err_pct, gate_p95=at.p95_err_pct, parity_acc=ap.accuracy_pct,
                   parity_median=ap.median_err_pct)
        rows.append(row)
        print(f"SIZE {n:6d}: transistor val " + " ".join(f"{c} {row[f'{c}_val_mean_median']:.2f}%" for c in C.CASES)
              + f" | gates val {av.accuracy_pct:.2f}% test {at.accuracy_pct:.2f}% | parity {ap.accuracy_pct:.2f}%", flush=True)
        pd.DataFrame(rows).to_csv(OUT / ("size_study_smoke.csv" if SMOKE else "size_study.csv"),
                                  index=False, float_format="%.4g")
    print("SIZE STUDY DONE", flush=True)


if __name__ == "__main__":
    main()

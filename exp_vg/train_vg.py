"""Fix 1, step 2: train the gate-voltage-aware models and compare them with the current final
models (GPX) on validation data only.

Model = the final model type unchanged (averaged GP; plain-voltage features for ON, official
features for OFF). The features already contain the gate voltage (Vgs, Vgd), so only the
training data changes: the 87k-row pool (gate on its rail) + 40k rows with the gate off its rail.

Validation (as the project document prescribes, transistor level only):
  - the original validation split (1,500 rows, gate on its rail)
  - the new off-rail validation set (2,000 rows per case)
Writes out/models/VG_<case>.joblib and out/transistor_validation.csv. SMOKE=1 -> tiny run.
"""
import glob
import time

import joblib
import numpy as np
import pandas as pd

from common_vg import DATA, OUT, ON, SMOKE, C
from common import GPBag
from common2 import T, TMX, pool87
import ml_utils as M

SUF = "_smoke" if SMOKE else ""


def gpx(case):
    """The current final model of this case (read-only)."""
    d = OUT.parent.parent / "exp_acc" / "out" / "models"
    return joblib.load(d / f"{'GPBag87X' if case in ON else 'GPBag87'}_{case}.joblib")


def metrics(m, d):
    P, Y = m.predict(d), d[C.CURRENTS].to_numpy()
    r = T.metrics(P, Y)
    e = 100 * np.abs(P - Y) / np.abs(Y)
    big = np.abs(Y) >= 1e-11
    return dict(mean_median=r["mean_med%"], **{f"{c}_median": r[f"{c}_med%"] for c in C.CURRENTS},
                median_ge_1e11=np.median(e[big]), p95_ge_1e11=np.quantile(e[big], .95))


def main():
    rows = []
    for case in C.CASES:
        vg_files = sorted(glob.glob(str(DATA / f"{case}_vg_0*{SUF}.csv")))
        vg = pd.concat([pd.read_csv(f) for f in vg_files], ignore_index=True)
        pos = (M.to_frame(vg) > 0).mean(axis=0)
        print(f"{case}: off-rail rows {len(vg)}; share positive (hi, g, lo, b) = {np.round(pos, 3)}", flush=True)
        f = OUT / "models" / f"VG_{case}{SUF}.joblib"
        if f.exists():
            m = joblib.load(f)
        else:
            pool = pd.concat([pool87(case).sample(2000, random_state=0) if SMOKE else pool87(case), vg],
                             ignore_index=True)
            reg = GPBag(slice_n=800, n_sub=200) if SMOKE else GPBag()
            t0 = time.time()
            m = (TMX(reg, case) if case in ON else M.TransistorModel(reg, case)).fit(pool)
            joblib.dump(m, f)
            print(f"  trained VG_{case} on {len(pool)} rows in {time.time() - t0:.0f} s; KCL-derived current "
                  f"index {m.kcl_idx}", flush=True)
        _, va, _ = T.split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
        vva = pd.read_csv(DATA / f"{case}_vg_val{SUF}.csv")
        for name, model in (("GPX (current final)", gpx(case)), ("VG (gate-voltage-aware)", m)):
            for vname, d in (("original validation (gate on rail)", va), ("off-rail validation", vva)):
                r = metrics(model, d)
                r.update(case=case, model=name, validation=vname)
                rows.append(r)
                print(f"  {name:24s} {vname:36s} mean-median {r['mean_median']:.3f}%  "
                      f">=1e-11 A median {r['median_ge_1e11']:.3f}% p95 {r['p95_ge_1e11']:.2f}%", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"transistor_validation{SUF}.csv", index=False, float_format="%.4g")
    print("TRAIN_VG DONE", flush=True)


if __name__ == "__main__":
    main()

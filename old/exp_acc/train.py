"""exp_acc step 3: train the averaged-GP transistor models.

  GPBag47  pool = original training split (7,000) + exp_extra/data (40,000)
  GPBag87  pool = GPBag47 pool + exp_acc/data (40,000 more)
Same TransistorModel wrapper as the official models (features, hi/lo frame, KCL,
two experts for OFF). Validation / test rows are the original splits, untouched.
Fitted models are cached in exp_acc/out/models/ (rerun skips them).
"""
import glob
import time
from importlib import import_module

import joblib
import pandas as pd

from common import DATA, HERE, OUT, C, M, GPBag

T = import_module("03_train_models")


def pools(case):
    tr, _, _ = T.split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
    ex = [pd.read_csv(f) for f in sorted(glob.glob(str(HERE.parent / "exp_extra" / "data" / f"{case}_extra_*.csv")))]
    more = [pd.read_csv(f) for f in sorted(glob.glob(str(DATA / f"{case}_more_*.csv")))]
    assert len(ex) == 4 and len(more) == 4, (case, len(ex), len(more))
    p47 = pd.concat([tr] + ex, ignore_index=True)
    return {"GPBag47": p47, "GPBag87": pd.concat([p47] + more, ignore_index=True)}


def main():
    for case in C.CASES:
        for name, pool in pools(case).items():
            f = OUT / "models" / f"{name}_{case}.joblib"
            if f.exists():
                print(f"exists: {f.name}", flush=True)
                continue
            t0 = time.time()
            m = M.TransistorModel(GPBag(), case).fit(pool)
            joblib.dump(m, f)
            print(f"{case} {name}: {len(pool)} rows, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

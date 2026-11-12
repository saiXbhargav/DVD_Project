"""One-time tool: copy the final trained models out of the experiment folder into models_trained/.

The models were trained in exp_acc (now old/exp_acc) and pickled with that folder's classes
(common.GPBag, common2.TMX). final_method.py has byte-for-byte the same code for both classes;
this tool loads each pickle with the old code, points the objects at the final_method classes
(no attribute is touched, so every number stays identical), and saves it as
models_trained/<case>.joblib. It also saves, from the OLD objects:
  expected_results/transistor_metrics.csv   validation / test scores (as step 3 writes them)
  tools/_convert_check/<case>.npy           predictions on validation + test, for check_converted.py
Usage: python3 tools/convert_models.py <old models dir> <old exp_acc dir>
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import final_method as F  # noqa: E402  (loads config / circuits / ml_utils from ROOT first)

OLD_MODELS, OLD_CODE = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.append(str(OLD_CODE))
import common  # noqa: E402  (old classes, only to read the pickles)
import common2  # noqa: E402

CLASSES = {common.GPBag: F.GPBag, common2.TMX: F.TMX}
FILES = {c: f"{'GPBag87X' if c in F.ON else 'GPBag87'}_{c}.joblib" for c in F.C.CASES}
CHECK = ROOT / "tools" / "_convert_check"
CHECK.mkdir(exist_ok=True)
(ROOT / "expected_results").mkdir(exist_ok=True)


def retarget(o, seen):
    if id(o) in seen:
        return
    seen.add(id(o))
    if type(o) in CLASSES:
        o.__class__ = CLASSES[type(o)]
    if isinstance(o, (list, tuple)):
        for x in o:
            retarget(x, seen)
    elif isinstance(o, dict):
        for x in o.values():
            retarget(x, seen)
    elif type(o).__module__ in ("final_method", "ml_utils", "common", "common2"):
        for x in vars(o).values():
            retarget(x, seen)


def leftovers(o, seen, found):
    if id(o) in seen:
        return
    seen.add(id(o))
    if type(o).__module__ in ("common", "common2"):
        found.append(type(o).__qualname__)
    if isinstance(o, (list, tuple)):
        for x in o:
            leftovers(x, seen, found)
    elif isinstance(o, dict):
        for x in o.values():
            leftovers(x, seen, found)
    elif hasattr(o, "__dict__") and not type(o).__module__.startswith(("sklearn", "numpy")):
        for x in vars(o).values():
            leftovers(x, seen, found)


def main():
    rows = []
    for case, fname in FILES.items():
        m = joblib.load(OLD_MODELS / fname)
        print(f"{case}: loaded {fname} ({type(m).__module__}.{type(m).__name__})", flush=True)
        _, va, te = F.split(pd.read_csv(F.DATA / f"{case}.csv"))
        P = []
        for name, d in (("validation", va), ("test", te)):
            r, p = F.transistor_metrics(m, d)
            rows.append(dict(case=case, split=name, n=len(d), **r))
            P.append(p)
        np.save(CHECK / f"{case}.npy", np.vstack(P))
        retarget(m, set())
        left = []
        leftovers(m, set(), left)
        assert not left, f"old classes left in {case}: {left}"
        P2 = np.vstack([m.predict(va), m.predict(te)])
        assert np.array_equal(P2, np.vstack(P)), case
        f = F.MODELS / f"{case}.joblib"
        joblib.dump(m, f)
        print(f"  -> models_trained/{f.name} ({type(m).__module__}.{type(m).__name__}); "
              f"predictions identical in this process", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "expected_results" / "transistor_metrics.csv", index=False, float_format="%.6g")
    print("CONVERT DONE", flush=True)


if __name__ == "__main__":
    main()

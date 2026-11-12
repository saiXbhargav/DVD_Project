"""One-time check (fresh process, no old code on the path): the converted models in
models_trained/ load with final_method alone and predict exactly what the original pickles did."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import final_method as F  # noqa: E402

models = F.load_models()
assert "common" not in sys.modules and "common2" not in sys.modules
ok = True
for case, m in models.items():
    _, va, te = F.split(pd.read_csv(F.DATA / f"{case}.csv"))
    P = np.vstack([m.predict(va), m.predict(te)])
    ref = np.load(ROOT / "tools" / "_convert_check" / f"{case}.npy")
    same = np.array_equal(P, ref)
    ok &= same
    print(f"{case}: {type(m).__module__}.{type(m).__name__}, {len(P)} rows x 4 currents, "
          f"{'IDENTICAL' if same else 'DIFFERENT (max rel %.2e)' % np.nanmax(np.abs(P / ref - 1))}", flush=True)
print("CHECK", "PASSED" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)

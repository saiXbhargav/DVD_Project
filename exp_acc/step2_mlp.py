"""exp_acc round 2, step 2: MLP ensemble (3 seeds x linear+MLP per output) on the 87k pool,
FEATS_X features, same TransistorModel wrapper. Usage: python3 step2_mlp.py <case>
"""
import sys
import time

import joblib

from common import OUT
from common2 import MLPEns, TMX, pool87


def main(case):
    f = OUT / "models" / f"MLP87_{case}.joblib"
    if f.exists():
        print(f"exists: {f.name}", flush=True)
        return
    t0 = time.time()
    pool = pool87(case)
    m = TMX(MLPEns(3), case).fit(pool)
    joblib.dump(m, f)
    print(f"{case} MLP87: {len(pool)} rows, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

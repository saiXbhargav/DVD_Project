"""exp_acc step 1: 40,000 more training rows per case (our own SPICE data).

Same generator, distribution, model card and options as step 1 of the project
(01_gen_transistor_data.sample_case / netlist), uniform sampling; only the seeds
differ: SEED + 100 * case_index + block (7001-7304), disjoint from every other seed
(data 2026-2029, exp_extra 5001-5304, split 42, models 7, corners 1234, val corners 4321).
Output: exp_acc/data/<case>_more_<block>.csv (training-only rows).
"""
import time
from importlib import import_module

import numpy as np

from common import DATA, C
import spice_utils as S

gen = import_module("01_gen_transistor_data")
BLOCKS, ROWS, SEED = 4, 10_000, 7000


def main():
    for k, case in enumerate(C.CASES):
        dev = C.CASES[case][0]
        for b in range(1, BLOCKS + 1):
            f = DATA / f"{case}_more_{b:02d}.csv"
            if f.exists():
                print(f"exists: {f.name}", flush=True)
                continue
            df = gen.sample_case(case, ROWS, np.random.default_rng(SEED + 100 * k + b))
            t0 = time.time()
            I = S.run_op([gen.netlist(r, dev, case) for r in df.to_dict("records")],
                         ["i(vd)", "i(vg)", "i(vs)", "i(vb)"])
            for j, c in enumerate(C.CURRENTS):
                df[c] = I[:, j]
            df = df.dropna(subset=C.CURRENTS)
            tmp = f.with_suffix(".tmp")
            df[C.CSV_COLUMNS].to_csv(tmp, index=False, float_format="%.10e")
            tmp.replace(f)
            print(f"{case} block {b}: {len(df)} rows, max KCL residual "
                  f"{df[C.CURRENTS].sum(axis=1).abs().max():.1e} A, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

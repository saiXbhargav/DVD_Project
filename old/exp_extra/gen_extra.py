"""Experiment (exp_extra): generate our own extra training rows.

Same generator, distribution, model card and netlist options as step 1
(01_gen_transistor_data.sample_case / netlist); only the seeds differ:
block seed = SEED_EXTRA + 100 * case_index + block, disjoint from every other
seed in the project (data 2026-2029, split 42, models 7, corners 1234).
Output: exp_extra/data/<case>_extra_<block>.csv (training-only rows).
Usage: python3 gen_extra.py [blocks per case, default 4]
"""
import sys
import time
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import config as C  # noqa: E402
import spice_utils as S  # noqa: E402

gen = import_module("01_gen_transistor_data")
BLOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ROWS = 10_000
SEED_EXTRA = 5000


def main():
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    for k, case in enumerate(C.CASES):
        dev = C.CASES[case][0]
        for b in range(1, BLOCKS + 1):
            f = out / f"{case}_extra_{b:02d}.csv"
            if f.exists():
                print(f"exists: {f.name}", flush=True)
                continue
            rng = np.random.default_rng(SEED_EXTRA + 100 * k + b)
            df = gen.sample_case(case, ROWS, rng)
            t0 = time.time()
            I = S.run_op([gen.netlist(r, dev, case) for r in df.to_dict("records")],
                         ["i(vd)", "i(vg)", "i(vs)", "i(vb)"])
            for j, c in enumerate(C.CURRENTS):
                df[c] = I[:, j]
            df = df.dropna(subset=C.CURRENTS)
            tmp = f.with_suffix(".tmp")
            df[C.CSV_COLUMNS].to_csv(tmp, index=False, float_format="%.10e")
            tmp.replace(f)
            kcl = df[C.CURRENTS].sum(axis=1).abs().max()
            print(f"{case} block {b}: {len(df)} rows, max KCL residual {kcl:.1e} A, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

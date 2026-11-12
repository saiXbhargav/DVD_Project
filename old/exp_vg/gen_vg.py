"""Fix 1, step 1: training and validation rows with the gate slightly off its rail.

Same generator, distributions, model card and SPICE options as step 1 of the project
(01_gen_transistor_data.sample_case / netlist); the only change is the gate voltage:
  cases with Vg = VDD (NMOS ON, PMOS OFF): Vg = VDD - delta
  cases with Vg = 0   (NMOS OFF, PMOS ON): Vg = delta
  delta log-uniform in [0.1 mV, 150 mV]
Training: 4 blocks x 10,000 rows per case (seeds 8001-8304) -> data/<case>_vg_<b>.csv
Validation: 2,000 rows per case (seeds 8901-8904)          -> data/<case>_vg_val.csv
SMOKE=1 -> 300 rows, files with a _smoke suffix.
"""
import time
from importlib import import_module

import numpy as np

from common_vg import DATA, DELTA_MAX, DELTA_MIN, SMOKE, C
import spice_utils as S

gen = import_module("01_gen_transistor_data")


def sample(case, n, rng):
    df = gen.sample_case(case, n, rng)
    delta = np.exp(rng.uniform(np.log(DELTA_MIN), np.log(DELTA_MAX), n))
    df["vg"] = df["vdd"] - delta if C.CASES[case][1] == "vdd" else delta
    return df


def make(case, k, f, n, seed):
    if f.exists():
        print(f"exists: {f.name}", flush=True)
        return
    dev = C.CASES[case][0]
    df = sample(case, n, np.random.default_rng(seed))
    t0 = time.time()
    I = S.run_op([gen.netlist(r, dev, case) for r in df.to_dict("records")], ["i(vd)", "i(vg)", "i(vs)", "i(vb)"])
    for j, c in enumerate(C.CURRENTS):
        df[c] = I[:, j]
    df = df.dropna(subset=C.CURRENTS)
    tmp = f.with_suffix(".tmp")
    df[C.CSV_COLUMNS].to_csv(tmp, index=False, float_format="%.10e")
    tmp.replace(f)
    print(f"{f.name}: {len(df)} rows, max KCL residual {df[C.CURRENTS].sum(axis=1).abs().max():.1e} A, "
          f"{time.time() - t0:.0f} s", flush=True)


def main():
    suf = "_smoke" if SMOKE else ""
    n_tr, n_va, blocks = (300, 100, 1) if SMOKE else (10_000, 2_000, 4)
    for k, case in enumerate(C.CASES):
        for b in range(1, blocks + 1):
            make(case, k, DATA / f"{case}_vg_{b:02d}{suf}.csv", n_tr, 8000 + 100 * k + b)
        make(case, k, DATA / f"{case}_vg_val{suf}.csv", n_va, 8900 + k + 1)


if __name__ == "__main__":
    main()

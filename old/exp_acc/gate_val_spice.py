"""exp_acc step 2: gate VALIDATION set - 100 new corners (seed 4321), SPICE for all 18
gate cases, same netlists / leakage definition as 05_gate_spice.py.

Used only to choose between experiment options (model variant, cut rule). The
official 200 corners stay untouched as the test set for gates and the parity checker.
Output: exp_acc/data/val_corners.csv, exp_acc/out/gate_spice_val.csv
"""
import time

import numpy as np
import pandas as pd

from common import DATA, OUT, C, G
import spice_utils as S

SEED_VAL_CORNERS, N_VAL = 4321, 100


def corners():
    rng = np.random.default_rng(SEED_VAL_CORNERS)
    df = pd.DataFrame({
        "corner": np.arange(N_VAL),
        "vdd": rng.uniform((1 - C.VDD_SPREAD) * C.VDD_NOM, (1 + C.VDD_SPREAD) * C.VDD_NOM, N_VAL),
        "temp": rng.uniform(*C.T_RANGE, N_VAL),
    })
    for dev in ("n", "p"):
        for p in C.PROC_PARAMS:
            df[f"{dev}_{p}"] = S.NOMINAL[dev][p] * rng.uniform(1 - C.PROC_SPREAD, 1 + C.PROC_SPREAD, N_VAL)
    return df


def main():
    if (OUT / "gate_spice_val.csv").exists():
        print("exists: gate_spice_val.csv", flush=True)
        return
    cor = corners()
    cor.to_csv(DATA / "val_corners.csv", index=False, float_format="%.10e")
    rows, t0 = [], time.time()
    for gate, spec in G.GATES.items():
        pats = G.patterns(gate)
        probes = ["i(vdd)", "i(vss)"] + [f"i(v{i})" for i in spec["inputs"]] + [f"v({n})" for n in spec["nodes"]]
        jobs = [(G.gate_netlist(gate, r),
                 [[f"v{i} = {S.fmt(r['vdd'] if bit == '1' else 0.0)}" for i, bit in zip(spec["inputs"], p)]
                  for p in pats]) for r in cor.to_dict("records")]
        res = pd.DataFrame(S.run_sweeps(jobs, probes), columns=probes)
        res["corner"] = np.repeat(cor["corner"].to_numpy(), len(pats))
        res["pattern"] = np.tile(pats, len(cor))
        res["gate"] = gate
        for p in pats:
            m = res["pattern"] == p
            res.loc[m, "leak_spice"] = G.spice_leakage(res[m], spec["inputs"], p)
        rows.append(res.rename(columns={f"v({n})": f"v_{n}" for n in spec["nodes"]}))
        print(f"{gate}: {len(res)} sims, failed {int(res.leak_spice.isna().sum())} ({time.time() - t0:.0f} s)", flush=True)
    out = pd.concat(rows, ignore_index=True).reindex(
        columns=["gate", "pattern", "corner", "leak_spice", "v_out", "v_x", "v_y", "v_nd", "v_nr"])
    out.to_csv(OUT / "gate_spice_val.csv", index=False, float_format="%.10e")
    print("saved exp_acc/out/gate_spice_val.csv", flush=True)


if __name__ == "__main__":
    main()

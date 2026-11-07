"""Part C step 6: SPICE reference for the 8-bit parity checker (28 NAND2, 112 FETs).

One complete transistor-level netlist per corner (same 200 corners as Part B);
all 256 input vectors are applied with `alter` on the 8 input sources.
Leakage = -( I(Vdd) + sum of I(Vdi) over inputs at 1 ), as for the gates.
Also checks the logic output against the expected parity.

Runs in batches of corners and saves after each batch (results/circuit_spice_raw.csv),
so an interrupted run resumes where it stopped.
"""
import time
from itertools import product

import numpy as np
import pandas as pd

import circuits as G
import config as C
import spice_utils as S

BATCH = 20                       # corners per checkpoint
RAW = C.RESULTS_DIR / "circuit_spice_raw.csv"


def main():
    corners = G.load_corners()
    vectors = ["".join(v) for v in product("01", repeat=8)]       # d0..d7
    probes = ["i(vdd)", "i(vss)"] + [f"i(v{d})" for d in G.PARITY_INPUTS] + ["v(p)"]
    (C.NETLIST_DIR / "parity8.cir").write_text(G.parity_netlist(corners.iloc[0].to_dict()))

    done = set()
    if RAW.exists():
        done = set(pd.read_csv(RAW, usecols=["corner"])["corner"].unique())
        print(f"resuming: {len(done)} corners already simulated", flush=True)
    todo = corners[~corners["corner"].isin(done)]

    t0 = time.time()
    for b in range(0, len(todo), BATCH):
        part = todo.iloc[b:b + BATCH]
        jobs = []
        for r in part.to_dict("records"):
            alter_sets = [[f"v{d} = {S.fmt(r['vdd'] if bit == '1' else 0.0)}" for d, bit in zip(G.PARITY_INPUTS, v)]
                          for v in vectors]
            jobs.append((G.parity_netlist(r), alter_sets))
        R = S.run_sweeps(jobs, probes, chunk=1)
        res = pd.DataFrame(R, columns=probes)
        res.insert(0, "vector", np.tile(vectors, len(part)))
        res.insert(0, "corner", np.repeat(part["corner"].to_numpy(), len(vectors)))
        res.to_csv(RAW, mode="a", header=not RAW.exists(), index=False, float_format="%.12e")
        n_done = len(done) + b + len(part)
        print(f"  {n_done}/{len(corners)} corners  ({time.time() - t0:.0f} s)", flush=True)

    res = pd.read_csv(RAW, dtype={"vector": str}).drop_duplicates(["corner", "vector"]).sort_values(["corner", "vector"])
    leak = -res["i(vdd)"].to_numpy().copy()
    gnd = res["i(vss)"].to_numpy().copy()
    for i, d in enumerate(G.PARITY_INPUTS):
        one = res["vector"].str[i].to_numpy() == "1"
        leak[one] -= res.loc[one, f"i(v{d})"].to_numpy()
        gnd[~one] += res.loc[~one, f"i(v{d})"].to_numpy()
    res["leak_spice"], res["leak_spice_gnd"] = leak, gnd
    res["v_p"] = res["v(p)"]
    expect = res["vector"].map(lambda v: v.count("1") % 2 == 1)
    vdd = res["corner"].map(corners.set_index("corner")["vdd"])
    logic_ok = (res["v_p"] > vdd / 2) == expect
    print(f"{len(res)} simulations; failed: {int(res.leak_spice.isna().sum())}, logic errors: {int((~logic_ok).sum())}, "
          f"max KCL mismatch: {np.nanmax(np.abs(res.leak_spice - res.leak_spice_gnd) / res.leak_spice):.1e}", flush=True)
    res[["corner", "vector", "leak_spice", "leak_spice_gnd", "v_p"]].to_csv(
        C.RESULTS_DIR / "circuit_spice.csv", index=False, float_format="%.10e")
    print("saved results/circuit_spice.csv; leakage median %.3e A, range %.3e .. %.3e A"
          % (res.leak_spice.median(), res.leak_spice.min(), res.leak_spice.max()), flush=True)


if __name__ == "__main__":
    main()

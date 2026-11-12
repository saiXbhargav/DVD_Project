"""Step 7 (Part C, evaluation only): SPICE reference for the 8-bit parity checker (28 NAND2).

One complete transistor-level netlist per corner; all 256 input vectors applied with `alter`.
Leakage = -(I(Vdd) + sum of I(Vdi) over inputs at 1), as for the gates; the output is checked
against the expected parity.
  results/circuit_spice_w.csv  main setup: per-corner widths   (~1.5 h)
  results/circuit_spice.csv    reference: fixed sizing         (~1.5 h)
Saved after every 20 corners (results/circuit_spice*_raw.csv), so an interrupted run resumes.
Existing final files are kept (skip-if-exists).
"""
import time
from itertools import product

import numpy as np
import pandas as pd

from final_method import RES, SMOKE, C, G, S, corners, parity_netlist_w


def parity_spice(cor, netlist_fn, f, raw, example):
    if f.exists():
        print(f"exists: {f.relative_to(C.ROOT)}", flush=True)
        return
    vectors = ["".join(v) for v in product("01", repeat=8)]
    probes = ["i(vdd)", "i(vss)"] + [f"i(v{d})" for d in G.PARITY_INPUTS] + ["v(p)"]
    if not SMOKE:
        (C.NETLIST_DIR / example).write_text(netlist_fn(cor.iloc[0].to_dict()))
    done = set(pd.read_csv(raw, usecols=["corner"])["corner"].unique()) if raw.exists() else set()
    todo = cor[~cor["corner"].isin(done)]
    t0 = time.time()
    for b in range(0, len(todo), 20):
        part = todo.iloc[b:b + 20]
        jobs = [(netlist_fn(r), [[f"v{d} = {S.fmt(r['vdd'] if bit == '1' else 0.0)}"
                                  for d, bit in zip(G.PARITY_INPUTS, v)] for v in vectors])
                for r in part.to_dict("records")]
        res = pd.DataFrame(S.run_sweeps(jobs, probes, chunk=1), columns=probes)
        res.insert(0, "vector", np.tile(vectors, len(part)))
        res.insert(0, "corner", np.repeat(part["corner"].to_numpy(), len(vectors)))
        res.to_csv(raw, mode="a", header=not raw.exists(), index=False, float_format="%.12e")
        print(f"  {len(done) + b + len(part)}/{len(cor)} corners ({time.time() - t0:.0f} s)", flush=True)
    res = pd.read_csv(raw, dtype={"vector": str}).drop_duplicates(["corner", "vector"]).sort_values(["corner", "vector"])
    leak = -res["i(vdd)"].to_numpy().copy()
    gnd = res["i(vss)"].to_numpy().copy()
    for k, d in enumerate(G.PARITY_INPUTS):
        one = res["vector"].str[k].to_numpy() == "1"
        leak[one] -= res.loc[one, f"i(v{d})"].to_numpy()
        gnd[~one] += res.loc[~one, f"i(v{d})"].to_numpy()
    res["leak_spice"], res["leak_spice_gnd"], res["v_p"] = leak, gnd, res["v(p)"]
    vdd = res["corner"].map(cor.set_index("corner")["vdd"])
    ok = (res["v_p"] > vdd / 2) == res["vector"].map(lambda v: v.count("1") % 2 == 1)
    print(f"  {len(res)} sims, failed {int(res.leak_spice.isna().sum())}, logic errors {int((~ok).sum())}, "
          f"max KCL mismatch {np.nanmax(np.abs(res.leak_spice - res.leak_spice_gnd) / res.leak_spice):.1e}", flush=True)
    res[["corner", "vector", "leak_spice", "leak_spice_gnd", "v_p"]].to_csv(f, index=False, float_format="%.10e")
    print(f"saved {f.relative_to(C.ROOT)}", flush=True)


def main():
    print("varying widths (main setup)", flush=True)
    parity_spice(corners("varying"), parity_netlist_w, RES / "circuit_spice_w.csv",
                 RES / "circuit_spice_w_raw.csv", "parity8_w.cir")
    print("fixed sizes (reference)", flush=True)
    parity_spice(corners("fixed"), G.parity_netlist, RES / "circuit_spice.csv",
                 RES / "circuit_spice_raw.csv", "parity8.cir")
    print("STEP 7 DONE", flush=True)


if __name__ == "__main__":
    main()

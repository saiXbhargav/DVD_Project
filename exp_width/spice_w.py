"""Varying widths, step 1: SPICE references for the gates and the parity checker.

Same netlists, options, input sources and leakage definition as 05_gate_spice.py and
07_circuit_spice.py; only the transistor widths come from the corner (wn / wp).
  gates : 18 cases x 200 corners  -> out/gate_spice_w.csv   (~1 min)
  parity: 256 vectors x 200 corners -> out/circuit_spice_w.csv (~1.5 h, resumable via
          out/circuit_spice_w_raw.csv)
SMOKE=1 -> first 3 corners, files with a _smoke suffix.
"""
import time
from itertools import product

import numpy as np
import pandas as pd

from common_w import OUT, SMOKE, G, S, corners_w, gate_netlist_w, parity_netlist_w

SUF = "_smoke" if SMOKE else ""


def gate_spice(cor):
    f = OUT / f"gate_spice_w{SUF}.csv"
    if f.exists():
        print(f"exists: {f.name}", flush=True)
        return
    rows, t0 = [], time.time()
    for gate, spec in G.GATES.items():
        pats = G.patterns(gate)
        probes = ["i(vdd)", "i(vss)"] + [f"i(v{i})" for i in spec["inputs"]] + [f"v({n})" for n in spec["nodes"]]
        jobs = [(gate_netlist_w(gate, r),
                 [[f"v{i} = {S.fmt(r['vdd'] if bit == '1' else 0.0)}" for i, bit in zip(spec["inputs"], p)]
                  for p in pats]) for r in cor.to_dict("records")]
        if not SMOKE:
            (OUT / f"gate_{gate}_w_example.cir").write_text(jobs[0][0])
        res = pd.DataFrame(S.run_sweeps(jobs, probes), columns=probes)
        res["corner"] = np.repeat(cor["corner"].to_numpy(), len(pats))
        res["pattern"] = np.tile(pats, len(cor))
        res["gate"] = gate
        for p in pats:
            m = res["pattern"] == p
            res.loc[m, "leak_spice"] = G.spice_leakage(res[m], spec["inputs"], p)
        gnd = res["i(vss)"].copy()                      # KCL check: current into ground-level sources
        for k, name in enumerate(spec["inputs"]):
            at0 = res["pattern"].str[k] == "0"
            gnd[at0] += res.loc[at0, f"i(v{name})"]
        res["leak_spice_gnd"] = gnd
        rows.append(res.rename(columns={f"v({n})": f"v_{n}" for n in spec["nodes"]}))
        print(f"{gate}: {len(res)} sims, failed {int(res.leak_spice.isna().sum())}, max KCL mismatch "
              f"{np.nanmax(np.abs(res.leak_spice - res.leak_spice_gnd) / res.leak_spice):.1e} "
              f"({time.time() - t0:.0f} s)", flush=True)
    out = pd.concat(rows, ignore_index=True).reindex(
        columns=["gate", "pattern", "corner", "leak_spice", "leak_spice_gnd", "v_out", "v_x", "v_y", "v_nd", "v_nr"])
    vdd = out.corner.map(cor.set_index("corner").vdd)
    logic = {"INV": lambda a, b: 1 - a, "NAND2": lambda a, b: 1 - (a & b), "NOR2": lambda a, b: 1 - (a | b),
             "AND2": lambda a, b: a & b, "OR2": lambda a, b: a | b}
    exp = [logic[g](int(p[0]), int(p[-1])) for g, p in zip(out.gate, out.pattern)]
    bad = int((np.abs(out.v_out - np.array(exp) * vdd) > 0.05 * vdd).sum())
    print(f"gate outputs off their logic level (> 5 % of VDD): {bad}", flush=True)
    out.to_csv(f, index=False, float_format="%.10e")
    print(f"saved {f.name}", flush=True)


def parity_spice(cor):
    f, raw = OUT / f"circuit_spice_w{SUF}.csv", OUT / f"circuit_spice_w_raw{SUF}.csv"
    if f.exists():
        print(f"exists: {f.name}", flush=True)
        return
    vectors = ["".join(v) for v in product("01", repeat=8)]
    probes = ["i(vdd)", "i(vss)"] + [f"i(v{d})" for d in G.PARITY_INPUTS] + ["v(p)"]
    if not SMOKE:
        (OUT / "parity8_w_example.cir").write_text(parity_netlist_w(cor.iloc[0].to_dict()))
    done = set(pd.read_csv(raw, usecols=["corner"])["corner"].unique()) if raw.exists() else set()
    todo = cor[~cor["corner"].isin(done)]
    t0 = time.time()
    for b in range(0, len(todo), 20):
        part = todo.iloc[b:b + 20]
        jobs = [(parity_netlist_w(r), [[f"v{d} = {S.fmt(r['vdd'] if bit == '1' else 0.0)}"
                                        for d, bit in zip(G.PARITY_INPUTS, v)] for v in vectors])
                for r in part.to_dict("records")]
        res = pd.DataFrame(S.run_sweeps(jobs, probes, chunk=1), columns=probes)
        res.insert(0, "vector", np.tile(vectors, len(part)))
        res.insert(0, "corner", np.repeat(part["corner"].to_numpy(), len(vectors)))
        res.to_csv(raw, mode="a", header=not raw.exists(), index=False, float_format="%.12e")
        print(f"  parity SPICE: {len(done) + b + len(part)}/{len(cor)} corners ({time.time() - t0:.0f} s)", flush=True)
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
    print(f"parity SPICE: {len(res)} sims, failed {int(res.leak_spice.isna().sum())}, logic errors {int((~ok).sum())}, "
          f"max KCL mismatch {np.nanmax(np.abs(res.leak_spice - res.leak_spice_gnd) / res.leak_spice):.1e}", flush=True)
    res[["corner", "vector", "leak_spice", "leak_spice_gnd", "v_p"]].to_csv(f, index=False, float_format="%.10e")
    print(f"saved {f.name}", flush=True)


def main():
    cor = corners_w()
    print(f"{len(cor)} corners; Wn {cor.wn.min()*1e9:.1f}-{cor.wn.max()*1e9:.1f} nm, "
          f"Wp {cor.wp.min()*1e9:.1f}-{cor.wp.max()*1e9:.1f} nm", flush=True)
    gate_spice(cor)
    parity_spice(cor)


if __name__ == "__main__":
    main()

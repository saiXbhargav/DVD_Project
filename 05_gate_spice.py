"""Step 5 (Part B, evaluation only): SPICE reference for INV, NAND2, NOR2, AND2, OR2.

Every gate is simulated as a full transistor-level netlist at all 200 corners and for every input
pattern (18 cases x 200 corners); inputs are ideal sources at exactly 0 or VDD.
  results/gate_spice_w.csv  main setup: per-corner widths wn / wp      (data/corners_w.csv)
  results/gate_spice.csv    reference: fixed sizing of circuits.py   (data/corners.csv)
Columns: leakage (current drawn from VDD and the inputs at 1), the same current returned through
ground (KCL check), output and internal node voltages. Never used for training.
Existing files are kept (skip-if-exists).
"""
import time

import numpy as np
import pandas as pd

from final_method import RES, SMOKE, C, G, S, corners, gate_netlist_w

LOGIC = {"INV": lambda a, b: 1 - a, "NAND2": lambda a, b: 1 - (a & b), "NOR2": lambda a, b: 1 - (a | b),
         "AND2": lambda a, b: a & b, "OR2": lambda a, b: a | b}


def gate_spice(cor, netlist_fn, f, example):
    if f.exists():
        print(f"exists: {f.relative_to(C.ROOT)}", flush=True)
        return
    rows, t0 = [], time.time()
    for gate, spec in G.GATES.items():
        pats = G.patterns(gate)
        probes = ["i(vdd)", "i(vss)"] + [f"i(v{i})" for i in spec["inputs"]] + [f"v({n})" for n in spec["nodes"]]
        jobs = [(netlist_fn(gate, r),
                 [[f"v{i} = {S.fmt(r['vdd'] if bit == '1' else 0.0)}" for i, bit in zip(spec["inputs"], p)]
                  for p in pats]) for r in cor.to_dict("records")]
        if not SMOKE:
            (C.NETLIST_DIR / example.format(gate=gate)).write_text(jobs[0][0])
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
        print(f"  {gate:5s}: {len(res)} sims, failed {int(res.leak_spice.isna().sum())}, max KCL mismatch "
              f"{np.nanmax(np.abs(res.leak_spice - res.leak_spice_gnd) / res.leak_spice):.1e} "
              f"({time.time() - t0:.0f} s)", flush=True)
    out = pd.concat(rows, ignore_index=True).reindex(
        columns=["gate", "pattern", "corner", "leak_spice", "leak_spice_gnd", "v_out", "v_x", "v_y", "v_nd", "v_nr"])
    vdd = out.corner.map(cor.set_index("corner").vdd)
    exp = [LOGIC[g](int(p[0]), int(p[-1])) for g, p in zip(out.gate, out.pattern)]
    bad = int((np.abs(out.v_out - np.array(exp) * vdd) > 0.05 * vdd).sum())
    print(f"  gate outputs off their logic level by > 5 % of VDD: {bad}", flush=True)
    out.to_csv(f, index=False, float_format="%.10e")
    print(f"saved {f.relative_to(C.ROOT)}", flush=True)


def main():
    print("varying widths (main setup)", flush=True)
    gate_spice(corners("varying"), gate_netlist_w, RES / "gate_spice_w.csv", "gate_{gate}_w.cir")
    print("fixed sizes (reference)", flush=True)
    gate_spice(corners("fixed"), G.gate_netlist, RES / "gate_spice.csv", "gate_{gate}.cir")
    print("STEP 5 DONE", flush=True)


if __name__ == "__main__":
    main()

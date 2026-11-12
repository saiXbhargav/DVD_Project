"""Part B step 3: SPICE reference for INV, NAND2, NOR2, AND2, OR2.

Every gate is simulated as a full transistor-level netlist at all 200 corners and
for every input pattern (18 cases x 200 corners). Inputs are ideal sources at
exactly 0 or VDD. Records the leakage, the output and internal node voltages and,
as a KCL sanity check, the current returned through the ground rail (Vss).
These results are for evaluation only (never used for training).
"""
import time

import numpy as np
import pandas as pd

import circuits as G
import config as C
import spice_utils as S


def main():
    corners = G.load_corners()
    rows = []
    t0 = time.time()
    for gate, spec in G.GATES.items():
        pats = G.patterns(gate)
        probes = ["i(vdd)", "i(vss)"] + [f"i(v{i})" for i in spec["inputs"]] + [f"v({n})" for n in spec["nodes"]]
        jobs = []
        for r in corners.to_dict("records"):
            alter_sets = [[f"v{i} = {S.fmt(r['vdd'] if bit == '1' else 0.0)}" for i, bit in zip(spec["inputs"], p)]
                          for p in pats]
            jobs.append((G.gate_netlist(gate, r), alter_sets))
        (C.NETLIST_DIR / f"gate_{gate}.cir").write_text(jobs[0][0])
        R = S.run_sweeps(jobs, probes)
        res = pd.DataFrame(R, columns=probes)
        res["corner"] = np.repeat(corners["corner"].to_numpy(), len(pats))
        res["pattern"] = np.tile(pats, len(corners))
        res["gate"] = gate
        for p in pats:
            m = res["pattern"] == p
            res.loc[m, "leak_spice"] = G.spice_leakage(res[m], spec["inputs"], p)
        # current into ground-level sources (vss + inputs at 0): equals leak by KCL
        gnd = res["i(vss)"].copy()
        for i_, name in enumerate(spec["inputs"]):
            at0 = res["pattern"].str[i_] == "0"
            gnd[at0] += res.loc[at0, f"i(v{name})"]
        res["leak_spice_gnd"] = gnd
        res = res.rename(columns={f"v({n})": f"v_{n}" for n in spec["nodes"]})
        rows.append(res)
        print(f"{gate:5s}: {len(res)} sims, failed {int(res['leak_spice'].isna().sum())}, "
              f"max KCL mismatch {np.nanmax(np.abs(res.leak_spice - res.leak_spice_gnd) / res.leak_spice):.1e}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(rows, ignore_index=True)
    keep = ["gate", "pattern", "corner", "leak_spice", "leak_spice_gnd", "v_out", "v_x", "v_y", "v_nd", "v_nr"]
    out = out.reindex(columns=keep)
    out.to_csv(C.RESULTS_DIR / "gate_spice.csv", index=False, float_format="%.10e")
    print("saved results/gate_spice.csv")
    print(out.groupby(["gate", "pattern"])["leak_spice"].median().to_string())


if __name__ == "__main__":
    main()

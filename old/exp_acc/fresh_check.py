"""Overnight job A: fresh independent check on data never used anywhere before.

  transistors : 2,000 new samples per case (seeds 9000-9003), same generator as step 1
  corners     : 100 new PVT corners (seed 9999)
  gate SPICE  : 18 cases x 100 corners;  parity SPICE: 256 vectors x 100 corners
Every available model set (common3.build_sets) is scored once on this data; nothing is
tuned or chosen here. Also measures the composition error (sum of SPICE NAND2 vs SPICE
circuit) on the new corners. Output: exp_acc/fresh/ (resumable; SMOKE=1 -> tiny run in fresh_smoke/).
"""
import os
import time
from importlib import import_module
from itertools import product

import numpy as np
import pandas as pd

from common import HERE, C, G, summary
from common2 import T
from common3 import PATS, build_sets, gates, parity_sum, vec_counts
import gate_val_spice as GV
import spice_utils as S

gen = import_module("01_gen_transistor_data")
SMOKE = os.environ.get("SMOKE") == "1"
N_T, N_C = (30, 2) if SMOKE else (2000, 100)
SEED_T, SEED_C = 9000, 9999
FRESH = HERE / ("fresh_smoke" if SMOKE else "fresh")
FRESH.mkdir(exist_ok=True)


def transistor_data():
    for k, case in enumerate(C.CASES):
        f = FRESH / f"{case}.csv"
        if f.exists():
            continue
        dev = C.CASES[case][0]
        df = gen.sample_case(case, N_T, np.random.default_rng(SEED_T + k))
        I = S.run_op([gen.netlist(r, dev, case) for r in df.to_dict("records")], ["i(vd)", "i(vg)", "i(vs)", "i(vb)"])
        for j, c in enumerate(C.CURRENTS):
            df[c] = I[:, j]
        df = df.dropna(subset=C.CURRENTS)
        df[C.CSV_COLUMNS].to_csv(f, index=False, float_format="%.10e")
        print(f"fresh {case}: {len(df)} rows", flush=True)


def corners():
    f = FRESH / "corners.csv"
    if not f.exists():
        GV.SEED_VAL_CORNERS, GV.N_VAL = SEED_C, N_C
        GV.corners().to_csv(f, index=False, float_format="%.10e")
    return pd.read_csv(f)


def gate_spice(cor):
    f = FRESH / "gate_spice.csv"
    if f.exists():
        return
    rows = []
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
    out = pd.concat(rows, ignore_index=True).reindex(
        columns=["gate", "pattern", "corner", "leak_spice", "v_out", "v_x", "v_y", "v_nd", "v_nr"])
    out.to_csv(f, index=False, float_format="%.10e")
    print(f"fresh gate SPICE: {len(out)} sims, failed {int(out.leak_spice.isna().sum())}", flush=True)


def parity_spice(cor):
    f, raw = FRESH / "circuit_spice.csv", FRESH / "circuit_spice_raw.csv"
    if f.exists():
        return
    vectors = ["".join(v) for v in product("01", repeat=8)]
    probes = ["i(vdd)", "i(vss)"] + [f"i(v{d})" for d in G.PARITY_INPUTS] + ["v(p)"]
    done = set(pd.read_csv(raw, usecols=["corner"])["corner"].unique()) if raw.exists() else set()
    todo = cor[~cor["corner"].isin(done)]
    t0 = time.time()
    for b in range(0, len(todo), 20):
        part = todo.iloc[b:b + 20]
        jobs = [(G.parity_netlist(r), [[f"v{d} = {S.fmt(r['vdd'] if bit == '1' else 0.0)}"
                                        for d, bit in zip(G.PARITY_INPUTS, v)] for v in vectors])
                for r in part.to_dict("records")]
        res = pd.DataFrame(S.run_sweeps(jobs, probes, chunk=1), columns=probes)
        res.insert(0, "vector", np.tile(vectors, len(part)))
        res.insert(0, "corner", np.repeat(part["corner"].to_numpy(), len(vectors)))
        res.to_csv(raw, mode="a", header=not raw.exists(), index=False, float_format="%.12e")
        print(f"  fresh parity SPICE: {len(done) + b + len(part)}/{len(cor)} corners ({time.time() - t0:.0f} s)", flush=True)
    res = pd.read_csv(raw, dtype={"vector": str}).drop_duplicates(["corner", "vector"])
    leak = -res["i(vdd)"].to_numpy().copy()
    for i, d in enumerate(G.PARITY_INPUTS):
        one = res["vector"].str[i].to_numpy() == "1"
        leak[one] -= res.loc[one, f"i(v{d})"].to_numpy()
    res["leak_spice"] = leak
    vdd = res["corner"].map(cor.set_index("corner")["vdd"])
    ok = (res["v(p)"] > vdd / 2) == res["vector"].map(lambda v: v.count("1") % 2 == 1)
    print(f"fresh parity SPICE: {len(res)} sims, failed {int(res.leak_spice.isna().sum())}, logic errors {int((~ok).sum())}",
          flush=True)
    res[["corner", "vector", "leak_spice"]].to_csv(f, index=False, float_format="%.10e")


def evaluate(cor):
    cor = cor.set_index("corner", drop=False)
    gsp = pd.read_csv(FRESH / "gate_spice.csv", dtype={"pattern": str})
    psp = pd.read_csv(FRESH / "circuit_spice.csv", dtype={"vector": str})
    cnt = vec_counts()
    sp_nand = gsp[gsp.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_spice")
    comp = summary(parity_sum(sp_nand, cnt, psp).assign(all=1), ["all"]).iloc[0]
    print(f"composition (sum of SPICE NAND2 vs SPICE circuit): accuracy {comp.accuracy_pct:.2f}%, "
          f"median {comp.median_err_pct:.2f}%, bias {comp.bias_pct:+.2f}%", flush=True)
    rows, trows = [], []
    for name, models in build_sets().items():
        for case in C.CASES:
            d = pd.read_csv(FRESH / f"{case}.csv")
            P, Y = models[case].predict(d), d[C.CURRENTS].to_numpy()
            r = T.metrics(P, Y)
            e = 100 * np.abs(P - Y) / np.abs(Y)
            big = np.abs(Y) >= 1e-11
            trows.append(dict(set=name, case=case, mean_median_pct=r["mean_med%"],
                              **{f"{c}_median_pct": r[f"{c}_med%"] for c in C.CURRENTS},
                              median_ge_1e11_pct=np.median(e[big]), p95_ge_1e11_pct=np.quantile(e[big], .95)))
        g = gates(gsp, cor, models)
        tag = name.replace("|", "_").replace(":", "-").replace("+", "_")
        summary(g, ["gate", "pattern"]).to_csv(FRESH / f"gate_results_{tag}.csv", index=False, float_format="%.4g")
        nand = g[g.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_ml")
        ag = summary(g.assign(all=1), ["all"]).iloc[0]
        ap = summary(parity_sum(nand, cnt, psp).assign(all=1), ["all"]).iloc[0]
        rows.append(dict(set=name, gate_acc=ag.accuracy_pct, gate_median=ag.median_err_pct, gate_p95=ag.p95_err_pct,
                         gate_bias=ag.bias_pct, parity_acc=ap.accuracy_pct, parity_median=ap.median_err_pct,
                         parity_p95=ap.p95_err_pct, parity_bias=ap.bias_pct,
                         composition_acc=comp.accuracy_pct, composition_bias=comp.bias_pct))
        print(f"FRESH {name:13s} gates {ag.accuracy_pct:.2f}% (med {ag.median_err_pct:.2f}%, p95 {ag.p95_err_pct:.2f}%) | "
              f"parity {ap.accuracy_pct:.2f}% (med {ap.median_err_pct:.2f}%, bias {ap.bias_pct:+.2f}%)", flush=True)
    pd.DataFrame(trows).to_csv(FRESH / "fresh_transistor_metrics.csv", index=False, float_format="%.4g")
    pd.DataFrame(rows).to_csv(FRESH / "fresh_summary.csv", index=False, float_format="%.4g")
    print("FRESH CHECK DONE", flush=True)


def main():
    t0 = time.time()
    transistor_data()
    cor = corners()
    gate_spice(cor)
    parity_spice(cor)
    evaluate(cor)
    print(f"fresh check total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

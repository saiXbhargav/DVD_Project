"""exp_acc round 2, evaluation. Official cut rule only (round 1: the cut choice changes < 0.1 %).

Model sets (per case):
  GPBag87        round-1 best (all four cases)
  GPX            GPBag87, ON cases replaced by GPBag87X (plain-voltage features)
  MLP87          MLP ensembles, all four cases
  GPX+MLP        average of GPX and MLP87 currents, all four cases
  GPX|ON:MLP     GPX, ON cases replaced by MLP87
  GPX|ON:blend   GPX, ON cases replaced by the GPX/MLP87 average
Chosen on the 100 validation corners (gate accuracy); reported on the official corners.
Writes exp_acc/out/round2_{summary,transistor_metrics}.csv and round2_gate_results_<set>.csv.
"""
from itertools import product

import joblib
import pandas as pd

from common import DATA, OUT, C, G, gate_leakage, summary
from common2 import T, Blend

ON = ("nmos_on", "pmos_on")


def ld(name, case):
    return joblib.load(OUT / "models" / f"{name}_{case}.joblib")


def gates(sp, corners, models):
    parts = []
    for (gate, pat), d in sp.groupby(["gate", "pattern"], sort=False):
        d = d.sort_values("corner").copy()
        cs = corners.loc[d.corner.to_numpy()]
        node_v = {n: d[f"v_{n}"].to_numpy() for n in G.GATES[gate]["nodes"]}
        d["leak_ml"] = gate_leakage(gate, pat, cs, node_v, models, "rule")
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    base = {c: ld("GPBag87", c) for c in C.CASES}
    gpx = {c: ld("GPBag87X", c) if c in ON else base[c] for c in C.CASES}
    mlp = {c: ld("MLP87", c) for c in C.CASES}
    blend = {c: Blend([gpx[c], mlp[c]]) for c in C.CASES}
    sets = {"GPBag87": base, "GPX": gpx, "MLP87": mlp, "GPX+MLP": blend,
            "GPX|ON:MLP": {c: mlp[c] if c in ON else gpx[c] for c in C.CASES},
            "GPX|ON:blend": {c: blend[c] if c in ON else gpx[c] for c in C.CASES}}

    rows = []
    for case in C.CASES:
        _, va, te = T.split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
        for name, m in (("GPBag87", base[case]), ("GPBag87X", gpx[case]), ("MLP87", mlp[case]), ("GPX+MLP", blend[case])):
            for split, d in (("val", va), ("test", te)):
                r = T.metrics(m.predict(d), d[C.CURRENTS].to_numpy())
                r.update(case=case, model=name, split=split)
                rows.append(r)
            print(f"{case:9s} {name:9s} val mean-median {rows[-2]['mean_med%']:.3f}%  test {rows[-1]['mean_med%']:.3f}%",
                  flush=True)
    pd.DataFrame(rows).to_csv(OUT / "round2_transistor_metrics.csv", index=False, float_format="%.4g")

    vcor = pd.read_csv(DATA / "val_corners.csv").set_index("corner", drop=False)
    vsp = pd.read_csv(OUT / "gate_spice_val.csv", dtype={"pattern": str})
    tcor = G.load_corners().set_index("corner", drop=False)
    tsp = pd.read_csv(C.RESULTS_DIR / "gate_spice.csv", dtype={"pattern": str})
    csp = pd.read_csv(C.RESULTS_DIR / "circuit_spice.csv", dtype={"vector": str})
    vectors = ["".join(v) for v in product("01", repeat=8)]
    cnt = pd.DataFrame({v: pd.Series([p for _, p in G.parity_logic(v)[1]]).value_counts()
                        .reindex(["00", "01", "10", "11"], fill_value=0) for v in vectors}).T
    res = []
    for s, models in sets.items():
        gv, gt = gates(vsp, vcor, models), gates(tsp, tcor, models)
        tag = s.replace("|", "_").replace(":", "-").replace("+", "_")
        summary(gt, ["gate", "pattern"]).merge(summary(gv, ["gate", "pattern"]), on=["gate", "pattern"],
                                              suffixes=("_test", "_val")).to_csv(
            OUT / f"round2_gate_results_{tag}.csv", index=False, float_format="%.4g")
        nand = gt[gt.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_ml")[["00", "01", "10", "11"]]
        ml = pd.DataFrame(nand.to_numpy() @ cnt.to_numpy().T, index=nand.index, columns=cnt.index)
        ml = ml.stack().rename("leak_ml").reset_index()
        ml.columns = ["corner", "vector", "leak_ml"]
        p = csp.merge(ml, on=["corner", "vector"])
        a_v = summary(gv.assign(all=1), ["all"]).iloc[0]
        a_t = summary(gt.assign(all=1), ["all"]).iloc[0]
        a_p = summary(p.assign(all=1), ["all"]).iloc[0]
        nor = summary(gt[(gt.gate == "NOR2") & (gt.pattern == "11")].assign(all=1), ["all"]).iloc[0]
        res.append(dict(set=s, val_gate_acc=a_v.accuracy_pct, val_gate_median=a_v.median_err_pct,
                        gate_acc=a_t.accuracy_pct, gate_median=a_t.median_err_pct, gate_p95=a_t.p95_err_pct,
                        gate_bias=a_t.bias_pct, nor2_11_acc=nor.accuracy_pct, parity_acc=a_p.accuracy_pct,
                        parity_median=a_p.median_err_pct, parity_p95=a_p.p95_err_pct, parity_bias=a_p.bias_pct))
        print(f"{s:13s} VAL gates {a_v.accuracy_pct:.2f}% | TEST gates {a_t.accuracy_pct:.2f}% med {a_t.median_err_pct:.2f}% "
              f"p95 {a_t.p95_err_pct:.2f}% NOR2-11 {nor.accuracy_pct:.2f}% | parity {a_p.accuracy_pct:.2f}% "
              f"med {a_p.median_err_pct:.2f}% bias {a_p.bias_pct:+.2f}%", flush=True)
    out = pd.DataFrame(res)
    out.to_csv(OUT / "round2_summary.csv", index=False, float_format="%.4g")
    best = out.loc[out.val_gate_acc.idxmax()]
    print(f"\nchosen on validation corners: {best['set']} -> official corners: gates {best.gate_acc:.2f}%, "
          f"parity {best.parity_acc:.2f}%")
    print("ROUND2 DONE", flush=True)


if __name__ == "__main__":
    main()

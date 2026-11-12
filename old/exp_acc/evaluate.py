"""exp_acc step 4: score every (transistor-model set x cut rule) option.

Model sets: GP_7k (official), GP_cap (exp_extra), GPBag47, GPBag87.
Cut rules : rule (official), flip, avg (see common.gate_leakage).
  1. transistor metrics on the original validation and test splits
  2. gates on the 100 VALIDATION corners (exp_acc/out/gate_spice_val.csv)
  3. gates + parity checker on the official 200 corners (results/*.csv; test only)
The option with the best gate accuracy on the validation corners is the one to adopt;
the official-corner numbers are reported for every option for transparency.
Writes exp_acc/out/{transistor_metrics,summary}.csv and gate_results_<option>.csv.
"""
import sys
from importlib import import_module
from itertools import product

import joblib
import numpy as np
import pandas as pd

from common import DATA, HERE, OUT, C, G, GPCap, gate_leakage, summary  # noqa: F401  (GPCap: unpickling)

sys.modules["__main__"].GPCap = GPCap          # exp_extra models were pickled from __main__
T = import_module("03_train_models")
SETS = ["GP_7k", "GP_cap", "GPBag47", "GPBag87"]
MODES = ["rule", "flip", "avg"]


def load(name, case):
    if name == "GP_7k":
        return joblib.load(C.MODEL_DIR / f"{case}.joblib")
    if name == "GP_cap":
        return joblib.load(HERE.parent / "exp_extra" / "out" / "models" / f"GP_cap_{case}.joblib")
    return joblib.load(OUT / "models" / f"{name}_{case}.joblib")


def gates(sp, corners, models, mode):
    parts = []
    for (gate, pat), d in sp.groupby(["gate", "pattern"], sort=False):
        d = d.sort_values("corner").copy()
        cs = corners.loc[d.corner.to_numpy()]
        node_v = {n: d[f"v_{n}"].to_numpy() for n in G.GATES[gate]["nodes"]}
        d["leak_ml"] = gate_leakage(gate, pat, cs, node_v, models, mode)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    models = {s: {c: load(s, c) for c in C.CASES} for s in SETS}
    rows = []
    for case in C.CASES:
        _, va, te = T.split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
        for s in SETS:
            for split, d in (("val", va), ("test", te)):
                r = T.metrics(models[s][case].predict(d), d[C.CURRENTS].to_numpy())
                r.update(case=case, model=s, split=split)
                rows.append(r)
            print(f"{case:9s} {s:8s} val mean-median {rows[-2]['mean_med%']:.3f}%  test {rows[-1]['mean_med%']:.3f}%",
                  flush=True)
    pd.DataFrame(rows).to_csv(OUT / "transistor_metrics.csv", index=False, float_format="%.4g")

    vcor = pd.read_csv(DATA / "val_corners.csv").set_index("corner", drop=False)
    vsp = pd.read_csv(OUT / "gate_spice_val.csv", dtype={"pattern": str})
    tcor = G.load_corners().set_index("corner", drop=False)
    tsp = pd.read_csv(C.RESULTS_DIR / "gate_spice.csv", dtype={"pattern": str})
    csp = pd.read_csv(C.RESULTS_DIR / "circuit_spice.csv", dtype={"vector": str})
    vectors = ["".join(v) for v in product("01", repeat=8)]
    cnt = pd.DataFrame({v: pd.Series([p for _, p in G.parity_logic(v)[1]]).value_counts()
                        .reindex(["00", "01", "10", "11"], fill_value=0) for v in vectors}).T
    res = []
    for s, mode in product(SETS, MODES):
        gv = gates(vsp, vcor, models[s], mode)
        gt = gates(tsp, tcor, models[s], mode)
        tag = f"{s}_{mode}"
        summary(gt, ["gate", "pattern"]).merge(summary(gv, ["gate", "pattern"]), on=["gate", "pattern"],
                                              suffixes=("_test", "_val")).to_csv(
            OUT / f"gate_results_{tag}.csv", index=False, float_format="%.4g")
        nand = gt[gt.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_ml")[["00", "01", "10", "11"]]
        ml = pd.DataFrame(nand.to_numpy() @ cnt.to_numpy().T, index=nand.index, columns=cnt.index)
        ml = ml.stack().rename("leak_ml").reset_index()
        ml.columns = ["corner", "vector", "leak_ml"]
        p = csp.merge(ml, on=["corner", "vector"])
        a_v = summary(gv.assign(all=1), ["all"]).iloc[0]
        a_t = summary(gt.assign(all=1), ["all"]).iloc[0]
        a_p = summary(p.assign(all=1), ["all"]).iloc[0]
        res.append(dict(option=tag, models=s, cut=mode,
                        val_gate_acc=a_v.accuracy_pct, val_gate_median=a_v.median_err_pct, val_gate_p95=a_v.p95_err_pct,
                        gate_acc=a_t.accuracy_pct, gate_median=a_t.median_err_pct, gate_p95=a_t.p95_err_pct,
                        gate_bias=a_t.bias_pct, parity_acc=a_p.accuracy_pct, parity_median=a_p.median_err_pct,
                        parity_p95=a_p.p95_err_pct, parity_bias=a_p.bias_pct))
        print(f"{tag:14s} VAL gates acc {a_v.accuracy_pct:.2f}% | TEST gates acc {a_t.accuracy_pct:.2f}% "
              f"med {a_t.median_err_pct:.2f}% p95 {a_t.p95_err_pct:.2f}% | parity acc {a_p.accuracy_pct:.2f}% "
              f"med {a_p.median_err_pct:.2f}% bias {a_p.bias_pct:+.2f}%", flush=True)
    out = pd.DataFrame(res)
    out.to_csv(OUT / "summary.csv", index=False, float_format="%.4g")
    best = out.loc[out.val_gate_acc.idxmax()]
    print(f"\nchosen on validation corners: {best.option} -> official corners: gates {best.gate_acc:.2f}%, "
          f"parity {best.parity_acc:.2f}%")
    print("EXP_ACC DONE", flush=True)


if __name__ == "__main__":
    main()
